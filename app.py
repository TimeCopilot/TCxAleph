"""
Segment ETA Forecast app.

Replicates the notebook pipeline (TC_implementation.ipynb) inside Streamlit.
Data: data_valid_trips.csv (preloaded). Required columns: trip_id, real_departure_origin, eta_error_minutes.
Segment series: same logic as notebook — filter by trip_id, signed-log, resample(window).median(),
  fill missing buckets with weekday x slot median, then global median. Output: unique_id, ds, y (ds timezone-naive).
Forecast: AsyncTimeCopilot.analyze(); display fcst_df, result.output, and predictions reverted to ETA error minutes.
Performance: build_even_series_for_segment is cached by (csv_path, trip_id, window) only so the full DataFrame is never hashed (hashing it was causing 10+ min slowdown).
"""

import asyncio
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from timecopilot.agent import AsyncTimeCopilot


st.set_page_config(page_title="Segment ETA Forecast", layout="wide")
st.title("Segment ETA Forecast")

if not os.getenv("OPENAI_API_KEY", "").strip():
    st.error("OPENAI_API_KEY is missing.")
    st.stop()


REQUIRED_COLS = ["trip_id", "real_departure_origin", "eta_error_minutes"]


@st.cache_data(show_spinner=True)
def load_valid_trips(csv_path: str) -> pd.DataFrame:
    """Load data_valid_trips.csv. Validate required columns; parse real_departure_origin once."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"data_valid_trips.csv is missing required columns: {missing}. "
            "Need: trip_id, real_departure_origin, eta_error_minutes."
        )
    df["real_departure_origin"] = pd.to_datetime(df["real_departure_origin"], utc=True, errors="coerce")
    df["eta_error_minutes"] = pd.to_numeric(df["eta_error_minutes"], errors="coerce")
    df = df.dropna(subset=["real_departure_origin", "eta_error_minutes"]).copy()
    return df


def run_coro(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run async coroutine from Streamlit (no running loop: asyncio.run)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return asyncio.create_task(coro)


def parse_window_hours(window: str) -> int:
    w = window.strip().upper()
    if not w.endswith("H"):
        raise ValueError("Window must end with 'H' (e.g. 6H).")
    hours = int(w[:-1])
    if 24 % hours != 0:
        raise ValueError("Window must divide 24 (e.g. 3H, 4H, 6H, 12H).")
    return hours


@st.cache_data(show_spinner=False)
def build_even_series_for_segment(csv_path: str, trip_id: str, window: str = "6H") -> pd.DataFrame:
    """
    Notebook logic: filter trip_id, datetime UTC, drop missing, sort, set index.
    signed_log_error = sign(e) * log1p(abs(e)). Resample(window).median().
    Fill missing: weekday + slot (index.hour // hours), baseline median by (weekday, slot), then overall median.
    Output: unique_id, ds (timezone-naive), y.
    Cache key is (csv_path, trip_id, window) only — avoids hashing the full DataFrame (was causing 10+ min slowdown).
    """
    base_df = load_valid_trips(csv_path)
    seg = base_df[base_df["trip_id"] == trip_id].copy()
    if seg.empty:
        raise ValueError("trip_id not found in valid data.")

    seg["real_departure_origin"] = pd.to_datetime(seg["real_departure_origin"], utc=True, errors="coerce")
    seg = (
        seg.dropna(subset=["real_departure_origin"])
        .sort_values("real_departure_origin")
        .set_index("real_departure_origin")
    )
    if seg.empty:
        raise ValueError("Segment has no valid real_departure_origin after cleaning.")

    e = seg["eta_error_minutes"].astype(float)
    seg["signed_log_error"] = np.sign(e) * np.log1p(np.abs(e))

    y = seg["signed_log_error"].resample(window).median()
    missing = y.isna()

    hours = parse_window_hours(window)
    tmp = pd.DataFrame({"y": y})
    tmp["weekday"] = tmp.index.dayofweek
    tmp["slot"] = (tmp.index.hour // hours).astype(int)

    slot_median = tmp.loc[~missing].groupby(["weekday", "slot"])["y"].median()
    y_filled = y.copy()
    fill_vals = tmp.loc[missing, ["weekday", "slot"]].apply(
        lambda r: slot_median.get((int(r["weekday"]), int(r["slot"]))),
        axis=1,
    )
    y_filled.loc[missing] = fill_vals.values
    y_filled = y_filled.fillna(y.median())

    out = pd.DataFrame({"unique_id": trip_id, "ds": y_filled.index, "y": y_filled.values})
    out["ds"] = pd.to_datetime(out["ds"]).dt.tz_localize(None)
    return out


def empty_bucket_ratio_before_fill(_base_df: pd.DataFrame, trip_id: str, window: str) -> float:
    """From raw segment: counts = resample(window).size(); ratio of zero-count buckets."""
    seg = _base_df[_base_df["trip_id"] == trip_id].copy()
    if seg.empty:
        return 1.0
    seg = seg.set_index("real_departure_origin").sort_index()
    counts = seg.resample(window).size()
    return float((counts == 0).mean()) if len(counts) else 1.0


def inv_signed_log(x: pd.Series | np.ndarray) -> np.ndarray:
    """Inverse signed-log: eta_error_minutes = sign(x) * (exp(abs(x)) - 1)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.expm1(np.abs(x))


def add_eta_error_scale(fcst_df: pd.DataFrame) -> pd.DataFrame:
    """Add eta_error_minutes_pred and, if present, eta_error_minutes_lower/upper."""
    out = fcst_df.copy()
    pred_col = None
    for c in ["y_hat", "yhat", "mean", "forecast", "y_pred"]:
        if c in out.columns:
            pred_col = c
            break
    if pred_col is not None:
        out["eta_error_minutes_pred"] = inv_signed_log(out[pred_col])
    lo_col = next((c for c in ["y_hat_lower", "yhat_lower", "lo", "lower", "y_lower"] if c in out.columns), None)
    hi_col = next((c for c in ["y_hat_upper", "yhat_upper", "hi", "upper", "y_upper"] if c in out.columns), None)
    if lo_col is not None:
        out["eta_error_minutes_lower"] = inv_signed_log(out[lo_col])
    if hi_col is not None:
        out["eta_error_minutes_upper"] = inv_signed_log(out[hi_col])
    return out


@st.cache_resource
def get_tc() -> AsyncTimeCopilot:
    return AsyncTimeCopilot(llm="openai:gpt-4o-mini", retries=3)


if "tc" not in st.session_state:
    st.session_state.tc = get_tc()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "base_df" not in st.session_state:
    st.session_state.base_df = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# Preloaded dataset path
default_csv = str(Path(__file__).resolve().parent / "data_valid_trips.csv")

try:
    base_df = load_valid_trips(default_csv)
    st.session_state.base_df = base_df
    st.caption(f"Loaded: {len(base_df):,} rows, {base_df['trip_id'].nunique():,} segments")
except Exception as e:
    st.error(f"Failed to load data_valid_trips.csv: {e}")
    st.stop()

trip_id_input = st.text_input("trip_id", value="")
col_w, col_h = st.columns(2)
with col_w:
    window = st.selectbox("window", ["3H", "4H", "6H", "12H"], index=2)
with col_h:
    h = st.number_input("h (forecast horizon points)", min_value=1, max_value=500, value=20, step=1)
query = st.text_input("query (optional)", value="Forecast the next points and summarize.")

if st.button("Run", type="primary"):
    if not trip_id_input.strip():
        st.error("Enter a trip_id.")
        st.stop()

    try:
        final_df = build_even_series_for_segment(default_csv, trip_id_input.strip(), window=window)

        st.subheader("Final dataset (first 10 rows)")
        st.dataframe(final_df[["unique_id", "ds", "y"]].head(10), use_container_width=True)

        empty_ratio = empty_bucket_ratio_before_fill(base_df, trip_id_input.strip(), window)
        st.write("Empty bucket ratio (before fill):", f"{empty_ratio:.3f}")

        tc = st.session_state.tc
        with st.spinner("Running analysis and forecast..."):
            result = run_coro(
                tc.analyze(
                    df=final_df,
                    freq=window,
                    h=int(h),
                    query=query.strip() or "Forecast and summarize.",
                )
            )

        st.session_state.analyzed = True

        fcst_df = getattr(result, "fcst_df", None)
        if not (isinstance(fcst_df, pd.DataFrame) and not fcst_df.empty):
            fcst_df = getattr(tc, "fcst_df", None)

        # Raw result.fcst_df (same as notebook)
        st.subheader("result.fcst_df")
        if isinstance(fcst_df, pd.DataFrame) and not fcst_df.empty:
            st.dataframe(fcst_df.head(50), use_container_width=True)
        else:
            st.write("No forecast dataframe in result.")

        if isinstance(fcst_df, pd.DataFrame) and not fcst_df.empty:
            fcst_df = add_eta_error_scale(fcst_df)

        # st.subheader("Forecast dataframe (first 50 rows, with ETA scale)")
        # st.dataframe(fcst_df.head(50), use_container_width=True)

        # st.subheader("Reverted ETA error (minutes)")
        # rev_cols = [c for c in ["ds", "eta_error_minutes_pred", "eta_error_minutes_lower", "eta_error_minutes_upper"] if c in fcst_df.columns]
        # if rev_cols:
        #     st.dataframe(fcst_df[rev_cols].head(50), use_container_width=True)
        # else:
        #     st.write("No reverted columns (check forecast output columns).")

        out = getattr(result, "output", None)
        if out is not None:
            st.subheader("Analysis summary")
            st.write(out)

    except Exception as e:
        st.session_state.analyzed = False
        st.exception(e)
        st.stop()

st.subheader("Forecast discussion")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about the forecast or data")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    tc = st.session_state.tc
    with st.chat_message("assistant"):
        placeholder = st.empty()
        if not st.session_state.analyzed or not tc.is_queryable():
            msg = "Run the forecast first, then you can ask questions."
            placeholder.markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
        else:
            async def get_response() -> str:
                async with tc.query_stream(user_input) as result_stream:
                    final = ""
                    async for text in result_stream.stream(debounce_by=0.02):
                        final = text
                        placeholder.markdown(text)
                return final or "(No response.)"

            try:
                answer = run_coro(get_response())
            except Exception as e:
                st.exception(e)
                answer = f"Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
