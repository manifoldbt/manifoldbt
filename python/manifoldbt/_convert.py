"""Convert Arrow / RecordBatch data from BacktestResult to numpy arrays.

Lives here rather than under `plot/` even though the charts are its biggest
consumer. `plot/__init__.py` refuses to import without plotly, and importing
any module inside that package runs it -- which made `Result.equity_df()` and
the exposure diagnostics, neither of which draws anything, require the
plotting extra. Nothing in this file imports plotly, and nothing should.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa


def arrow_to_numpy(arr: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """Convert a PyArrow array to a numpy array, combining chunks if needed."""
    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy(zero_copy_only=False)
    return np.array(arr.to_pylist())


def _ts_to_int64(arr: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """Convert a PyArrow Timestamp column to int64 nanoseconds via Arrow cast."""
    import pyarrow as pa

    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()
    # Cast Timestamp → int64 inside Arrow (no Python Timestamp objects)
    if pa.types.is_timestamp(arr.type):
        return arr.cast(pa.int64()).to_numpy(zero_copy_only=False)
    raw = arr.to_numpy(zero_copy_only=False)
    if raw.dtype == np.int64 or raw.dtype.kind == "i":
        return raw
    # Fallback: already datetime64
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw.view(np.int64)
    return np.array(arr.to_pylist(), dtype="int64")


def timestamps_to_dates(arr: "pa.ChunkedArray | pa.Array") -> np.ndarray:
    """Convert Timestamp(ns, UTC) Arrow array to numpy datetime64[ns]."""
    ns = _ts_to_int64(arr)
    return ns.view("datetime64[ns]")


def equity_with_dates(result) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (dates, equity_values) from a BacktestResult.

    The positions RecordBatch has one row per (timestamp, symbol). Equity is
    portfolio-level (same value across symbols at a given timestamp), so we
    deduplicate on timestamp.
    """
    positions = result.positions
    ts_col = positions.column("timestamp")
    eq_col = positions.column("equity")

    eq_raw = arrow_to_numpy(eq_col)

    # Get timestamps as int64 nanoseconds for deduplication
    ts_ns = _ts_to_int64(ts_col)

    _, unique_idx = np.unique(ts_ns, return_index=True)
    unique_idx.sort()

    dates = ts_ns[unique_idx].view("datetime64[ns]")
    values = eq_raw[unique_idx].astype(np.float64)
    return dates, values


def daily_returns_array(result) -> np.ndarray:
    """Extract daily_returns as a numpy float64 array."""
    return arrow_to_numpy(result.daily_returns).astype(np.float64)


def positions_arrays(result) -> dict:
    """Extract positions RecordBatch as a dict of numpy arrays.

    Returns dict with keys: timestamp, symbol_id, position, close, capital, equity.
    """
    positions = result.positions
    out = {}
    for name in positions.schema.names:
        col = positions.column(name)
        if name == "timestamp":
            out[name] = timestamps_to_dates(col)
        else:
            out[name] = arrow_to_numpy(col)
    return out


def trades_arrays(result) -> dict:
    """Extract trades RecordBatch as a dict of numpy arrays.

    Returns dict with keys matching the trades schema.
    """
    trades = result.trades
    out = {}
    for name in trades.schema.names:
        col = trades.column(name)
        if "timestamp" in name:
            out[name] = timestamps_to_dates(col)
        else:
            out[name] = arrow_to_numpy(col)
    return out


def run_currency(result) -> str:
    """Currency the run is denominated in, from its manifest.

    The manifest embeds the full BacktestConfig, so nothing is guessed. "USD"
    is only the last resort for a result that has no manifest at all (mock
    objects in tests).
    """
    try:
        code = result.manifest["config"]["currency"]
        return str(code) if code else "USD"
    except Exception:
        return "USD"


_CURRENCY_PREFIX = {"USD": "$", "EUR": "€", "GBP": "£"}


def money_hovertemplate(values: np.ndarray, currency: str) -> str:
    """Hover template for a money series, currency- and magnitude-aware.

    The old template was a hardcoded "$%{y:,.0f}": a 10-BTC equity hovered as
    "$10" - wrong currency, and a precision that erased every variation the
    chart existed to show. Decimals follow the magnitude of the series, and
    the currency is written as a symbol when it has one, as a suffix code
    (10.0443 BTC) when it does not.
    """
    peak = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    decimals = 0 if peak >= 10_000 else 2 if peak >= 100 else 4
    code = (currency or "USD").upper()
    amount = "%{y:,." + str(decimals) + "f}"
    prefix = _CURRENCY_PREFIX.get(code)
    amount = prefix + amount if prefix else amount + " " + code
    return "%{x|%d %b %Y}   " + amount + "<extra></extra>"


def date_tickformat(dates: np.ndarray) -> str:
    """Date-axis tick format adapted to the span of the series.

    Hardcoding "%b %Y" labelled every tick of a two-month backtest "May 2025"
    (and every tick of a 30-day synthetic run "Jan 2024"): the format must
    follow the span, not assume it. Thresholds are where the coarser format
    stops producing distinct labels for ~6 ticks.
    """
    if len(dates) < 2:
        return "%b %Y"
    span_days = float(
        (np.datetime64(dates[-1], "ns") - np.datetime64(dates[0], "ns"))
        / np.timedelta64(1, "D")
    )
    if span_days <= 3:
        return "%d %b %H:%M"
    if span_days <= 180:
        return "%d %b"
    # Beyond ~6 months the historical "%b %Y" is already distinct per tick,
    # whatever the span: plotly widens the tick spacing with the range. Only
    # the short end was ever broken.
    return "%b %Y"


def percent_tickformat(magnitude: float) -> str:
    """Percent-axis tick format adapted to the magnitude of the series.

    The date-axis disease, on the value axis: ".0%" labelled every tick of a
    -0.9% max-drawdown chart "0%". Decimals follow the extreme value, so the
    ticks always spell out distinct numbers.
    """
    m = abs(float(magnitude))
    if m >= 0.05:
        return ".0%"
    if m >= 0.005:
        return ".1%"
    return ".2%"
