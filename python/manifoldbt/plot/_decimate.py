"""Min/max time-series decimation for plotting — pure numpy.

A chart is ~1000-2500 px wide, so plotting 10^5-10^6 samples draws hundreds of
sub-pixel points per column and bloats saved HTML. Per pixel column we keep the
bucket's min and max in time order, which preserves peaks and troughs exactly
(max drawdown survives untouched) at O(n) cost (~1 ms for 1M points).
"""
from __future__ import annotations

import numpy as np

#: Series shorter than this are plotted as-is.
DECIMATE_THRESHOLD = 20_000


def decimate_minmax(x: np.ndarray, y: np.ndarray, n_cols: int = 2500):
    """Per-column min/max envelope. Returns (x, y) unchanged when small."""
    n = len(y)
    if n <= 2 * n_cols:
        return x, y
    bucket = n // n_cols
    m = bucket * n_cols
    yb = y[:m].reshape(n_cols, bucket)
    cols = np.arange(n_cols)
    idx_min = yb.argmin(axis=1) + cols * bucket
    idx_max = yb.argmax(axis=1) + cols * bucket
    lo = np.minimum(idx_min, idx_max)
    hi = np.maximum(idx_min, idx_max)
    idx = np.empty(n_cols * 2, dtype=np.int64)
    idx[0::2] = lo
    idx[1::2] = hi
    if m < n:
        idx = np.append(idx, n - 1)  # keep the true last sample
    idx = np.unique(idx)  # dedupe flat buckets (lo == hi)
    return x[idx], y[idx]


def maybe_decimate(x: np.ndarray, y: np.ndarray, n_cols: int = 2500):
    """Decimate only when the series is longer than DECIMATE_THRESHOLD."""
    if len(y) <= DECIMATE_THRESHOLD:
        return x, y
    return decimate_minmax(x, y, n_cols)
