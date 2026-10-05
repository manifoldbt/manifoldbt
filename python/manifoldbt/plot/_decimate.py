"""Min/max time-series decimation for plotting - pure numpy.

A chart is ~1000-2500 px wide, so plotting 10^5-10^6 samples draws hundreds of
sub-pixel points per column and bloats saved HTML. Per pixel column we keep the
bucket's min and max in time order, which preserves peaks and troughs exactly
(max drawdown survives untouched) at O(n) cost (~1 ms for 1M points).

The bucket width is rounded UP, so the last bucket is the short one and every
sample falls inside a bucket. Rounding down instead left ``n % n_cols`` samples
past the last bucket, and those were dropped except the very last one: a series
of 29,927 points lost its final 2,427 samples, which the chart then drew as one
straight segment across a tenth of its width, hiding both the series maximum and
9.5% of its range.
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
    bucket = -(-n // n_cols)  # ceil, so the remainder stays inside the buckets
    m = bucket * n_cols
    if m > n:
        # Pad the short last bucket with the final value: any index the padding
        # wins is clipped back onto that same last sample below.
        y = np.concatenate([y, np.full(m - n, y[-1], dtype=y.dtype)])
    yb = y[:m].reshape(n_cols, bucket)
    cols = np.arange(n_cols)
    idx_min = yb.argmin(axis=1) + cols * bucket
    idx_max = yb.argmax(axis=1) + cols * bucket
    lo = np.minimum(idx_min, idx_max)
    hi = np.maximum(idx_min, idx_max)
    idx = np.empty(n_cols * 2, dtype=np.int64)
    idx[0::2] = lo
    idx[1::2] = hi
    idx = np.append(idx, n - 1)  # keep the true last sample
    idx = np.unique(np.clip(idx, 0, n - 1))  # dedupe flat buckets (lo == hi)
    return x[idx], y[idx]


def maybe_decimate(x: np.ndarray, y: np.ndarray, n_cols: int = 2500):
    """Decimate only when the series is longer than DECIMATE_THRESHOLD."""
    if len(y) <= DECIMATE_THRESHOLD:
        return x, y
    return decimate_minmax(x, y, n_cols)
