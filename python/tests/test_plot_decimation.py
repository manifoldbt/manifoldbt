"""The plotting decimator must not drop the tail of a series.

Rounding the bucket width down left ``n % n_cols`` samples past the last bucket
with only the final one kept, so the chart drew one straight segment over them.
On a 29,927-point equity curve that segment covered 101 days, swallowed the
series maximum and 9.5% of its range, while the file size, the trace count and
the point count all looked normal. The checks below pin the two properties the
module promises: every peak survives, and no drawn segment spans much more than
one bucket.
"""
import numpy as np
import pytest

from manifoldbt.plot._decimate import (
    DECIMATE_THRESHOLD,
    decimate_minmax,
    maybe_decimate,
)

N_COLS = 2500

# n % n_cols != 0 is where the tail used to disappear; 7499 was the worst case
# (bucket = 2, a third of the series dropped).
RAGGED = [7499, 12345, 29927, 100_003]


def _series(n, seed=0):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0.0, 1.0, n)) + 1000.0
    return np.arange(n, dtype=np.int64), y


@pytest.mark.parametrize("n", RAGGED)
def test_extremes_survive(n):
    x, y = _series(n)
    _, yd = decimate_minmax(x, y, N_COLS)
    assert yd.min() == y.min()
    assert yd.max() == y.max()


@pytest.mark.parametrize("n", RAGGED)
def test_no_long_straight_segment(n):
    """No drawn segment may span more than a couple of buckets."""
    x, y = _series(n)
    xd, _ = decimate_minmax(x, y, N_COLS)
    bucket = -(-n // N_COLS)
    assert np.diff(xd).max() <= 2 * bucket


@pytest.mark.parametrize("n", RAGGED)
def test_endpoints_and_order(n):
    x, y = _series(n)
    xd, yd = decimate_minmax(x, y, N_COLS)
    assert xd[-1] == x[-1] and yd[-1] == y[-1]
    assert np.all(np.diff(xd) > 0)
    assert len(yd) <= 2 * N_COLS + 1


@pytest.mark.parametrize("n", RAGGED)
def test_span_covers_head_and_tail(n):
    """The kept indices must reach both ends: no head or tail cut off."""
    x, y = _series(n)
    xd, _ = decimate_minmax(x, y, N_COLS)
    bucket = -(-n // N_COLS)
    assert xd[0] <= bucket - 1
    assert xd[-1] == n - 1


def test_short_series_untouched():
    x, y = _series(2 * N_COLS)
    xd, yd = decimate_minmax(x, y, N_COLS)
    assert xd is x and yd is y


def test_threshold_gate():
    x, y = _series(DECIMATE_THRESHOLD)
    assert maybe_decimate(x, y)[1] is y
    x, y = _series(DECIMATE_THRESHOLD + 1)
    assert len(maybe_decimate(x, y)[1]) < DECIMATE_THRESHOLD + 1
