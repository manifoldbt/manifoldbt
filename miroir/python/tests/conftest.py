import glob
import os

import pytest

_CRATE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The 1-minute fixtures sit with the engine's own tests, one folder beside this
# package in the source repository; absent anywhere else.
_GOLDEN = sorted(glob.glob(os.path.join(_CRATE_ROOT, "..", "*", "tests", "fixtures", "golden")))
GOLDEN_ROOT = _GOLDEN[0] if _GOLDEN else None


@pytest.fixture
def golden_buy_hold_dir():
    """Path to the buy_and_hold golden fixture directory.

    Skips instead of failing when the directory is absent: the 1-minute fixtures
    are part of the engine's own test data and are not distributed with the
    package, so a clone that only has the Python suite should report "not
    applicable" rather than an error it cannot act on. The golden test that runs
    anywhere is `test_golden_daily.py`, whose fixtures live next to it.
    """
    path = os.path.join(GOLDEN_ROOT, "buy_and_hold", "v1") if GOLDEN_ROOT else ""
    if not path or not os.path.isdir(path):
        pytest.skip("1-minute golden fixtures are not distributed with the package")
    return path


def _same(a, b):
    """``a == b`` with NaN equal to NaN, recursively through dicts and lists."""
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture
def same_metrics():
    """Compare two metrics dicts value for value, a NaN matching a NaN.

    A run shorter than two days reports its Sharpe, CAGR and the other daily
    statistics as NaN, and ``nan == nan`` is False: two identical short runs
    would compare unequal with ``==``. Every other value is compared with
    ``==``, as before.
    """
    return _same
