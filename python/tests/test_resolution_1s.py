"""One-second bars: Pro simulates and returns them, Community stops at the minute.

The store holds a one-second tier since 0.24.0. Two rules sit on top of it:

* simulating on bars finer than one minute needs Pro (``resolution_1s``),
  enforced by the engine core on every runner and answered early by the
  bindings;
* the output floor is one second on Pro and one day on Community, so on Pro a
  one-second run comes back with one point per bar.

The Pro cases run on a Pro machine and in the engine's own debug builds, which
is what the CI runs; the Community cases skip there, like ``test_combo_limit``.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt

IS_PRO = bt.license_info()[0] == "Pro"
N_BARS = 3 * 3600  # three hours of one-second bars


@pytest.fixture(scope="module")
def one_second_store():
    root = tempfile.mkdtemp()
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="1s", tz="UTC")
    close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.01, N_BARS))
    df = pd.DataFrame({
        "timestamp": idx, "open": close, "high": close + 0.02,
        "low": close - 0.02, "close": close, "volume": np.full(N_BARS, 10.0),
    })
    return bt.import_dataframe(
        df, symbol="XS", symbol_id=1, interval="1s",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )


def _strategy():
    return bt.Strategy(
        name="always_long",
        signals={"signal": bt.lit(1.0)},
        position_sizing=bt.lit(1.0) * bt.param("size", default=1.0),
        parameters={"size": bt.param("size", default=1.0)},
    )


def _config(interval):
    t0, t1 = bt.time_range("2024-01-01", "2024-01-02")
    return bt.BacktestConfig(
        universe=[1], time_range_start=t0, time_range_end=t1, bar_interval=interval,
    )


def test_one_second_bars_land_in_their_own_tier(one_second_store):
    root = one_second_store.data_root()
    tiers = {
        d for _, dirs, _ in os.walk(root) for d in dirs if d in ("1s", "1m", "bars_1s", "bars_1m")
    }
    assert "1s" in tiers or "bars_1s" in tiers, tiers
    # Nothing was silently rewritten as one-minute data.
    assert "1m" not in tiers and "bars_1m" not in tiers, tiers


@pytest.mark.skipif(not IS_PRO, reason="Pro (or the debug override): one-second simulation")
def test_pro_simulates_and_returns_every_second(one_second_store):
    result = bt.run(_strategy(), _config({"Seconds": 1}), one_second_store)
    # One point per bar: the output floor is one second, not one minute.
    assert len(result.equity_curve) == N_BARS


@pytest.mark.skipif(not IS_PRO, reason="Pro (or the debug override): one-second simulation")
def test_one_minute_run_on_the_same_store_is_coarser(one_second_store):
    result = bt.run(_strategy(), _config({"Minutes": 1}), one_second_store)
    assert len(result.equity_curve) == N_BARS // 60 + 1


@pytest.mark.skipif(IS_PRO, reason="Community-only; needs a machine without a Pro licence")
def test_community_is_refused_below_one_minute(one_second_store):
    with pytest.raises(PermissionError, match="Pro"):
        bt.run(_strategy(), _config({"Seconds": 1}), one_second_store)


@pytest.mark.skipif(IS_PRO, reason="Community-only; needs a machine without a Pro licence")
def test_community_still_runs_at_one_minute_on_a_one_second_store(one_second_store):
    result = bt.run(_strategy(), _config({"Minutes": 1}), one_second_store)
    # Daily output floor: three hours collapse onto a single day.
    assert len(result.equity_curve) <= 2
