"""Smoke test — verifies the installed wheel works end-to-end."""
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval


def test_import():
    assert hasattr(mbt, "__version__")
    print(f"  version: {mbt.__version__}")


def test_strategy_build():
    fast = ema(close, 12)
    slow = ema(close, 26)
    signal = mbt.when(fast > slow, mbt.lit(1.0), mbt.lit(-1.0))

    strategy = (
        mbt.Strategy.create("smoke_test")
        .signal("fast", fast)
        .signal("slow", slow)
        .signal("signal", signal)
        .size(mbt.col("signal") * mbt.lit(0.25))
    )

    j = strategy.to_json()
    assert "smoke_test" in j
    print(f"  strategy JSON length: {len(j)}")


def test_backtest_run():
    fast = ema(close, 12)
    slow = ema(close, 26)
    signal = mbt.when(fast > slow, mbt.lit(1.0), mbt.lit(-1.0))

    strategy = (
        mbt.Strategy.create("smoke_test")
        .signal("fast", fast)
        .signal("slow", slow)
        .signal("signal", signal)
        .size(mbt.col("signal") * mbt.lit(0.25))
    )

    start, end = time_range("2024-01-01", "2025-01-01")
    config = mbt.BacktestConfig(
        universe=[1],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.hours(12),
        initial_capital=10_000,
        execution=mbt.ExecutionConfig(allow_short=True, max_position_pct=0.5),
        fees=mbt.FeeConfig.binance_perps(),
        slippage=Slippage.fixed_bps(2),
        warmup_bars=30,
    )

    import os
    root = os.path.join(os.path.dirname(__file__), "..")
    store = mbt.DataStore(
        data_root=os.path.abspath(os.path.join(root, "data")),
        metadata_db=os.path.abspath(os.path.join(root, "metadata", "metadata.sqlite")),
    )

    result = mbt.run(strategy, config, store)
    assert result is not None
    print(f"  result: {repr(result)}")


if __name__ == "__main__":
    tests = [test_import, test_strategy_build, test_backtest_run]
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {name}: {e}")
