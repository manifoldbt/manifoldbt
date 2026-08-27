"""Diagnostics -- look-ahead bias detection and exposure stability checks.

Demonstrates:
  - detect_lookahead(): split-test for look-ahead bias
  - check_exposure_stability(): verify positions are consistent across time windows
  - risk_check(): post-run risk metrics validation

Data: shared store — real market data from `data/` (see examples/README.md)

Usage:
    python examples/12_diagnostics.py
"""
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

from _bootstrap import open_store

# -- Strategy -----------------------------------------------------------------
fast = ema(close, 12)
slow = ema(close, 50)

signal = mbt.when(fast > slow, 0.5, 0.0)

strategy = (
    mbt.Strategy.create("ema_trend")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(signal)
    .stop_loss(pct=3.0)
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(12),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        allow_short=False,
        max_position_pct=0.5,
    ),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    store = open_store()

    # -- 1. Look-ahead bias detection -----------------------------------------
    # Splits the time range and compares trades from shorter runs against
    # the full run. If trades differ, the strategy uses future data.
    print("1. Look-ahead bias detection")
    print("-" * 40)
    t0 = time.perf_counter()
    lookahead = mbt.diagnostics.detect_lookahead(strategy, config, store)
    print(lookahead)
    print(f"   Elapsed: {time.perf_counter() - t0:.2f}s\n")

    # -- 2. Exposure stability -------------------------------------------------
    # Verifies that utilization and per-symbol exposure are identical
    # across different time windows. Catches position sizing that leaks
    # future data (e.g. z-score over the entire series).
    print("2. Exposure stability")
    print("-" * 40)
    t0 = time.perf_counter()
    stability = mbt.diagnostics.check_exposure_stability(strategy, config, store)
    print(stability)
    print(f"   Elapsed: {time.perf_counter() - t0:.2f}s\n")

    # -- 3. Backtest + risk check ----------------------------------------------
    # Run the strategy, then validate risk metrics against thresholds.
    print("3. Backtest + risk check")
    print("-" * 40)
    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    print(result.summary())
    print(f"   Elapsed: {time.perf_counter() - t0:.2f}s\n")

    risk = mbt.diagnostics.risk_check(result)
    print("Risk check:")
    print(risk)
