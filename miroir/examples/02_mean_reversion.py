"""Mean Reversion -- EMA crossover long/short.

Demonstrates:
  - EMA crossover signal
  - Long and short positions
  - Continuous sizing (signal * 0.25)

Data: shared store — real market data from `data/` (see examples/README.md)

Usage:
    python examples/02_mean_reversion.py
"""
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

from _bootstrap import open_store, plots_available

# -- Indicators ---------------------------------------------------------------
fast = ema(close, 12)
slow = ema(close, 26)

# -- Strategy -----------------------------------------------------------------
signal = mbt.when(fast > slow, 1.0, -1.0)

strategy = (
    mbt.Strategy.create("ema_crossover")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(signal * 0.25)
    .describe("EMA 12/26 crossover")
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2026-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(12),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        allow_short=True,
        max_position_pct=0.5,
    ),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=30,
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    store = open_store()

    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\nElapsed: {elapsed:.3f}s")
    if plots_available():
        mbt.plot.summary(result)
