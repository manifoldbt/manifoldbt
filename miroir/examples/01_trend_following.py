"""Trend Following -- EMA crossover with stop-loss and dynamic sizing.

Demonstrates:
  - Fluent Strategy builder
  - EMA indicators
  - Conditional sizing with when()
  - Stop-loss via .stop_loss()
  - Diagnostics (lookahead, exposure stability, risk)
  - result.summary() rich output

Data: shared store — real market data from `data/` (see examples/README.md)

Usage:
    python examples/01_trend_following.py
"""
import time
import manifoldbt as mbt
from manifoldbt.indicators import ema, close, volume
from manifoldbt.helpers import time_range, Slippage, Interval

from _bootstrap import open_store, plots_available

# -- Indicators ---------------------------------------------------------------
fast = ema(close, 12)
slow = ema(close, 26)
trend = fast - slow                          # MACD-like spread
vol_ma = volume.rolling_mean(20)             # average volume filter

# -- Strategy -----------------------------------------------------------------
strategy = (
    mbt.Strategy.create("trend_following")
    .signal("fast", fast)
    .signal("slow", slow)
    .signal("trend", trend)
    .signal("vol_filter", volume > vol_ma)   # only trade on above-average volume
    .size(mbt.when((trend > 0.0) & (volume > vol_ma), 0.5, 0.0))
    .stop_loss(pct=3.0)
    .describe("EMA(12/26) crossover, volume filter, 3% stop-loss")
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(1),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        allow_short=False,
        max_position_pct=0.5,
        position_sizing_mode="FractionOfInitialCapital",
    ),
    output_resolution=Interval.hours(1),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=30,
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    store = open_store()


    # Backtest
    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\nElapsed: {elapsed:.3f}s")

    # Plot
    if plots_available():
        mbt.plot.summary(result)
