"""Multi-Timeframe Strategy -- trend on 12h, entry on 1h.

Demonstrates:
  - bt.tf() for referencing higher-timeframe columns
  - extra_timeframes config to inject resampled OHLCV
  - Combining slow trend filter (12h EMA) with faster entry (1h RSI)

Logic:
  - 12h trend: EMA(20) > EMA(50) → bullish regime
  - 1h entry:  RSI(14) < 35 during bullish regime → buy the dip
  - Size: 50% of initial capital when conditions met, else flat

Usage:
    python examples/14_multi_timeframe.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import ema, rsi, close
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Higher timeframe references ---------------------------------------------
h12 = mbt.tf("12h")  # references columns like "12h.close"

# -- Indicators ---------------------------------------------------------------
# Trend filter on 12-hour bars (forward-filled onto 1h grid)
trend_fast = ema(h12.close, 20)
trend_slow = ema(h12.close, 50)
bullish = trend_fast > trend_slow

# Entry signal on 1-hour bars (native resolution)
entry_rsi = rsi(close, 14)
dip = entry_rsi < 35.0

# -- Strategy -----------------------------------------------------------------
strategy = (
    mbt.Strategy.create("multi_tf_trend_dip")
    .signal("bullish", bullish)
    .signal("entry_rsi", entry_rsi)
    .signal("dip", dip)
    .size(mbt.when(mbt.col("bullish") & mbt.col("dip"), 0.5, 0.0))
    .stop_loss(pct=3.0)
    .describe("12h EMA trend + 1h RSI dip-buy, 3% stop-loss")
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
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=50,
    extra_timeframes={
        "12h": Interval.hours(12),
    },
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..")
    data_root = os.path.abspath(os.path.join(root, "data"))
    store = mbt.DataStore(
        data_root=data_root,
        metadata_db=os.path.abspath(os.path.join(root, "metadata", "metadata.sqlite")),
        arrow_dir=os.path.join(data_root, "mega"),
    )

    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    elapsed = time.perf_counter() - t0
    

    print(result.summary())
    print(f"\nElapsed: {elapsed:.3f}s")

    mbt.plot.equity(result)
