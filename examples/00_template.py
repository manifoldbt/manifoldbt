"""Strategy template — copy this file and modify.

Usage:
    python examples/00_template.py
"""

import os
from time import perf_counter
import manifoldbt as mbt
from manifoldbt.indicators import close
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Indicators ---------------------------------------------------------------
# All 45+ indicators available: rsi, ema, sma, bollinger, macd, atr, etc.
# See: from manifoldbt.indicators import <tab> for full list

zscore = close.zscore(60)

# -- Strategy -----------------------------------------------------------------
# mbt.when(condition, value_if_true, value_if_false)
#   - Omit 3rd arg → hold current position
#   - Nest mbt.when() for multiple conditions
#
# Examples:
#   signal = mbt.when(rsi < 30, 0.5, mbt.when(rsi > 70, 0.0))
#   signal = mbt.when(fast_ema > slow_ema, 1.0, -1.0)

signal = mbt.when(zscore < -1.0, 1.0,       # oversold → long
         mbt.when(zscore > 1.0, 0.0))        # overbought → exit, else hold

strategy = (
    mbt.Strategy.create("my_strategy")
    .signal("zscore", zscore)
    .size(signal)
    .describe("Z-score mean reversion")
    # .stop_loss(pct=3.0)
    # .take_profit(pct=5.0)
    # .trailing_stop(pct=2.0)
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2026-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.minutes(1),      # bar resolution
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        allow_short=False,
        max_position_pct=1.0,
    ),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,
    output_resolution=Interval.hours(1),   # Pro: sub-daily, Community: capped to daily
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

    t0 = perf_counter()
    result = mbt.run(strategy, config, store)
    print(result.summary())
    print(f"\nElapsed: {perf_counter() - t0:.2f}s")

    mbt.plot.tearsheet(result)
