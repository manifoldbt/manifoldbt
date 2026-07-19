"""Multi-Asset Momentum -- relative strength across 5 assets.

Demonstrates:
  - Multi-asset universe (5 symbols)
  - Momentum via smoothed ROC on 12h bars
  - Volatility-adjusted sizing

Usage:
    python examples/03_multi_asset_momentum.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, ema, roc, high, low
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Indicators ---------------------------------------------------------------
mom = ema(roc(close, 14), 6)                       # 7-day momentum, smoothed
avg_range = (high - low).rolling_mean(14)
norm_vol = avg_range / (close + mbt.lit(1e-12))    # normalized volatility
safe_vol = mbt.when(norm_vol > 0.0005, norm_vol, 0.0005)

# -- Strategy -----------------------------------------------------------------
signal = mbt.when(mom > 0.0, mom / safe_vol, 0.0)

strategy = (
    mbt.Strategy.create("multi_momentum")
    .signal("momentum", mom)
    .signal("norm_vol", norm_vol)
    .size(signal * 0.01)
    .describe("Multi-asset momentum with volatility-adjusted sizing")
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe={
        "binance": ["BTC-USDT:perp", "ETH-USDT:perp", "LTC-USDT:perp",
                     "DOT-USDT:perp", "XRP-USDT:perp"],
    },
    # Legacy equivalent: universe=[201, 202, 204, 206, 208]
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(12),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        signal_delay=1,
        max_position_pct=0.3,
        allow_short=False,
    ),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=25,
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
    mbt.plot.summary(result)
