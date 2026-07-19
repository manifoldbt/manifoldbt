"""Statistical Arbitrage -- spread z-score vs ETH anchor.

Demonstrates:
  - symbol_ref() for cross-asset signals
  - Kalman filter for spread equilibrium
  - Z-score mean-reversion sizing

Usage:
    python examples/05_stat_arb.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, kalman
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Spread construction ------------------------------------------------------
pair_close = mbt.symbol_ref("binance:ETH-USDT:perp", "close")
ratio = close / (pair_close + mbt.lit(1e-12))

# -- Kalman equilibrium -------------------------------------------------------
equilibrium = kalman(ratio, q=1e-4, r=1e-2)
spread = ratio - equilibrium

# -- Z-score signal -----------------------------------------------------------
spread_z = spread.zscore(28)
signal = -spread_z  # mean-revert: short when z > 0, long when z < 0

# -- Strategy -----------------------------------------------------------------
strategy = (
    mbt.Strategy.create("stat_arb")
    .signal("pair_close", pair_close)
    .signal("spread", spread)
    .signal("spread_z", spread_z)
    .signal("signal", signal)
    .size(mbt.col("signal"))
    .describe("Spread z-score mean reversion vs ETH")
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2022-01-01", "2026-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp", "ETH-USDT:perp", "BNB-USDT:perp"]},  # BTC, ETH, BNB
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(24),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        allow_short=True,
        max_position_pct=5,
    ),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=30,
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
