"""Monte Carlo Simulation -- confidence intervals on equity paths (Pro).

Demonstrates:
  - py_run_monte_carlo() for bootstrapped equity paths
  - Monte Carlo fan chart visualization
  - Risk metrics from simulated distributions

Usage:
    python examples/10_monte_carlo.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Strategy -----------------------------------------------------------------
fast = ema(close, 12)
slow = ema(close, 26)

trend = fast - slow

strategy = (
    mbt.Strategy.create("mc_ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .signal("trend", trend)
    .size(mbt.when(trend > 0.0, 0.5, 0.0))
    .stop_loss(pct=3.0)
    .describe("EMA crossover for Monte Carlo analysis")
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2025-01-01")

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

    # 1. Run base backtest
    print("Running base backtest...")
    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    print(result.summary())
    print(f"Elapsed: {time.perf_counter() - t0:.3f}s\n")

    # 2. Monte Carlo fan chart
    mbt.plot.monte_carlo(result, n_simulations=10000, seed=42)
