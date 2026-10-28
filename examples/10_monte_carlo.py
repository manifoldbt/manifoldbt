"""Monte Carlo Simulation -- confidence intervals on equity paths.

Demonstrates:
  - mbt.monte_carlo() for bootstrapped equity paths, computed in the engine
  - Risk metrics from simulated distributions
  - Monte Carlo fan chart visualization

Community runs up to 1,000 paths; Pro has no ceiling.

Data: shared store — real market data from `data/` (see examples/README.md)

Usage:
    python examples/10_monte_carlo.py
"""
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

from _bootstrap import open_store, plots_available

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
    store = open_store()

    # 1. Run base backtest
    print("Running base backtest...")
    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    print(result.summary())
    print(f"Elapsed: {time.perf_counter() - t0:.3f}s\n")

    # 2. Risk distribution, resampled in the engine: blocks of 20 bars keep the
    #    short-term autocorrelation that an IID bootstrap would destroy.
    mc = mbt.monte_carlo(result, n_paths=1000, method="block", block_size=20, seed=42)
    final = dict(mc["final_return"]["percentiles"])
    drawdown = dict(mc["max_drawdown"]["percentiles"])
    print(f"Final return   5th pct {final[0.05]:+.1%}   median {final[0.5]:+.1%}")
    print(f"Max drawdown   median {drawdown[0.5]:.1%}   95th pct {drawdown[0.95]:.1%}")
    print(f"P(ruin) {mc['prob_of_ruin']:.1%}\n")

    # 3. Monte Carlo fan chart
    if plots_available():
        mbt.plot.monte_carlo(result, n_simulations=10000, seed=42)
