"""Multi-Strategy Portfolio -- combine strategies with risk management.

Demonstrates:
  - Portfolio builder with weighted strategies
  - Importing strategies from separate files
  - Risk rules (max drawdown, gross exposure cap)
  - Periodic rebalancing
  - Per-strategy breakdown

Usage:
    python examples/11_portfolio.py
"""
import os
import sys
import time

# Allow importing sibling example files as modules
sys.path.insert(0, os.path.dirname(__file__))

import manifoldbt as mbt
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Import strategies from dedicated files -----------------------------------
from importlib import import_module

strategy_a = import_module("01_trend_following").strategy
strategy_b = import_module("02_mean_reversion").strategy

# -- Portfolio ----------------------------------------------------------------
portfolio = (
    mbt.Portfolio()
    .strategy(strategy_a, weight=0.6)
    .strategy(strategy_b, weight=0.4)
    .max_drawdown(pct=20.0)
    .max_gross_exposure(pct=150.0)
    .rebalance_periodic(every_n_bars=30)
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp", "ETH-USDT:perp"]},
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
    warmup_bars=60,
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

    print(f"Running portfolio: {portfolio}\n")
    t0 = time.perf_counter()
    result = mbt.run_portfolio(portfolio, config, store)
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\nElapsed: {elapsed:.3f}s")

    mbt.plot.tearsheet(result)
