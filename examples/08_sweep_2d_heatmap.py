"""2D Parameter Sweep Heatmap -- EMA crossover t-stat(alpha).

Demonstrates:
  - param() in indicator periods (engine re-compiles per combo)
  - run_sweep() for Cartesian grid search
  - Heatmap visualization with mbt.plot.heatmap_2d()

Usage:
    python examples/08_sweep_2d_heatmap.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Strategy (single definition, param() in periods) ------------------------
fast = ema(close, mbt.param("fast"))
slow = ema(close, mbt.param("slow"))

signal = mbt.when(fast > slow, 0.25, mbt.when(fast < slow, -0.25, 0.0))

strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(signal)
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2026-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(1),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        allow_short=True,
        max_position_pct=0.5,
    ),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=80,
    output_resolution=Interval.days(1),
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

    fast_values = list(range(5, 1000, 5))
    slow_values = list(range(10, 5000, 5))

    print(f"Running 2D sweep ({len(fast_values)*len(slow_values)} combos)...")
    t0 = time.perf_counter()
    batch = mbt.run_sweep_lite(
        strategy,
        {"fast": fast_values, "slow": slow_values},
        config,
        store,
    )
    elapsed = time.perf_counter() - t0

    # run_sweep_lite iterates sorted keys: fast (outer) × slow (inner)
    # Reshape into grid[slow][fast] for heatmap (y=slow, x=fast)
    metric_grid = [[0.0] * len(fast_values) for _ in slow_values]
    idx = 0
    for fi, f_val in enumerate(fast_values):
        for si, s_val in enumerate(slow_values):
            metric_grid[si][fi] = batch[idx].metrics.get("tstat_alpha", 0.0)
            idx += 1

    print(f"\n{len(batch)} combos in {elapsed:.2f}s")

    mbt.plot.heatmap_2d({
        "x_param": "fast",
        "y_param": "slow",
        "x_values": fast_values,
        "y_values": slow_values,
        "metric": "t-stat(alpha)",
        "metric_grid": metric_grid,
    })
