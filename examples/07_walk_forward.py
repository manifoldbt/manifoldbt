"""Walk-Forward Optimization -- find robust parameters across time (Pro).

Demonstrates:
  - run_walk_forward() with anchored method
  - param() for sweep-able parameters
  - Walk-forward fold results inspection

Usage:
    python examples/07_walk_forward.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Strategy with tunable parameters ----------------------------------------
# The sweep engine substitutes each grid value into the param() references at
# runtime. The period MUST be param("..."): a hardcoded int compiles every
# combo to the same strategy, so the sweep becomes a silent no-op.
fast = ema(close, mbt.param("fast", default=12))
slow = ema(close, mbt.param("slow", default=26))

signal = mbt.when(fast > slow, 1.0, mbt.when(fast < slow, -1.0, 0.0))

strategy = (
    mbt.Strategy.create("wfo_ema")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(signal * 0.25)
    .describe("EMA crossover with walk-forward parameter optimization")
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

    wf_config = {
        "method": "Anchored",
        "n_splits": 5,
        "train_ratio": 0.7,
        "optimize_metric": "sharpe",
        "param_grid": {
            "fast": [5, 8, 12, 16, 20],
            "slow": [25, 35, 50],
        },
        "max_parallelism": 0,
    }

    print("Running walk-forward optimization (Pro)...\n")
    t0 = time.perf_counter()
    result = mbt.run_walk_forward(strategy, wf_config, config, store)
    elapsed = time.perf_counter() - t0

    folds = result.get("folds", [])
    metric = wf_config["optimize_metric"]

    def unwrap(params):
        # best_params values are ScalarValue dicts, e.g. {"Int64": 20} -> 20
        return {k: (next(iter(v.values())) if isinstance(v, dict) else v)
                for k, v in params.items()}

    for i, fold in enumerate(folds):
        is_m = fold["is_metrics"][metric]
        oos_m = fold["oos_metrics"][metric]
        params = unwrap(fold["best_params"])
        print(f"  Fold {i+1}: IS {metric}={is_m:+.3f}  OOS {metric}={oos_m:+.3f}  params={params}")

    print(f"\n{len(folds)} folds in {elapsed:.2f}s")

    if folds:
        mbt.plot.walk_forward({"optimize_metric": metric, "folds": folds}, show=True)
