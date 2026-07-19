"""Full Visualization Suite -- RSI mean-reversion + all plots.

Strategy:
  - Long when RSI < 30 (oversold)
  - Short when RSI > 70 (overbought)
  - Exit long when RSI > 50, exit short when RSI < 50

Demonstrates every plotting function available in manifoldbt.

Usage:
    python examples/06_full_visualization.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, rsi
from manifoldbt.helpers import time_range, Slippage, Interval

rsi_14 = rsi(close, 14)

# -- Strategy -----------------------------------------------------------------
# Entry: RSI < 30 → long, RSI > 70 → short
# Exit:  RSI crosses 50

long_entry  = rsi_14 < mbt.lit(30.0)
short_entry = rsi_14 > mbt.lit(70.0)

signal = mbt.when(
    long_entry,   1.0,
    mbt.when(short_entry, -1.0, 0.0),
)

strategy = (
    mbt.Strategy.create("RSI_strategy")
    .signal("rsi14", rsi_14)
    .size(signal * 0.25)
    .describe(
        "RSI(14) mean-reversion: long when RSI<30, short when RSI>70, "
        "exit when RSI crosses 50."
    )
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2026-01-01")

ALL_SYMBOLS = {"binance": [
    "BTC-USDT:perp", "ETH-USDT:perp", "LTC-USDT:perp", "BNB-USDT:perp",
    "DOT-USDT:perp", "XRP-USDT:perp", "ADA-USDT:perp", "LINK-USDT:perp",
    "DOGE-USDT:perp", "AVAX-USDT:perp",
]}

config = mbt.BacktestConfig(
    universe=ALL_SYMBOLS,
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.minutes(120),
    initial_capital=100_000,
    execution=mbt.ExecutionConfig(
        allow_short=True,
        max_position_pct=0.5,
        position_sizing_mode="FractionOfInitialCapital",
    ),
    fees=mbt.FeeConfig.zero(),
    slippage=Slippage.fixed_bps(0),
    warmup_bars=20,
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..")
    os.makedirs(os.path.join(root, "output"), exist_ok=True)
    data_root = os.path.abspath(os.path.join(root, "data"))
    store = mbt.DataStore(
        data_root=data_root,
        metadata_db=os.path.abspath(os.path.join(root, "metadata", "metadata.sqlite")),
        arrow_dir=os.path.join(data_root, "mega"),
    )

    # -- 1. Single backtest --------------------------------------------------
    print("Running backtest...")
    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    elapsed = time.perf_counter() - t0
    print(result.summary())
    print(f"Elapsed: {elapsed:.3f}s\n")

    # -- 2. Tearsheet (3 figures: overview, returns, rolling) ---------------
    print("Generating tearsheet...")
    mbt.plot.tearsheet(
        result, show=True,
        save=os.path.join(root, "output", "tearsheet.html"),
    )

    # -- 3. Summary 3-panel ---------------------------------------------------
    mbt.plot.summary(result)

    # -- 4. Candlestick chart (first symbol in universe) --------------------
    mbt.plot.chart(
        result, store, symbol_id=201,
        emas=[10, 25],
        smas=[50],
        n_bars=120,
        interactive=False,
    )

    # -- 5. Individual charts -------------------------------------------------
    mbt.plot.equity(result)
    mbt.plot.drawdown(result)
    mbt.plot.monthly_returns(result)
    mbt.plot.annual_returns(result)
    mbt.plot.returns_histogram(result)
    mbt.plot.var_chart(result)
    mbt.plot.rolling_sharpe(result)
    mbt.plot.rolling_volatility(result)

    # -- 6. Sweep heatmap 2D -------------------------------------------------
    # Sweep over RSI period and oversold threshold
    print("\nRunning 2D sweep (RSI period × oversold threshold)...")
    t0 = time.perf_counter()

    periods    = [7, 10, 14, 21]
    thresholds = [20, 25, 30, 35]   # oversold level (overbought = 100 - threshold)
    sweep_strategies = []
    for p in periods:
        for thr in thresholds:
            r14 = rsi(close, p)
            ob  = mbt.lit(float(100 - thr))
            os_ = mbt.lit(float(thr))
            sig = mbt.when(
                r14 < os_,  1.0,
                mbt.when(r14 > ob, -1.0, 0.0),
            )
            s = (
                mbt.Strategy.create(f"rsi_p{p}_t{thr}")
                .signal("rsi", r14)
                .size(sig * 0.05)
                .stop_loss(pct=2.0)
                .take_profit(pct=4.0)
            )
            sweep_strategies.append(s)

    batch_results = mbt.run_batch_lite(sweep_strategies, config, store)
    metric_grid = []
    idx = 0
    for _ in periods:
        row = []
        for _ in thresholds:
            r = batch_results[idx]
            row.append(r.metrics.get("sharpe", 0.0))
            idx += 1
        metric_grid.append(row)

    sweep_result = {
        "x_param": "oversold_thr",
        "y_param": "period",
        "x_values": thresholds,
        "y_values": periods,
        "metric": "sharpe",
        "metric_grid": metric_grid,
    }
    print(f"Sweep done in {time.perf_counter() - t0:.1f}s")
    mbt.plot.heatmap_2d(sweep_result)

    # -- 7. Walk-forward validation -------------------------------------------
    print("\nRunning walk-forward (manual folds)...")
    t0 = time.perf_counter()

    fold_months = [
        ("2024-01-01", "2024-07-01", "2024-07-01", "2024-09-01"),
        ("2024-01-01", "2024-08-01", "2024-08-01", "2024-10-01"),
        ("2024-01-01", "2024-09-01", "2024-09-01", "2024-11-01"),
        ("2024-01-01", "2024-10-01", "2024-10-01", "2024-12-01"),
        ("2024-01-01", "2024-11-01", "2024-11-01", "2025-01-01"),
    ]
    wf_folds = []
    for train_start, train_end, test_start, test_end in fold_months:
        ts, te = time_range(train_start, train_end)
        train_cfg = mbt.BacktestConfig(
            universe=ALL_SYMBOLS, time_range_start=ts, time_range_end=te,
            bar_interval=Interval.minutes(60), initial_capital=100_000,
            execution=config.execution, fees=config.fees,
            slippage=config.slippage, warmup_bars=20,
        )
        ts2, te2 = time_range(test_start, test_end)
        test_cfg = mbt.BacktestConfig(
            universe=ALL_SYMBOLS, time_range_start=ts2, time_range_end=te2,
            bar_interval=Interval.minutes(60), initial_capital=100_000,
            execution=config.execution, fees=config.fees,
            slippage=config.slippage, warmup_bars=20,
        )
        train_r = mbt.run(strategy, train_cfg, store)
        test_r  = mbt.run(strategy, test_cfg, store)
        wf_folds.append({
            "train_metric": train_r.metrics.get("sharpe", 0.0),
            "test_metric":  test_r.metrics.get("sharpe", 0.0),
        })

    wf_result = {
        "metric": "sharpe",
        "folds": wf_folds,
    }
    print(f"Walk-forward done in {time.perf_counter() - t0:.1f}s")
    mbt.plot.walk_forward(wf_result)

    # -- 8. Monte Carlo -------------------------------------------------------
    print("\nRunning Monte Carlo (1000 paths)...")
    mbt.plot.monte_carlo(result, n_simulations=1000, seed=42)

    # -- 9. Parameter stability -----------------------------------------------
    print("\nRunning stability analysis (RSI period)...")
    t0 = time.perf_counter()
    stability_periods = [5, 7, 9, 11, 14, 18, 21, 28]
    stability_metrics = []
    for p in stability_periods:
        r14 = rsi(close, p)
        sig = mbt.when(
            r14 < mbt.lit(30.0),  1.0,
            mbt.when(r14 > mbt.lit(70.0), -1.0, 0.0),
        )
        s = (
            mbt.Strategy.create(f"rsi_stab_{p}")
            .signal("rsi", r14)
            .size(sig * 0.05)
            .stop_loss(pct=2.0)
            .take_profit(pct=4.0)
        )
        r = mbt.run(s, config, store)
        stability_metrics.append(r.metrics.get("sharpe", 0.0))

    import numpy as np
    mean_m = float(np.mean(stability_metrics))
    std_m  = float(np.std(stability_metrics))
    stab_result = {
        "param_name": "period",
        "metric": "sharpe",
        "values": stability_periods,
        "metric_values": stability_metrics,
        "mean_metric": mean_m,
        "std_metric": std_m,
        "stability_score": 1.0 - (std_m / abs(mean_m)) if mean_m != 0 else 0.0,
    }
    print(f"Stability done in {time.perf_counter() - t0:.1f}s")
    mbt.plot.stability(stab_result)

    # -- 10. Research report (composite) --------------------------------------
    print("\nGenerating research report...")
    mbt.plot.research_report(
        sweep_result=sweep_result,
        wf_result=wf_result,
        stability_result=stab_result,
        show=True,
        save=os.path.join(root, "output", "research.png"),
    )

    print("\nDone — all visualizations generated.")
    print(f"PNGs saved to {os.path.join(root, 'output')}")
