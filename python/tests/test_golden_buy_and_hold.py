"""Python mirror of the Rust golden_buy_and_hold test.

Verifies that the Python DSL + Rust engine produce identical results
to the Rust-only golden test fixtures.
"""
import json
import os

import pytest

import manifoldbt as bt
from manifoldbt import run_with_parquet

# The golden fixtures assert on 1-second output resolution, below even the Pro
# floor (60s) — exactly like the Rust golden test, which sets BT_UNLOCKED=1.
# The override is only honored by debug builds (cargo test / maturin develop),
# so this needs BOTH: a dev build and BT_UNLOCKED=1 in the environment.
pytestmark = pytest.mark.skipif(
    os.environ.get("BT_UNLOCKED") != "1",
    reason="requires BT_UNLOCKED=1 on a dev (debug) build: fixtures assert 1s output, below the Pro 60s floor",
)


def test_golden_buy_and_hold_matches_fixtures(golden_buy_hold_dir):
    """Mirror of Rust golden_buy_and_hold_equity_trade_metrics_and_manifest_match_fixture."""

    # Build strategy using Python DSL — same as Rust golden test
    signal_expr = bt.lit(1.0)
    sizing_expr = bt.col("signal")

    strategy = bt.Strategy(
        name="golden_buy_and_hold",
        signals={"signal": signal_expr},
        position_sizing=sizing_expr,
    )

    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=4_000_000_000,
        # The fixture is 4 bars at 1-second spacing; the Rust golden test runs
        # them at Seconds(1) with per-bar output. Days(1) would resample the
        # whole range into a single bar and the comparison would be meaningless.
        bar_interval={"Seconds": 1},
        output_resolution={"Seconds": 1},
        initial_capital=1000.0,
        currency="USD",
        risk_free_rate=0.025,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            allow_fractional=True,
            skip_gap_bars=False,
            position_sizing_mode="Units",
        ),
        fees=bt.FeeConfig(),
        slippage={"FixedBps": {"bps": 0.0}},
        data_version="golden_v1",
        rng_seed=7,
    )

    parquet_path = os.path.join(golden_buy_hold_dir, "bars_1m.parquet")
    result = run_with_parquet(
        strategy.to_json(),
        config.to_json(),
        parquet_path,
        "golden_v1",
    )

    # -- Assert equity curve matches --
    with open(os.path.join(golden_buy_hold_dir, "expected_equity.json")) as f:
        expected_equity = json.load(f)

    equity = result.equity_curve.to_pylist()
    assert equity == expected_equity, f"Equity mismatch: {equity} != {expected_equity}"

    # -- Assert trades match --
    with open(os.path.join(golden_buy_hold_dir, "expected_trades.json")) as f:
        expected_trades = json.load(f)

    trades_batch = result.trades
    actual_trades = []
    for i in range(trades_batch.num_rows):
        actual_trades.append({
            "symbol_id": trades_batch.column("symbol_id")[i].as_py(),
            "side": trades_batch.column("side")[i].as_py(),
            "quantity": trades_batch.column("quantity")[i].as_py(),
            "fill_price": trades_batch.column("fill_price")[i].as_py(),
        })
    assert actual_trades == expected_trades, (
        f"Trade mismatch: {actual_trades} != {expected_trades}"
    )

    # -- Assert metrics match --
    with open(os.path.join(golden_buy_hold_dir, "expected_metrics.json")) as f:
        expected_metrics = json.load(f)

    # Mirror the Rust golden test: annualized metrics (CAGR, volatility,
    # sharpe, sortino, calmar) are not compared because the fixture uses 4
    # synthetic 1-second bars, making annualization numerically extreme.
    metrics = result.metrics
    for key in ("total_return", "max_drawdown"):
        assert abs(metrics[key] - expected_metrics[key]) <= 1e-12, (
            f"Metric {key}: {metrics[key]} != {expected_metrics[key]}"
        )

    # -- Assert manifest snapshot fields match --
    with open(os.path.join(golden_buy_hold_dir, "expected_manifest_snapshot.json")) as f:
        expected_manifest = json.load(f)

    # Mirror the Rust golden test: engine_version is excluded from the snapshot
    # (it tracks the crate version and would break on every release bump);
    # assert only that it is populated.
    manifest = result.manifest
    assert manifest["strategy_name"] == expected_manifest["strategy_name"]
    assert manifest["engine_version"], "engine_version should be populated"
    assert manifest["data_versions"].get("bars_1m", "") == expected_manifest["data_version"]
    assert manifest["config"] == expected_manifest["config"]
