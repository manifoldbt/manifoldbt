"""Python mirror of the Rust golden_buy_and_hold test.

Verifies that the Python DSL + Rust engine produce identical results
to the Rust-only golden test fixtures.
"""
import json
import os

import pytest

import manifoldbt as bt
from manifoldbt import run_with_parquet

# The golden fixtures were generated at full (Pro) resolution; the Community
# resolution cap changes the equity-point count and the comparison is
# meaningless. CI unlocks via BT_UNLOCKED=1 (debug builds); locally this needs
# an activated Pro license.
pytestmark = pytest.mark.skipif(
    bt.license_info()[0] != "Pro",
    reason="requires Pro (fixtures generated at sub-daily resolution); activate a license or use a BT_UNLOCKED dev build",
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
        bar_interval={"Days": 1},
        initial_capital=1000.0,
        currency="USD",
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

    metrics = result.metrics
    for key in expected_metrics:
        assert abs(metrics[key] - expected_metrics[key]) <= 1e-12, (
            f"Metric {key}: {metrics[key]} != {expected_metrics[key]}"
        )

    # -- Assert manifest snapshot fields match --
    with open(os.path.join(golden_buy_hold_dir, "expected_manifest_snapshot.json")) as f:
        expected_manifest = json.load(f)

    manifest = result.manifest
    assert manifest["strategy_name"] == expected_manifest["strategy_name"]
    assert manifest["engine_version"] == expected_manifest["engine_version"]
    assert manifest["config"] == expected_manifest["config"]
