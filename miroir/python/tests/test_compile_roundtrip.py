"""Round-trip test: Python DSL -> JSON -> Rust strategy compiler."""
import json

import manifoldbt as bt


def test_dsl_strategy_compiles_via_rust():
    """Strategy built with Python DSL successfully compiles through Rust."""
    signal = bt.when(
        bt.col("close") > bt.col("close").lag(1),
        bt.lit(1.0),
        bt.lit(0.0),
    )

    strategy = bt.Strategy(
        name="compile_test",
        signals={"signal": signal},
        position_sizing=bt.col("signal"),
    )

    summary_json = bt.compile_strategy_json(strategy.to_json())
    summary = json.loads(summary_json)

    assert summary["name"] == "compile_test"
    assert "signal" in summary["signal_names"]
    assert "close" in summary["required_columns"]


def test_strategy_with_params_compiles():
    """Strategy with parameters compiles correctly."""
    size = bt.param("size", default=1.0, range=(0.5, 2.0))

    strategy = bt.Strategy(
        name="param_test",
        signals={"signal": bt.lit(1.0)},
        position_sizing=bt.col("signal") * size,
        parameters={"size": size},
    )

    summary_json = bt.compile_strategy_json(strategy.to_json())
    summary = json.loads(summary_json)

    assert summary["name"] == "param_test"
    assert "size" in summary["parameters"]
