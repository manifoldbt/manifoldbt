"""Tests for Strategy serialization."""
import json

from manifoldbt.expr import col, lit, param, when
from manifoldbt.strategy import Strategy


def test_strategy_serializes_to_valid_json():
    size = param("size", default=1.0, range=(0.5, 2.0))
    signal = when(col("close") > col("close").lag(1), lit(1.0), lit(0.0))

    strategy = Strategy(
        name="test_strategy",
        signals={"trend": signal},
        position_sizing=col("trend") * size,
        parameters={"size": size},
    )

    result = json.loads(strategy.to_json())

    assert result["name"] == "test_strategy"
    assert "trend" in result["signals"]
    assert result["parameters"]["size"]["default"] == {"Float64": 1.0}
    assert result["parameters"]["size"]["range"] == [
        {"Float64": 0.5},
        {"Float64": 2.0},
    ]


def test_strategy_no_params():
    strategy = Strategy(
        name="simple",
        signals={"signal": lit(1.0)},
        position_sizing=col("signal"),
    )

    result = json.loads(strategy.to_json())
    assert result["name"] == "simple"
    assert result["parameters"] == {}
    assert result["constraints"] == []
    assert result["signals"]["signal"] == {"Literal": {"Float64": 1.0}}
    assert result["position_sizing"] == {"Column": "signal"}


def test_strategy_metadata():
    strategy = Strategy(
        name="documented",
        signals={"s": lit(1.0)},
        position_sizing=col("s"),
        description="A documented strategy",
    )

    result = json.loads(strategy.to_json())
    assert result["metadata"]["description"] == "A documented strategy"
