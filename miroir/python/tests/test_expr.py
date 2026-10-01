"""Pure-Python tests for the Expr DSL serialization.

These tests verify that the Python DSL produces the externally-tagged JSON
format the engine reads.  No compiled extension needed.
"""
import pytest

from manifoldbt.expr import Expr, col, lit, param, when


def test_column_serializes():
    assert col("close").to_json() == {"Column": "close"}


def test_literal_float():
    assert lit(1.0).to_json() == {"Literal": {"Float64": 1.0}}


def test_literal_int():
    assert lit(42).to_json() == {"Literal": {"Int64": 42}}


def test_literal_bool():
    assert lit(True).to_json() == {"Literal": {"Bool": True}}


def test_literal_null():
    assert lit(None).to_json() == {"Literal": "Null"}


def test_parameter():
    assert param("size", default=1.0).to_json() == {"Parameter": "size"}


def test_add():
    expr = col("close") + lit(1.0)
    assert expr.to_json() == {
        "Add": [{"Column": "close"}, {"Literal": {"Float64": 1.0}}]
    }


def test_sub():
    expr = col("close") - col("open")
    assert expr.to_json() == {
        "Sub": [{"Column": "close"}, {"Column": "open"}]
    }


def test_mul_with_raw_float():
    expr = col("signal") * 0.5
    assert expr.to_json() == {
        "Mul": [{"Column": "signal"}, {"Literal": {"Float64": 0.5}}]
    }


def test_rmul():
    expr = 2.0 * col("signal")
    assert expr.to_json() == {
        "Mul": [{"Literal": {"Float64": 2.0}}, {"Column": "signal"}]
    }


def test_neg():
    expr = -col("x")
    assert expr.to_json() == {
        "Mul": [{"Literal": {"Float64": -1.0}}, {"Column": "x"}]
    }


def test_div():
    expr = col("a") / col("b")
    assert expr.to_json() == {
        "Div": [{"Column": "a"}, {"Column": "b"}]
    }


def test_gt():
    expr = col("close") > lit(100.0)
    assert expr.to_json() == {
        "Gt": [{"Column": "close"}, {"Literal": {"Float64": 100.0}}]
    }


def test_lt():
    expr = col("close") < 50.0
    assert expr.to_json() == {
        "Lt": [{"Column": "close"}, {"Literal": {"Float64": 50.0}}]
    }


def test_eq():
    expr = col("side") == lit(1)
    assert expr.to_json() == {
        "Eq": [{"Column": "side"}, {"Literal": {"Int64": 1}}]
    }


def test_and_or():
    a = col("x") > lit(0.0)
    b = col("y") < lit(1.0)
    expr = a & b
    assert expr.to_json()["And"][0] == {"Gt": [{"Column": "x"}, {"Literal": {"Float64": 0.0}}]}
    expr2 = a | b
    assert "Or" in expr2.to_json()


def test_not():
    expr = ~(col("flag") == lit(True))
    assert expr.to_json()["Not"]["Eq"][0] == {"Column": "flag"}


def test_rolling_mean():
    expr = col("close").rolling_mean(20)
    assert expr.to_json() == {"RollingMean": [{"Column": "close"}, 20]}


def test_rolling_std():
    expr = col("close").rolling_std(30)
    assert expr.to_json() == {"RollingStd": [{"Column": "close"}, 30]}


def test_lag():
    expr = col("close").lag(5)
    assert expr.to_json() == {"Lag": [{"Column": "close"}, 5]}


def test_diff():
    expr = col("close").diff()
    assert expr.to_json() == {"Diff": [{"Column": "close"}, 1]}


def test_pct_change():
    expr = col("close").pct_change(3)
    assert expr.to_json() == {"PctChange": [{"Column": "close"}, 3]}


def test_ewm_mean():
    expr = col("close").ewm_mean(10.0)
    assert expr.to_json() == {"EwmMean": [{"Column": "close"}, 10.0]}


def test_zscore():
    expr = col("close").zscore(20)
    assert expr.to_json() == {"ZScore": [{"Column": "close"}, 20]}


def test_cumsum():
    expr = col("volume").cumsum()
    assert expr.to_json() == {"CumSum": {"Column": "volume"}}


def test_cumprod():
    expr = col("returns").cumprod()
    assert expr.to_json() == {"CumProd": {"Column": "returns"}}


def test_full_series_rank():
    expr = col("score").full_series_rank()
    assert expr.to_json() == {"Rank": {"Column": "score"}}


def test_rank_reste_un_alias_mais_previent():
    """L'ancien nom continue de marcher, en disant pourquoi il a change.

    Le JSON est le meme : c'est le nom Python qui bouge, pas le format de
    serialisation, donc une strategie deja enregistree se relit.
    """
    with pytest.warns(DeprecationWarning, match="whole series"):
        expr = col("score").rank()
    assert expr.to_json() == {"Rank": {"Column": "score"}}


def test_if_else():
    cond = col("x") > lit(0.0)
    expr = when(cond, lit(1.0), lit(-1.0))
    expected = {
        "IfElse": [
            {"Gt": [{"Column": "x"}, {"Literal": {"Float64": 0.0}}]},
            {"Literal": {"Float64": 1.0}},
            {"Literal": {"Float64": -1.0}},
        ]
    }
    assert expr.to_json() == expected


def test_complex_sma_cross():
    """SMA crossover — the canonical DSL example."""
    close = col("close")
    sma_fast = close.rolling_mean(20)
    sma_slow = close.rolling_mean(60)
    signal = when(sma_fast > sma_slow, lit(1.0), lit(-1.0))

    result = signal.to_json()
    assert result["IfElse"][0]["Gt"][0] == {"RollingMean": [{"Column": "close"}, 20]}
    assert result["IfElse"][0]["Gt"][1] == {"RollingMean": [{"Column": "close"}, 60]}
    assert result["IfElse"][1] == {"Literal": {"Float64": 1.0}}
    assert result["IfElse"][2] == {"Literal": {"Float64": -1.0}}


def test_param_with_meta():
    p = param("size", default=1.0, range=(0.5, 2.0), description="position size")
    assert p.to_json() == {"Parameter": "size"}
    meta = p._param_meta
    assert meta["name"] == "size"
    assert meta["default"] == 1.0
    assert meta["range"] == (0.5, 2.0)
    assert meta["description"] == "position size"


# -- Datetime extraction tests -----------------------------------------------


def test_hour():
    expr = col("timestamp").hour()
    assert expr.to_json() == {"Hour": {"Column": "timestamp"}}


def test_minute():
    expr = col("timestamp").minute()
    assert expr.to_json() == {"Minute": {"Column": "timestamp"}}


def test_day_of_week():
    expr = col("timestamp").day_of_week()
    assert expr.to_json() == {"DayOfWeek": {"Column": "timestamp"}}


def test_month():
    expr = col("timestamp").month()
    assert expr.to_json() == {"Month": {"Column": "timestamp"}}


def test_day_of_month():
    expr = col("timestamp").day_of_month()
    assert expr.to_json() == {"DayOfMonth": {"Column": "timestamp"}}


def test_datetime_in_filter_expression():
    """Datetime functions compose with arithmetic and boolean ops."""
    ts = col("timestamp")
    # US market hours filter: hour >= 14 AND hour < 21
    in_us = (ts.hour() > lit(13.5)) & (ts.hour() < lit(21.0))
    result = in_us.to_json()
    assert "And" in result
    assert "Gt" in result["And"][0]
    assert result["And"][0]["Gt"][0] == {"Hour": {"Column": "timestamp"}}


def test_datetime_indicators_module():
    """indicators.hour() etc. default to timestamp column."""
    from manifoldbt.indicators import hour, day_of_week, month

    assert hour().to_json() == {"Hour": {"Column": "timestamp"}}
    assert day_of_week().to_json() == {"DayOfWeek": {"Column": "timestamp"}}
    assert month().to_json() == {"Month": {"Column": "timestamp"}}


# -- Scan (stateful fold) tests ----------------------------------------------


def test_scan_prev_json():
    from manifoldbt.expr import s

    assert s.prev("x").to_json() == {"ScanPrev": "x"}


def test_scan_var_json():
    from manifoldbt.expr import s

    assert s.var("k").to_json() == {"ScanVar": "k"}


def test_scan_cumsum_json():
    from manifoldbt.expr import s, scan

    cumsum = scan(
        state={"total": lit(0.0)},
        update={"total": s.prev("total") + col("value")},
        output="total",
    )
    result = cumsum.to_json()
    assert "Scan" in result
    data = result["Scan"]
    assert data["state_names"] == ["total"]
    assert data["update_names"] == ["total"]
    assert data["output"] == "total"
    # init_exprs should be [Literal(Float64(0.0))]
    assert data["init_exprs"] == [{"Literal": {"Float64": 0.0}}]
    # update_exprs should be [Add(ScanPrev("total"), Column("value"))]
    assert data["update_exprs"] == [
        {"Add": [{"ScanPrev": "total"}, {"Column": "value"}]}
    ]


def test_scan_kalman_json():
    from manifoldbt.expr import s, scan

    kalman = scan(
        state={"x": col("close"), "p": lit(1.0)},
        update={
            "p_pred": s.prev("p") + param("q"),
            "k": s.var("p_pred") / (s.var("p_pred") + param("r")),
            "x": s.prev("x") + s.var("k") * (col("close") - s.prev("x")),
            "p": (lit(1.0) - s.var("k")) * s.var("p_pred"),
        },
        output="x",
    )
    result = kalman.to_json()
    assert "Scan" in result
    data = result["Scan"]
    assert data["state_names"] == ["x", "p"]
    assert data["update_names"] == ["p_pred", "k", "x", "p"]
    assert data["output"] == "x"
    assert len(data["init_exprs"]) == 2
    assert len(data["update_exprs"]) == 4


def test_kalman_indicator():
    from manifoldbt.indicators import kalman

    result = kalman().to_json()
    assert "Scan" in result
    data = result["Scan"]
    assert data["state_names"] == ["x", "p"]
    assert data["output"] == "x"


def test_garch_indicator():
    from manifoldbt.indicators import garch

    result = garch().to_json()
    assert "Scan" in result
    data = result["Scan"]
    assert "sigma2" in data["state_names"]
    assert data["output"] == "sigma"


def test_scan_exports():
    """Verify scan and s are accessible from the top-level package."""
    import manifoldbt as bt

    assert hasattr(bt, "scan")
    assert hasattr(bt, "s")
    assert bt.s.prev("x").to_json() == {"ScanPrev": "x"}
