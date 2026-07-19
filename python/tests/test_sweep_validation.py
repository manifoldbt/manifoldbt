"""Sweeping a parameter the strategy never declares must fail loudly.

It used to be a silent no-op: the unknown name landed in a parameter map
nothing reads, so every combination ran the same backtest and the sweep
returned N identical results with no warning. An "optimisation" over
thousands of combos looked like it had worked.

These tests only exercise the Python-side guard, so they need no data store:
validation happens before any native call.
"""
import pytest

import manifoldbt as bt
from manifoldbt.exceptions import StrategyError
from manifoldbt.indicators import close, ema


def _declared():
    """Strategy whose 'fast' comes from mbt.param() inside an indicator."""
    fast = ema(close, bt.param("fast"))
    return (
        bt.Strategy.create("declared")
        .signal("fast", fast)
        .size(bt.when(close > fast, 1.0, 0.0))
    )


def _hardcoded():
    """The shape that caused the bug: the period is a literal, not a param."""
    fast = ema(close, 12)
    return (
        bt.Strategy.create("hardcoded")
        .signal("fast", fast)
        .size(bt.when(close > fast, 1.0, 0.0))
    )


def _cfg():
    # Never reaches the engine: the guard raises before config is used.
    return bt.BacktestConfig(universe={"binance": ["BTC-USDT:perp"]})


def test_sweep_lite_rejects_undeclared_param():
    with pytest.raises(StrategyError) as exc:
        bt.run_sweep_lite(_hardcoded(), {"fast": [10, 20, 30]}, _cfg(), None)
    msg = str(exc.value)
    assert "fast" in msg
    # The message must say what to do, not just that it failed.
    assert "mbt.param" in msg


def test_sweep_rejects_undeclared_param():
    with pytest.raises(StrategyError):
        bt.run_sweep(_hardcoded(), {"fast": [10, 20]}, _cfg(), None)


def test_walk_forward_rejects_undeclared_param():
    wf = {
        "method": "Rolling", "n_splits": 2, "train_ratio": 0.7,
        "optimize_metric": "sharpe", "param_grid": {"fast": [10, 20]},
    }
    with pytest.raises((StrategyError, bt.LicenseError)) as exc:
        bt.run_walk_forward(_hardcoded(), wf, _cfg(), None)
    # Walk-forward is Pro-gated first; only assert our message when we got past it.
    if isinstance(exc.value, StrategyError):
        assert "fast" in str(exc.value)


def test_sweep_2d_rejects_undeclared_params():
    sweep = {
        "x_param": "fast", "y_param": "slow",
        "x_values": [5, 10], "y_values": [20, 40], "metric": "sharpe",
    }
    with pytest.raises(StrategyError) as exc:
        bt.run_sweep_2d(_hardcoded(), sweep, _cfg(), None)
    assert "fast" in str(exc.value) and "slow" in str(exc.value)


def test_stability_rejects_undeclared_param():
    stab = {"param_name": "fast", "values": [5, 10, 15], "metric": "sharpe"}
    with pytest.raises(StrategyError) as exc:
        bt.run_stability(_hardcoded(), stab, _cfg(), None)
    assert "fast" in str(exc.value)


def test_declared_param_passes_validation():
    """A declared param must get past the guard (it then fails on the store)."""
    with pytest.raises(Exception) as exc:
        bt.run_sweep_lite(_declared(), {"fast": [10, 20]}, _cfg(), None)
    # Whatever stops it next, it must not be our guard.
    assert "not declared" not in str(exc.value)


def test_explicit_param_call_counts_as_declared():
    """.param() declares a name even when no expression references it."""
    strat = _hardcoded().param("fast", default=12)
    with pytest.raises(Exception) as exc:
        bt.run_sweep_lite(strat, {"fast": [10, 20]}, _cfg(), None)
    assert "not declared" not in str(exc.value)


def test_message_lists_only_the_unknown_names():
    """A mixed grid must blame the unknown name, not the good one."""
    with pytest.raises(StrategyError) as exc:
        bt.run_sweep_lite(_declared(), {"fast": [10], "slow": [50]}, _cfg(), None)
    msg = str(exc.value)
    assert "slow" in msg
    # 'fast' is declared, so it must appear as available, never as unknown.
    assert "['slow']" in msg
