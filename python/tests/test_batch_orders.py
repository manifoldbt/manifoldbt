"""Regression tests for per-strategy orders in batch runs.

History: run_batch/run_batch_lite once dropped SL/TP entirely (they called
_prepare_config(config, None)). They were then fixed by merging orders into a
grouped config. Now orders travel INSIDE the strategy JSON (StrategyDef.orders)
and the engine applies them per-strategy, so a single native call handles a
batch of strategies with DIFFERENT brackets over one data load — the config
carries no orders and there is no per-profile grouping.

Native calls are monkeypatched, so no market data is needed.
"""
import json

import pytest

import manifoldbt as bt


class _DummyStore:
    """Minimal store: no metadata DB, default dataset (all lookups fall back)."""

    def dataset(self):
        raise RuntimeError("no dataset")

    def metadata_db(self):
        raise RuntimeError("no metadata db")


def _strategy(name, sl=None, tp=None):
    s = bt.Strategy.create(name).signal("sig", bt.lit(1.0)).size(bt.lit(0.1))
    if sl is not None:
        s = s.stop_loss(pct=sl)
    if tp is not None:
        s = s.take_profit(pct=tp)
    return s


def _config():
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=10_000_000_000,
        initial_capital=10_000,
    )


@pytest.fixture()
def captured(monkeypatch):
    """Patch both native batch entry points; record (config_dict, [strategy_dict])."""
    calls = []

    def fake_batch_lite(strategy_jsons, config_json, store, max_parallelism=0):
        strats = [json.loads(s) for s in strategy_jsons]
        calls.append((json.loads(config_json), strats))
        return [f"lite:{s['name']}" for s in strats]

    def fake_batch(strategy_jsons, config_json, store, max_parallelism=0):
        strats = [json.loads(s) for s in strategy_jsons]
        calls.append((json.loads(config_json), strats))
        return [object() for _ in strats]

    monkeypatch.setattr(bt, "_run_batch_lite_native", fake_batch_lite)
    monkeypatch.setattr(bt, "_run_batch_native", fake_batch)
    return calls


def _config_orders(cfg_json):
    return (cfg_json.get("execution") or {}).get("orders")


def _names(strats):
    return [s["name"] for s in strats]


def _sl_of(strat):
    orders = strat.get("orders")
    return orders["stop_loss"]["stop_pct"] if orders and "stop_loss" in orders else None


def test_batch_lite_carries_sl_tp_in_strategy_json(captured):
    strats = [_strategy(f"s{i}", sl=2.0, tp=4.0) for i in range(3)]
    out = bt.run_batch_lite(strats, _config(), _DummyStore())

    assert len(captured) == 1, "one native call handles the whole batch"
    cfg, sent = captured[0]
    assert _config_orders(cfg) is None, "orders travel in the strategy JSON, not the config"
    for s in sent:
        assert s["orders"]["stop_loss"]["stop_pct"] == 2.0
        assert s["orders"]["take_profit"]["profit_pct"] == 4.0
    assert _names(sent) == ["s0", "s1", "s2"]
    assert out == ["lite:s0", "lite:s1", "lite:s2"]


def test_batch_lite_no_orders_absent_from_json(captured):
    strats = [_strategy(f"s{i}") for i in range(2)]
    bt.run_batch_lite(strats, _config(), _DummyStore())

    assert len(captured) == 1
    cfg, sent = captured[0]
    assert _config_orders(cfg) is None
    for s in sent:
        assert s.get("orders") is None


def test_batch_lite_mixed_orders_single_call_in_order(captured):
    strats = [
        _strategy("a", sl=2.0),
        _strategy("b"),          # no orders
        _strategy("c", sl=2.0),
        _strategy("d", sl=5.0),
    ]
    out = bt.run_batch_lite(strats, _config(), _DummyStore())

    # Heterogeneous brackets now run in ONE native call over one data load,
    # each strategy carrying its own orders — no grouping, no reordering.
    assert len(captured) == 1
    cfg, sent = captured[0]
    assert _config_orders(cfg) is None
    assert _names(sent) == ["a", "b", "c", "d"]
    assert [_sl_of(s) for s in sent] == [2.0, None, 2.0, 5.0]
    assert out == ["lite:a", "lite:b", "lite:c", "lite:d"]


def test_run_batch_carries_sl_tp(captured):
    strats = [_strategy("x", sl=1.5), _strategy("y", sl=1.5)]
    bt.run_batch(strats, _config(), _DummyStore())

    assert len(captured) == 1
    cfg, sent = captured[0]
    assert _config_orders(cfg) is None
    assert all(_sl_of(s) == 1.5 for s in sent)
    assert _names(sent) == ["x", "y"]


def test_portfolio_warns_on_ignored_orders():
    with pytest.warns(UserWarning, match="IGNORED"):
        bt.Portfolio().strategy(_strategy("p", sl=2.0), weight=1.0)
