"""Tests for bt.choice() — sweeping a CHOICE of expression, not just a number.

The contract under test: `choice("sel", {...})` resolves to exactly one branch
per combo BEFORE simulation, so a sweep over the selector must be
bit-identical to running each branch inlined by hand. The selector must count
as a declared parameter (otherwise `_validate_swept_params` would reject the
sweep), and an unknown branch must fail with a message naming the known ones.
"""
import json
import os

import pytest

import manifoldbt as bt
from manifoldbt.indicators import close, sma

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

N_BARS = 3_000


def _store(tmp_path):
    ts = pd.date_range("2022-01-01", periods=N_BARS, freq="1min", tz="UTC")
    rng = np.random.default_rng(11)
    px = 100.0 + np.cumsum(np.sin(np.arange(N_BARS) / 90.0) * 0.3 + rng.normal(0, 0.2, N_BARS)) * 0.05
    px = np.maximum(px, 1.0)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": px,
            "high": px * 1.0005,
            "low": px * 0.9995,
            "close": px,
            "volume": [1000.0] * N_BARS,
        }
    )
    root = tmp_path / "choice_store"
    return bt.import_dataframe(
        df,
        symbol="ZC",
        symbol_id=1,
        interval="1m",
        asset_class="equity",
        exchange="TEST",
        data_root=str(root / "data"),
        metadata_db=str(root / "metadata.sqlite"),
    )


def _config():
    start, end = bt.time_range("2022-01-01", "2022-01-03")
    cfg = bt.BacktestConfig(
        universe=[1],
        time_range_start=start,
        time_range_end=end,
        initial_capital=10_000.0,
        provider="TEST",
        bar_interval=bt.Interval.minutes(1),
        symbol_names={"ZC": 1},
    )
    cfg.warmup_bars = 0
    return cfg


def _strategy_with_choice():
    band = bt.choice(
        "pick",
        {
            "fast": sma(close, 5),
            "slow": sma(close, 20),
        },
    )
    return (
        bt.Strategy.create("choice_e2e")
        .signal("band", band)
        .size(bt.when(close > bt.col("band"), 1.0, 0.0))
    )


def _strategy_inlined(period):
    return (
        bt.Strategy.create(f"inline_{period}")
        .signal("band", sma(close, period))
        .size(bt.when(close > bt.col("band"), 1.0, 0.0))
    )


def test_serializes_as_ordered_pairs():
    """serde expects Choice(String, Vec<(String, Expr)>): a list of pairs,
    order preserved — the first branch is the compile-time default."""
    e = bt.choice("pick", {"a": close, "b": sma(close, 3)})
    payload = json.loads(json.dumps(e.to_json()))
    assert list(payload) == ["Choice"]
    name, branches = payload["Choice"]
    assert name == "pick"
    assert [k for k, _ in branches] == ["a", "b"]


def test_empty_branches_rejected():
    with pytest.raises(ValueError, match="at least one branch"):
        bt.choice("pick", {})


def test_selector_counts_as_declared_parameter():
    """Sweeping the selector must pass strategy-side validation: choice()
    declares it via _param_meta exactly like param() does."""
    strat = _strategy_with_choice()
    assert "pick" in (strat.to_json_dict().get("parameters") or {})


def test_sweep_over_choice_matches_inlined_branches(tmp_path):
    """The money test: each combo of the selector sweep is bit-identical to
    the strategy with that branch written directly."""
    store = _store(tmp_path)
    cfg = _config()

    sweep = bt.run_sweep_lite(
        _strategy_with_choice(), {"pick": ["fast", "slow"]}, cfg, store, device="cpu"
    )
    assert len(sweep) == 2

    by_branch = dict(zip(["fast", "slow"], sweep))
    for name, period in (("fast", 5), ("slow", 20)):
        ref = bt.run_sweep_lite(
            _strategy_inlined(period), {}, cfg, store, device="cpu"
        )[0]
        got, want = by_branch[name].metrics, ref.metrics
        for key in ("total_return", "sharpe", "max_drawdown"):
            assert got.get(key) == want.get(key), (
                f"branch {name!r}: {key} diverged ({got.get(key)} vs {want.get(key)})"
            )


def test_unknown_branch_names_the_known_ones(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(Exception, match="fast"):
        bt.run_sweep_lite(
            _strategy_with_choice(), {"pick": ["nope"]}, _config(), store, device="cpu"
        )
