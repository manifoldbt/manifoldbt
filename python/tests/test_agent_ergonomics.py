"""Ergonomics an agent (or a first-time user) hits before any strategy code.

Three walls measured in benchmarks/mcp_tokens/RESULTATS.md, each of which cost
LLM agents dozens of turns:

1. A store written by ``import_dataframe`` could not be reopened with
   ``DataStore(data_root, metadata_db)``: the Arrow IPC layout required an
   undocumented ``arrow_dir`` argument, and the failure surfaced later as
   "empty bar dataset", pointing at the data instead of the store handle.
2. An empty time window (the classic wrong-units mistake: seconds or
   milliseconds instead of nanoseconds) produced the same "empty bar dataset"
   message with no mention of the window.
3. Bare ``symbol_ref()`` strategies needed three pieces of orchestrator lore
   (dict universe, provider-qualified names, no ref in sizing) that nothing
   documented; agents discovered them by trial and error.
"""
import os

import pytest

import manifoldbt as bt
from manifoldbt.expr import col, lit, when, symbol_ref
from manifoldbt.helpers import Interval

pd = pytest.importorskip("pandas")

N = 400


def _frame(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=N, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, N))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": [10.0] * N,
    })


@pytest.fixture
def two_symbol_store(tmp_path):
    root = str(tmp_path / "data")
    db = str(tmp_path / "meta.sqlite")
    bt.import_dataframe(_frame(1), symbol="AAA", symbol_id=1, interval="1h",
                        data_root=root, metadata_db=db)
    store = bt.import_dataframe(_frame(2), symbol="BBB", symbol_id=2, interval="1h",
                                data_root=root, metadata_db=db)
    return store, root, db


def _cfg(**kw):
    base = dict(
        universe=["AAA"],
        time_range_start=0,
        time_range_end=4_102_444_800_000_000_000,
        bar_interval=Interval.hours(1),
        initial_capital=10_000.0,
        warmup_bars=0,
        fees=bt.FeeConfig.zero(),
    )
    base.update(kw)
    return bt.BacktestConfig(**base)


def _buy_and_hold():
    pos = when(col("close") > lit(0.0), lit(1.0), lit(0.0))
    return bt.Strategy.create("bh").signal("position", pos).size(pos)


# ---------------------------------------------------------------------------
# 1. Reopening an imported store with the default constructor
# ---------------------------------------------------------------------------

def test_imported_store_reopens_with_defaults(two_symbol_store):
    store, root, db = two_symbol_store
    reopened = bt.DataStore(root, db)
    assert reopened.dataset() == "arrow_ipc"
    assert dict(reopened.list_symbols()) == dict(store.list_symbols())

    res_live = bt.run(_buy_and_hold(), _cfg(), store)
    res_reop = bt.run(_buy_and_hold(), _cfg(), reopened)
    assert res_live.metrics["total_return"] == res_reop.metrics["total_return"]


def test_parquet_store_still_opens_without_mega_dir(tmp_path):
    # No mega/ directory: the constructor must fall through to Parquet exactly
    # as before, not error out on a missing Arrow store.
    root = tmp_path / "data"
    root.mkdir()
    store = bt.DataStore(str(root), str(tmp_path / "meta.sqlite"))
    assert store.dataset() == "bars_1m"


# ---------------------------------------------------------------------------
# 2. Empty window names the window and the unit
# ---------------------------------------------------------------------------

def test_empty_window_error_names_nanoseconds(two_symbol_store):
    store, _, _ = two_symbol_store
    # The classic mistake: an end that is 4 seconds after the epoch because the
    # value was not nanoseconds. The message must point at the window, not the data.
    cfg = _cfg(time_range_end=4_000_000_000)
    with pytest.raises(bt.DataError, match="NANOSECONDS"):
        bt.run(_buy_and_hold(), cfg, store)


# ---------------------------------------------------------------------------
# 3. Bare symbol_ref() runs without orchestrator lore
# ---------------------------------------------------------------------------

def _pair_strategy(ref_name):
    ratio = col("close") / symbol_ref(ref_name, "close")
    pos = when(ratio > lit(1.0), lit(1.0), lit(0.0))
    return (bt.Strategy.create("pair")
            .signal("ratio", ratio)
            .signal("position", pos)
            .size(pos))


def test_bare_symbol_ref_matches_qualified_form(two_symbol_store):
    store, _, _ = two_symbol_store
    naive = bt.run(_pair_strategy("BBB"), _cfg(universe=["AAA"]), store)
    explicit = bt.run(
        _pair_strategy("dataframe:BBB"),
        _cfg(universe={"dataframe": ["AAA", "BBB"]}),
        store,
    )
    assert naive.metrics["total_return"] == explicit.metrics["total_return"]
    assert naive.metrics["sharpe"] == explicit.metrics["sharpe"]


def test_symbol_ref_inside_sizing_is_hoisted(two_symbol_store):
    store, _, _ = two_symbol_store
    # The reference sits in .size() directly: evaluated per-symbol it is refused,
    # so the rewrite must hoist it into a synthetic signal.
    sizing = when(col("close") / symbol_ref("BBB", "close") > lit(1.0),
                  lit(1.0), lit(0.0))
    strat = (bt.Strategy.create("pair-sizing")
             .signal("position", sizing)
             .size(sizing))
    res = bt.run(strat, _cfg(universe=["AAA"]), store)
    assert "total_return" in res.metrics


def test_explicit_dict_universe_is_not_second_guessed(two_symbol_store):
    store, _, _ = two_symbol_store
    res = bt.run(
        _pair_strategy("dataframe:BBB"),
        _cfg(universe={"dataframe": ["AAA", "BBB"]}),
        store,
    )
    assert "total_return" in res.metrics


# ---------------------------------------------------------------------------
# 4. The cheat sheet exists and names the essentials
# ---------------------------------------------------------------------------

def test_guide_prints_and_names_the_essentials(capsys):
    # It must PRINT, not return: an agent running
    # `python -c "import manifoldbt as bt; bt.guide()"` sees nothing when the
    # sheet is only returned, and pays a turn reading the source instead.
    assert bt.guide() is None
    text = capsys.readouterr().out
    for needle in ("DataStore", "time_range", "NANOSECONDS", "symbol_ref",
                   "hold()", "trade_stats", "round_trips"):
        assert needle in text, needle


def test_guide_shows_a_composed_strategy_not_just_primitives(capsys):
    # Listing hold() and symbol_ref() separately did not stop an agent from
    # concluding the DSL could not express a stateful cross-asset strategy and
    # rewriting the whole backtest in pandas. The sheet has to show the
    # composition, not only the parts.
    bt.guide()
    text = capsys.readouterr().out
    recettes = text.split("Worked recipes")[-1]
    assert "hold()" in recettes and "symbol_ref" in recettes
    assert "zscore" in recettes
    assert "cs_zscore" in recettes
