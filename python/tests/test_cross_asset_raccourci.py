"""``_apply_cross_asset`` answers from the strategy's JSON text when it can.

Every ``mbt.run`` rebuilt the strategy's dict and walked it looking for a
``SymbolRef``, half of the Python cost of a small run. The memoised JSON text
says whether one is there: without the word, nothing is rewritten, and the
function returns what the walk returned. Held here against the function as it
was, kept below: the same strategy JSON, the same config (the same object when
nothing is rewritten), the same exception, for strategies without a
reference, with a bare one, with the word elsewhere in the text, and invalid.
"""
import copy
import json
import os

import numpy as np
import pytest

import manifoldbt as bt
from manifoldbt.crossasset import prepare_cross_asset

pd = pytest.importorskip("pandas")


def old_apply_cross_asset(strategy, config, store):
    """``_apply_cross_asset`` as it was."""
    try:
        doc = strategy.to_json_dict()
    except Exception:
        return strategy.to_json(), config
    try:
        meta_db = store.metadata_db()
    except Exception:
        meta_db = None
    try:
        prepared = prepare_cross_asset(doc, config.universe, meta_db)
    except ValueError:
        prepared = None
    if prepared is None:
        return strategy.to_json(), config
    new_doc, dict_universe = prepared
    if dict_universe is None:
        return json.dumps(new_doc), config
    cfg = copy.deepcopy(config)
    cfg.universe = dict_universe
    return json.dumps(new_doc), cfg


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("xasset")
    n = 500
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    st = None
    for sid, name in ((1, "PAIRA"), (2, "PAIRB")):
        close = 100.0 + np.arange(n) * 0.01 * sid
        df = pd.DataFrame({"timestamp": ts, "open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": np.full(n, 1.0)})
        st = bt.import_dataframe(df, symbol=name, symbol_id=sid, interval="1h",
                                 data_root=os.path.join(tmp, "data"),
                                 metadata_db=os.path.join(tmp, "meta.sqlite"))
    return st


def config():
    return bt.BacktestConfig(universe=[1], time_range_start=0, time_range_end=2**62,
                             bar_interval=bt.Interval.hours(1))


def strategies():
    c = bt.col("close")
    plain = bt.Strategy.create("plain").signal("m", c).size(bt.when(c > 100.0, 1.0, 0.0))
    bare = (bt.Strategy.create("bare").signal("spread", c - bt.symbol_ref("PAIRB", "close"))
            .size(bt.when(bt.col("spread") > 0.0, 1.0, 0.0)))
    in_sizing = bt.Strategy.create("sizing").signal("m", c).size(
        bt.when(bt.symbol_ref("PAIRB", "close") > 100.0, 1.0, 0.0))
    qualified = (bt.Strategy.create("qualified")
                 .signal("x", bt.symbol_ref("dataframe:PAIRB", "close"))
                 .size(bt.lit(0.5)))
    word = (bt.Strategy.create("SymbolRef").signal("SymbolRef", c).size(bt.lit(1.0))
            .describe('mentions "SymbolRef" but holds none'))
    conflict = (bt.Strategy.create("conflict")
                .signal("a", bt.indicators.ema(c, bt.param("n", default=5)))
                .signal("b", bt.indicators.ema(c, bt.param("n", default=9)))
                .size(bt.lit(1.0)))
    return {"plain": plain, "bare": bare, "sizing": in_sizing, "qualified": qualified,
            "word": word, "conflict": conflict}


@pytest.mark.parametrize("name", ["plain", "bare", "sizing", "qualified", "word", "conflict"])
def test_the_text_answers_as_the_walk_answered(store, name):
    cfg = config()
    new_s, old_s = strategies()[name], strategies()[name]
    try:
        old = old_apply_cross_asset(old_s, cfg, store)
    except Exception as exc:  # the invalid strategy
        with pytest.raises(type(exc)) as got:
            bt._apply_cross_asset(new_s, cfg, store)
        assert str(got.value) == str(exc)
        return
    new = bt._apply_cross_asset(new_s, cfg, store)
    assert new[0] == old[0]
    if old[1] is cfg:
        assert new[1] is cfg
    else:
        assert new[1] is not cfg
        assert new[1].to_json() == old[1].to_json()
    if name == "plain":
        assert new == (new_s.to_json(), cfg)
    if name in ("bare", "sizing"):
        assert new[0] != new_s.to_json(), "the reference was rewritten"
