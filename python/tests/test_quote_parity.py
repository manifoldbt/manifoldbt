"""The maker of example 30 as quotes (example 31) against itself as a function
(example 30), on real days of book and tape: the same orders, fills, order
events, equity, cash and fees, float for float.

Each run's journal is written as text, every float as the hexadecimal of its
bits, and the two texts are compared whole. Skipped unless ``BT_DUEL_STORES``
names the folder of the 0.27 duel's stores (they are not distributed with the
package).
"""
import dataclasses
import hashlib
import importlib.util
import os
import sys
from datetime import date, timedelta

import pytest

import manifoldbt as bt

# Folder holding the duel's stores; the module is skipped when it is unset.
DUEL = os.environ.get("BT_DUEL_STORES", "")
EXAMPLES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "examples"
)
CASES = [
    ("store", "2025-08-13"),
    ("store", "2025-08-23"),
    ("store", "2025-08-30"),
    ("store200", "2025-08-23"),
]


def _layer_open():
    try:
        bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    except PermissionError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not DUEL or not os.path.isdir(DUEL), reason="the duel's stores are not here"),
    pytest.mark.skipif(not _layer_open(), reason="order-level simulation locked"),
]


def _example(name, day):
    """An example module, imported as is (its argv[1] pins the day)."""
    path = os.path.join(EXAMPLES, name)
    argv = sys.argv
    sys.argv = [path, day]
    try:
        spec = importlib.util.spec_from_file_location(name[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def _store(name):
    root = os.path.join(DUEL, name)
    return bt.DataStore(
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
        arrow_dir=os.path.join(root, "data", "mega"),
    )


def _bits(x):
    return "NA" if x is None else float(x).hex()


def journal(raw):
    """The run's orders, fills, events and equity as one text per table."""
    out = {}
    for name in ("orders_columns", "fills_columns", "events_columns"):
        cols = getattr(raw, name)()
        rows = zip(*cols.values())
        out[name] = "\n".join(
            ",".join(_bits(v) if isinstance(v, float) or v is None else str(v) for v in row)
            for row in rows
        )
    ts, eq = raw.equity_columns()
    out["equity"] = "\n".join(f"{t},{e.hex()}" for t, e in zip(ts, eq))
    out["totals"] = f"{raw.cash.hex()},{raw.fees.hex()},{raw.position}"
    return out


@pytest.mark.parametrize("store, day", CASES)
def test_the_quotes_give_the_function_s_journal_on_a_real_day(store, day):
    if not os.path.isdir(os.path.join(DUEL, store)):
        pytest.skip(f"{store} is not here")
    ex30 = _example("30_order_api_market_maker.py", day)
    ex31 = _example("31_dsl_market_maker.py", day)
    nxt = str(date.fromisoformat(day) + timedelta(days=1))
    # Example 30's venue, with example 31's capital: the metrics of mbt.run
    # need one, and nothing the venue does depends on it.
    config = dataclasses.replace(ex30.CONFIG, initial_cash=ex31.INITIAL_CAPITAL)
    ref = bt.sim.run(ex30.maker, config, _store(store), symbols=[ex30.SYMBOL],
                     start=day, end=nxt)
    res = bt.run(ex31.build_strategy(), ex31.build_config(), _store(store))
    a, b = journal(res.sim.raw), journal(ref.raw)
    assert len(res.sim.raw.orders_columns()["order_id"]) > 1_000
    for table in a:
        ha = hashlib.sha256(a[table].encode()).hexdigest()
        hb = hashlib.sha256(b[table].encode()).hexdigest()
        assert ha == hb, f"{store} {day}: {table} differs"
