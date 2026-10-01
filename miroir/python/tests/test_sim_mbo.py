"""A market stored order by order in the order-level simulation.

The rules of the exact queue are pinned by the engine's own tests and held
against a direct FIFO replay of a real day. This file checks that Python
reaches them: a day ingested from ITCH messages written here, read by
``bt.sim.run`` and by ``mbt.run`` with ``Strategy.quote``, the volume ahead of
an order read exactly, the queue model a run names, and the refusals.

With ``BT_ITCH_SAMPLE`` set to a stream of one symbol's raw ITCH messages (an
extract of a real day), the last test runs a maker over its first minutes.
"""
import os
import struct

import pytest

import manifoldbt as bt
from manifoldbt import _native, book
from manifoldbt.helpers import Interval
from manifoldbt.result import QuoteResult

DATE = "2025-12-09"
H = 3_600_000_000_000
MS = 1_000_000
# 09:30 in New York on 2025-12-09, as an instant (UTC).
OPEN_UTC = 1_765_238_400_000_000_000 + 5 * H + 9 * H + H // 2
ITCH_OPEN = 9 * H + H // 2


def _layer_open():
    try:
        bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    except PermissionError:
        return False
    from manifoldbt._native import py_attach_quotes_arrays

    try:
        py_attach_quotes_arrays("", "", 1, [], [], [])
    except PermissionError:
        return False
    except Exception:
        return True
    return True


pytestmark = pytest.mark.skipif(
    not _layer_open(),
    reason="order-by-order layer locked (runs in the engine's own debug builds)",
)


def _frame(body):
    return struct.pack(">H", len(body)) + body


def _head(typ, ms):
    return typ + struct.pack(">HH", 1, 0) + (ITCH_OPEN + ms * MS).to_bytes(6, "big")


def add(ms, ref, buy, qty, px):
    return _frame(_head(b"A", ms) + struct.pack(">Q", ref) + (b"B" if buy else b"S")
                  + struct.pack(">I", qty) + b"XYZ     " + struct.pack(">I", int(round(px * 10_000))))


def execute(ms, ref, qty):
    return _frame(_head(b"E", ms) + struct.pack(">QIQ", ref, qty, ref))


def cancel(ms, ref, qty):
    return _frame(_head(b"X", ms) + struct.pack(">QI", ref, qty))


def _store(tmp_path, msgs):
    path = tmp_path / "day.itch"
    path.write_bytes(b"".join(msgs))
    return bt.ingest_mbo(str(path), {"XYZ": 1}, DATE,
                         data_root=str(tmp_path / "data"),
                         metadata_db=str(tmp_path / "meta.sqlite"),
                         progress=False)


def queue_day():
    """100 bid at 10.00 (order 1) and offered at 10.01 (order 2) at 09:30;
    200 more bid at 10.00 at 3 ms (order 3, behind an order posted at 2 ms);
    order 1 executed at 4 and 5 ms, order 3 at 6 and 7 ms."""
    return [
        add(0, 1, True, 100, 10.00),
        add(0, 2, False, 100, 10.01),
        add(3, 3, True, 200, 10.00),
        execute(4, 1, 60),
        execute(5, 1, 40),
        execute(6, 3, 30),
        execute(7, 3, 50),
        add(20, 4, True, 10, 9.99),
    ]


def _run_queue(store, config=None):
    seen = {}

    def strategy(sim):
        sim.elapse(OPEN_UTC + 2 * MS - sim.now())
        oid = sim.post("XYZ", "buy", 10.00, 50.0)
        for ms in (2, 4, 5):
            sim.elapse(OPEN_UTC + ms * MS - sim.now())
            seen[ms] = sim.order(oid).queue_ahead
        sim.elapse(OPEN_UTC + 10 * MS - sim.now())

    res = bt.sim.run(strategy, config, store, symbols=["XYZ"],
                     start=OPEN_UTC, end=OPEN_UTC + 60 * 1_000 * MS)
    return res, seen


def test_an_order_waits_in_its_exact_place(tmp_path):
    store = _store(tmp_path, queue_day())
    res, seen = _run_queue(store)
    # The volume ahead, read exactly as the queue moves.
    assert seen == {2: 100.0, 4: 40.0, 5: 0.0}
    fills = res.fills_df(backend="pandas")
    assert list(fills["qty"]) == [30.0, 20.0]
    assert set(fills["channel"]) == {"queue"}
    assert res.fill_fragility["queue"]["queue_decided_fills"] == 2


def test_a_market_stored_order_by_order_says_so(tmp_path):
    store = _store(tmp_path, queue_day())
    m = bt.sim.Market.from_store(store, "XYZ", OPEN_UTC, OPEN_UTC + H)
    assert m.by_order
    # The eight messages, and the three orders left when the stream ends,
    # which leave the book then.
    assert m.mbo_events == 11
    assert m.book_states == 8
    assert m.trades == 4
    hand = bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    assert not hand.by_order


def test_the_queue_a_run_names_is_the_one_served(tmp_path):
    # Order 1 cancelled ahead of ours: the exact queue sees it; a book by
    # price shows the level as full as before (order 3 is behind).
    msgs = [
        add(0, 1, True, 100, 10.00),
        add(0, 2, False, 100, 10.01),
        add(3, 3, True, 100, 10.00),
        cancel(4, 1, 100),
        execute(6, 3, 10),
        add(20, 4, True, 10, 9.99),
    ]
    store = _store(tmp_path, msgs)
    exact, _ = _run_queue(store)
    assert list(exact.fills_df(backend="pandas")["qty"]) == [10.0]
    named, _ = _run_queue(store, bt.sim.Config(queue={"model": "fifo"}))
    assert list(named.fills_df(backend="pandas")["qty"]) == [10.0]
    level2, _ = _run_queue(store, bt.sim.Config(queue={"model": "risk_adverse"}))
    assert len(level2.fills_df(backend="pandas")) == 0


def test_the_exact_queue_is_refused_on_a_book_stored_by_price():
    market = bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    with pytest.raises(Exception, match="stored order by order"):
        bt.sim.run(lambda sim: None, bt.sim.Config(queue={"model": "fifo"}), markets=[market])


# ---------------------------------------------------------------------------
# mbt.run with Strategy.quote, on the same market
# ---------------------------------------------------------------------------

CLIP = 100.0


def walk_day(n=3_000, seed=5):
    """Orders coming and going around 10.00 for a few minutes."""
    import random

    rng = random.Random(seed)
    msgs, live, ref, ms = [], {}, 10, 0
    for k in range(4):
        live[ref] = (True, 100, 10.00 - 0.01 * k)
        msgs.append(add(0, ref, True, 100, 10.00 - 0.01 * k))
        ref += 1
        live[ref] = (False, 100, 10.01 + 0.01 * k)
        msgs.append(add(0, ref, False, 100, 10.01 + 0.01 * k))
        ref += 1
    for _ in range(n):
        ms += 1 + rng.randrange(40)
        if rng.random() < 0.5 or len(live) < 6:
            buy = rng.random() < 0.5
            px = 10.00 - 0.01 * rng.randrange(4) if buy else 10.01 + 0.01 * rng.randrange(4)
            q = 100 * (1 + rng.randrange(5))
            live[ref] = (buy, q, px)
            msgs.append(add(ms, ref, buy, q, px))
            ref += 1
        else:
            r = rng.choice(sorted(live))
            buy, q, px = live[r]
            take = q if rng.random() < 0.5 else 100 * (1 + rng.randrange(max(1, q // 100)))
            take = min(take, q)
            msgs.append(execute(ms, r, take) if rng.random() < 0.6 else cancel(ms, r, take))
            if take == q:
                del live[r]
            else:
                live[r] = (buy, q - take, px)
    return msgs


def dsl_maker(clip=CLIP):
    # Join the best of each side; step away when too much is queued ahead.
    c = bt.param("clip", default=clip)
    crowded_bid = bt.queue_ahead("bid") > 5_000.0
    crowded_ask = bt.queue_ahead("ask") > 5_000.0
    return (
        bt.Strategy.create("mbo_maker")
        .quote("buy", book.bid_price_at(1), c, tif="GTX",
               enabled=(bt.position() < 5 * c) & ~crowded_bid)
        .quote("sell", book.ask_price_at(1), c, tif="GTX",
               enabled=(bt.position() > -5 * c) & ~crowded_ask)
    )


def quote_config(fill_model=None):
    execution = dict(latency={"order": 2 * MS, "cancel": 2 * MS, "response": 1 * MS, "feed": 1 * MS})
    if fill_model is not None:
        execution["fill_model"] = fill_model
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=OPEN_UTC,
        time_range_end=OPEN_UTC + 10 * 60 * 1_000 * MS,
        bar_interval=Interval.millis(100),
        initial_capital=1_000_000.0,
        execution=bt.ExecutionConfig(**execution),
    )


def run_quotes(store, config):
    raw, sim_raw = _native.run_quotes(dsl_maker().to_json(), config.to_json(), store, None)
    sim = bt.sim.SimResult(sim_raw, bt.sim.Config(initial_cash=config.initial_capital))
    return QuoteResult(raw, sim)


def test_a_strategy_that_quotes_runs_on_a_market_stored_order_by_order(tmp_path):
    store = _store(tmp_path, walk_day())
    res = run_quotes(store, quote_config())
    fills = res.sim.fills_df(backend="pandas")
    assert len(fills) > 5
    assert "queue" in set(fills["channel"])
    # Named, the exact queue is the same run; a queue of the book by price
    # is another.
    named = run_quotes(store, quote_config({"queue": {"model": "fifo"}}))
    assert named.sim.fills_df(backend="pandas").equals(fills)
    level2 = run_quotes(store, quote_config({"queue": {"depth_source": "book"}}))
    assert not level2.sim.fills_df(backend="pandas").equals(fills)


def test_a_sweep_of_quoting_runs_reads_the_market_stored_order_by_order(tmp_path):
    import json

    store = _store(tmp_path, walk_day(n=1_500))
    cfg = quote_config()
    grid = {"clip": [100.0, 200.0]}
    lite = bt.run_sweep_lite(dsl_maker(), grid, cfg, store, max_parallelism=2)
    full = bt.run_sweep(dsl_maker(), grid, cfg, store, max_parallelism=2)
    assert len(lite) == len(full) == 2
    for i, clip in enumerate(grid["clip"]):
        one = bt.run(dsl_maker(clip), cfg, store)
        assert one.sim.metrics["fills"] > 0
        m = json.dumps(one.metrics, sort_keys=True)
        assert json.dumps(full[i].metrics, sort_keys=True) == m, f"combination {i}"
        assert json.dumps(lite[i].metrics, sort_keys=True) == m, f"combination {i}"


@pytest.mark.skipif(not os.environ.get("BT_ITCH_SAMPLE"),
                    reason="BT_ITCH_SAMPLE names no extract of a real day")
def test_a_real_day_quoted_in_its_exact_queue(tmp_path):
    path = os.environ["BT_ITCH_SAMPLE"]
    ticker = os.environ.get("BT_ITCH_SAMPLE_TICKER", "AAPL")
    store = bt.ingest_mbo(path, {ticker: 1}, DATE, data_root=str(tmp_path / "data"),
                          metadata_db=str(tmp_path / "meta.sqlite"), progress=False)
    m = bt.sim.Market.from_store(store, ticker, OPEN_UTC, OPEN_UTC + 5 * 60 * 1_000 * MS)
    assert m.by_order and m.mbo_events > 10_000
    config = quote_config()
    config.time_range_end = OPEN_UTC + 5 * 60 * 1_000 * MS
    res = run_quotes(store, config)
    fills = res.sim.fills_df(backend="pandas")
    assert len(fills) > 0
    assert res.sim.fill_fragility["queue"]["queue_decided_fills"] > 0
