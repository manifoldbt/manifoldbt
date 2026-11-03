"""Brackets held by the simulated venue: ``sim.post_bracket(...)``.

The rules are pinned by the engine's own tests (``brackets.rs``); this file
checks that Python reaches them, that the legs and the tables say what they
are, that every word is refused by name when it is not one the venue reads,
and the fills on a set of cases where the trade-driven top-of-book convention
and this venue meet, or part on a rule named below.

The market of the first tests is the one of ``test_sim.py``: bids 100.0 x 5,
99.9 x 9, 99.0 x 9, asks 100.1 x 5, 100.5 x 6, on a 0.1 tick.
"""
import datetime

import pytest

import manifoldbt as bt
from manifoldbt.helpers import Interval

MS = 1_000_000
POST, END = 2 * MS, 10 * MS


def _layer_open():
    try:
        bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    except PermissionError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _layer_open(),
    reason="order-level simulation locked (runs in the engine's own debug builds)",
)

BIDS = [(100.0, 5.0), (99.9, 9.0), (99.0, 9.0)]
ASKS = [(100.1, 5.0), (100.5, 6.0)]
SOLD_AT_99_4 = (4 * MS, 99.4, 1.0, "sell")


def _market(trades, states=None):
    states = states if states is not None else [(MS, BIDS, ASKS)]
    return bt.sim.Market.from_ladders("XYZ", 0.1, states, list(trades))


def _bracket(side, qty, trades, config=None, states=None, at=POST, then=None, **kw):
    """One bracket sent at ``at``, the run to END: the parent's id, what the
    strategy saw (``then(sim, parent)``, called once the bracket is sent), and
    the result."""
    out = {}

    def strategy(sim):
        sim.elapse(at - sim.now())
        out["id"] = sim.post_bracket("XYZ", side, qty, **kw)
        if then is not None:
            out["then"] = then(sim, out["id"])
        sim.elapse(END - sim.now())

    res = bt.sim.run(strategy, config, markets=[_market(trades, states)])
    return out["id"], out.get("then"), res


def _fills(res, order_id):
    cols = res.fills_df(backend="arrow")
    return [
        (round(p, 6), round(q, 6))
        for i, p, q in zip(cols["order_id"], cols["price"], cols["qty"])
        if i == order_id
    ]


def _journal(res, order_id):
    cols = res.events_df(backend="arrow")
    return [
        (t, e)
        for t, i, e in zip(cols["timestamp"], cols["order_id"], cols["event"])
        if i == order_id
    ]


def _orders(res):
    o = res.orders_df(backend="arrow")
    return list(zip(o["order_id"], o["side"], o["qty"], o["status"], o["parent"], o["role"]))


def test_the_venue_arms_the_legs_once_the_parent_has_filled():
    def then(sim, pid):
        sim.elapse(8 * MS - sim.now())
        time_leg = sim.order(4)
        return time_leg.triggered_at, time_leg.trigger_price, sim.order(2).triggered_at

    pid, seen, res = _bracket(
        "buy", 2.0, [], then=then, stop=99.5, target=101.0, exit_after=Interval.millis(5)
    )
    # The exit at a time: no trigger price, the instant its time set it off;
    # the stop it took off was never set off.
    assert seen == (7 * MS, None, None)
    assert pid == 1
    assert _fills(res, 1) == [(100.1, 2.0)]
    # The legs: the next ids, the other side, the size filled; the exit at a
    # time went, and took the other two off.
    assert _orders(res) == [
        (1, "buy", 2.0, "filled", None, None),
        (2, "sell", 2.0, "cancelled", 1, "stop"),
        (3, "sell", 2.0, "cancelled", 1, "target"),
        (4, "sell", 2.0, "filled", 1, "time"),
    ]
    # The exit after 5 ms is due at 7 ms, a sell of 2 at the bid; it takes
    # the stop and the target off.
    assert _fills(res, 4) == [(100.0, 2.0)]
    assert _journal(res, 4) == [(POST, "armed"), (7 * MS, "triggered"), (7 * MS, "fill")]
    assert _journal(res, 2) == [(POST, "armed"), (7 * MS, "cancelled")]
    pd = pytest.importorskip("pandas")
    orders = res.orders_df(backend="pandas")
    assert list(orders.columns) == [
        "order_id", "symbol", "side", "price", "qty", "filled", "avg_price", "status",
        "tif", "trigger_price", "trigger_on", "parent", "role", "sent_at",
    ]
    assert orders["trigger_price"].tolist()[1:3] == [99.5, 101.0]
    assert orders["role"].tolist()[1:] == ["stop", "target", "time"]
    assert pd.isna(orders["trigger_price"].iloc[3])
    # A leg was not sent by the strategy: it was armed at the venue's instant.
    assert orders["sent_at"].iloc[1].value == POST


def test_the_strategy_learns_of_its_legs_with_the_answers():
    config = bt.sim.Config(latency={"entry": MS, "response": MS})

    def then(sim, pid):
        seen = []
        for t in (3 * MS, 4 * MS):
            sim.elapse(t - sim.now())
            seen.append(([o.id for o in sim.orders()], sim.order(2), sim.position("XYZ")))
        armed = sim.orders(status="armed")
        leg = sim.order(3)
        return seen, [o.id for o in armed], (leg.parent, leg.role, leg.trigger_price)

    _, (seen, armed, leg), res = _bracket(
        "buy", 2.0, [], config=config, then=then, stop=99.5, target=101.0
    )
    # Filled and armed at 3 ms at the venue; the strategy hears of it at 4 ms.
    assert seen[0] == ([1], None, 0.0)
    assert seen[1][0] == [1, 2, 3]
    assert seen[1][1].status == "armed"
    assert seen[1][2] == 2.0
    assert armed == [2, 3]
    assert leg == (1, "target", 101.0)
    assert _journal(res, 2) == [(3 * MS, "armed")]


def test_the_first_leg_set_off_takes_the_others_off():
    expected = {
        None: [(100.0, 2.0)],
        "aggressed_side": [(99.4, 1.0), (99.0, 1.0)],
        "both_sides": [(99.4, 1.0), (99.3, 1.0)],
    }
    for rule, fills in expected.items():
        config = bt.sim.Config(prints_update_book=rule, taker_fee_bps=4.0)
        _, _, res = _bracket(
            "buy", 2.0, [SOLD_AT_99_4], config=config, stop=99.5, target=101.0,
            exit_after=Interval.millis(5),
        )
        assert _fills(res, 2) == fills, rule
        assert [s for *_, s, _, _ in _orders(res)] == ["filled", "filled", "cancelled",
                                                        "cancelled"]
        assert _journal(res, 3) == [(POST, "armed"), (4 * MS, "cancelled")]
        assert res.position["XYZ"] == pytest.approx(0.0, abs=1e-12)
        legs = res.fills_df(backend="arrow")
        assert all(c == "taker" for c in legs["channel"])
        assert res.fees == pytest.approx(sum(p * q * 4e-4 for p, q in
                                             _fills(res, 1) + fills))


def test_what_the_book_does_not_fill_of_an_exit_is_carried_to_the_next_event():
    states = [
        (MS, BIDS, ASKS),
        (3 * MS + MS // 2, [(99.4, 2.0)], ASKS),
        (5 * MS, [(99.3, 4.0), (99.2, 3.0)], ASKS),
    ]
    prints = [SOLD_AT_99_4, (6 * MS, 99.2, 1.0, "sell")]
    _, _, res = _bracket("buy", 11.0, prints, states=states, stop=99.5)
    assert _fills(res, 2) == [(99.4, 2.0), (99.3, 4.0), (99.2, 3.0), (99.3, 2.0)]
    ev = res.events_df(backend="arrow")
    carried = [(t, q) for t, e, q in zip(ev["timestamp"], ev["event"], ev["qty"]) if e == "carried"]
    assert carried == [(4 * MS, 9.0), (5 * MS, 2.0)]
    assert _orders(res)[1][3] == "filled"


def test_exit_at_takes_an_instant_a_list_or_the_nth_of_a_list():
    closes = [3 * MS, 6 * MS, 9 * MS]
    for exit_at, due in (
        (6 * MS, 6 * MS),
        (closes, 3 * MS),
        ((closes, 2), 6 * MS),
        (tuple(closes), 3 * MS),
    ):
        _, _, res = _bracket("buy", 2.0, [], exit_at=exit_at)
        assert _journal(res, 2)[:2] == [(POST, "armed"), (due, "triggered")], exit_at
    # The instants as numpy gives them.
    np = pytest.importorskip("numpy")
    for exit_at, due in (
        (np.int64(6 * MS), 6 * MS),
        (np.array(closes), 3 * MS),
        ((np.array(closes), np.int64(2)), 6 * MS),
    ):
        _, _, res = _bracket("buy", 2.0, [], exit_at=exit_at)
        assert _journal(res, 2)[1] == (due, "triggered"), exit_at
    # Past the last instant: armed, never due.
    _, _, res = _bracket("buy", 2.0, [], exit_at=(closes, 4))
    assert _journal(res, 2) == [(POST, "armed")]
    # A duration in any of its spellings.
    for after in (Interval.millis(3), datetime.timedelta(milliseconds=3), 3 * MS):
        _, _, res = _bracket("buy", 2.0, [], exit_after=after)
        assert _journal(res, 2)[1] == (5 * MS, "triggered"), after


def test_a_leg_is_cancelled_alone_and_is_not_replaced():
    def then(sim, pid):
        sim.elapse(3 * MS - sim.now())
        sim.cancel(2)
        with pytest.raises(ValueError, match="leg of a bracket"):
            sim.replace(3, price=100.9)
        return None

    prints = [(5 * MS, 99.0, 1.0, "sell"), (6 * MS, 101.0, 1.0, "buy")]
    _, _, res = _bracket("buy", 2.0, prints, then=then, stop=99.5, target=101.0)
    assert _journal(res, 2) == [(POST, "armed"), (3 * MS, "cancel_sent"), (3 * MS, "cancelled")]
    assert _fills(res, 3) == [(100.0, 2.0)]


def test_a_parent_that_fills_nothing_arms_no_leg():
    def then(sim, pid):
        sim.elapse(3 * MS - sim.now())
        sim.cancel(pid)

    _, _, res = _bracket("buy", 1.0, [], then=then, price=99.9, stop=99.5, target=101.0)
    assert [o[3] for o in _orders(res)] == ["cancelled"]
    _, _, res = _bracket("buy", 1.0, [], price=100.1, tif="GTX", stop=99.5)
    assert [o[3] for o in _orders(res)] == ["rejected"]


def test_a_stop_parent_and_a_limit_parent():
    stop_set_off = [(3 * MS, 100.3, 1.0, "buy"), SOLD_AT_99_4]
    _, _, res = _bracket("buy", 2.0, stop_set_off, trigger=100.3, stop=99.5, target=101.0)
    assert _journal(res, 1)[:3] == [(POST, "sent"), (POST, "armed"), (3 * MS, "triggered")]
    assert _journal(res, 2)[:2] == [(3 * MS, "armed"), (4 * MS, "triggered")]
    # A stop parent on the bid and the ask; its legs on prints.
    _, _, res = _bracket("buy", 2.0, [], trigger=100.1, trigger_on="bid_ask", stop=99.5)
    assert _fills(res, 1) == [(100.1, 2.0)]
    assert _journal(res, 2) == [(POST, "armed")]
    # A limit parent served by a seller through it.
    _, _, res = _bracket("buy", 2.0, [(3 * MS, 99.9, 3.0, "sell")], price=100.0, stop=99.5)
    assert _fills(res, 1) == [(100.0, 2.0)]
    assert _journal(res, 2) == [(3 * MS, "armed")]


def test_legs_on_the_bid_and_the_ask():
    states = [(MS, BIDS, ASKS), (4 * MS, [(99.5, 3.0), (99.0, 9.0)], ASKS)]
    _, _, res = _bracket(
        "buy", 2.0, [(3 * MS, 99.0, 1.0, "sell")], states=states, stop=99.5, target=101.0,
        legs_on="bid_ask",
    )
    assert _fills(res, 2) == [(99.5, 2.0)]
    o = res.orders_df(backend="arrow")
    assert list(o["trigger_on"]) == [None, "bid_ask", "bid_ask"]


# ---------------------------------------------------------------------------
# Refused by name
# ---------------------------------------------------------------------------


def _refused(error, match, *args, **kw):
    def strategy(sim):
        with pytest.raises(error, match=match):
            sim.post_bracket("XYZ", "buy", 1.0, *args, **kw)
        # Nothing was sent.
        assert sim.orders() == []

    bt.sim.run(strategy, markets=[_market([])])


def test_a_bracket_is_refused_by_name_when_the_venue_cannot_hold_it():
    _refused(ValueError, "one with none is an order")
    _refused(ValueError, "its stop below its target", stop=101.0, target=100.5)
    _refused(ValueError, "no one price reaches both", stop=100.43, target=100.37)
    _refused(ValueError, "a bracket's stop must be finite", stop=float("nan"))
    _refused(ValueError, "below the first tick", stop=0.04)
    _refused(ValueError, "give one", stop=99.5, exit_after=MS, exit_at=5 * MS)
    _refused(ValueError, "strictly ascending", exit_at=[6 * MS, 3 * MS])
    _refused(ValueError, "counts from 1", exit_at=([3 * MS], 0))
    _refused(ValueError, "counts n from 1", exit_at=([3 * MS], -1))
    _refused(ValueError, "duration of|duration >= 0|>= 0 ns", exit_after=-1)
    _refused(TypeError, "exit_at takes a venue instant", exit_at="09:30")
    _refused(TypeError, "exit_at takes a venue instant", exit_at=[3.5 * MS])
    _refused(TypeError, "exit_at takes a venue instant", exit_at=True)
    _refused(ValueError, "trigger_on is what sets off a stop parent", trigger_on="last",
             stop=99.5)
    _refused(ValueError, 'legs_on must be "last"', stop=99.5, legs_on="mid")
    _refused(ValueError, "GTC or GTD", trigger=100.3, tif="IOC", stop=99.5)
    _refused(TypeError, "positional", 100.0)  # the price is a keyword
    _refused(TypeError, "unexpected keyword", stop_loss=99.5)

    # A leg the venue armed, the answer still on its way: not the strategy's
    # to cancel yet.
    def strategy(sim):
        sim.post_bracket("XYZ", "buy", 1.0, stop=99.5)
        sim.elapse(0)
        assert sim.order(2) is None
        with pytest.raises(ValueError, match="not heard of yet"):
            sim.cancel(2)
        with pytest.raises(ValueError, match="no order 3 was sent"):
            sim.cancel(3)

    bt.sim.run(strategy, bt.sim.Config(latency={"response": MS}), markets=[_market([])])


# ---------------------------------------------------------------------------
# Against a trade-driven top-of-book engine
# ---------------------------------------------------------------------------

# One equity on a 0.01 tick, one case per run: L1 quotes (ts ms, bid, size,
# ask, size) and prints (ts ms, price, size, aggressor); at one instant the
# print comes before the quote. The bracket is sent with zero latency on the
# first quote, or right after print `k` ("print:k"): the entry (side, kind,
# price or trigger, size, GTD expiry in ms or None), the stop, the target,
# what sets them off, an exit after so many ms, and a queue model when the
# case needs the resting entry at the front of its queue. The venue runs
# prints_update_book="both_sides" ("stored_book": without it).
Q0 = (1, 10.00, 500, 10.01, 300)
CASES = {
    "stop_set_off": ([Q0], [(3, 10.00, 100, "sell"), (5, 9.94, 200, "sell")], "quote",
                     ("buy", "market", None, 100, None), 9.95, 10.10, "last", None, None),
    "target_set_off": ([Q0], [(3, 10.05, 100, "buy"), (5, 10.12, 50, "buy")], "quote",
                       ("buy", "market", None, 100, None), 9.95, 10.10, "last", None, None),
    "both_at_one_instant": ([Q0], [(5, 9.94, 50, "sell"), (5, 10.12, 50, "buy")], "quote",
                            ("buy", "market", None, 100, None), 9.95, 10.10, "last", None,
                            None),
    "limit_entry": ([Q0], [(3, 9.99, 200, "sell"), (5, 9.94, 200, "sell")], "quote",
                    ("buy", "limit", 10.00, 100, None), 9.95, 10.10, "last", None, None),
    "partial_entry_expires": ([Q0, (5, 10.00, 500, 10.01, 300)],
                              [(3, 10.00, 60, "sell"), (6, 9.94, 200, "sell")], "quote",
                              ("buy", "limit", 10.00, 100, 4), 9.95, 10.10, "last", None,
                              "front"),
    "partial_entry_then_stop": ([Q0], [(3, 10.00, 60, "sell"), (5, 9.94, 200, "sell")],
                                "quote", ("buy", "limit", 10.00, 100, None), 9.95, 10.10,
                                "last", None, "front"),
    "stop_entry": ([Q0], [(3, 10.03, 200, "buy"), (5, 10.10, 100, "buy")], "quote",
                   ("buy", "stop_market", 10.03, 100, None), 9.98, 10.10, "last", None, None),
    "stop_reached_when_armed": ([Q0], [(2, 9.90, 100, "sell")], "print:0",
                                ("buy", "market", None, 100, None), 9.95, 10.10, "last",
                                None, None),
    "target_reached_when_armed": ([Q0], [(2, 10.15, 100, "buy")], "print:0",
                                  ("buy", "market", None, 100, None), 9.95, 10.10, "last",
                                  None, None),
    "thin_bid": ([Q0, (4, 9.96, 30, 10.01, 300), (6, 9.95, 500, 9.96, 300)],
                 [(5, 9.94, 10, "sell")], "quote", ("buy", "market", None, 100, None), 9.95,
                 10.10, "last", None, None),
    "exit_at_a_time": ([Q0, (3, 10.02, 500, 10.03, 300)], [(4, 10.02, 100, "sell")], "quote",
                       ("buy", "market", None, 100, None), 9.95, 10.10, "last", 5, None),
    "exit_due_when_the_stop_is_set_off": ([Q0], [(6, 9.94, 200, "sell")], "quote",
                                          ("buy", "market", None, 100, None), 9.95, 10.10,
                                          "last", 5, None),
    "legs_on_the_bid_and_the_ask": ([Q0, (5, 9.95, 500, 9.96, 300)], [(3, 9.90, 10, "sell")],
                                    "quote", ("buy", "market", None, 100, None), 9.95, 10.10,
                                    "bid_ask", None, None),
}

# The fills of nautilus_trader 1.231.0 and 2.0.0rc5 on the same data
# (BacktestEngine, BookType.L1_MBP, trade_execution=True, a margin account,
# FillModel left as it is; the bracket an order list, the entry OTO, the
# stop-market and the market-if-touched target OCO, triggered on the last
# price or on the bid and the ask; the exit at a time a time alert that
# cancels the legs and sends a market order), as (role, price, size, ms).
# The two versions give the same here.
BOTH_VERSIONS = {
    "stop_set_off": [("entry", 10.01, 100.0, 1), ("stop", 9.94, 100.0, 5)],
    "limit_entry": [("entry", 10.0, 100.0, 3), ("stop", 9.94, 100.0, 5)],
    "stop_entry": [("entry", 10.03, 100.0, 3), ("target", 10.1, 100.0, 5)],
    "exit_at_a_time": [("entry", 10.01, 100.0, 1), ("time", 10.02, 100.0, 6)],
    "exit_due_when_the_stop_is_set_off": [("entry", 10.01, 100.0, 1), ("stop", 9.94, 100.0, 6)],
}
# Here they part, and this venue gives what one of them gives.
ONE_VERSION = {
    # 1.231.0 cancels the rest of the leg that filled in part (its OCO
    # sibling's cancellation cancels it back), and leaves it open.
    "target_set_off": ("2.0.0rc5", [("entry", 10.01, 100.0, 1), ("target", 10.12, 50.0, 5),
                                    ("target", 10.11, 50.0, 5)]),
    "thin_bid": ("2.0.0rc5", [("entry", 10.01, 100.0, 1), ("stop", 9.94, 10.0, 5),
                              ("stop", 9.93, 90.0, 5)]),
    # 2.0.0rc5 sets off the stop, grown to the 100 filled, only at the next
    # quote; and refuses a target already reached, the stop left armed.
    "partial_entry_then_stop": ("1.231.0", [("entry", 10.0, 60.0, 3), ("entry", 10.0, 40.0, 5),
                                            ("stop", 9.94, 100.0, 5)]),
    "target_reached_when_armed": ("1.231.0", [("entry", 10.15, 100.0, 2),
                                              ("target", 10.15, 100.0, 2)]),
}


def _case(name, rule="both_sides"):
    quotes, prints, send, entry, sl, tp, legs_on, exit_ms, queue = CASES[name]
    side, kind, level, qty, expire_ms = entry
    states = [(t * MS, [(b, float(bq))], [(a, float(aq))]) for t, b, bq, a, aq in quotes]
    trades = [(t * MS, p, float(q), agg) for t, p, q, agg in prints]
    market = bt.sim.Market.from_ladders("XYZ", 0.01, states, trades)
    out = {}

    def strategy(sim):
        kw = {"stop": sl, "target": tp, "legs_on": legs_on}
        if kind == "limit":
            kw["price"] = level
        elif kind == "stop_market":
            kw["trigger"] = level
        if expire_ms is not None:
            kw.update(tif="GTD", expire_at=expire_ms * MS)
        if exit_ms is not None:
            kw["exit_after"] = exit_ms * MS
        if send.startswith("print:"):
            k, seen = int(send.split(":")[1]), 0
            while seen <= k:
                sim.wait(trade=True)
                seen += len(sim.trades("XYZ"))
        out["id"] = sim.post_bracket("XYZ", side, float(qty), **kw)
        sim.elapse(30 * MS)

    front = {"depth_source": "assumed", "assumed_queue": 0.0} if queue == "front" else None
    config = bt.sim.Config(prints_update_book=rule, queue=front)
    res = bt.sim.run(strategy, config, markets=[market])
    o = res.orders_df(backend="arrow")
    role = {i: ("entry" if i == out["id"] else r) for i, r in zip(o["order_id"], o["role"])}
    cols = res.fills_df(backend="arrow")
    return [
        (role[i], round(p, 6), q, t // MS)
        for t, i, p, q in zip(cols["timestamp"], cols["order_id"], cols["price"], cols["qty"])
    ]


def test_where_the_two_conventions_meet_the_fills_are_the_same():
    for name, fills in BOTH_VERSIONS.items():
        assert _case(name) == fills, name
    for name, (_, fills) in ONE_VERSION.items():
        assert _case(name) == fills, name


def test_where_they_part_each_part_is_a_rule_of_this_venue():
    # Two prints at one instant reach the stop, then the target: the stop was
    # set off first and the target went with it; the stop executes at the end
    # of the instant, against what the last print made of the book. The
    # trade-driven engine executes it on the first print.
    assert _case("both_at_one_instant") == [
        ("entry", 10.01, 100.0, 1), ("stop", 10.12, 50.0, 5), ("stop", 10.11, 50.0, 5),
    ]
    # An entry that expires after a fill in part arms the legs for what it
    # filled; the trade-driven engine cancels them with the entry, the 60
    # left open.
    assert _case("partial_entry_expires") == [
        ("entry", 10.0, 60.0, 3), ("stop", 9.94, 60.0, 6),
    ]
    # A stop the market already reaches when the legs are armed is set off at
    # once; the trade-driven engine refuses it, and the target with it.
    assert _case("stop_reached_when_armed") == [
        ("entry", 9.9, 100.0, 2), ("stop", 9.9, 100.0, 2),
    ]
    # Legs on the bid and the ask read the stored quotes alone: the print at
    # 9.90 sets nothing off, the quote at 5 ms does. The trade-driven engine
    # reads its bid and ask, which the print moved.
    assert _case("legs_on_the_bid_and_the_ask") == [
        ("entry", 10.01, 100.0, 1), ("stop", 9.95, 100.0, 5),
    ]
    # On the stored book alone, what the thin bid cannot fill waits for the
    # next event; the trade-driven engine fills it a tick worse (2.0.0rc5).
    assert _case("thin_bid", rule=None) == [
        ("entry", 10.01, 100.0, 1), ("stop", 9.96, 30.0, 5), ("stop", 9.95, 70.0, 6),
    ]
