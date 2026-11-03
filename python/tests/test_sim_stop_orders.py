"""Stop and stop-limit orders held by the simulated venue: ``sim.post(...,
trigger=p)``.

The rules are pinned by the engine's own tests (``stop_orders.rs``); this file
checks that Python reaches them, that the order and the tables carry the
stop, that every word is refused by name when it is not one the venue reads,
and the fills on a set of cases where the trade-driven top-of-book convention
and this venue meet, or part on a rule named below.

The market of the first tests is the one of ``test_sim.py``: bids 100.0 x 5,
99.9 x 9, 99.0 x 9, asks 100.1 x 5, 100.5 x 6, on a 0.1 tick.
"""
import json

import pytest

import manifoldbt as bt

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
BOUGHT_AT_100_3 = (3 * MS, 100.3, 1.0, "buy")


def _market(trades, states=None):
    states = states if states is not None else [(MS, BIDS, ASKS)]
    return bt.sim.Market.from_ladders("XYZ", 0.1, states, list(trades))


def _stop(side, price, qty, trades, config=None, states=None, at=POST, **stop):
    """One stop sent at ``at``, the run to END: its fills, the order as the
    strategy last knew it, and the result."""
    out = {}

    def strategy(sim):
        sim.elapse(at - sim.now())
        out["id"] = sim.post("XYZ", side, price, qty, **stop)
        sim.elapse(END - sim.now())
        out["order"] = sim.order(out["id"])

    res = bt.sim.run(strategy, config, markets=[_market(trades, states)])
    cols = res.fills_df(backend="arrow")
    fills = [
        (round(p, 6), round(q, 6))
        for i, p, q in zip(cols["order_id"], cols["price"], cols["qty"])
        if i == out["id"]
    ]
    return fills, out["order"], res


def _journal(res, order_id):
    e = res.events_df(backend="arrow")
    return [(t, k) for t, i, k in zip(e["timestamp"], e["order_id"], e["event"]) if i == order_id]


# ---------------------------------------------------------------------------
# The venue's verdicts, reached from Python
# ---------------------------------------------------------------------------


def test_a_stop_market_waits_for_a_print_at_or_through_its_trigger():
    prints = [(2 * MS + MS // 2, 100.2, 1.0, "buy"), BOUGHT_AT_100_3]
    expected = {
        None: [(100.1, 2.0)],
        "aggressed_side": [(100.3, 1.0), (100.5, 1.0)],
        "both_sides": [(100.3, 1.0), (100.4, 1.0)],
    }
    for rule, want in expected.items():
        config = bt.sim.Config(prints_update_book=rule)
        fills, order, res = _stop("buy", None, 2.0, prints, config, trigger=100.3)
        assert fills == want, rule
        assert order.status == "filled"
        stop = (order.trigger_price, order.trigger_on, order.triggered_at)
        assert stop == (100.3, "last", 3 * MS)
        # A stop without a limit has nothing to rest: no after_trigger.
        assert order.after_trigger is None
        kinds = [k for _, k in _journal(res, order.id)]
        assert kinds[:3] == ["sent", "armed", "triggered"]
        assert set(kinds[3:]) == {"fill"}
        assert res.fills_df(backend="arrow")["channel"] == ["taker"] * len(want)


def test_a_stop_on_the_bid_and_the_ask_reads_the_stored_states_alone():
    states = [
        (MS, BIDS, ASKS),
        (3 * MS, BIDS, [(100.2, 4.0), (100.5, 6.0)]),
        (4 * MS, BIDS, [(100.3, 4.0), (100.5, 6.0)]),
    ]
    # A buyer at 101.0 between the states sets off no stop on the quotes.
    prints = [(3 * MS + MS // 2, 101.0, 1.0, "buy")]
    for rule in (None, "aggressed_side", "both_sides"):
        config = bt.sim.Config(prints_update_book=rule)
        fills, order, _ = _stop(
            "buy", None, 2.0, prints, config, states, trigger=100.3, trigger_on="bid_ask"
        )
        assert fills == [(100.3, 2.0)], rule
        assert (order.trigger_on, order.triggered_at) == ("bid_ask", 4 * MS)


def test_a_stop_limit_that_does_not_execute_rests_or_expires():
    through = [BOUGHT_AT_100_3, (5 * MS, 99.9, 1.0, "sell")]
    # It rests at 100.0, behind the 5 shown there; a seller through 100.0
    # serves it whole, as the maker it is by then.
    fills, order, res = _stop("buy", 100.0, 8.0, through, trigger=100.3)
    assert fills == [(100.0, 8.0)]
    assert res.fills_df(backend="arrow")["channel"] == ["traverse"]
    assert order.after_trigger == "rest"
    e = res.events_df(backend="arrow")
    accepted = [q for k, q in zip(e["event"], e["queue_ahead"]) if k == "accepted"]
    assert accepted == [5.0]
    # Or it expires once set off.
    fills, order, _ = _stop("buy", 100.0, 8.0, through, trigger=100.3, after_trigger="ioc")
    assert (fills, order.status, order.after_trigger) == ([], "expired", "ioc")
    # A limit that executes takes the ladder within it, its rest expires.
    fills, order, _ = _stop("buy", 100.5, 8.0, [BOUGHT_AT_100_3], trigger=100.3)
    assert (fills, order.status) == ([(100.1, 5.0), (100.5, 3.0)], "filled")


def test_a_stop_already_in_the_market_is_set_off_or_refused():
    prints = [(3 * MS, 100.6, 1.0, "buy")]
    fills, order, res = _stop("buy", None, 2.0, prints, at=4 * MS, trigger=100.5)
    assert (fills, order.status) == ([(100.1, 2.0)], "filled")
    assert [k for _, k in _journal(res, order.id)] == ["sent", "triggered", "fill"]
    reject = bt.sim.Config(stop_in_market="reject")
    fills, order, res = _stop("buy", None, 2.0, prints, reject, at=4 * MS, trigger=100.5)
    assert (fills, order.status) == ([], "rejected")
    assert [k for _, k in _journal(res, order.id)] == ["sent", "rejected"]


def test_a_gtd_stop_expires_after_the_market_of_its_instant():
    late = [(5 * MS, 100.3, 1.0, "buy")]
    fills, order, res = _stop("buy", None, 2.0, late, trigger=100.3, tif="GTD", expire_at=4 * MS)
    assert (fills, order.status) == ([], "expired")
    assert _journal(res, order.id)[-1] == (4 * MS, "expired")
    on_time = [(4 * MS, 100.3, 1.0, "buy")]
    fills, order, _ = _stop("buy", None, 2.0, on_time, trigger=100.3, tif="GTD", expire_at=4 * MS)
    assert (fills, order.status) == ([(100.1, 2.0)], "filled")


def test_the_strategy_sees_its_stop_armed_and_can_cancel_it():
    seen = {}

    def strategy(sim):
        sim.elapse(POST - sim.now())
        oid = sim.post("XYZ", "buy", None, 2.0, trigger=100.3)
        assert sim.order(oid).status == "in_flight"
        sim.elapse(MS // 2)
        seen["armed"] = [o.id for o in sim.orders(status="armed")]
        seen["live"] = [o.id for o in sim.orders(status="live")]
        seen["resting"] = [o.id for o in sim.orders(status="resting")]
        with pytest.raises(ValueError, match="cancel it and post another"):
            sim.replace(oid, price=100.4)
        sim.cancel(oid)
        sim.elapse(END - sim.now())
        seen["order"] = sim.order(oid)

    bt.sim.run(strategy, markets=[_market([(4 * MS, 100.3, 1.0, "buy")])])
    assert seen["armed"] == seen["live"] == [1]
    assert seen["resting"] == []
    assert (seen["order"].status, seen["order"].filled) == ("cancelled", 0.0)


def test_the_tables_carry_the_stop():
    _, order, res = _stop("buy", 100.5, 2.0, [BOUGHT_AT_100_3], trigger=100.33)
    # The trigger asked at 100.33 is at 100.4, away from the market: the
    # print at 100.3 is short of it.
    assert (order.trigger_price, order.status, order.triggered_at) == (100.4, "armed", None)
    assert res.levels_snapped == 1
    pd = pytest.importorskip("pandas")
    orders = res.orders_df(backend="pandas")
    assert list(orders.columns) == [
        "order_id", "symbol", "side", "price", "qty", "filled", "avg_price", "status",
        "tif", "trigger_price", "trigger_on", "parent", "role", "sent_at",
    ]
    assert (orders["trigger_price"].iloc[0], orders["trigger_on"].iloc[0]) == (100.4, "last")
    assert orders["status"].tolist() == ["armed"]
    # An order that is not a stop carries none.
    _, plain, res = _stop("buy", 100.0, 1.0, [])
    assert (plain.trigger_price, plain.trigger_on, plain.triggered_at) == (None, None, None)
    assert pd.isna(res.orders_df(backend="pandas")["trigger_price"].iloc[0])


# ---------------------------------------------------------------------------
# Refused by name
# ---------------------------------------------------------------------------


def _refused(match, **kw):
    def strategy(sim):
        with pytest.raises(ValueError, match=match):
            sim.post("XYZ", "buy", kw.pop("price", None), 1.0, **kw)

    bt.sim.run(strategy, markets=[_market([])])


def test_a_stop_is_refused_by_name_when_the_venue_cannot_hold_it():
    for tif in ("IOC", "FOK", "GTX"):
        _refused("GTC or GTD", trigger=100.3, tif=tif)
    _refused("after_trigger is what a stop-limit", trigger=100.3, after_trigger="ioc")
    _refused('trigger_on must be "last"', trigger=100.3, trigger_on="mid")
    _refused('after_trigger must be "rest"', price=100.5, trigger=100.3, after_trigger="later")
    _refused("belong to a stop order", trigger_on="last")
    _refused("belong to a stop order", price=100.0, after_trigger="rest")
    _refused("trigger must be finite", trigger=float("nan"))
    _refused("trigger must be finite", trigger=-1.0)

    def strategy(sim):
        with pytest.raises(TypeError, match="unexpected keyword argument 'trigger_price'"):
            sim.post("XYZ", "buy", None, 1.0, trigger_price=100.3)
        with pytest.raises(TypeError, match="trigger_on takes a str"):
            sim.post("XYZ", "buy", None, 1.0, trigger=100.3, trigger_on=1)
        # None is a keyword left out.
        oid = sim.post("XYZ", "buy", None, 1.0, trigger=None, trigger_on=None)
        assert sim.order(oid).trigger_price is None

    bt.sim.run(strategy, markets=[_market([])])


def test_stop_in_market_is_one_of_two_words():
    market = _market([])
    for value in ("refuse", True, None, "REJECT"):
        with pytest.raises(ValueError, match='stop_in_market takes "trigger"'):
            bt.sim.run(lambda s: None, bt.sim.Config(stop_in_market=value), markets=[market])
    assert "stop_in_market" not in json.loads(bt.sim.Config()._to_json())
    reject = bt.sim.Config(stop_in_market="reject")
    assert json.loads(reject._to_json())["stop_in_market"] == "reject"


# ---------------------------------------------------------------------------
# Against a trade-driven top-of-book engine
# ---------------------------------------------------------------------------

# One equity on a 0.01 tick, one case per run: L1 quotes (ts ms, bid, size,
# ask, size) and prints (ts ms, price, size, aggressor); at one instant the
# print comes before the quote. The stop is sent with zero latency on the
# first quote, or right after print `k` ("print:k"): (side, trigger, limit,
# size, expire_at in ms or None, trigger_on). The venue runs
# prints_update_book="both_sides"; "reject" is stop_in_market="reject".
CASES = {
    "stop_market_buy": ([(1, 10.00, 500, 10.01, 300)],
                        [(3, 10.02, 200, "buy"), (5, 10.04, 50, "buy")],
                        "quote", ("buy", 10.03, None, 100, None, "last"), False),
    "stop_market_sell": ([(1, 10.00, 500, 10.01, 300)],
                         [(3, 9.99, 100, "sell"), (5, 9.97, 300, "sell")],
                         "quote", ("sell", 9.98, None, 100, None, "last"), False),
    "stop_limit_executes": ([(1, 10.00, 500, 10.01, 300), (8, 10.04, 100, 10.05, 100)],
                            [(5, 10.04, 50, "buy")],
                            "quote", ("buy", 10.03, 10.05, 100, None, "last"), False),
    "stop_limit_rests": ([(1, 10.00, 500, 10.01, 300), (8, 10.01, 100, 10.02, 200)],
                         [(5, 10.04, 50, "buy")],
                         "quote", ("buy", 10.03, 10.02, 100, None, "last"), False),
    "in_the_market": ([(1, 10.00, 500, 10.01, 300)], [(2, 10.04, 100, "buy")],
                      "print:0", ("buy", 10.03, None, 100, None, "last"), False),
    "in_the_market_rejected": ([(1, 10.00, 500, 10.01, 300)], [(2, 10.04, 100, "buy")],
                               "print:0", ("buy", 10.03, None, 100, None, "last"), True),
    "gtd_quote_before_the_print": ([(1, 10.00, 500, 10.01, 300), (5, 10.00, 500, 10.01, 300)],
                                   [(6, 10.04, 50, "buy")],
                                   "quote", ("buy", 10.03, None, 100, 4, "last"), False),
    "gtd_nothing_before_the_print": ([(1, 10.00, 500, 10.01, 300)], [(6, 10.04, 50, "buy")],
                                     "quote", ("buy", 10.03, None, 100, 4, "last"), False),
    "gtd_print_at_its_expiry": ([(1, 10.00, 500, 10.01, 300)], [(6, 10.04, 50, "buy")],
                                "quote", ("buy", 10.03, None, 100, 6, "last"), False),
    "print_and_quote_at_one_instant": ([(1, 10.00, 500, 10.01, 300), (5, 10.06, 100, 10.07, 100)],
                                       [(5, 10.04, 50, "buy")],
                                       "quote", ("buy", 10.03, None, 100, None, "last"), False),
    "ask_through_no_print": ([(1, 10.00, 500, 10.01, 300), (5, 10.03, 100, 10.04, 200)], [],
                             "quote", ("buy", 10.03, None, 100, None, "bid_ask"), False),
}

# The fills of nautilus_trader 1.231.0 and 2.0.0rc5 on the same data
# (BacktestEngine, BookType.L1_MBP, trade_execution=True, a margin account,
# FillModel left as it is, reject_stop_orders as the case says, the stop sent
# from the quote or trade callback), as (price, size, ms), and how it ended.
# The two versions give the same, but on "stop_limit_executes".
TRADE_DRIVEN = {
    "stop_market_buy": ([(10.04, 50.0, 5), (10.05, 50.0, 5)], "filled"),
    "stop_market_sell": ([(9.97, 100.0, 5)], "filled"),
    "stop_limit_rests": ([(10.02, 100.0, 8)], "filled"),
    "in_the_market": ([(10.04, 100.0, 2)], "filled"),
    "in_the_market_rejected": ([], "rejected"),
    "gtd_quote_before_the_print": ([], "expired"),
    "gtd_nothing_before_the_print": ([(10.04, 50.0, 6), (10.05, 50.0, 6)], "filled"),
    "gtd_print_at_its_expiry": ([(10.04, 50.0, 6), (10.05, 50.0, 6)], "filled"),
    "print_and_quote_at_one_instant": ([(10.04, 50.0, 5), (10.05, 50.0, 5)], "filled"),
    "ask_through_no_print": ([(10.04, 100.0, 5)], "filled"),
}
TRADE_DRIVEN_STOP_LIMIT_EXECUTES = {
    "1.231.0": ([(10.03, 50.0, 5), (10.04, 50.0, 5)], "filled"),
    "2.0.0rc5": ([(10.04, 50.0, 5), (10.05, 50.0, 8)], "filled"),
}


def _case(name, trigger_on=None):
    quotes, prints, send, order, reject = CASES[name]
    side, trigger, limit, qty, expire_ms, on = order
    states = [(t * MS, [(b, float(bq))], [(a, float(aq))]) for t, b, bq, a, aq in quotes]
    trades = [(t * MS, p, float(q), agg) for t, p, q, agg in prints]
    market = bt.sim.Market.from_ladders("XYZ", 0.01, states, trades)
    out = {}

    def strategy(sim):
        kw = {"trigger": trigger, "trigger_on": trigger_on or on}
        if expire_ms is not None:
            kw.update(tif="GTD", expire_at=expire_ms * MS)
        if send.startswith("print:"):
            k, seen = int(send.split(":")[1]), 0
            while seen <= k:
                sim.wait(trade=True)
                seen += len(sim.trades("XYZ"))
        out["id"] = sim.post("XYZ", side, limit, float(qty), **kw)
        sim.elapse(20 * MS)

    config = bt.sim.Config(
        prints_update_book="both_sides", stop_in_market="reject" if reject else "trigger"
    )
    res = bt.sim.run(strategy, config, markets=[market])
    cols = res.fills_df(backend="arrow")
    fills = [
        (round(p, 6), q, t // MS)
        for t, i, p, q in zip(cols["timestamp"], cols["order_id"], cols["price"], cols["qty"])
        if i == out["id"]
    ]
    return fills, res.orders_df(backend="arrow")["status"][0]


def test_where_the_two_conventions_meet_the_fills_are_the_same():
    for name in (
        "stop_market_buy",
        "stop_market_sell",
        "stop_limit_rests",
        "in_the_market",
        "in_the_market_rejected",
        "gtd_quote_before_the_print",
        "gtd_print_at_its_expiry",
        "ask_through_no_print",
    ):
        assert _case(name) == TRADE_DRIVEN[name], name


def test_where_they_part_each_part_is_a_rule_of_this_venue():
    # A stop-limit whose limit executes takes what the book offers within it
    # and its rest expires, as any limit executing on arrival here; the
    # trade-driven engine rests it (2.0.0rc5) or fills it at the trigger
    # (1.231.0).
    assert _case("stop_limit_executes") == ([(10.04, 50.0, 5)], "expired")
    assert _case("stop_limit_executes") not in TRADE_DRIVEN_STOP_LIMIT_EXECUTES.values()
    # A GTD expires at its instant here; the trade-driven engine expires it at
    # its next market event, after matching it against that event.
    assert _case("gtd_nothing_before_the_print") == ([], "expired")
    # A quote stamped with the print that set the stop off is the newer word:
    # the stop meets it, at the end of the instant. The trade-driven engine
    # executes on the print.
    assert _case("print_and_quote_at_one_instant") == ([(10.07, 100.0, 5)], "filled")
    # On prints alone, an ask that moves through the trigger without a print
    # sets nothing off; the trade-driven engine reads its bid and ask.
    assert _case("ask_through_no_print", trigger_on="last") == ([], "armed")
