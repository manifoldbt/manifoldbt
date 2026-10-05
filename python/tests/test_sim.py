"""``bt.sim``: a strategy function driving a simulated venue.

The venue's rules are pinned by the engine's own tests: the duel's
minimal queue scenarios, the four latencies, our own orders at one price, the
order types. This file checks that Python sees the same verdicts through the
binding, the views a strategy reads, the tables a run hands back, and the
refusals, by name.

Every market here is written out by hand (``bt.sim.Market.from_ladders``): bids
100.0 (the size under test), 99.9 x 9 and 99.0 x 9, asks 100.1 x 5 and 100.5 x
6, on a 0.1 tick; a buy at 100.0 posted at 2 ms and read at 10 ms.
"""
import os

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


def _ladder(ts, size_at_100):
    return (ts, [(100.0, size_at_100), (99.9, 9.0), (99.0, 9.0)], [(100.1, 5.0), (100.5, 6.0)])


def _market(states, trades=()):
    return bt.sim.Market.from_ladders("BTCUSDT", 0.1, list(states), list(trades))


def _served(states, trades, config=None):
    """Post a buy of 1 at 100.0 at POST, run to END: what filled."""
    out = {}

    def strategy(sim):
        sim.elapse(POST - sim.now())
        out["id"] = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sim.elapse(END - sim.now())
        out["order"] = sim.order(out["id"])

    bt.sim.run(strategy, config, markets=[_market(states, trades)])
    return round(out["order"].filled, 6)


# ---------------------------------------------------------------------------
# The same verdicts as the venue's own tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "states, trades, filled",
    [
        # A. A seller through our price serves the order.
        ([_ladder(MS, 5.0)], [(3 * MS, 99.9, 10.0, "sell")], 1.0),
        # C / D. Prints past a short queue serve it, short of a long one do not.
        ([_ladder(MS, 2.0)], [(3 * MS, 100.0, 3.0, "sell")], 1.0),
        ([_ladder(MS, 10.0)], [(3 * MS, 100.0, 3.0, "sell")], 0.0),
        # H. A buyer through our price serves it too.
        ([_ladder(MS, 5.0)], [(3 * MS, 99.9, 10.0, "buy")], 1.0),
        # K. What a print leaves past the queue is a partial fill.
        ([_ladder(MS, 2.995)], [(3 * MS, 100.0, 3.0, "sell")], 0.005),
        # G. The market of the arrival instant comes before the order.
        ([_ladder(MS, 5.0), _ladder(POST, 3.0)],
         [(POST, 100.0, 2.0, "sell"), (4 * MS, 100.0, 2.5, "sell")], 0.0),
        # N. A print and the state that shows it count once.
        ([_ladder(MS, 5.0), _ladder(3 * MS, 3.0)],
         [(3 * MS, 100.0, 2.0, "sell"), (4 * MS, 100.0, 1.5, "sell")], 0.0),
    ],
    ids=["A", "C", "D", "H", "K", "G", "N"],
)
def test_the_duel_scenarios_give_the_venue_s_verdicts(states, trades, filled):
    assert _served(states, trades) == filled


def test_an_order_is_in_nobody_s_queue_before_it_arrives():
    config = bt.sim.Config(latency={"entry": bt.Interval.millis(2)})
    assert _served([_ladder(MS, 5.0)], [(3 * MS, 99.9, 10.0, "sell")], config) == 0.0


def test_the_strategy_learns_of_a_fill_one_response_later():
    seen = []

    def strategy(sim):
        sim.elapse(POST - sim.now())
        oid = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        for _ in range(2):
            w = sim.wait(order=True)
            assert (w.kind, w.order_id) == ("order", oid)
            seen.append((sim.now(), sim.order(oid).status, sim.position("BTCUSDT")))

    config = bt.sim.Config(latency={"response": MS})
    bt.sim.run(strategy, config, markets=[_market([_ladder(MS, 5.0)], [(3 * MS, 99.9, 10.0, "sell")])])
    assert seen == [(3 * MS, "resting", 0.0), (4 * MS, "filled", 1.0)]


# ---------------------------------------------------------------------------
# What the strategy reads
# ---------------------------------------------------------------------------


def test_the_book_is_seen_one_feed_later():
    books = []

    def strategy(sim):
        books.append(sim.book("BTCUSDT"))
        sim.elapse(MS)
        books.append(sim.book("BTCUSDT"))

    config = bt.sim.Config(latency={"feed": MS})
    bt.sim.run(strategy, config, markets=[_market([_ladder(MS, 5.0)])])
    empty, book = books
    assert (empty.best_bid, empty.ts_ns) == (None, None)
    assert (book.best_bid, book.best_ask, book.ts_ns, book.tick) == (100.0, 100.1, MS, 0.1)
    assert (book.best_bid_qty, book.best_ask_qty) == (5.0, 5.0)
    assert round(book.mid, 6) == 100.05 and round(book.spread, 6) == 0.1
    assert (book.qty_at(99.9), book.qty_at(100.5), book.qty_at(100.3)) == (9.0, 6.0, 0.0)
    assert book.levels(1) == ([(100.0, 5.0)], [(100.1, 5.0)])
    assert book.depth_through(99.9) == 14.0
    assert book.depth_through(100.5) == 11.0
    assert book.depth_through(100.05) == 0.0


def test_trades_hands_over_the_prints_not_yet_read():
    batches = []
    trades = [(3 * MS, 100.0, 1.0, "sell"), (4 * MS, 100.1, 2.0, "buy")]

    def strategy(sim):
        sim.elapse(3 * MS - sim.now())
        batches.append(sim.trades("BTCUSDT"))
        batches.append(sim.trades("BTCUSDT"))
        sim.elapse(MS)
        batches.append(sim.trades(0))

    bt.sim.run(strategy, markets=[_market([_ladder(MS, 5.0)], trades)])
    assert batches == [
        [(3 * MS, 100.0, 1.0, "sell")],
        [],
        [(4 * MS, 100.1, 2.0, "buy")],
    ]


def test_orders_filter_by_side_and_status():
    views = {}

    def strategy(sim):
        sim.elapse(POST - sim.now())
        bid = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        ask = sim.post("BTCUSDT", "sell", 100.5, 2.0, tif="GTX")
        sim.elapse(0)
        views["live"] = [o.id for o in sim.orders(status="live")]
        views["buys"] = [o.id for o in sim.orders("BTCUSDT", side="buy")]
        views["resting"] = [o.id for o in sim.orders(status="resting")]
        o = sim.order(ask)
        views["ask"] = (o.side, o.price, o.qty, o.remaining, o.tif, o.status, o.queue_ahead)
        sim.cancel(bid)
        views["pending"] = sim.order(bid).cancel_pending
        sim.elapse(MS)
        views["after"] = [o.id for o in sim.orders(status="cancelled")]
        views["ids"] = (bid, ask)

    bt.sim.run(strategy, markets=[_market([_ladder(MS, 5.0)])])
    bid, ask = views["ids"]
    assert views["live"] == [bid, ask]
    assert views["buys"] == [bid]
    assert views["resting"] == [bid, ask]
    assert views["ask"] == ("sell", 100.5, 2.0, 2.0, "GTX", "resting", 6.0)
    assert views["pending"] is True
    assert views["after"] == [bid]


def test_an_order_read_again_shows_what_changed_and_a_kept_one_does_not():
    seen = {}

    def strategy(sim):
        sim.elapse(POST - sim.now())
        oid = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sent = sim.order(oid)
        again = sim.order(oid)
        sim.cancel(oid)
        cancelling = sim.order(oid)
        sim.elapse(MS)
        done = sim.order(oid)
        seen["kept"] = [(o.status, o.cancel_pending) for o in (sent, again, cancelling, done)]
        seen["listed"] = [o.status for o in sim.orders(status="cancelled")]

    config = bt.sim.Config(latency={"entry": MS // 2, "response": MS // 4})
    bt.sim.run(strategy, config, markets=[_market([_ladder(MS, 5.0)])])
    assert seen["kept"] == [
        ("in_flight", False),
        ("in_flight", False),
        ("in_flight", True),
        ("cancelled", False),
    ]
    assert seen["listed"] == ["cancelled"]


def test_wait_says_why_it_returned():
    kinds = []

    def strategy(sim):
        # The clock starts on the first state, at 1 ms: the print of 3 ms
        # comes next, then the state of 5 ms, then nothing.
        kinds.append((sim.wait(trade=True), sim.now()))
        kinds.append((sim.wait(book=True), sim.now()))
        kinds.append((sim.wait(timeout=bt.Interval.millis(1)), sim.now()))
        kinds.append((sim.wait(), sim.now()))

    states = [_ladder(MS, 5.0), _ladder(5 * MS, 5.0)]
    bt.sim.run(strategy, markets=[_market(states, [(3 * MS, 100.0, 1.0, "sell")])])
    assert [(w.kind, w.symbol, t) for w, t in kinds] == [
        ("trade", "BTCUSDT", 3 * MS),
        ("book", "BTCUSDT", 5 * MS),
        ("timeout", None, 6 * MS),
        ("end", None, 6 * MS),
    ]


# ---------------------------------------------------------------------------
# What a run hands back
# ---------------------------------------------------------------------------


def test_the_tables_of_a_run():
    pd = pytest.importorskip("pandas")

    def strategy(sim):
        sim.elapse(POST - sim.now())
        sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sim.post("BTCUSDT", "buy", None, 2.0, tif="IOC")
        sim.elapse(END - sim.now())

    config = bt.sim.Config(maker_fee_bps=1.0, taker_fee_bps=5.0, initial_cash=1_000.0)
    res = bt.sim.run(strategy, config,
                     markets=[_market([_ladder(MS, 5.0)], [(3 * MS, 99.9, 10.0, "sell")])])

    orders = res.orders_df(backend="pandas")
    assert list(orders.columns) == [
        "order_id", "symbol", "side", "price", "qty", "filled", "avg_price", "status",
        "tif", "sent_at",
    ]
    assert orders["status"].tolist() == ["filled", "filled"]
    assert pd.isna(orders["price"].iloc[1])

    fills = res.fills_df(backend="pandas")
    assert fills["channel"].tolist() == ["taker", "traverse"]
    assert fills["maker"].tolist() == [False, True]
    assert str(fills["timestamp"].dt.tz) == "UTC"
    assert [round(f, 6) for f in fills["fee"]] == [round(2 * 100.1 * 5e-4, 6), 0.01]

    events = res.events_df(backend="pandas")
    assert events["event"].tolist()[:2] == ["sent", "sent"]
    assert {"accepted", "fill"} <= set(events["event"])

    equity = res.equity_df(backend="pandas")
    assert list(equity.columns) == ["timestamp", "equity"]

    assert res.position == {"BTCUSDT": 3.0}
    assert res.fill_marks["anchor"] == "fill"
    assert len(res.fill_marks_df(backend="pandas")) == 2

    m = res.metrics
    assert (m["orders"], m["orders_filled"], m["fills"]) == (2, 2, 2)
    assert (m["maker_fills"], m["taker_fills"], m["volume"]) == (1, 1, 3.0)
    assert round(m["fees"], 6) == round(0.01 + 2 * 100.1 * 5e-4, 6)
    assert "orders" in res.summary()


# ---------------------------------------------------------------------------
# Refusals, by name
# ---------------------------------------------------------------------------


def _expect(error, match, call):
    def strategy(sim):
        call(sim)

    with pytest.raises(error, match=match):
        bt.sim.run(strategy, markets=[_market([_ladder(MS, 5.0)])])


def test_what_the_strategy_cannot_do_is_refused_by_name():
    _expect(KeyError, "no symbol", lambda s: s.book("ETHUSDT"))
    _expect(ValueError, "side", lambda s: s.post("BTCUSDT", "long", 100.0, 1.0))
    _expect(ValueError, "tif", lambda s: s.post("BTCUSDT", "buy", 100.0, 1.0, tif="DAY"))
    _expect(ValueError, "expire_at", lambda s: s.post("BTCUSDT", "buy", 100.0, 1.0, tif="GTD"))
    _expect(ValueError, "size", lambda s: s.post("BTCUSDT", "buy", 100.0, 0.0))
    _expect(ValueError, "no order", lambda s: s.cancel(42))
    _expect(ValueError, "status", lambda s: s.orders(status="open"))
    _expect(TypeError, "duration", lambda s: s.elapse("1ms"))


def test_an_error_in_the_strategy_reaches_the_caller():
    def strategy(sim):
        raise RuntimeError("the strategy's own bug")

    with pytest.raises(RuntimeError, match="own bug"):
        bt.sim.run(strategy, markets=[_market([_ladder(MS, 5.0)])])


def test_a_malformed_run_is_refused_before_it_starts():
    market = _market([_ladder(MS, 5.0)])
    with pytest.raises(TypeError, match="latency"):
        bt.sim.run(lambda s: None, bt.sim.Config(latency={"order": 5}), markets=[market])
    with pytest.raises(ValueError, match="zero or more"):
        bt.sim.run(lambda s: None, bt.sim.Config(latency={"entry": -1}), markets=[market])
    with pytest.raises(ValueError, match="invalid json|unknown field"):
        bt.sim.run(lambda s: None, bt.sim.Config(queue={"modle": "log"}), markets=[market])
    with pytest.raises(TypeError, match="store, symbols, start and end"):
        bt.sim.run(lambda s: None)
    with pytest.raises(TypeError, match="either markets"):
        bt.sim.run(lambda s: None, store=object(), markets=[market])
    with pytest.raises(ValueError, match="twice"):
        bt.sim.run(lambda s: None, markets=[market, market])
    with pytest.raises(ValueError, match="time order"):
        bt.sim.Market.from_ladders("X", 0.1, [_ladder(2 * MS, 1.0), _ladder(MS, 1.0)])


# ---------------------------------------------------------------------------
# A stored market
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MANIFOLDBT_TEST_TAPE_STORE"),
    reason="needs a store holding a tape and a book: MANIFOLDBT_TEST_TAPE_STORE="
    "<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD>",
)
def test_a_stored_day_runs_end_to_end():
    import datetime as dt

    root, meta, sid, day = os.environ["MANIFOLDBT_TEST_TAPE_STORE"].split("|")
    store = bt.DataStore(root, meta)
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    posted = []

    def strategy(sim):
        while sim.elapse(bt.Interval.seconds(60)):
            book = sim.book(0)
            if book.best_bid is not None and not sim.orders(status="live"):
                posted.append(sim.post(0, "buy", book.best_bid, 0.001, tif="GTX"))

    res = bt.sim.run(strategy, store=store, symbols=[int(sid)], start=day, end=nxt)
    assert posted, "the stored day never showed a book"
    assert res.metrics["orders"] == len(posted)
