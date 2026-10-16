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


def test_a_refusal_names_the_parameter_and_the_value(tmp_path):
    market = _market([_ladder(MS, 5.0)])
    nan, inf = float("nan"), float("inf")
    for field, value in (
        ("maker_fee_bps", nan),
        ("taker_fee_bps", inf),
        ("initial_cash", nan),
    ):
        config = bt.sim.Config(**{field: value})
        with pytest.raises(ValueError, match=rf"Config\.{field} must be a finite number, got {value}"):
            bt.sim.run(lambda s: None, config, markets=[market])
    with pytest.raises(ValueError, match="assumed_queue.*finite number, got nan"):
        bt.sim.run(
            lambda s: None,
            bt.sim.Config(queue={"model": "risk_adverse", "assumed_queue": nan}),
            markets=[market],
        )
    with pytest.raises(TypeError, match="latency.feed takes a duration, got inf"):
        bt.sim.run(lambda s: None, bt.sim.Config(latency={"feed": inf}), markets=[market])
    with pytest.raises(ValueError, match="latency.entry must be a finite duration"):
        bt.sim.run(
            lambda s: None, bt.sim.Config(latency={"entry": bt.Interval.millis(nan)}), markets=[market]
        )
    with pytest.raises(ValueError, match=r"Config\.seed must be zero or more, got -1"):
        bt.sim.run(lambda s: None, bt.sim.Config(seed=-1), markets=[market])

    # An id no `post` returned is an order never sent, not an OverflowError.
    def strategy(sim):
        for call in (lambda: sim.cancel(-1), lambda: sim.replace(-1, price=100.0)):
            with pytest.raises(ValueError, match="no order -1 was sent"):
                call()
        with pytest.raises(ValueError, match="no order 36893488147419103232 was sent"):
            sim.cancel(2**65)
        assert sim.order(-1) is None
        with pytest.raises(ValueError, match="levels.*0 or more, got -1"):
            sim.book("BTCUSDT").levels(-1)
        assert sim.book("BTCUSDT").levels(0) == ([], [])

    bt.sim.run(strategy, markets=[market])

    # `levels` of a stored market is a number of levels, 1 or more.
    store = bt.DataStore(
        data_root=str(tmp_path / "data"), metadata_db=str(tmp_path / "meta.sqlite")
    )
    for levels in (-1, 0):
        with pytest.raises(ValueError, match=rf"levels must be at least 1.*got {levels}"):
            bt.sim.Market.from_store(store, "BTCUSDT", 0, MS, levels)


def test_a_tick_finer_than_the_price_grid_is_refused_by_name():
    # The venue compares prices on a grid of 1e-8: a tick of 1e-9 would make
    # an order take a level beyond its limit.
    state = (MS, [(9.0e-8, 5.0)], [(1.0e-7, 5.0), (1.1e-7, 5.0)])
    for tick in (1e-9, 1.5e-8):
        with pytest.raises(ValueError, match=r"tick size .*1e-8"):
            bt.sim.Market.from_ladders("T", tick, [state])
    # No tick, and ticks on the grid, stand.
    for tick in (0.0, 1e-8, 0.07, 0.1, 1.0):
        bt.sim.Market.from_ladders("T", tick, [_ladder(MS, 5.0)])


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


# ---------------------------------------------------------------------------
# An order past the levels read refuses itself, not the run
# ---------------------------------------------------------------------------


def test_an_order_past_the_levels_read_is_refused_alone_with_a_warning():
    out = {}

    def strategy(sim):
        sim.elapse(POST - sim.now())
        out["far"] = sim.post("BTCUSDT", "buy", 90.0, 1.0)      # 3 levels read: 99.0 is the last
        out["near"] = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        out["again"] = sim.post("BTCUSDT", "buy", 80.0, 1.0)
        sim.elapse(END - sim.now())
        out["status"] = {k: sim.order(out[k]).status for k in ("far", "near", "again")}

    with pytest.warns(UserWarning, match=r"90.*Bid.*beyond the 3 level.*assumed_queue") as seen:
        res = bt.sim.run(strategy, markets=[_market([_ladder(MS, 5.0)])])
    # Said once per run and market, for the first order refused.
    assert len([w for w in seen if "beyond the 3 level" in str(w.message)]) == 1
    assert out["status"] == {"far": "rejected", "near": "resting", "again": "rejected"}
    events = res.events_df(backend="pandas")
    assert list(events[events["order_id"] == out["far"]]["event"]) == ["sent", "rejected"]


def test_a_replacement_past_the_levels_read_is_refused_and_the_order_stays():
    out = {}

    def strategy(sim):
        sim.elapse(POST - sim.now())
        out["id"] = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sim.elapse(MS)
        sim.replace(out["id"], price=90.0)
        sim.elapse(END - sim.now())
        out["order"] = sim.order(out["id"])

    with pytest.warns(UserWarning, match="replacement of order"):
        res = bt.sim.run(strategy, markets=[_market([_ladder(MS, 5.0)], [(6 * MS, 100.0, 20.0, "sell")])])
    assert (out["order"].status, out["order"].price, round(out["order"].filled, 6)) == ("filled", 100.0, 1.0)
    events = res.events_df(backend="pandas")
    kinds = list(events[events["order_id"] == out["id"]]["event"])
    assert "replace_rejected" in kinds and "replaced" not in kinds


def test_an_order_past_the_levels_read_joins_the_assumed_queue_without_a_warning(recwarn):
    def strategy(sim):
        sim.elapse(POST - sim.now())
        sim.post("BTCUSDT", "buy", 90.0, 1.0)
        sim.elapse(END - sim.now())

    cfg = bt.sim.Config(queue={"depth_source": "book", "assumed_queue": 2.0})
    bt.sim.run(strategy, cfg, markets=[_market([_ladder(MS, 5.0)])])
    assert not [w for w in recwarn if "beyond" in str(w.message)]


def test_queue_ahead_is_that_of_the_last_answer_after_a_fill_and_after_a_replacement():
    out = {}

    def strategy(sim):
        sim.elapse(POST - sim.now())
        oid = sim.post("BTCUSDT", "buy", 100.0, 2.0)
        sim.elapse(MS)
        out["posted"] = sim.order(oid).queue_ahead
        sim.elapse(3 * MS)                      # a print of 2 at 100.0 burns the queue to 3
        sim.replace(oid, qty=1.5)               # smaller, same price: keeps its place
        sim.elapse(MS)
        out["replaced"] = sim.order(oid).queue_ahead
        sim.elapse(END - sim.now())             # a print of 6 serves 1.0 of it
        o = sim.order(oid)
        out["filled"] = (round(o.filled, 6), o.queue_ahead)

    trades = [(5 * MS, 100.0, 2.0, "sell"), (8 * MS, 100.0, 6.0, "sell")]
    bt.sim.run(strategy, markets=[_market([_ladder(MS, 5.0)], trades)])
    assert out == {"posted": 5.0, "replaced": 3.0, "filled": (1.5, 0.0)}
