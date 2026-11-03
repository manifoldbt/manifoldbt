"""``mbt.sim.Config(prints_update_book=...)``: each print updates the book that
whatever executes against it meets, until the next book state:
``"aggressed_side"`` the side the print traded against, ``"both_sides"`` both
sides, the print alone, with a market order's rest one tick worse.

The rules are pinned by the engine's own tests (``prints_update_book.rs``); this
file checks that Python reaches them, that the setting is refused by name when
it is not one of the two, that leaving it out is the run without it, that
``"aggressed_side"`` gives the run a strategy got by writing a book state at
every print by hand, and that ``"both_sides"`` gives the fills of a
trade-driven top-of-book engine on a scenario where the two rules part.

The market is the one of ``test_sim.py``: bids 100.0 x 5, 99.9 x 9, 99.0 x 9,
asks 100.1 x 5, 100.5 x 6, on a 0.1 tick.
"""
import json
import random

import pytest

import manifoldbt as bt

MS = 1_000_000


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


def _market(trades, states=None):
    states = states if states is not None else [(MS, BIDS, ASKS)]
    return bt.sim.Market.from_ladders("XYZ", 0.1, states, trades)


def _on_print(side, qty, config, trades, states=None, price=None, tif="IOC"):
    """Wait for the first print, send one order at once: its fills and status."""
    out = {}

    def strategy(sim):
        sim.wait(trade=True)
        out["id"] = sim.post("XYZ", side, price, qty, tif=tif)
        sim.elapse(10 * MS)
        out["order"] = sim.order(out["id"])

    res = bt.sim.run(strategy, config, markets=[_market(trades, states)])
    cols = res.fills_df(backend="arrow")
    fills = [
        (round(p, 6), round(q, 6))
        for i, p, q in zip(cols["order_id"], cols["price"], cols["qty"])
        if i == out["id"]
    ]
    return fills, out["order"].status


ON = bt.sim.Config(prints_update_book="aggressed_side")
BOTH = bt.sim.Config(prints_update_book="both_sides")
OFF = bt.sim.Config()
SOLD_THROUGH = [(3 * MS, 99.5, 2.0, "sell")]


def test_a_market_order_sent_on_a_print_fills_at_the_print():
    assert _on_print("sell", 1.0, ON, SOLD_THROUGH) == ([(99.5, 1.0)], "filled")
    assert _on_print("sell", 1.0, OFF, SOLD_THROUGH) == ([(100.0, 1.0)], "filled")
    # Past the print's size: the stored levels beyond its price.
    assert _on_print("sell", 5.0, ON, SOLD_THROUGH) == ([(99.5, 2.0), (99.0, 3.0)], "filled")
    # The print traded against the bids: the asks are as stored.
    assert _on_print("buy", 1.0, ON, SOLD_THROUGH) == ([(100.1, 1.0)], "filled")
    bought = [(3 * MS, 100.5, 1.0, "buy")]
    assert _on_print("buy", 2.0, ON, bought) == ([(100.5, 2.0)], "filled")
    assert _on_print("buy", 2.0, OFF, bought) == ([(100.1, 2.0)], "filled")


def test_both_sides_is_the_print_alone_and_the_rest_one_tick_worse():
    # Both sides are the print, whoever crossed.
    assert _on_print("buy", 1.0, BOTH, SOLD_THROUGH, tif="GTC") == ([(99.5, 1.0)], "filled")
    # A GTC larger than the print: the rest a tick worse, the stored 99.0 not walked.
    fills = _on_print("sell", 5.0, BOTH, SOLD_THROUGH, tif="GTC")
    assert fills == ([(99.5, 2.0), (99.4, 3.0)], "filled")
    # An IOC lets the rest expire, a FOK takes nothing.
    assert _on_print("sell", 5.0, BOTH, SOLD_THROUGH, tif="IOC") == ([(99.5, 2.0)], "expired")
    assert _on_print("sell", 5.0, BOTH, SOLD_THROUGH, tif="FOK") == ([], "expired")


def test_a_book_state_at_the_print_s_instant_comes_after_it():
    states = [(MS, BIDS, ASKS), (3 * MS, BIDS, ASKS)]
    for config in (ON, BOTH):
        assert _on_print("sell", 1.0, config, SOLD_THROUGH, states) == ([(100.0, 1.0)], "filled")


def test_post_only_and_fok_read_the_book_the_print_left():
    assert _on_print("sell", 1.0, ON, SOLD_THROUGH, price=99.9, tif="GTX")[1] == "resting"
    assert _on_print("sell", 1.0, OFF, SOLD_THROUGH, price=99.9, tif="GTX")[1] == "rejected"
    assert _on_print("sell", 10.0, ON, SOLD_THROUGH, price=99.5, tif="FOK") == ([], "expired")
    assert _on_print("buy", 1.0, BOTH, SOLD_THROUGH, price=99.5, tif="GTX")[1] == "rejected"


def test_the_setting_is_one_of_two_rules_and_left_out_it_is_off():
    market = _market(SOLD_THROUGH)
    for value in (True, False, "yes", "BOTH_SIDES", 1):
        with pytest.raises(ValueError, match='takes "aggressed_side" or "both_sides"'):
            bt.sim.run(lambda s: None, bt.sim.Config(prints_update_book=value), markets=[market])
    assert "prints_update_book" not in json.loads(bt.sim.Config()._to_json())
    assert json.loads(ON._to_json())["prints_update_book"] == "aggressed_side"
    assert json.loads(BOTH._to_json())["prints_update_book"] == "both_sides"


# One equity on a 0.01 tick, four L1 quotes, seven prints. On each print but
# the last, an order with zero latency: a market order (GTC) for cases 0 to 4,
# a buy limit at the bid for case 5, which rests; the last print sells at that
# limit, 60 against the 100 the quote shows there.
QUOTES = [
    (1, 10.00, 500, 10.01, 300),
    (5, 10.02, 400, 10.03, 300),
    (10, 10.05, 100, 10.06, 100),
    (18, 10.05, 100, 10.06, 100),
]
PRINTS = [  # ts ms, price, size, aggressor, order
    (3, 10.03, 200, "buy", ("buy", None, 100)),
    (7, 10.02, 300, "sell", ("sell", None, 100)),
    (8, 10.03, 50, "buy", ("buy", None, 100)),
    (12, 10.04, 100, "sell", ("buy", None, 100)),
    (14, 10.03, 500, "sell", ("sell", None, 200)),
    (16, 10.06, 10, "buy", ("buy", 10.05, 100)),
    (20, 10.05, 60, "sell", None),
]
# The fills of nautilus_trader 1.231.0 and 2.0.0rc5 on the same data, the two
# alike (BacktestEngine, BookType.L1_MBP, trade_execution=True, FillModel left
# as it is, each order sent from the trade callback), per case.
TRADE_DRIVEN = [
    [(10.03, 100.0)],
    [(10.02, 100.0)],
    [(10.03, 50.0), (10.04, 50.0)],
    [(10.04, 100.0)],
    [(10.03, 200.0)],
    [(10.05, 60.0)],
]


def _cases(config):
    """The fills of each case's order under ``config``."""
    states = [(t * MS, [(b, float(bq))], [(a, float(aq))]) for t, b, bq, a, aq in QUOTES]
    trades = [(t * MS, p, float(q), agg) for t, p, q, agg, _ in PRINTS]
    market = bt.sim.Market.from_ladders("XYZ", 0.01, states, trades)
    sent = []

    def strategy(sim):
        seen = 0
        while sim.wait(trade=True).kind != "end":
            for _ in sim.trades("XYZ"):
                order = PRINTS[seen][4]
                seen += 1
                if order is not None:
                    side, price, qty = order
                    sent.append(sim.post("XYZ", side, price, float(qty)))

    res = bt.sim.run(strategy, config, markets=[market])
    cols = res.fills_df(backend="arrow")
    fills = list(zip(cols["order_id"], cols["price"], cols["qty"]))
    return [[(round(p, 6), q) for i, p, q in fills if i == id_] for id_ in sent]


def test_both_sides_gives_the_fills_of_a_trade_driven_top_of_book_engine():
    ours = _cases(BOTH)
    assert ours[:5] == TRADE_DRIVEN[:5]
    # 5: a resting limit. A print at its price is served by the queue model,
    # whatever the rule: the 60 printed leave 40 of the 100 ahead, nothing
    # fills. The trade-driven engine gives it the 60, without a queue.
    assert ours[5] == []
    # At the front of its queue (an assumed queue of nothing), it gets them.
    front = bt.sim.Config(
        prints_update_book="both_sides", queue={"depth_source": "assumed", "assumed_queue": 0.0}
    )
    assert _cases(front)[5] == TRADE_DRIVEN[5]


def test_aggressed_side_parts_from_it_on_two_rules():
    ours = _cases(ON)
    for case in (0, 1, 4):
        assert ours[case] == TRADE_DRIVEN[case], case
    # 2: an order larger than the print. Here 10.03 shows the stored 300,
    # larger than the 50 printed; the trade-driven engine fills the rest one
    # tick worse.
    assert ours[2] == [(10.03, 100.0)]
    # 3: a buy on a print where the seller crossed. Here the print says
    # nothing of the asks, the stored 10.06 stands; the trade-driven engine
    # fills at the print.
    assert ours[3] == [(10.06, 100.0)]
    assert ours[5] == []


def _day(seed, n=400):
    """A synthetic session: a quote every few ms, prints between them, some
    through the touch."""
    rng = random.Random(seed)
    mid, t = 1000, MS
    states, trades = [], []
    for _ in range(n):
        mid += rng.choice((-1, 0, 0, 1))
        bids = [(round((mid - k) * 0.1, 1), float(rng.randint(1, 9))) for k in (1, 2, 3)]
        asks = [(round((mid + k) * 0.1, 1), float(rng.randint(1, 9))) for k in (1, 2, 3)]
        states.append((t, bids, asks))
        nxt = t + rng.randint(2, 5) * MS
        for _ in range(rng.randint(0, 2)):
            sell = rng.random() < 0.5
            away = rng.randint(0, 2)
            px = round(((mid - 1 - away) if sell else (mid + 1 + away)) * 0.1, 1)
            ts = rng.randint(t + 1, nxt - 1)
            trades.append((ts, px, float(rng.randint(1, 6)), "sell" if sell else "buy"))
        t = nxt
    trades.sort(key=lambda p: p[0])
    return states, trades


def _follow(market, config):
    """A taker that follows every print it sees with a market order of 3 the
    way the print went: what it filled, order by order."""

    def strategy(sim):
        while True:
            w = sim.wait(trade=True)
            for _, _, _, aggressor in sim.trades("XYZ"):
                sim.post("XYZ", aggressor, None, 3.0, tif="IOC")
            if w.kind == "end":
                # The orders sent on the last print reach the venue.
                sim.elapse(bt.Interval.millis(1))
                return

    res = bt.sim.run(strategy, config, markets=[market])
    cols = res.fills_df(backend="arrow")
    return list(zip(cols["timestamp"], cols["order_id"], cols["price"], cols["qty"]))


def test_left_out_it_is_the_run_without_it():
    states, trades = _day(7)
    market = bt.sim.Market.from_ladders("XYZ", 0.1, states, trades)
    off = _follow(market, bt.sim.Config())
    assert len(off) > 100
    assert off == _follow(market, bt.sim.Config(prints_update_book=None))
    assert off != _follow(market, ON)
    assert off != _follow(market, BOTH)


def _hand_built(states, trades):
    """Each print written into the book by hand, as a state at the print's
    instant: the side it traded against starts at its price, with its size or
    the stored size there when larger, the stored levels past it behind; the
    other side is the last state's (or the last such print's)."""
    out = []
    i = 0
    stored = None
    made = {"bid": None, "ask": None}

    def side(levels, print_, better):
        price, qty = print_
        size = qty
        rest = []
        for p, q in levels:
            if better(p, price):
                continue
            if abs(p - price) < 1e-9:
                size = max(size, q)
            else:
                rest.append((p, q))
        return [(price, size)] + rest

    for t, px, qty, aggressor in trades:
        while i < len(states) and states[i][0] < t:
            stored = states[i]
            out.append(stored)
            made = {"bid": None, "ask": None}
            i += 1
        if aggressor == "sell":
            made["bid"] = (px, qty)
        else:
            made["ask"] = (px, qty)
        bids, asks = stored[1], stored[2]
        if made["bid"]:
            bids = side(bids, made["bid"], lambda p, x: p > x + 1e-9)
        if made["ask"]:
            asks = side(asks, made["ask"], lambda p, x: p < x - 1e-9)
        if out and out[-1][0] == t:
            out[-1] = (t, bids, asks)
        else:
            out.append((t, bids, asks))
    out.extend(states[i:])
    return out


def test_it_gives_the_run_of_a_book_state_written_at_every_print():
    states, trades = _day(11)
    assert len({t for t, *_ in trades}) == len(trades), "one print per instant here"
    engine = _follow(bt.sim.Market.from_ladders("XYZ", 0.1, states, trades), ON)
    by_hand = _follow(
        bt.sim.Market.from_ladders("XYZ", 0.1, _hand_built(states, trades), trades), OFF
    )
    assert len(engine) > 100
    assert [(t, i, round(p, 6), round(q, 6)) for t, i, p, q in engine] == [
        (t, i, round(p, 6), round(q, 6)) for t, i, p, q in by_hand
    ]
