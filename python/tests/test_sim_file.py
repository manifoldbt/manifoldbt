"""``bt.sim``: the queue of a resting order on a book stored by price.

The rules are pinned by the engine's own tests (``bt-sim/tests/lifecycle.rs``,
``bt-core``); this file checks that Python sees the same verdicts: our own
orders at one price, the probabilistic queues with two of them, the prints that
explain a fall of the level, the assumed queue as a swept axis, and what is
refused by name.

Every market is written out by hand (``bt.sim.Market.from_ladders``), on a 0.1
tick, bids 100.0 (the size under test), 99.9 x 9 and 99.0 x 9, asks 100.1 x 5
and 100.5 x 6.
"""
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


def _ladder(ts, size_at_100):
    return (ts, [(100.0, size_at_100), (99.9, 9.0), (99.0, 9.0)], [(100.1, 5.0), (100.5, 6.0)])


def _market(states, trades=()):
    return bt.sim.Market.from_ladders("BTCUSDT", 0.1, list(states), list(trades))


# ---------------------------------------------------------------------------
# Our own orders at one price
# ---------------------------------------------------------------------------


def _b_behind_a(act, states, a_qty=1.0, sold=2.5):
    """2 stand at 100.0; A then B (1) rest behind them; at 3 ms ``act`` is done
    to A, at 4 ms ``sold`` are sold into the bid at 100.0. What B got."""
    out = {}

    def strategy(sim):
        sim.elapse(2 * MS - sim.now())
        a = sim.post("BTCUSDT", "buy", 100.0, a_qty)
        out["b"] = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sim.elapse(MS)
        act(sim, a)
        sim.elapse(10 * MS - sim.now())
        out["filled"] = sim.order(out["b"]).filled

    bt.sim.run(strategy, markets=[_market(states, [(4 * MS, 100.0, sold, "sell")])])
    return round(out["filled"], 6)


@pytest.mark.parametrize("grows", [False, True], ids=["quiet", "level grows behind"])
@pytest.mark.parametrize(
    "act, a_qty, expected",
    [
        (lambda s, a: s.cancel(a), 1.0, 0.5),
        (lambda s, a: s.replace(a, price=99.9), 1.0, 0.5),
        (lambda s, a: s.replace(a, qty=2.0), 1.0, 0.5),
        (lambda s, a: s.replace(a, qty=0.4), 1.0, 0.1),
        (lambda s, a: None, 1.0, 0.0),
    ],
    ids=["cancelled", "other price", "larger, to the back", "smaller, keeps its place", "stays"],
)
def test_an_order_of_ours_that_leaves_shortens_the_queue_behind_it(act, a_qty, expected, grows):
    states = [_ladder(MS, 2.0)] + ([_ladder(5 * MS // 2, 8.0)] if grows else [])
    assert _b_behind_a(act, states, a_qty) == expected


# ---------------------------------------------------------------------------
# The probabilistic queues
# ---------------------------------------------------------------------------

PROBABILISTIC = [
    {"model": "log"},
    {"model": "power", "n": 2.0},
    {"model": "power", "n": 3.0},
]


def _two_orders(queue, states, trades):
    """A then B (1 each) post at 100.0 at 2 ms; what each got by 10 ms."""
    out = {}

    def strategy(sim):
        sim.elapse(2 * MS - sim.now())
        a = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        b = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sim.elapse(10 * MS - sim.now())
        out["a"], out["b"] = sim.order(a).filled, sim.order(b).filled

    bt.sim.run(strategy, bt.sim.Config(queue=queue), markets=[_market(states, trades)])
    return round(out["a"], 6), round(out["b"], 6)


@pytest.mark.parametrize("queue", [{"model": "risk_adverse"}] + PROBABILISTIC, ids=str)
def test_two_of_our_orders_are_not_served_by_the_same_volume(queue):
    # 4 shown, the level falls to 2 with no print, 2.5 are sold: 2 stand ahead
    # of A at most, and those 2 and A ahead of B.
    got = _two_orders(queue, [_ladder(MS, 4.0), _ladder(3 * MS, 2.0)],
                      [(7 * MS // 2, 100.0, 2.5, "sell")])
    assert got == (0.5, 0.0)


@pytest.mark.parametrize("queue", PROBABILISTIC, ids=str)
@pytest.mark.parametrize(
    "prints",
    [
        [(3 * MS, 100.0, 4.0, "sell")],
        [(3 * MS, 100.0, 2.0, "sell"), (4 * MS, 100.0, 2.0, "sell")],
        [(3 * MS, 100.0, 4.0, "sell"), (4 * MS, 100.1, 0.1, "buy")],
        [(5 * MS, 100.0, 1.0, "sell")] * 4,
    ],
    ids=["one print", "two prints", "a print of the other side", "four prints at the state's instant"],
)
def test_a_print_that_explains_a_fall_is_counted_once(queue, prints):
    # 10 shown, 4 print at the level, the book then shows 6: the prints explain
    # the whole fall, so 6.5 sold afterwards reach the order for 0.5.
    out = {}

    def strategy(sim):
        sim.elapse(2 * MS - sim.now())
        out["id"] = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sim.elapse(10 * MS - sim.now())
        out["filled"] = sim.order(out["id"]).filled

    market = _market([_ladder(MS, 10.0), _ladder(5 * MS, 6.0)], prints + [(6 * MS, 100.0, 6.5, "sell")])
    bt.sim.run(strategy, bt.sim.Config(queue=queue), markets=[market])
    assert round(out["filled"], 6) == 0.5


# ---------------------------------------------------------------------------
# The assumed queue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "assumed, expected", [(0.5, 1.0), (3.0, 0.5), (100.0, 0.0), (1000.0, 0.0)]
)
def test_an_assumed_queue_is_not_brought_down_to_the_displayed_size(assumed, expected):
    # 1 shown, a later state that leaves the level alone, 3.5 sold at 100.0.
    # Whatever the book shows, the order waits behind `assumed` times its size:
    # the axis the guide asks to sweep has to move the result.
    out = {}

    def strategy(sim):
        sim.elapse(2 * MS - sim.now())
        out["id"] = sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sim.elapse(10 * MS - sim.now())
        out["filled"] = sim.order(out["id"]).filled

    later = (5 * MS // 2, [(100.0, 1.0), (99.9, 8.0), (99.0, 9.0)], [(100.1, 5.0), (100.5, 6.0)])
    market = _market([_ladder(MS, 1.0), later], [(4 * MS, 100.0, 3.5, "sell")])
    config = bt.sim.Config(queue={"depth_source": "assumed", "assumed_queue": assumed})
    bt.sim.run(strategy, config, markets=[market])
    assert round(out["filled"], 6) == expected


@pytest.mark.parametrize("model", [{"model": "log"}, {"model": "power", "n": 2.0}], ids=str)
def test_a_probabilistic_queue_on_an_assumed_depth_is_refused_by_name(model):
    ran = []
    config = bt.sim.Config(queue=dict(model, depth_source="assumed", assumed_queue=3.0))
    with pytest.raises(Exception, match=r"queue\.model=.*assumed"):
        bt.sim.run(lambda sim: ran.append(1), config, markets=[_market([_ladder(MS, 5.0)])])
    assert ran == [], "refused before the strategy is called"
    # The historical rule on an assumed queue stays what it was.
    ok = bt.sim.Config(queue={"model": "risk_adverse", "depth_source": "assumed", "assumed_queue": 3.0})
    bt.sim.run(lambda sim: None, ok, markets=[_market([_ladder(MS, 5.0)])])
