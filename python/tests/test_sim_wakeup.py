"""``sim.wait`` returns a ``Wakeup`` that is false once the data has ended,
so ``while sim.wait(...)`` stops with the run instead of turning forever on
``Wakeup("end")``; every other wake-up stays true, a timeout included.
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


def _market():
    states = [(MS, [(9.9, 5.0)], [(10.1, 5.0)])]
    trades = [(t * MS, 10.0, 1.0, "buy") for t in (2, 3, 4)]
    return bt.sim.Market.from_ladders("XYZ", 0.1, states, trades)


def test_while_wait_stops_at_the_end_of_the_data():
    seen = []

    def strategy(sim):
        turns = 0
        while (w := sim.wait(trade=True)):
            seen.append(w.kind)
            turns += 1
            assert turns < 50, "while sim.wait() did not stop at the end"
        seen.append(w.kind)

    bt.sim.run(strategy, markets=[_market()])
    assert seen[-1] == "end"
    assert seen[:-1] and set(seen[:-1]) == {"trade"}


def test_end_is_false_every_other_kind_true():
    out = {}

    def strategy(sim):
        out["trade"] = bool(sim.wait(trade=True))
        out["timeout"] = bool(sim.wait(timeout=1))
        while sim.wait():
            pass
        end = sim.wait()
        out["end_kind"] = end.kind
        out["end"] = bool(end)

    bt.sim.run(strategy, markets=[_market()])
    assert out == {"trade": True, "timeout": True, "end_kind": "end", "end": False}
