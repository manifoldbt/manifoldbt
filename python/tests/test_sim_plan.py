"""An order plan run by ``mbt.sim.run``: ``mbt.sim.Plan`` and its parts.

The rules of a plan are pinned by the engine's own tests (``plan_sim.rs``);
this file checks that Python reaches them: the plan as it is written down,
``mbt.param`` as the value of a field, durations, sessions set against the
market, setups from a table, every refusal by name, and the periodic plan of
the benchmark against the same strategy written as a Python function, fill
for fill.
"""
import datetime

import numpy as np
import pytest

import manifoldbt as bt
from manifoldbt.helpers import Interval

S = bt.sim
MS = 1_000_000
SEC = 1_000 * MS
MIN = 60 * SEC
END = 10 * MS


def _layer_open():
    try:
        S.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    except PermissionError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _layer_open(),
    reason="order-level simulation locked (runs in the engine's own debug builds)",
)

BIDS = [(100.0, 5.0), (99.9, 9.0), (99.0, 9.0)]
ASKS = [(100.1, 5.0), (100.5, 6.0)]


def _market(trades=(), states=None):
    states = states if states is not None else [(0, BIDS, ASKS), (END, BIDS, ASKS)]
    return S.Market.from_ladders("XYZ", 0.1, states, list(trades))


def _sent(res):
    o = res.orders_df(backend="arrow")
    return [t for t, r in zip(o["sent_at"], o["role"]) if r is None]


def test_a_plan_runs_in_the_engine_and_gives_the_record_of_a_run():
    plan = S.Plan(when=S.every(Interval.millis(2)), entry=S.market(), side="buy", size=1)
    res = S.run(plan, markets=[_market()])
    assert _sent(res) == [2 * MS, 4 * MS, 6 * MS, 8 * MS, 10 * MS]
    f = res.fills_df(backend="arrow")
    assert f["price"] == [100.1] * 5
    assert set(f["channel"]) == {"taker"}
    assert res.metrics["orders"] == 5
    assert res.position == {"XYZ": 5.0}
    # The record of a run: a journal, an equity curve, the markouts.
    assert "sent" in res.events_df(backend="arrow")["event"]
    assert len(res.equity_df(backend="arrow")["equity"]) > 0
    assert res.fill_marks["anchor"] == "fill"


def test_a_field_takes_a_number_or_mbt_param_and_its_default_runs():
    plan = S.Plan(
        when=S.every(bt.param("every", default=Interval.millis(3))),
        entry=S.market(),
        side=S.alternate(bt.param("first", default="sell")),
        size=bt.param("qty", default=2),
    )
    res = S.run(plan, markets=[_market()])
    assert _sent(res) == [3 * MS, 6 * MS, 9 * MS]
    f = res.fills_df(backend="arrow")
    assert f["side"] == ["sell", "buy", "sell"]
    assert f["qty"] == [2.0, 2.0, 2.0]
    # Any other expression is refused by name.
    with pytest.raises(TypeError, match=r"takes a number or mbt.param\(\.\.\.\)"):
        S.Plan(when=S.every(MS), entry=S.market(), side="buy", size=bt.col("close"))._to_json(
            [_market()]
        )
    # A parameter with no default and no grid value is refused by the engine.
    plan = S.Plan(when=S.every(MS), entry=S.market(), side="buy", size=bt.param("q"))
    with pytest.raises(ValueError, match=r"mbt.param\(\"q\"\), which has no default"):
        S.run(plan, markets=[_market()])
    # Two defaults for one name in one plan.
    plan = S.Plan(
        when=S.every(MS),
        entry=S.limit(bt.param("x", default=1)),
        side="buy",
        size=bt.param("x", default=2),
    )
    with pytest.raises(ValueError, match="two defaults"):
        plan._to_json([_market()])


@pytest.mark.parametrize(
    "period",
    [Interval.millis(2), datetime.timedelta(milliseconds=2), 2 * MS, np.int64(2 * MS)],
)
def test_a_duration_is_an_interval_a_timedelta_or_nanoseconds(period):
    plan = S.Plan(when=S.every(period), entry=S.market(), side="buy", size=1)
    assert _sent(S.run(plan, markets=[_market()]))[:2] == [2 * MS, 4 * MS]


def test_entries_set_from_the_reference_and_exits_from_the_entry_price():
    prints = [(MS, 100.0, 1.0, "sell")]
    table = {"ts_ns": [2 * MS], "level": [100.03]}
    plan = S.Plan(when=S.setups(table), entry=S.stop(0), side="buy", size=1)
    o = S.run(plan, markets=[_market(prints)]).orders_df(backend="arrow")
    assert o["trigger_price"] == [100.1]
    # A bracket in percent of the price filled at (100.1): 0.1 % = 10 bps.
    plan = S.Plan(
        when=S.setups({"ts_ns": [2 * MS]}),
        entry=S.market(),
        side="buy",
        size=1,
        exit=S.bracket(stop=0.1, target=0.1, unit="pct"),
    )
    o = S.run(plan, markets=[_market(prints)]).orders_df(backend="arrow")
    legs = {r: p for r, p in zip(o["role"], o["trigger_price"]) if r is not None}
    assert legs == {"stop": 99.9, "target": 100.3}
    # In ticks, a short: the stop above, the target below.
    plan = S.Plan(
        when=S.setups({"ts_ns": [2 * MS]}),
        entry=S.market(),
        side="sell",
        size=1,
        exit=S.bracket(stop=2, target=3),
    )
    o = S.run(plan, markets=[_market(prints)]).orders_df(backend="arrow")
    legs = {r: p for r, p in zip(o["role"], o["trigger_price"]) if r is not None}
    assert legs == {"stop": 100.2, "target": 99.7}


def test_setups_read_a_dataframe_a_pyarrow_table_or_a_dict():
    import pyarrow as pa

    rows = {
        "ts_ns": [2 * MS, 3 * MS, 4 * MS],
        "side": ["buy", "sell", "buy"],
        "level": [100.0, float("nan"), None],
        "size": [1.0, 2.0, 3.0],
    }
    tables = [rows, pa.table(rows)]
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df["ts_ns"] = pd.to_datetime(df["ts_ns"], unit="ns", utc=True)
        tables.append(df)
    except ImportError:
        pass
    prints = [(MS, 100.0, 1.0, "sell")]
    want = None
    for table in tables:
        plan = S.Plan(
            when=S.setups(table),
            entry=S.limit(1),
            side="from_setups",
            size="from_setups",
        )
        o = S.run(plan, markets=[_market(prints)]).orders_df(backend="arrow")
        got = (o["side"], o["qty"], o["price"])
        assert got == (["buy", "sell", "buy"], [1.0, 2.0, 3.0], [100.1, 99.9, 100.1])
        want = want or got
        assert got == want
    with pytest.raises(ValueError, match="no 'ts' column"):
        S.setups({"t": [1]}, ts="ts")
    with pytest.raises(ValueError, match="takes \"buy\" / \"sell\""):
        S.setups({"ts_ns": [1], "side": ["long"]})


def test_sessions_are_set_against_the_market_and_bound_the_entries():
    day = 1_773_100_800_000_000_000  # 2026-03-10 00:00 UTC, a Tuesday
    states = [(day + t * MIN, BIDS, ASKS) for t in (9 * 60 + 30, 10 * 60, 16 * 60)]
    m = S.Market.from_ladders("XYZ", 0.1, states, [])
    sessions = S.sessions(open="09:30", close="16:00", tz="UTC", entries=("09:40", "15:45"))
    spec = sessions._spec(m.span)
    opens = [o for o, _ in spec["open_close"]]
    assert day + 570 * MIN in opens
    assert (spec["entries_from_ns"], spec["entries_until_ns"]) == (10 * MIN, 15 * MIN)
    plan = S.Plan(
        when=S.every(Interval.minutes(30), anchor="session"),
        entry=S.market(),
        side="buy",
        size=1,
        sessions=sessions,
    )
    sent = _sent(S.run(plan, markets=[m]))
    # 10:00, 10:30, ... 15:30: inside 09:40 to 15:45.
    assert [(t - day) // MIN for t in sent] == list(range(600, 931, 30))
    with pytest.raises(ValueError, match="local time of day"):
        S.sessions(open="9h30", close="16:00")
    with pytest.raises(ValueError, match="not known to this Python"):
        S.sessions(open="09:30", close="16:00", tz="Not/AZone")._spec(m.span)
    # closes(n) without sessions.
    plan = S.Plan(
        when=S.every(MIN), entry=S.market(), side="buy", size=1, exit=S.bracket(hold=S.closes(1))
    )
    with pytest.raises(ValueError, match=r"closes\(n\) counts the session closes"):
        S.run(plan, markets=[m])


def test_every_word_of_a_plan_is_refused_by_name():
    with pytest.raises(ValueError, match='every\\(on=\\) takes "clock", "print"'):
        S.every(MS, on="trade")
    with pytest.raises(ValueError, match='limit\\(unit=\\) takes "ticks"'):
        S.limit(1, unit="percent")
    with pytest.raises(ValueError, match='limit\\(tif=\\) takes "IOC", "FOK"'):
        S.limit(1, tif="GTC")
    with pytest.raises(ValueError, match='stop\\(on=\\) takes "last", "bid_ask"'):
        S.stop(1, on="mid")
    with pytest.raises(ValueError, match='after_trigger=\\) takes "rest", "ioc"'):
        S.stop_limit(1, 1, after_trigger="cancel")
    with pytest.raises(ValueError, match='Plan\\(side=\\) takes "buy"'):
        S.Plan(when=S.every(MS), entry=S.market(), side="long", size=1)
    with pytest.raises(TypeError, match="Plan\\(when=\\)"):
        S.Plan(when=MS, entry=S.market(), side="buy", size=1)
    with pytest.raises(TypeError, match="Plan\\(entry=\\)"):
        S.Plan(when=S.every(MS), entry="market", side="buy", size=1)
    with pytest.raises(ValueError, match="must be a duration of zero or more"):
        S.Plan(when=S.every(-MS), entry=S.market(), side="buy", size=1)._to_json([_market()])
    with pytest.raises(TypeError, match="a number, not a bool"):
        S.Plan(when=S.every(MS), entry=S.market(), side="buy", size=True)._to_json([_market()])
    # And the engine's own refusals, with the field they name.
    plan = S.Plan(when=S.every(0), entry=S.market(), side="buy", size=1)
    with pytest.raises(ValueError, match="when.period must be a duration > 0"):
        S.run(plan, markets=[_market()])
    plan = S.Plan(when=S.every(MS, anchor="session"), entry=S.market(), side="buy", size=1)
    with pytest.raises(ValueError, match='anchor="session"'):
        S.run(plan, markets=[_market()])
    plan = S.Plan(when=S.every(MS), entry=S.market(), side="from_setups", size=1)
    with pytest.raises(ValueError, match="side column of the setups"):
        S.run(plan, markets=[_market()])
    plan = S.Plan(when=S.every(MS), entry=S.market(), side="buy", size=1, exit=S.bracket())
    with pytest.raises(ValueError, match="one with none is no exit"):
        S.run(plan, markets=[_market()])
    plan = S.Plan(when=S.every(MS), entry=S.market(), side="buy", size=1)
    with pytest.raises(ValueError, match="a plan runs on one market"):
        S.run(plan, markets=[_market(), S.Market.from_ladders("ABC", 0.1, [(0, BIDS, ASKS)])])


# The periodic plan of the benchmark: a market order at the first print at or
# after each deadline (open + k x interval), sides in turn, under both_sides.
OPEN = 1_773_149_400_000_000_000  # 2026-03-10 13:30 UTC


def _synthetic(seed):
    rng = np.random.default_rng(seed)
    n_q, n_t = 20_000, 6_000
    span = 2 * 3_600 * SEC
    q_ts = np.sort(OPEN + rng.integers(0, span, n_q))
    t_ts = np.sort(OPEN + rng.integers(0, span, n_t))
    mid = 100.0 + np.cumsum(rng.integers(-1, 2, n_q)) * 0.01
    quotes = {
        "ts_ns": q_ts,
        "bid_px": np.round(mid, 2),
        "bid_qty": rng.integers(1, 10, n_q) * 100.0,
        "ask_px": np.round(mid + 0.01, 2),
        "ask_qty": rng.integers(1, 10, n_q) * 100.0,
    }
    idx = np.searchsorted(q_ts, t_ts).clip(0, n_q - 1)
    buy = rng.integers(0, 2, n_t)
    trades = {
        "ts_ns": t_ts,
        "price": np.round(np.where(buy == 1, mid[idx] + 0.01, mid[idx]), 2),
        "qty": rng.integers(1, 20, n_t) * 50.0,
        "aggressor": np.where(buy == 1, 1, -1).astype(np.int8),
    }
    return S.Market.from_arrays("XYZ", 0.01, quotes=quotes, trades=trades)


def _bench_strategy(period, qty, first):
    other = "sell" if first == "buy" else "buy"

    def strategy(sim):
        n = 0
        deadline = OPEN + period
        while True:
            dt = deadline - 1 - sim.now()
            if dt > 0 and not sim.elapse(dt):
                return
            if sim.wait(trade=True).kind != "trade":
                return
            sim.post("XYZ", first if n % 2 == 0 else other, None, qty)
            n += 1
            k = (sim.now() - OPEN) // period + 1
            deadline = OPEN + max(k, 1) * period

    return strategy


def _bench_plan(period, qty, first):
    return S.Plan(
        when=S.every(period, on="print", anchor="session"),
        entry=S.market(),
        side=S.alternate(first),
        size=qty,
        sessions=S.sessions(open="13:30", close="20:00", tz="UTC"),
    )


def _bits(res):
    f = res.fills_df(backend="arrow")
    return [
        (t, float(p).hex(), float(q).hex(), s)
        for t, p, q, s in zip(f["timestamp"], f["price"], f["qty"], f["side"])
    ]


@pytest.mark.parametrize("engine", ["sim", "direct"])
def test_the_periodic_plan_swept_fills_as_the_python_strategy_on_either_engine(engine):
    m = _synthetic(20261030)
    cfg = S.Config(prints_update_book="both_sides")
    plan = S.Plan(
        when=S.every(bt.param("every", default=MIN), on="print", anchor="session"),
        entry=S.market(),
        side=S.alternate(bt.param("side", default="buy")),
        size=bt.param("qty", default=100),
        sessions=S.sessions(open="13:30", close="20:00", tz="UTC"),
    )
    grid = {"every": [MIN, 3 * MIN, 7 * MIN], "qty": [100, 2000], "side": ["buy", "sell"]}
    res = S.sweep(plan, grid, cfg, markets=[m], engine=engine, keep="fills")
    assert res.engine == engine
    for i in range(len(res)):
        p = res.params(i)
        by_hand = S.run(_bench_strategy(p["every"], float(p["qty"]), p["side"]), cfg,
                        markets=[m])
        f = res.fills_df(i, backend="arrow")
        swept = [
            (t, float(x).hex(), float(q).hex(), s)
            for t, x, q, s in zip(f["timestamp"], f["price"], f["qty"], f["side"])
        ]
        assert _bits(by_hand), "the strategy trades"
        assert swept == _bits(by_hand), (engine, p)
        assert res.to_df(backend="arrow")["position"][i] == by_hand.position["XYZ"]


@pytest.mark.parametrize("seed", [20261030, 7])
def test_the_periodic_plan_fills_as_the_python_strategy_to_the_bit(seed):
    m = _synthetic(seed)
    cfg = S.Config(prints_update_book="both_sides")
    for period, qty, first in [(MIN, 100, "buy"), (3 * MIN, 500, "sell"), (7 * MIN, 2000, "buy")]:
        by_hand = S.run(_bench_strategy(period, float(qty), first), cfg, markets=[m])
        plan = S.run(_bench_plan(period, qty, first), cfg, markets=[m])
        assert _bits(by_hand), "the strategy trades"
        assert _bits(plan) == _bits(by_hand), (seed, period, qty, first)
        assert plan.cash == by_hand.cash
        assert plan.position == by_hand.position
