"""``mbt.sim.sweep``: every variant of an order plan, in the engine.

What it holds: the enumeration (the parameters in name order, the last the
fastest, the execution axes the slowest), an execution grid over the keys of
``mbt.sim.Config``, the same results on 1, 2 and 7 threads, ``keep``, the
fills of a variant on demand, and a variant run again being the variant;
the engine each sweep takes, and the two engines giving the same bits.
A test that holds the simulation's own sweep forces ``engine="sim"``: the
plan of this file is one ``"auto"`` runs on ``"direct"``.
"""
import json
import os

import numpy as np
import pytest

import manifoldbt as bt
from manifoldbt.helpers import Interval

S = bt.sim
MS = 1_000_000
SEC = 1_000 * MS
OPEN = 1_773_149_400_000_000_000  # 2026-03-10 13:30 UTC


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


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(20261030)
    n_q, n_t = 8_000, 2_500
    span = 1_800 * SEC
    q_ts = np.sort(OPEN + rng.integers(0, span, n_q))
    t_ts = np.sort(OPEN + rng.integers(0, span, n_t))
    mid = 100.0 + np.cumsum(rng.integers(-1, 2, n_q)) * 0.01
    idx = np.searchsorted(q_ts, t_ts).clip(0, n_q - 1)
    buy = rng.integers(0, 2, n_t)
    return S.Market.from_arrays(
        "XYZ",
        0.01,
        quotes={
            "ts_ns": q_ts,
            "bid_px": np.round(mid, 2),
            "bid_qty": rng.integers(1, 10, n_q) * 100.0,
            "ask_px": np.round(mid + 0.01, 2),
            "ask_qty": rng.integers(1, 10, n_q) * 100.0,
        },
        trades={
            "ts_ns": t_ts,
            "price": np.round(np.where(buy == 1, mid[idx] + 0.01, mid[idx]), 2),
            "qty": rng.integers(1, 20, n_t) * 50.0,
            "aggressor": np.where(buy == 1, 1, -1).astype(np.int8),
        },
    )


def _plan():
    return S.Plan(
        when=S.every(bt.param("every", default=Interval.seconds(60)), on="print"),
        entry=S.market(),
        side=S.alternate(bt.param("side", default="buy")),
        size=bt.param("qty", default=100),
    )


GRID = {
    "side": ["buy", "sell"],
    "every": [Interval.seconds(30), Interval.seconds(90), Interval.seconds(240)],
    "qty": [100, 300],
}
CFG = S.Config(prints_update_book="both_sides")


def _fills(res, i):
    f = res.fills_df(i, backend="arrow")
    return [
        (t, float(p).hex(), float(q).hex(), s)
        for t, p, q, s in zip(f["timestamp"], f["price"], f["qty"], f["side"])
    ]


def test_the_variants_come_in_the_order_of_the_bar_sweep(market):
    res = S.sweep(_plan(), GRID, CFG, markets=[market], engine="sim")
    assert len(res) == 12
    assert (res.engine, res.engine_reason) == ("sim", 'engine="sim" was asked')
    # Names sorted: every, qty, side; the last the fastest.
    want = [
        (e * SEC, q, s) for e in (30, 90, 240) for q in (100, 300) for s in ("buy", "sell")
    ]
    got = [(p["every"], p["qty"], p["side"]) for p in (res.params(i) for i in range(12))]
    assert got == want
    df = res.to_df(backend="arrow")
    assert df["variant"] == list(range(12))
    assert df["param_every"] == [w[0] for w in want]
    assert df["param_side"] == [w[2] for w in want]
    for key in ("orders", "entries", "fills", "volume", "pnl", "max_drawdown", "round_trips",
                "win_rate", "exposure", "position", "exits_carried"):
        assert key in df
    # A shorter interval decides more often.
    assert df["entries"][0] > df["entries"][4] > df["entries"][8] > 0


def _stop_plan():
    return S.Plan(
        when=S.every(Interval.seconds(60), on="print"),
        entry=S.stop(offset=1, unit="ticks"),
        side=S.alternate("buy"),
        size=100,
    )


def _bracket_plan():
    return S.Plan(
        when=S.every(Interval.seconds(60), on="print"),
        entry=S.market(),
        side=S.alternate("buy"),
        size=100,
        exit=S.bracket(stop=2, target=3, unit="ticks"),
    )


def _resting_plan():
    return S.Plan(
        when=S.every(Interval.seconds(60), on="print"),
        entry=S.stop_limit(offset=1, limit_offset=1, unit="ticks"),
        side=S.alternate("buy"),
        size=100,
    )


def test_auto_takes_the_direct_way_for_a_plan_it_computes_and_the_simulation_otherwise(market):
    for plan in (_plan(), _stop_plan(), _bracket_plan()):
        res = S.sweep(plan, None, CFG, markets=[market])
        assert res.engine == "direct"
        assert 'engine="auto" reads the market only at the instants' in res.engine_reason
        res = S.sweep(plan, None, CFG, markets=[market], engine="direct")
        assert (res.engine, res.engine_reason) == ("direct", 'engine="direct" was asked')
    # A stop-limit that rests once set off waits in the queue of the book.
    plan = _resting_plan()
    res = S.sweep(plan, None, CFG, markets=[market])
    assert res.engine == "sim"
    assert res.engine_reason.startswith('engine="auto" runs the plan on the order-level')
    assert 'entry.after_trigger="rest" leaves a limit order resting' in res.engine_reason
    with pytest.raises(ValueError, match='outside what engine="direct" computes: entry'):
        S.sweep(plan, None, CFG, markets=[market], engine="direct")
    # The fallback changes nothing but the way: the simulation's own run.
    sim = S.sweep(plan, None, CFG, markets=[market], engine="sim")
    assert sim.to_arrow().equals(res.to_arrow())
    with pytest.raises(ValueError, match='engine takes "auto" \\(the default\\), "sim" or "direct"'):
        S.sweep(_plan(), {"qty": [100]}, CFG, markets=[market], engine="fast")
    # The engine itself refuses a plan outside by name too.
    from manifoldbt._native import sim_sweep

    with pytest.raises(ValueError, match='engine="sim" runs it on the order-level simulation'):
        sim_sweep(_resting_plan()._to_json([market]), "{}", CFG._to_json(), [market], 0, "{}",
                  "metrics", "direct")


def _stops_and_brackets():
    """Stop entries that expire, stop-limits that execute once set off or
    expire, brackets with a stop, a target and an exit at a time, on prints
    and on the stored bid and ask: a grid of each."""
    sl, tp = bt.param("sl", default=3), bt.param("tp", default=4)
    return [
        (S.Plan(
            when=S.every(Interval.seconds(45), on="print"),
            entry=S.stop(offset=bt.param("off", default=1), unit="ticks", on=on,
                         expire_after=Interval.seconds(20)),
            side=S.alternate("buy"),
            size=200,
            exit=S.bracket(stop=sl, target=tp, unit="ticks", on=on,
                           hold=Interval.seconds(90)),
        ), {"off": [0, 2], "sl": [2, 5], "tp": [3, 6]})
        for on in ("last", "bid_ask")
    ] + [
        (S.Plan(
            when=S.every(Interval.seconds(30)),
            entry=S.stop_limit(offset=bt.param("off", default=1), limit_offset=1, unit="ticks",
                               after_trigger="ioc", expire_after=Interval.seconds(25)),
            side=S.alternate("sell"),
            size=bt.param("qty", default=300),
        ), {"off": [0, 1, 3], "qty": [300, 2000]}),
        (S.Plan(
            when=S.every(Interval.seconds(60), on="print"),
            entry=S.market(),
            side=S.alternate("buy"),
            size=1500,
            exit=S.bracket(stop=sl, unit="ticks", hold=Interval.seconds(20)),
        ), {"sl": [1, 2, 4]}),
    ]


def _bits(table):
    """A table, every value as its bits: a ``NaN`` (a variant without a
    round trip) equal to itself."""
    return {
        name: [float(v).hex() if isinstance(v, float) else v for v in table[name].to_pylist()]
        for name in table.column_names
    }


@pytest.mark.parametrize("rule", [None, "aggressed_side", "both_sides"])
def test_stops_and_brackets_give_the_same_bits_on_both_engines(market, rule):
    exec_grid = {"latency.entry": [0, Interval.millis(3)], "latency.response": [0, MS],
                 "stop_in_market": ["trigger", "reject"]}
    cfg = S.Config(prints_update_book=rule, taker_fee_bps=0.7)
    for plan, grid in _stops_and_brackets():
        runs = {
            engine: S.sweep(plan, grid, cfg, markets=[market], engine=engine,
                            execution_grid=exec_grid, keep="fills")
            for engine in ("sim", "direct")
        }
        assert {e: r.engine for e, r in runs.items()} == {"sim": "sim", "direct": "direct"}
        assert _bits(runs["direct"].to_arrow()) == _bits(runs["sim"].to_arrow())
        for i in range(len(runs["sim"])):
            assert _fills(runs["direct"], i) == _fills(runs["sim"], i)
        df = runs["direct"].to_df(backend="arrow")
        assert sum(df["fills"]) > 0
        assert sum(df["entries_filled"]) > 0


def test_the_stop_and_bracket_plans_set_off_expire_and_exit(market):
    # What the plans above reach on the market of this file: the coverage of
    # the test before.
    totals = {}
    for plan, grid in _stops_and_brackets():
        df = S.sweep(plan, grid, S.Config(), markets=[market], engine="direct").to_df(
            backend="arrow")
        for key in ("entries_expired", "exits_stop", "exits_target", "exits_time",
                    "exits_carried", "entries_skipped_busy"):
            totals[key] = totals.get(key, 0) + sum(df[key])
    assert all(v > 0 for v in totals.values()), totals


@pytest.mark.parametrize("rule", [None, "aggressed_side", "both_sides"])
def test_both_engines_give_the_same_table_and_the_same_fills(market, rule):
    exec_grid = {"latency.entry": [0, Interval.millis(3)], "latency.feed": [0, MS]}
    cfg = S.Config(prints_update_book=rule, taker_fee_bps=0.7)
    runs = {
        engine: S.sweep(_plan(), GRID, cfg, markets=[market], engine=engine,
                        execution_grid=exec_grid, keep="fills")
        for engine in ("sim", "direct")
    }
    assert {e: r.engine for e, r in runs.items()} == {"sim": "sim", "direct": "direct"}
    assert len(runs["sim"]) == 48
    assert runs["direct"].to_arrow().equals(runs["sim"].to_arrow())
    for i in range(len(runs["sim"])):
        assert _fills(runs["direct"], i) == _fills(runs["sim"], i)
    assert sum(runs["direct"].to_df(backend="arrow")["fills"]) > 0


def test_the_execution_axes_are_the_slowest(market):
    grid = {"qty": [100, 200]}
    exec_grid = {
        "latency.entry": [0, Interval.millis(5)],
        "prints_update_book": [None, "both_sides"],
    }
    res = S.sweep(_plan(), grid, CFG, markets=[market], execution_grid=exec_grid, keep="fills")
    assert len(res) == 8
    df = res.to_df(backend="arrow")
    assert df["exec_latency.entry"] == [0] * 4 + [5 * MS] * 4
    assert df["exec_prints_update_book"] == [None, None, "both_sides", "both_sides"] * 2
    assert df["param_qty"] == [100, 200] * 4
    assert res.config_of(5).latency == {"entry": 5 * MS}
    assert res.config_of(5).prints_update_book is None
    # Each variant is the run of its own config: 5 ms later, the same
    # decisions.
    early, late = _fills(res, 2), _fills(res, 6)
    assert late[0][0] == early[0][0] + 5 * MS
    with pytest.raises(ValueError, match="unknown field"):
        S.sweep(_plan(), grid, CFG, markets=[market], execution_grid={"latency.entrance": [1]})
    with pytest.raises(ValueError, match="has no value"):
        S.sweep(_plan(), grid, CFG, markets=[market], execution_grid={"taker_fee_bps": []})


def test_the_same_results_on_one_two_and_seven_threads(market):
    runs = [
        S.sweep(_plan(), GRID, CFG, markets=[market], max_parallelism=n, keep="fills")
        for n in (1, 2, 7)
    ]
    first = runs[0].to_arrow()
    for other in runs[1:]:
        assert other.to_arrow().equals(first)
        for i in range(len(other)):
            assert _fills(other, i) == _fills(runs[0], i)
    with pytest.raises(ValueError, match="max_parallelism must be at least 1"):
        S.sweep(_plan(), GRID, CFG, markets=[market], max_parallelism=0)


def test_keep_metrics_and_fills_give_the_same_metrics(market):
    lite = S.sweep(_plan(), GRID, CFG, markets=[market], keep="metrics")
    full = S.sweep(_plan(), GRID, CFG, markets=[market], keep="fills")
    assert lite.to_arrow().equals(full.to_arrow())
    # The fills of a variant: kept, or computed again.
    for i in (0, 7, 11):
        assert _fills(lite, i) == _fills(full, i)
    with pytest.raises(ValueError, match='keep takes "metrics" or "fills"'):
        S.sweep(_plan(), GRID, CFG, markets=[market], keep="all")


def test_run_i_is_variant_i(market):
    res = S.sweep(_plan(), GRID, CFG, markets=[market], keep="fills")
    df = res.to_df(backend="arrow")
    for i in (0, 5, 11):
        one = res.run(i)
        f = one.fills_df(backend="arrow")
        got = [
            (t, float(p).hex(), float(q).hex(), s)
            for t, p, q, s in zip(f["timestamp"], f["price"], f["qty"], f["side"])
        ]
        assert got == _fills(res, i)
        assert one.position["XYZ"] == df["position"][i]
        assert len(one.events_df(backend="arrow")["event"]) > 0
        # And the plan run alone with the same parameters.
        p = res.params(i)
        alone = S.Plan(
            when=S.every(p["every"], on="print"),
            entry=S.market(),
            side=S.alternate(p["side"]),
            size=p["qty"],
        )
        a = S.run(alone, CFG, markets=[market]).fills_df(backend="arrow")
        assert a["price"] == f["price"]
        assert a["timestamp"] == f["timestamp"]
    with pytest.raises(IndexError):
        res.run(12)


def test_a_grid_is_refused_by_name_when_no_field_reads_it(market):
    with pytest.raises(ValueError, match='"spread", which no field of the plan reads'):
        S.sweep(_plan(), {"spread": [1, 2]}, CFG, markets=[market])
    with pytest.raises(ValueError, match="has no value"):
        S.sweep(_plan(), {"qty": []}, CFG, markets=[market])
    with pytest.raises(TypeError, match="not a bool"):
        S.sweep(_plan(), {"qty": [True]}, CFG, markets=[market])
    with pytest.raises(TypeError, match="takes a list of values"):
        S.sweep(_plan(), {"qty": 100}, CFG, markets=[market])
    with pytest.raises(ValueError, match="a plan runs on one market"):
        S.sweep(_plan(), {}, CFG, markets=[market, market])
    with pytest.raises(TypeError, match="plan is a mbt.sim.Plan"):
        S.sweep(lambda sim: None, {}, CFG, markets=[market])


def test_a_sweep_without_a_grid_is_one_variant_the_defaults(market):
    res = S.sweep(_plan(), None, CFG, markets=[market])
    assert len(res) == 1
    assert res.params(0) == {}
    alone = S.run(_plan(), CFG, markets=[market])
    assert res.to_df(backend="arrow")["fills"] == [alone.metrics["fills"]]
    assert json.loads(res._raw.config_json(0))["prints_update_book"] == "both_sides"


def test_by_day_takes_auto_true_or_false():
    with pytest.raises(ValueError, match='by_day takes "auto", True or False'):
        S.sweep(_plan(), {}, CFG, store=object(), symbol="XYZ", start=OPEN, end=OPEN + SEC,
                by_day="sometimes")
    with pytest.raises(TypeError, match="either markets=, or store"):
        S.sweep(_plan(), {}, CFG, markets=[], store=object(), symbol="XYZ", start=OPEN,
                end=OPEN + SEC)


# A store holding several days of one symbol (a book and a tape a day), for the sweep held
# day by day: <data_root>|<metadata_db>|<arrow_dir>|<symbol>. A tape enters a store only
# through bt.ingest_trades, so the test is skipped when it is unset.
_DAYS = os.environ.get("MANIFOLDBT_TEST_SIM_DAYS_STORE", "")


@pytest.mark.skipif(not _DAYS, reason="set MANIFOLDBT_TEST_SIM_DAYS_STORE to sweep over days")
@pytest.mark.parametrize("engine", ["direct", "sim"])
def test_a_sweep_held_day_by_day_gives_the_bits_of_the_market_held_whole(engine):
    root, meta, arrow, symbol = _DAYS.split("|")
    store = bt.DataStore(data_root=root, metadata_db=meta, arrow_dir=arrow)
    whole = S.Market.from_store(store, symbol, 0, 4_102_444_800 * SEC, by_day=False)
    start, end = whole.span
    plan = S.Plan(
        when=S.every(bt.param("every", default=Interval.minutes(30)), on="print"),
        entry=S.stop(offset=bt.param("off", default=1), unit="ticks",
                     expire_after=Interval.minutes(20)),
        side=S.alternate("buy"),
        size=100,
        exit=S.bracket(stop=10, target=20, unit="ticks", hold=Interval.hours(20)),
    )
    grid = {"every": [Interval.minutes(17), Interval.minutes(45)], "off": [0, 3]}
    held = S.sweep(plan, grid, CFG, markets=[whole], engine="sim", keep="fills")
    days = S.sweep(plan, grid, CFG, store=store, symbol=symbol, start=start, end=end + 1,
                   by_day=True, engine=engine, keep="fills")
    assert days.engine == engine
    assert days.to_df(backend="arrow") == held.to_df(backend="arrow")
    for i in range(len(held)):
        assert _fills(days, i) == _fills(held, i)