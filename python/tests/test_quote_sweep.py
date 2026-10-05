"""Sweeps and batches of strategies that quote (``Strategy.quote``).

Each combination of a sweep is the run ``mbt.run`` makes of the same strategy
with those values as its defaults, under that execution config: the same
orders, fills, order events and equity, to the bit, whatever the number of
threads. The metrics-only sweep keeps the full run's metrics to the bit. The
enumeration, the labels and the result types are the bar sweep's.

The engine's own tests hold the bit-level proofs; this
file checks the Python door: the routing of ``run_sweep``, ``run_sweep_lite``,
``run_batch`` and ``run_batch_lite``, the refusals by name, the labels of
``to_df``, and, where the stores of the 0.27 duel are present, real days.
"""
import json
import os

import pytest

import manifoldbt as bt
from manifoldbt import _native, book
from manifoldbt.helpers import Interval
from manifoldbt.result import QuoteResult
from manifoldbt.sweep import SweepResult

from test_quote_dsl import MS, needs_layer, walk_market, _layer_open

# Folders of the real stores; the tests that read them are skipped when unset.
DUEL = os.environ.get("BT_DUEL_STORES", "")


def maker(clip=0.01, max_inv=0.05):
    """Example 31's maker with its clip and its inventory bound as parameters."""
    c = bt.param("clip", default=clip)
    m = bt.param("max_inv", default=max_inv)
    inventory = bt.position()
    clips = bt.round(inventory / c)
    bids, asks = book.bid_levels(), book.ask_levels()
    book_ok = (bids > 0) & (asks > 0)
    bid = book.bid_price_at(bt.clip(clips, 0, bids - 1) + 1)
    ask = book.ask_price_at(bt.clip(-clips, 0, asks - 1) + 1)
    return (
        bt.Strategy.create("maker")
        .quote("buy", bt.when(book_ok, bid), c, enabled=inventory < m)
        .quote("sell", bt.when(book_ok, ask), c, enabled=inventory > -m)
    )


def config(**over):
    kw = dict(
        universe=[1],
        time_range_start=0,
        time_range_end=86_400_000_000_000,
        bar_interval=Interval.millis(100),
        initial_capital=1_000.0,
        fees=bt.FeeConfig(maker_fee_bps=0.0, taker_fee_bps=4.0),
        execution=bt.ExecutionConfig(
            latency={"order": 5 * MS, "cancel": 5 * MS, "response": 5 * MS, "feed": 2 * MS},
            fill_model={"queue": {"depth_source": "book", "assumed_queue": 2.0}},
        ),
    )
    kw.update(over)
    return bt.BacktestConfig(**kw)


# Written out of order on purpose: the enumeration sorts by name.
GRID = {"max_inv": [0.03, 0.05], "clip": [0.01, 0.02]}
EXEC = {"latency.order": [0, 20 * MS], "fill_model.queue.assumed_queue": [1.0, 3.0]}


def combos():
    """The combinations in the engine's order: execution axes slowest, then
    the parameters, each set of axes sorted by name, the last fastest."""
    from manifoldbt.dataframe import exec_grid_combos, grid_combos

    return [(e, p) for e in exec_grid_combos(EXEC) for p in grid_combos(GRID)]


def with_exec(cfg, assignment):
    """``cfg`` with an execution assignment written into it, as the engine
    applies one."""
    execution = json.loads(cfg.to_json())["execution"]
    for path, value in assignment.items():
        node = execution
        *head, last = path.split(".")
        for seg in head:
            node = node.setdefault(seg, {})
        node[last] = value
    out = json.loads(cfg.to_json())
    out["execution"] = execution
    return json.dumps(out)


def sweep_native(market, threads, lite=False):
    fn = _native.run_quote_sweep_lite if lite else _native.run_quote_sweep
    grid_json = json.dumps({k: [{"Float64": v} for v in vs] for k, vs in GRID.items()})
    return fn(maker().to_json(), grid_json, config().to_json(), None, [market], threads,
              json.dumps(EXEC))


def alone(market, exec_assignment, params):
    raw, sim = _native.run_quotes(
        maker(**params).to_json(), with_exec(config(), exec_assignment), None, [market]
    )
    return raw, sim


def _bits(v):
    return float(v).hex() if isinstance(v, float) else v


def journal(sim):
    out = {}
    for name in ("orders_columns", "fills_columns", "events_columns"):
        cols = getattr(sim, name)()
        out[name] = [tuple(_bits(v) for v in row) for row in zip(*cols.values())]
    ts, eq = sim.equity_columns()
    out["equity"] = list(zip(ts, (e.hex() for e in eq)))
    out["totals"] = (sim.cash.hex(), sim.fees.hex(), sim.position)
    return out


@needs_layer
def test_every_combination_is_the_run_alone_to_the_bit_on_one_two_and_eight_threads():
    market = walk_market(seed=11, n=3_000)
    expected = [alone(market, e, p) for e, p in combos()]
    assert len({len(sim.orders_columns()["order_id"]) for _, sim in expected}) > 4
    for threads in (1, 2, 8):
        got = sweep_native(market, threads)
        assert len(got) == len(expected) == 16
        for i, ((raw, sim), (raw0, sim0)) in enumerate(zip(got, expected)):
            assert journal(sim) == journal(sim0), f"{threads} thread(s), combination {i}"
            assert json.dumps(raw.metrics, sort_keys=True) == json.dumps(
                raw0.metrics, sort_keys=True
            ), f"{threads} thread(s), combination {i}"
            assert raw.order_activity == raw0.order_activity


@needs_layer
def test_the_metrics_only_sweep_keeps_the_full_metrics_to_the_bit():
    market = walk_market(seed=5, n=2_000)
    full = sweep_native(market, 3)
    lite = sweep_native(market, 3, lite=True)
    assert len(full) == len(lite)
    for (raw, _), l in zip(full, lite):
        assert json.dumps(raw.metrics, sort_keys=True) == json.dumps(l.metrics, sort_keys=True)
        assert l.trade_count == raw.trade_count
        assert l.final_equity.hex() == float(raw.equity_curve[-1].as_py()).hex()


@needs_layer
def test_the_labels_are_read_from_each_run_s_manifest():
    market = walk_market(seed=3, n=600)
    pairs = sweep_native(market, 4)
    sweep = SweepResult(bt._quote_results(pairs, config()), GRID, EXEC)
    assert all(isinstance(r, QuoteResult) for r in sweep)
    pd = pytest.importorskip("pandas")
    df = sweep.to_df("pandas")
    assert isinstance(df, pd.DataFrame)
    for i, (e, p) in enumerate(combos()):
        row = df.iloc[i]
        assert row["param_clip"] == p["clip"] and row["param_max_inv"] == p["max_inv"]
        assert row["exec_latency.order"] == e["latency.order"]
        assert row["exec_fill_model.queue.assumed_queue"] == e["fill_model.queue.assumed_queue"]
    # A combination is a QuoteResult like any run's, its journal beside it.
    assert len(sweep[5].orders_df("pandas")) == len(pairs[5][1].orders_columns()["order_id"])


@needs_layer
def test_a_batch_of_strategies_that_quote_is_each_run_alone():
    market = walk_market(seed=9, n=1_500)
    strategies = [maker(0.01, 0.05), maker(0.02, 0.03)]
    got = _native.run_quote_batch([s.to_json() for s in strategies], config().to_json(),
                                  None, [market], 2)
    lite = _native.run_quote_batch_lite([s.to_json() for s in strategies],
                                        config().to_json(), None, [market], 2)
    for (raw, sim), l, s in zip(got, lite, strategies):
        raw0, sim0 = _native.run_quotes(s.to_json(), config().to_json(), None, [market])
        assert journal(sim) == journal(sim0)
        assert json.dumps(l.metrics, sort_keys=True) == json.dumps(raw0.metrics, sort_keys=True)


@needs_layer
def test_the_sweeps_refuse_by_name(tmp_path):
    store = bt.DataStore(data_root=str(tmp_path / "data"),
                         metadata_db=str(tmp_path / "meta.sqlite"))
    s, cfg = maker(), config()
    with pytest.raises(Exception, match="not declared"):
        bt.run_sweep(s, {"spread": [1.0]}, cfg, store)
    with pytest.raises(ValueError, match="on the CPU"):
        bt.run_sweep_lite(s, {"clip": [0.01]}, cfg, store, device="cuda")
    with pytest.raises(ValueError, match="fp64 only"):
        bt.run_sweep_lite(s, {"clip": [0.01]}, cfg, store, device="cpu", precision="fp32")
    plain = bt.Strategy.create("bars").signal("x", bt.col("close") > 1).size(bt.col("x"))
    with pytest.raises(Exception, match=r"'bars' do\(es\) not quote"):
        bt.run_batch([s, plain], cfg, store)
    # An execution axis the order-level simulation cannot honour, before a day
    # is read.
    with pytest.raises(Exception, match="latency.feed"):
        bt.run_sweep(s, {"clip": [0.01]}, cfg, store,
                     execution_grid={"latency.feed": [0, -1]})
    # A setting a quoting run does not read, on an execution axis.
    with pytest.raises(Exception, match="max_participation_rate"):
        bt.run_sweep(s, {}, cfg, store,
                     execution_grid={"fill_model.max_participation_rate": [0.1]})


@pytest.mark.skipif(_layer_open(), reason="the order-level layer is open here")
def test_a_sweep_of_quotes_is_behind_the_same_door_as_bt_sim(tmp_path):
    store = bt.DataStore(data_root=str(tmp_path / "data"),
                         metadata_db=str(tmp_path / "meta.sqlite"))
    for call in (
        lambda: bt.run_sweep(maker(), {"clip": [0.01]}, config(), store),
        lambda: bt.run_sweep_lite(maker(), {"clip": [0.01]}, config(), store),
        lambda: bt.run_batch([maker()], config(), store),
        lambda: bt.run_batch_lite([maker()], config(), store),
    ):
        with pytest.raises(PermissionError, match="Order-level simulation"):
            call()


# ---------------------------------------------------------------------------
# A real day
# ---------------------------------------------------------------------------


def _duel_store(name="store"):
    root = os.path.join(DUEL, name)
    return bt.DataStore(
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
        arrow_dir=os.path.join(root, "data", "mega"),
    )


@needs_layer
@pytest.mark.skipif(not DUEL or not os.path.isdir(os.path.join(DUEL, "store")),
                    reason="the duel's stores are not here")
def test_a_sweep_on_a_real_day_is_each_run_alone():
    from manifoldbt.helpers import time_range

    start, end = time_range("2025-08-23", "2025-08-24")
    cfg = config(time_range_start=start, time_range_end=end, initial_capital=100_000.0)
    grid = {"clip": [0.01, 0.02]}
    # Two latencies far enough apart to move the run on this day: a flat axis
    # would make the comparison trivial. (The feed latency is not one: this
    # day's book is stamped at a fixed phase of the wake-up grid, and 50 ms of
    # feed does not change which state a wake-up sees.)
    exec_grid = {"latency.order": [5 * MS, 50 * MS]}
    sweep = bt.run_sweep(maker(), grid, cfg, _duel_store(), execution_grid=exec_grid,
                         max_parallelism=4)
    lite = bt.run_sweep_lite(maker(), grid, cfg, _duel_store(), execution_grid=exec_grid,
                             max_parallelism=4)
    assert len(sweep) == len(lite) == 4
    i, journals = 0, set()
    for order in exec_grid["latency.order"]:
        for clip in grid["clip"]:
            c = config(time_range_start=start, time_range_end=end, initial_capital=100_000.0)
            c.execution.latency = dict(c.execution.latency, order=order)
            one = bt.run(maker(clip=clip), c, _duel_store())
            assert len(one.sim.raw.orders_columns()["order_id"]) > 1_000
            j = journal(one.sim.raw)
            journals.add(repr(j["fills_columns"]))
            assert journal(sweep[i].sim.raw) == j, f"combination {i}"
            m = json.dumps(one.metrics, sort_keys=True)
            assert json.dumps(sweep[i].metrics, sort_keys=True) == m
            assert json.dumps(lite[i].metrics, sort_keys=True) == m
            i += 1
    assert len(journals) == 4

ANNEE = os.environ.get("BT_ANNEE_STORE", "")


@needs_layer
@pytest.mark.skipif(not ANNEE or not os.path.isdir(ANNEE), reason="the year's store is not here")
def test_a_sweep_over_days_holds_them_day_by_day_and_is_each_run_alone():
    """Over more than one day the combinations run day by day, as ``run``
    holds them (its ``res.sim`` equity at the output resolution), each lot of
    ``max_parallelism`` combinations reading every day once."""
    from manifoldbt.helpers import time_range

    start, end = time_range("2026-08-01", "2026-08-03")
    cfg = config(time_range_start=start, time_range_end=end, initial_capital=100_000.0)
    store = bt.DataStore(
        data_root=os.path.join(ANNEE, "data"),
        metadata_db=os.path.join(ANNEE, "meta.sqlite"),
        arrow_dir=os.path.join(ANNEE, "data", "mega"),
    )
    grid = {"clip": [0.1, 0.2, 0.3]}
    # Three combinations on two threads: a lot of two, then a lot of one.
    sweep = bt.run_sweep(maker(max_inv=0.5), grid, cfg, store, max_parallelism=2)
    lite = bt.run_sweep_lite(maker(max_inv=0.5), grid, cfg, store, max_parallelism=2)
    assert len(sweep) == len(lite) == 3
    for i, clip in enumerate(grid["clip"]):
        one = bt.run(maker(clip=clip, max_inv=0.5), cfg, store)
        j = journal(one.sim.raw)
        assert len(j["orders_columns"]) > 1_000
        # Held day by day, the venue's curve is kept at the output resolution.
        assert len(j["equity"]) < 2 * 86_400 * 10 / 2
        assert journal(sweep[i].sim.raw) == j, f"combination {i}"
        m = json.dumps(one.metrics, sort_keys=True)
        assert json.dumps(sweep[i].metrics, sort_keys=True) == m
        assert json.dumps(lite[i].metrics, sort_keys=True) == m
