"""The tables and the totals of a result that quotes, read from Arrow.

``orders_df``, ``fills_df``, ``events_df``, ``equity_df`` and ``fill_marks_df``
used to be built from one Python list per column, and ``SimResult.metrics``
from a Python loop over every order, fill and equity sample. They are built
from Arrow columns and totals the engine computes now; the lists are still
there (``orders_columns`` and the rest), and this file holds the new reads to
what the old code built from them: the same columns, in the same order, of the
same types, with the same values to the bit, NaN and null included, for pandas,
polars and the plain dict.
"""
import math
import sys

import pytest

import manifoldbt as bt
from manifoldbt.sim import _to_frame
from test_quote_dsl import (
    MS,
    _layer_open,
    needs_layer,
    py_maker,
    run_on,
    sim_config,
    walk_market,
)

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

BACKENDS = ("pandas", "polars")


# ---------------------------------------------------------------------------
# What the old code built, and how two frames are the same
# ---------------------------------------------------------------------------


def old_frame(sim, table, backend):
    """The frame as the accessors built it before they read Arrow."""
    raw = sim.raw
    if table == "orders":
        cols = raw.orders_columns()
        cols["sent_at"] = cols.pop("sent_ns")
        return _to_frame(cols, ["sent_at"], backend)
    if table == "fills":
        return _to_frame(raw.fills_columns(), ["timestamp"], backend)
    if table == "events":
        return _to_frame(raw.events_columns(), ["timestamp"], backend)
    if table == "equity":
        ts, equity = raw.equity_columns()
        return _to_frame({"timestamp": ts, "equity": equity}, ["timestamp"], backend)
    if table == "fill_marks":
        return _to_frame(raw.fill_marks_detail(), ["timestamp", "anchor"], backend)
    raise AssertionError(table)


def old_result_fill_marks(res, backend):
    """``Result.fill_marks_df`` as it was, from ``fill_marks_detail``."""
    cols = res.raw.fill_marks_detail()
    if cols is None:
        return None
    return _to_frame(cols, ["timestamp", "anchor"], backend)


def old_metrics(sim):
    """``SimResult.metrics`` as it was: a Python loop over the lists."""
    orders = sim.raw.orders_columns()
    fills = sim.raw.fills_columns()
    _, equity = sim.raw.equity_columns()
    final = equity[-1] if equity else sim.config.initial_cash
    peak, drawdown = float("-inf"), 0.0
    for e in equity:
        peak = max(peak, e)
        drawdown = min(drawdown, e - peak)
    served = sum(1 for f in orders["filled"] if f > 0)
    return {
        "orders": len(orders["order_id"]),
        "orders_filled": served,
        "fills": len(fills["order_id"]),
        "maker_fills": sum(1 for m in fills["maker"] if m),
        "taker_fills": sum(1 for m in fills["maker"] if not m),
        "volume": float(sum(fills["qty"])),
        "notional": float(sum(q * p for q, p in zip(fills["qty"], fills["price"]))),
        "fees": sim.fees,
        "final_equity": final,
        "pnl": final - sim.config.initial_cash,
        "max_drawdown": drawdown,
        "position": sim.position,
    }


def same_cell(a, b):
    """The same value of the same type; a float to the bit, NaN to NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a.hex() == b.hex() or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same_cell(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_cell(x, y) for x, y in zip(a, b))
    return a == b


def assert_identical(new, old):
    """Same columns, order, types and values, NaN and null included."""
    if old is None:
        assert new is None
        return
    assert type(new) is type(old)
    if isinstance(old, pd.DataFrame):
        assert list(new.columns) == list(old.columns)
        assert [str(t) for t in new.dtypes] == [str(t) for t in old.dtypes]
        assert all(a == b for a, b in zip(new.dtypes, old.dtypes))
        assert type(new.index) is type(old.index) and new.index.equals(old.index)
        pd.testing.assert_frame_equal(new, old, check_exact=True)
        for c in old.columns:
            a, b = new[c].to_list(), old[c].to_list()
            assert len(a) == len(b), c
            bad = [i for i, (x, y) in enumerate(zip(a, b)) if not same_cell(x, y)]
            assert not bad, (c, bad[:5], [a[i] for i in bad[:5]], [b[i] for i in bad[:5]])
        return
    if isinstance(old, pl.DataFrame):
        assert new.schema == old.schema
        from polars.testing import assert_frame_equal

        assert_frame_equal(new, old, check_exact=True)
        for c in old.columns:
            assert same_cell(new[c].to_list(), old[c].to_list()), c
        return
    assert same_cell(new, old)


# ---------------------------------------------------------------------------
# A run that quotes, and the same maker through bt.sim
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quoted():
    if not _layer_open():
        pytest.skip("order-level simulation locked")
    return run_on(walk_market())


@pytest.fixture(scope="module")
def simulated():
    if not _layer_open():
        pytest.skip("order-level simulation locked")
    return bt.sim.run(py_maker, sim_config(), markets=[walk_market()])


TABLES = ("orders", "fills", "events", "equity", "fill_marks")


def _read(sim, table, backend):
    name = {"equity": "equity_df", "fill_marks": "fill_marks_df"}.get(table, f"{table}_df")
    return getattr(sim, name)(backend=backend)


@pytest.mark.parametrize("backend", BACKENDS + ("arrow",))
@pytest.mark.parametrize("table", TABLES)
def test_each_table_of_a_quoting_run_is_the_one_the_lists_built(quoted, table, backend):
    new = _read(quoted.sim, table, backend)
    assert len(old_frame(quoted.sim, table, "pandas")) > 0, "the table under test is empty"
    assert_identical(new, old_frame(quoted.sim, table, backend))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("table", ("orders", "fills", "events"))
def test_the_quote_result_hands_over_the_venue_s_tables(quoted, table, backend):
    new = getattr(quoted, f"{table}_df")(backend)
    assert_identical(new, old_frame(quoted.sim, table, backend))


@pytest.mark.parametrize("backend", BACKENDS + ("arrow",))
def test_the_result_s_fill_marks_are_the_ones_the_lists_built(quoted, backend):
    assert quoted.raw.fill_marks_detail() is not None
    assert_identical(quoted.fill_marks_df(backend), old_result_fill_marks(quoted, backend))


@pytest.mark.parametrize("backend", BACKENDS + ("arrow",))
@pytest.mark.parametrize("table", TABLES)
def test_each_table_of_a_simulation_is_the_one_the_lists_built(simulated, table, backend):
    assert_identical(_read(simulated, table, backend), old_frame(simulated, table, backend))


def test_the_metrics_of_a_quoting_run_are_the_loop_s(quoted):
    assert quoted.sim.metrics["fills"] > 10
    assert_identical(quoted.sim.metrics, old_metrics(quoted.sim))


def test_the_metrics_of_a_simulation_are_the_loop_s(simulated):
    assert_identical(simulated.metrics, old_metrics(simulated))
    assert simulated.summary() == _old_summary(old_metrics(simulated))


def _old_summary(m):
    return (
        f"{m['orders']} orders, {m['orders_filled']} filled; {m['fills']} fills "
        f"({m['maker_fills']} maker, {m['taker_fills']} taker), volume {m['volume']:g}; "
        f"P&L {m['pnl']:.2f} after {m['fees']:.2f} of fees"
    )


def test_the_metrics_are_counted_once_and_handed_out_fresh(simulated):
    first = simulated.metrics
    first["pnl"] = 0.0
    first["position"]["BTCUSDT"] = 1e9
    again = simulated.metrics
    assert_identical(again, old_metrics(simulated))
    # The P&L is read against the initial cash of the moment, as before.
    cash = simulated.config.initial_cash
    try:
        simulated.config.initial_cash = cash + 10.0
        assert_identical(simulated.metrics, old_metrics(simulated))
    finally:
        simulated.config.initial_cash = cash


def test_the_sums_follow_the_interpreter_s_own_sum(simulated):
    """Python 3.12 made ``sum()`` of floats compensated: the engine is told
    which one runs, and matches it to the bit, whichever it is."""
    qty = simulated.raw.fills_columns()["qty"]
    compensated = sys.version_info >= (3, 12)
    totals = simulated.raw.metrics_totals(compensated)
    assert totals[5].hex() == float(sum(qty)).hex()


# ---------------------------------------------------------------------------
# The corners: nothing filled, nothing at all, a market order
# ---------------------------------------------------------------------------


def _ladder(ts):
    return (ts, [(100.0, 5.0), (99.9, 9.0)], [(100.1, 5.0), (100.5, 6.0)])


def _run(strategy, trades=()):
    market = bt.sim.Market.from_ladders(
        "BTCUSDT", 0.1, [_ladder(MS), _ladder(5 * MS)], list(trades)
    )
    return bt.sim.run(strategy, bt.sim.Config(initial_cash=100.0), markets=[market])


@needs_layer
@pytest.mark.parametrize("backend", BACKENDS + ("arrow",))
def test_a_run_where_nothing_fills_keeps_its_all_missing_columns(backend):
    """No fill: every ``avg_price`` is missing, a column of ``None`` that pandas
    typed ``object`` and polars ``Null``; the fills and the marks are empty."""

    def strategy(sim):
        sim.elapse(2 * MS)
        # Behind the book's own size, and no print to burn it down.
        sim.post("BTCUSDT", "buy", 99.9, 1.0)
        sim.post("BTCUSDT", "sell", 100.5, 1.0)
        sim.elapse(8 * MS)

    res = _run(strategy)
    assert res.raw.fills_columns()["order_id"] == []
    assert set(res.raw.orders_columns()["avg_price"]) == {None}
    for table in TABLES:
        assert_identical(_read(res, table, backend), old_frame(res, table, backend))
    assert_identical(res.metrics, old_metrics(res))


@needs_layer
@pytest.mark.parametrize("backend", BACKENDS + ("arrow",))
def test_a_run_that_sends_nothing_has_the_empty_tables_it_had(backend):
    def strategy(sim):
        sim.elapse(8 * MS)

    res = _run(strategy)
    assert res.raw.orders_columns()["order_id"] == []
    for table in TABLES:
        assert_identical(_read(res, table, backend), old_frame(res, table, backend))
    assert_identical(res.metrics, old_metrics(res))


@needs_layer
@pytest.mark.parametrize("backend", BACKENDS + ("arrow",))
def test_a_market_order_leaves_a_missing_price_among_the_others(backend):
    def strategy(sim):
        sim.elapse(2 * MS)
        sim.post("BTCUSDT", "buy", 100.0, 1.0)
        sim.post("BTCUSDT", "buy", None, 2.0, tif="IOC")
        sim.post("BTCUSDT", "sell", 100.5, 0.5)
        sim.elapse(8 * MS)

    res = _run(strategy, [(3 * MS, 99.9, 10.0, "sell")])
    prices = res.raw.orders_columns()["price"]
    assert None in prices and any(p is not None for p in prices)
    for table in TABLES:
        assert_identical(_read(res, table, backend), old_frame(res, table, backend))
    assert_identical(res.metrics, old_metrics(res))


# ---------------------------------------------------------------------------
# final_equity and pnl among a run's metrics
# ---------------------------------------------------------------------------


def test_a_quoting_run_reports_its_final_equity_and_its_pnl(quoted):
    m = quoted.metrics
    last = quoted.raw.equity_curve[-1].as_py()
    capital = quoted.raw.manifest["config"]["initial_capital"]
    assert m["final_equity"].hex() == float(last).hex()
    assert m["pnl"].hex() == (last - capital).hex()
    assert m["pnl"] == pytest.approx(m["total_return"] * capital, rel=1e-9, abs=1e-9)
    # The venue's own record agrees on where the money ended.
    assert m["final_equity"] == pytest.approx(quoted.sim.metrics["final_equity"])


def test_a_bar_run_reports_its_final_equity_and_its_pnl(tmp_path):
    import os

    np = pytest.importorskip("numpy")
    n = 3 * 1440
    close = 100.0 + np.sin(np.arange(n) / 50.0)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000.0),
        }
    )
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"), metadata_db=os.path.join(root, "meta.sqlite"),
    )
    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=int(df["timestamp"].iloc[-1].value) + 60_000_000_000,
        bar_interval=bt.Interval.minutes(1),
        initial_capital=10_000.0,
        fees=bt.FeeConfig.zero(),
    )
    strategy = (
        bt.Strategy.create("sine")
        .signal("up", bt.col("close") > bt.col("close").lag(1))
        .size(bt.when(bt.col("up"), 1.0, -1.0))
    )
    res = bt.run(strategy, config, store)
    m = res.metrics
    last = res.raw.equity_curve[-1].as_py()
    assert m["final_equity"].hex() == float(last).hex()
    assert m["pnl"].hex() == (last - 10_000.0).hex()
    assert m["pnl"] == pytest.approx(m["total_return"] * 10_000.0, rel=1e-9, abs=1e-6)
    # The two new keys sit after the ones a run always reported.
    assert list(m)[-2:] == ["final_equity", "pnl"]
    # A lite result of the same run reports the same two, to the bit.
    (lite,) = bt.run_batch_lite([strategy], config, store)
    # (a lite run keeps no trade statistics)
    assert list(lite.metrics) == [k for k in m if k != "trade_stats"]
    assert lite.metrics["final_equity"].hex() == m["final_equity"].hex()
    assert lite.metrics["final_equity"].hex() == lite.final_equity.hex()
    assert lite.metrics["pnl"].hex() == m["pnl"].hex()
