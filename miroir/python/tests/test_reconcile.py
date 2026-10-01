"""`bt.reconcile`: a broker's fills against the backtest of the same days.

Three halves, the split `test_fill_marks.py` already makes.

The half that runs everywhere needs no tape: how a journal is read (the column
spellings, the sides, the sort), the refusals a wrong journal hits first, and
the chart, which reads nothing but the dict it is handed.

The end-to-end half needs a store holding a tape, because a mark is a price the
market printed and a tape enters a store only through `bt.ingest_trades`. It
skips by name unless `MANIFOLDBT_TEST_TAPE_STORE` points at one, as
`<root>|<meta.sqlite>|<symbol_id>|<day>` -- the same variable
`test_fill_marks.py` reads.

The measurement itself is pinned by the engine's own tests, unit and end to
end.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from manifoldbt.reconcile import Reconciliation, fills_to_columns  # noqa: E402

SYMBOL_ID = 1
HORIZONS = ("100ms", "1s", "10s")


# ---------------------------------------------------------------------------
# Reading a journal
# ---------------------------------------------------------------------------


def _journal(**over):
    base = {
        "timestamp": pd.to_datetime(
            ["2021-03-01T00:00:02Z", "2021-03-01T00:00:01Z"], utc=True
        ),
        "side": ["buy", "sell"],
        "price": [100.0, 101.0],
        "qty": [1.0, 2.0],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_a_journal_is_read_in_time_order_whatever_order_it_was_written_in():
    cols = fills_to_columns(_journal(), SYMBOL_ID)
    assert cols["ts"] == sorted(cols["ts"])
    # The row that was second is now first, with its own price and side.
    assert cols["price"] == [101.0, 100.0]
    assert cols["side"] == [2, 1]
    assert cols["qty"] == [2.0, 1.0]
    assert cols["symbol"] == [SYMBOL_ID, SYMBOL_ID]


def test_the_common_broker_spellings_are_read_without_a_rename():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2021-03-01T00:00:01Z"], utc=True),
            "direction": ["B"],
            "avg_price": [100.0],
            "quantity": [3.0],
        }
    )
    cols = fills_to_columns(frame, SYMBOL_ID)
    assert cols["side"] == [1]
    assert cols["price"] == [100.0]
    assert cols["qty"] == [3.0]


def test_a_side_that_reads_as_neither_is_refused_rather_than_taken_for_a_sell():
    with pytest.raises(ValueError, match="neither a buy nor a sell"):
        fills_to_columns(_journal(side=["buy", "hedge"]), SYMBOL_ID)


def test_the_engines_own_side_codes_round_trip():
    cols = fills_to_columns(_journal(side=[1, 2]), SYMBOL_ID)
    assert cols["side"] == [2, 1]  # sorted by time: the second row comes first


def test_a_missing_column_names_what_it_wanted_and_what_it_saw():
    frame = _journal().drop(columns=["price"])
    with pytest.raises(ValueError, match="no 'price' column"):
        fills_to_columns(frame, SYMBOL_ID)


def test_a_journal_with_no_symbol_asks_for_one():
    with pytest.raises(ValueError, match="symbol_id"):
        fills_to_columns(_journal(), None)


def test_a_journal_carrying_its_own_symbol_column_keeps_it():
    cols = fills_to_columns(_journal(symbol_id=[7, 9]), None)
    assert cols["symbol"] == [9, 7]


def test_a_dict_of_columns_is_a_journal_too():
    cols = fills_to_columns(
        {
            "timestamp": np.array([2_000_000_000, 1_000_000_000], dtype="int64"),
            "side": np.array([1, 2], dtype="int64"),
            "price": np.array([100.0, 101.0]),
            "qty": np.array([1.0, 2.0]),
        },
        SYMBOL_ID,
    )
    assert cols["ts"] == [1_000_000_000, 2_000_000_000]


def test_something_that_is_not_a_table_is_refused_by_name():
    with pytest.raises(TypeError, match="DataFrame"):
        fills_to_columns([1, 2, 3], SYMBOL_ID)


def test_a_tolerance_that_is_not_a_duration_is_refused_by_name():
    with pytest.raises(TypeError, match="duration"):
        bt.reconcile(None, _journal(), store=None, symbol_id=1, tolerance="soon")


# ---------------------------------------------------------------------------
# The accessors and the chart
# ---------------------------------------------------------------------------


def _stub():
    """The engine's answer, hand-made: the wrapper reads it and nothing else."""
    horizons = [
        {
            "horizon": h, "horizon_ns": ns,
            "live_mean_bps": -1.2, "live_stderr_bps": 0.05, "live_marked_fills": 400,
            "sim_mean_bps": -0.7, "sim_stderr_bps": 0.04, "sim_marked_fills": 660,
            "diff_mean_bps": -0.5, "paired_mean_bps": -0.5,
            "paired_stderr_bps": 0.01, "paired_fills": 400,
        }
        for h, ns in zip(HORIZONS, (100_000_000, 1_000_000_000, 10_000_000_000))
    ]
    return {
        "tolerance_ns": 1_000_000_000,
        "price_tolerance": None,
        "live_fills": 400, "sim_fills": 660, "matched": 400,
        "live_only": 0, "sim_only": 260,
        "matched_share_live": 1.0, "matched_share_sim": 400 / 660,
        "service": {"orders_posted": 5000, "live_fills": 400, "sim_fills": 660,
                    "live_rate": 0.08, "sim_rate": 0.132, "served_ratio": 400 / 660},
        "by_day": {"key": [0], "live_fills": [400], "sim_fills": [660],
                   "matched": [400], "live_only": [0], "sim_only": [260],
                   "served_ratio": [400 / 660]},
        "by_hour": {"key": [0, 1], "live_fills": [200, 200],
                    "sim_fills": [330, 330], "matched": [200, 200],
                    "live_only": [0, 0], "sim_only": [130, 130],
                    "served_ratio": [0.606, 0.606]},
        "horizons": horizons,
        "half_spread": {"live_captured_bps": 0.001, "live_fills": 400,
                        "sim_captured_bps": 0.006, "sim_fills": 660,
                        "diff_bps": -0.005, "book_half_spread_bps": 0.0064},
        "price_gap_bps": {"count": 400, "mean": -0.5, "min": -0.5, "p05": -0.5,
                          "p25": -0.5, "median": -0.5, "p75": -0.5, "p95": -0.5,
                          "max": -0.5},
        "time_gap_ns": {"count": 400, "mean": 0.0, "min": 0.0, "p05": 0.0,
                        "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0,
                        "max": 0.0},
        "verdict": ["service: the backtest serves 13.2% ... where the journal serves 8.0%"],
        "pairs": {"live_index": [0], "sim_index": [0], "symbol_id": [1], "side": [1],
                  "live_timestamp": [0], "sim_timestamp": [0], "time_gap_ns": [0],
                  "live_price": [100.005], "sim_price": [100.0],
                  "price_edge_bps": [-0.5], "qty_gap": [0.0],
                  "live_mark_100ms": [99.9], "sim_mark_100ms": [99.9],
                  "live_mark_1s": [99.9], "sim_mark_1s": [99.9],
                  "live_mark_10s": [99.9], "sim_mark_10s": [99.9]},
    }


def test_the_tables_come_back_shaped_and_named():
    rec = Reconciliation(_stub())
    assert rec.live_fills == 400 and rec.sim_fills == 660 and rec.matched == 400

    day = rec.by_day_df()
    assert list(day.columns)[0] == "day"
    assert str(day["day"].dtype).startswith("datetime64[ns")

    hour = rec.by_hour_df()
    assert list(hour.columns)[0] == "hour"
    assert list(hour["hour"]) == [0, 1]

    horizons = rec.horizons_df()
    assert list(horizons["horizon"]) == list(HORIZONS)
    assert "paired_stderr_bps" in horizons.columns

    pairs = rec.pairs_df()
    assert str(pairs["live_timestamp"].dtype).startswith("datetime64[ns")
    for h in HORIZONS:
        assert f"live_mark_{h}" in pairs.columns
        assert f"sim_mark_{h}" in pairs.columns


def test_the_summary_is_the_verdict_one_line_at_a_time():
    rec = Reconciliation(_stub())
    assert rec.summary() == "\n".join(rec.verdict)
    assert "Reconciliation(" in repr(rec)


def test_the_chart_is_exported_and_refuses_what_is_not_a_reconciliation():
    plot = pytest.importorskip("manifoldbt.plot")
    assert "reconcile" in plot.__all__
    with pytest.raises(ValueError, match="bt.reconcile"):
        plot.reconcile(object())


def test_the_chart_draws_both_sides_on_both_panels():
    pytest.importorskip("plotly")
    plot = pytest.importorskip("manifoldbt.plot")
    fig = plot.reconcile(Reconciliation(_stub()), show=False)
    # Two bars a panel, two panels: service by hour and markout by horizon.
    assert len(fig.data) == 4
    assert {t.name for t in fig.data} == {"backtest", "journal"}


def test_the_chart_refuses_a_reconciliation_of_two_empty_sides():
    plot = pytest.importorskip("manifoldbt.plot")
    empty = _stub()
    empty["live_fills"] = 0
    empty["sim_fills"] = 0
    with pytest.raises(ValueError, match="nothing"):
        plot.reconcile(Reconciliation(empty))


def test_the_name_is_exported():
    assert "reconcile" in bt.__all__
    assert "Reconciliation" in bt.__all__
    assert callable(bt.reconcile)


# ---------------------------------------------------------------------------
# End to end, on a store that holds a tape
# ---------------------------------------------------------------------------

_TAPE = os.environ.get("MANIFOLDBT_TEST_TAPE_STORE", "")
_needs_tape = pytest.mark.skipif(
    not _TAPE,
    reason="set MANIFOLDBT_TEST_TAPE_STORE=<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD> "
    "to reconcile real fills (a tape enters a store only through bt.ingest_trades, "
    "which needs the network)",
)


def _tape_store():
    root, meta, sid, day = _TAPE.split("|")
    store = bt.DataStore(
        data_root=root, metadata_db=meta, arrow_dir=os.path.join(root, "mega")
    )
    start = int(bt.date_to_ns(day))
    return store, int(sid), start, start + 86_400_000_000_000


def _tape_run(store, sid, start, end):
    strategy = (
        bt.Strategy.create("maker")
        .signal("level", col("close") * lit(1.0 - 1e-4))
        .size(lit(0.01))
        .limit_entry(signal="level", time_in_force={"GTB": 1})
        .take_profit(pct=0.02, side="long")
        .stop_loss(pct=0.06, side="long")
    )
    cfg = bt.BacktestConfig(
        universe=[sid],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.seconds(1),
        initial_capital=100_000.0,
        # Daily output, the way a run without a licence sees it: nothing here
        # asserts on an intraday row.
        output_resolution=Interval.days(1),
        warmup_bars=60,
        execution=bt.ExecutionConfig(
            signal_delay=1, allow_short=False, position_sizing_mode="Units",
            fill_resolution="ticks", fill_marks=True,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
    )
    return bt.run(strategy, cfg, store)


def _as_journal(trades, *, drop=0.0, worse_bps=0.0, seed=20260908):
    kept = trades
    if drop:
        rng = np.random.default_rng(seed)
        kept = trades.loc[rng.random(len(trades)) >= drop]
    side = kept["side"].to_numpy()
    price = kept["fill_price"].to_numpy()
    if worse_bps:
        price = price * (1.0 + np.where(side == 1, 1.0, -1.0) * worse_bps * 1e-4)
    return pd.DataFrame({
        "timestamp": kept["execution_timestamp"].to_numpy(),
        "side": np.where(side == 1, "buy", "sell"),
        "price": price,
        "qty": kept["quantity"].to_numpy(),
    })


@_needs_tape
def test_a_run_reconciled_with_its_own_fill_log_is_zero_everywhere():
    store, sid, start, end = _tape_store()
    result = _tape_run(store, sid, start, end)
    trades = result.trades_df()
    assert len(trades) > 0, "the day has to trade for this to say anything"

    rec = bt.reconcile(result, _as_journal(trades), store=store, symbol_id=sid,
                       tolerance=Interval.seconds(1))
    assert rec.matched == rec.live_fills == rec.sim_fills == len(trades)
    assert rec.live_only == 0 and rec.sim_only == 0
    for h in rec.horizons:
        assert h["paired_mean_bps"] == 0.0
        assert h["diff_mean_bps"] == 0.0
        assert h["live_mean_bps"] == h["sim_mean_bps"]
    assert rec.price_gap_bps["mean"] == 0.0
    assert rec.time_gap_ns["max"] == 0.0
    # The marks it took for the backtest side ARE the run's own marks.
    for own, both in zip(result.fill_marks["horizons"], rec.horizons):
        assert own["mean_bps"] == both["sim_mean_bps"]
        assert own["stderr_bps"] == both["sim_stderr_bps"]
        assert own["marked_fills"] == both["sim_marked_fills"]


@_needs_tape
def test_a_degraded_journal_reads_back_the_degradation_that_was_injected():
    store, sid, start, end = _tape_store()
    result = _tape_run(store, sid, start, end)
    trades = result.trades_df()
    journal = _as_journal(trades, drop=0.40, worse_bps=0.5)

    rec = bt.reconcile(result, journal, store=store, symbol_id=sid,
                       tolerance=Interval.seconds(1))
    assert rec.matched == rec.live_fills, "a kept fill lost its twin"
    assert rec.sim_only == rec.sim_fills - rec.live_fills
    assert abs(rec.live_fills / rec.sim_fills - 0.60) < 0.05
    assert abs(rec.price_gap_bps["mean"] + 0.5) < 1e-2
    for h in rec.horizons:
        if h["paired_fills"] < 20:
            continue
        assert abs(h["paired_mean_bps"] + 0.5) < 1e-2, h["horizon"]

    # The tables come back with rows in them, and the day is the run's day.
    assert len(rec.by_day_df()) >= 1
    assert len(rec.by_hour_df()) >= 1
    assert len(rec.pairs_df()) == rec.matched
    assert any(line.startswith("service:") for line in rec.verdict)


@_needs_tape
def test_the_service_rates_share_the_backtests_own_posting_count():
    store, sid, start, end = _tape_store()
    result = _tape_run(store, sid, start, end)
    posted = result.order_activity["orders_posted"]
    rec = bt.reconcile(result, _as_journal(result.trades_df(), drop=0.40),
                       store=store, symbol_id=sid)
    assert rec.service["orders_posted"] == posted
    assert rec.service["sim_rate"] == pytest.approx(rec.sim_fills / posted)
    assert rec.service["live_rate"] == pytest.approx(rec.live_fills / posted)
