"""The trade clock (``bar_interval=Interval.trades()``) seen from Python.

Two halves, and the split is the same one `test_fill_resolution.py` makes.

The half that runs everywhere needs no tape: the JSON shape, the DSL
guard, and every refusal the engine takes BEFORE it loads anything. Those are
the ones a user hits first, and they are the ones a wrong config must not slip
past.

The end-to-end half needs a store holding a tape, and a tape enters a store
only through ``bt.ingest_trades``, which fetches a venue's archive over the
network. Writing Arrow files into the store's private layout instead would test
the layout rather than the feature and would keep passing after the layout
moved. So it skips by name unless ``MANIFOLDBT_TEST_TAPE_STORE`` points at a
store that already holds one, as ``<root>|<meta.sqlite>|<symbol_id>|<day>``.
The per-print semantics themselves are pinned by the engine's own end-to-end
tests.
"""
import os

import pytest

import manifoldbt as bt
from manifoldbt.expr import col
from manifoldbt.helpers import Interval, Slippage

IS_PRO = bt.license_info()[0] == "Pro"


def _locked():
    """Probe the tick gate, not the licence banner.

    The gate answers before any I/O, so a path that is not a CSV separates the
    two cleanly: PermissionError = locked, anything else = we are through.
    """
    try:
        bt.ticks.tape_info(__file__)
    except PermissionError:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# The name and its JSON
# ---------------------------------------------------------------------------


def test_interval_trades_is_a_name_not_a_duration():
    # A unit variant on the Rust side, so the JSON of the sized intervals is
    # byte-for-byte what it always was.
    assert Interval.trades() == "Trades"
    assert Interval.seconds(1) == {"Seconds": 1}
    assert Interval.minutes(1) == {"Minutes": 1}
    assert Interval.days(1) == {"Days": 1}


def test_the_clock_survives_the_config_round_trip():
    cfg = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=86_400_000_000_000,
        bar_interval=Interval.trades(),
        output_resolution=Interval.trades(),
    )
    payload = cfg.to_json_dict()
    assert payload["bar_interval"] == "Trades"
    assert payload["output_resolution"] == "Trades"


def test_a_clock_is_not_a_window():
    """``rolling_sum(Interval.trades())`` is a category error, and says so.

    Under this clock an integer window counts EVENTS and a duration window
    counts TIME; "trades" is neither, it is how often the engine steps.
    """
    with pytest.raises(TypeError, match="not a window"):
        col("buy_volume").rolling_sum(Interval.trades())
    # And both real forms still work.
    assert col("buy_volume").rolling_sum(34) is not None
    assert col("buy_volume").rolling_sum(Interval.seconds(34)) is not None


# ---------------------------------------------------------------------------
# Refusals taken before anything is loaded
# ---------------------------------------------------------------------------


def _bar_store(tmp_path):
    """A store with ordinary bars: enough for refusals that fire before load."""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    n = 200
    ts = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    px = 100.0 + np.arange(n) * 0.01
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": px,
            "high": px + 0.05,
            "low": px - 0.05,
            "close": px,
            "volume": np.full(n, 10.0),
        }
    )
    return bt.import_dataframe(
        df,
        symbol="BTCUSDT",
        symbol_id=1,
        data_root=str(tmp_path / "data"),
        metadata_db=str(tmp_path / "meta.sqlite"),
    )


def _strategy():
    return bt.Strategy.create("plat").signal("pos", col("close") * 0.0 + 1.0).size(col("pos"))


def test_per_event_output_needs_the_clock(tmp_path):
    """``output_resolution="Trades"`` under a bar clock is a category error."""
    store = _bar_store(tmp_path)
    cfg = bt.BacktestConfig(
        universe=[1],
        time_range_start=int(bt.date_to_ns("2026-01-01")),
        time_range_end=int(bt.date_to_ns("2026-01-02")),
        bar_interval=Interval.minutes(1),
        output_resolution=Interval.trades(),
        slippage=Slippage.none(),
    )
    with pytest.raises(Exception) as err:
        bt.run(_strategy(), cfg, store)
    assert "output_resolution=trades" in str(err.value)


# ---------------------------------------------------------------------------
# End to end, on a store that really holds a tape
# ---------------------------------------------------------------------------

_TAPE = os.environ.get("MANIFOLDBT_TEST_TAPE_STORE", "")
_needs_tape = pytest.mark.skipif(
    not _TAPE or _locked(),
    reason="set MANIFOLDBT_TEST_TAPE_STORE=<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD> "
    "to run the trade clock end to end (a tape enters a store only through "
    "bt.ingest_trades, which needs the network), and hold the tick-layer key",
)


def _tape_store():
    root, meta, sid, day = _TAPE.split("|")
    store = bt.DataStore(data_root=root, metadata_db=meta, arrow_dir=os.path.join(root, "mega"))
    start = int(bt.date_to_ns(day))
    return store, int(sid), start, start + 86_400_000_000_000


def _config(sid, start, end, **kw):
    cfg = bt.BacktestConfig(
        universe=[sid],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.trades(),
        initial_capital=10_000.0,
        slippage=Slippage.none(),
        **kw,
    )
    cfg.execution.signal_delay = 1
    return cfg


def _flux(k_seconds=34, thr=0.9):
    """The flow strategy of the one-second study, with a window in TIME."""
    from manifoldbt.expr import lit, when

    b = col("buy_volume").rolling_sum(Interval.seconds(k_seconds))
    v = col("sell_volume").rolling_sum(Interval.seconds(k_seconds))
    imb = (b - v) / (b + v + lit(1e-12))
    armed = ((imb >= lit(thr)) | (imb <= lit(-thr))).value_when(
        when(imb >= lit(thr), lit(1.0), lit(-1.0))
    )
    return bt.Strategy.create("flux").signal("pos", armed).size(col("pos"))


@_needs_tape
def test_the_result_has_the_shape_of_a_run_over_events():
    store, sid, start, end = _tape_store()
    # Forced daily output from the first day, the way the CI without a licence
    # will see it: a test that asserts on intraday rows goes red there.
    cfg = _config(sid, start, end, output_resolution=Interval.days(1))
    res = bt.run(_flux(), cfg, store)

    pos = res.positions_df()
    assert {"timestamp", "position", "close", "equity"} <= set(pos.columns)
    assert len(pos) <= 2, "a single UTC day resamples to one row, plus the first"

    # The trade counters count FILLS, as they do on every clock.
    stats = res.metrics["trade_stats"]
    assert stats["total_trades"] >= 0
    assert res.metrics["sharpe"] == res.metrics["sharpe"]  # not NaN


@_needs_tape
@pytest.mark.skipif(not IS_PRO, reason="per-event output is not a Community shape")
def test_per_event_output_hands_back_one_row_per_trade():
    store, sid, start, end = _tape_store()
    cfg = _config(sid, start, end, output_resolution=Interval.trades())
    res = bt.run(_flux(), cfg, store)
    pos = res.positions_df()
    # Millions of prints in a liquid day: the only thing worth asserting is
    # that nothing was bucketed away, which strictly increasing sub-second
    # timestamps show.
    assert len(pos) > 100_000
    ts = pos["timestamp"].astype("int64").to_numpy()
    assert (ts[1:] >= ts[:-1]).all()
    assert (ts[1:] - ts[:-1]).min() < 1_000_000_000, "sub-second steps survived"


@_needs_tape
def test_the_flow_columns_are_readable_from_the_dsl():
    store, sid, start, end = _tape_store()
    cfg = _config(sid, start, end, output_resolution=Interval.days(1))
    cfg.execution.signal_delay = 0
    # Units, so the position IS the column rather than a fraction of equity,
    # and no notional cap, which would otherwise scale one whole coin down to
    # what the capital can carry.
    cfg.execution.position_sizing_mode = "Units"
    cfg.execution.max_position_pct = 1e9
    strat = bt.Strategy.create("side").signal("pos", col("side")).size(col("pos"))
    res = bt.run(strat, cfg, store)
    # +1 aggressive buy, -1 aggressive sell: never anything else, never zero.
    assert abs(res.positions_df()["position"].iloc[-1]) == 1.0


@_needs_tape
def test_the_lite_paths_refuse_the_clock_by_name():
    store, sid, start, end = _tape_store()
    cfg = _config(sid, start, end)
    with pytest.raises(Exception) as err:
        bt.run_batch_lite([_flux()], cfg, store)
    assert "trade clock" in str(err.value)


@_needs_tape
def test_two_runs_on_the_same_tape_agree_to_the_bit():
    store, sid, start, end = _tape_store()
    cfg = _config(sid, start, end, output_resolution=Interval.days(1))
    a = bt.run(_flux(), cfg, store)
    b = bt.run(_flux(), cfg, store)
    assert a.metrics["sharpe"] == b.metrics["sharpe"]
    assert a.metrics["total_return"] == b.metrics["total_return"]
    assert a.trades_df().equals(b.trades_df())


def test_per_event_output_is_never_silently_downgraded(monkeypatch):
    """The Community cap must not turn a refusal into a silent success.

    ``output_resolution`` is capped to daily for tiers that stop there, and the
    cap rewrites the field. Per-event output is not a resolution to round down
    but a shape the engine answers by name, so the cap has to leave it alone --
    on every tier, which is why the tier is forced here rather than read.
    """
    monkeypatch.setattr(bt, "_is_pro", lambda: False)
    cfg = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=86_400_000_000_000,
        bar_interval=Interval.trades(),
        output_resolution=Interval.trades(),
    )
    assert bt._cap_output_resolution(cfg).output_resolution == "Trades"

    # And a real sub-daily duration is still capped, as it always was.
    cfg.output_resolution = Interval.minutes(1)
    assert bt._cap_output_resolution(cfg).output_resolution is None
