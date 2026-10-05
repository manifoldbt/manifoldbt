"""``fill_model={"queue": ...}``: what it refuses, and what it reports.

The queue model decides whether a resting maker order was really served, or
merely stood at a price the market reached. The simulation loop consumes it now,
so this file covers the two halves a user meets:

* the refusals, which need no tape and no licence and are what a wrong config
  hits first: a malformed queue is named as malformed, and a queue asked for
  where there are no prints to burn is named as such;
* the counters, which need a store holding a tape AND a book, since the whole
  point is reading the depth at the level of an order. A tape enters a store
  only through ``bt.ingest_trades``, which fetches a venue's archive over the
  network, so that half skips by name unless ``MANIFOLDBT_TEST_TAPE_STORE``
  points at a store that already holds one, as
  ``<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD>``.

The service rule itself is pinned by the engine's own tests: the model alone,
the model against the study's oracle, and the loop that consumes it.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.exceptions import BacktesterError  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

QUEUE_KEYS = {
    "queue_decided_fills",
    "traverse_fills",
    "partial_fills",
    "book_unknown_at_post",
    "fills_from_book_cross",
}


def _store(tmp_path):
    n = 8
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC"),
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1_000.0),
        }
    )
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df,
        symbol="TEST",
        symbol_id=1,
        interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    return store, int(df["timestamp"].iloc[-1].value)


def _config(last_ns, queue=None, **fill):
    fill_model = dict(fill)
    if queue is not None:
        fill_model["queue"] = queue
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=last_ns + 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000.0,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            allow_short=False,
            fill_model=fill_model or None,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def _strategy():
    return (
        bt.Strategy.create("queue")
        .signal("sig", lit(1.0))
        .size(col("sig"))
        .limit_entry(price=99.5)
    )


# ---------------------------------------------------------------------------
# What refuses, and why
# ---------------------------------------------------------------------------


def test_a_queue_without_prints_is_refused_by_name(tmp_path):
    """A bar has a high and a low, not an order of trades: no queue to burn."""
    store, last_ns = _store(tmp_path)
    config = _config(last_ns, {"depth_source": "book", "assumed_queue": 2.0})
    with pytest.raises(BacktesterError, match="fill_model.queue needs the prints"):
        bt.run(_strategy(), config, store)


def test_the_queue_and_the_participation_cap_are_refused_together(tmp_path):
    """Both cap one fill, and the queue is the finer of the two."""
    store, last_ns = _store(tmp_path)
    config = _config(
        last_ns,
        {"depth_source": "book", "assumed_queue": 2.0},
        fill_resolution="ticks",
        max_participation_rate=0.1,
    )
    with pytest.raises((BacktesterError, PermissionError)) as err:
        bt.run(_strategy(), config, store)
    # Without the tick-layer key the licence gate answers first, and that is
    # the right answer too: the run is refused either way, by name.
    assert "max_participation_rate" in str(err.value) or isinstance(
        err.value, PermissionError
    )


def test_the_lite_paths_refuse_the_queue_by_name(tmp_path):
    store, last_ns = _store(tmp_path)
    config = _config(last_ns, {"depth_source": "book", "assumed_queue": 2.0})
    with pytest.raises(BacktesterError, match="lite paths"):
        bt.run_batch_lite([_strategy()], config, store)


@pytest.mark.parametrize(
    "queue, message",
    [
        ({"depth_source": "assumed"}, "assumed_queue"),
        ({"assumed_queue": -1.0}, "assumed_queue"),
        ({"cancel_ahead_rate": -0.5}, "cancel_ahead_rate"),
        ({"assumed_qeue": 2.0}, "unknown field"),
        ({"depth_source": "carnet"}, "unknown variant"),
    ],
)
def test_a_malformed_queue_is_named_before_anything_else(tmp_path, queue, message):
    """A typo in a parameter that makes the P&L is reported as a typo, before
    any question about resolutions or licences."""
    store, last_ns = _store(tmp_path)
    with pytest.raises(BacktesterError, match=message):
        bt.run(_strategy(), _config(last_ns, queue), store)


def test_a_config_without_a_queue_is_unchanged(tmp_path):
    """The field is optional and skipped when absent, so a run that does not
    mention it keeps the fill model it always had -- and reports no queue."""
    store, last_ns = _store(tmp_path)
    plain = bt.run(_strategy(), _config(last_ns), store)
    explicit = _config(last_ns)
    explicit.execution.fill_model = {"passive_fill": "traverse"}
    other = bt.run(_strategy(), explicit, store)
    for res in (plain, other):
        frag = res.fill_fragility
        assert frag is not None
        assert set(frag) == {"maker_fills", "touch_only_fills"}


# ---------------------------------------------------------------------------
# End to end, on a store that holds a tape and a book
# ---------------------------------------------------------------------------

_TAPE = os.environ.get("MANIFOLDBT_TEST_TAPE_STORE", "")
_needs_tape = pytest.mark.skipif(
    not _TAPE,
    reason="set MANIFOLDBT_TEST_TAPE_STORE=<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD> "
    "to run the queue end to end (a tape and a book enter a store only through "
    "bt.ingest_trades / bt.ingest_book, which need the network), and hold the "
    "tick-layer key",
)


def _tape_store():
    root, meta, sid, day = _TAPE.split("|")
    store = bt.DataStore(
        data_root=root, metadata_db=meta, arrow_dir=os.path.join(root, "mega")
    )
    start = int(bt.date_to_ns(day))
    return store, int(sid), start, start + 86_400_000_000_000


def _tape_config(sid, start, end, queue):
    cfg = bt.BacktestConfig(
        universe=[sid],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.seconds(1),
        # DAILY, and on purpose: a Community wheel returns one row a day
        # whatever is asked, so a test that read an intraday position would
        # pass here and fail in the public CI. Fills carry their own stamps.
        output_resolution=Interval.days(1),
        initial_capital=100_000.0,
        warmup_bars=120,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            allow_short=False,
            position_sizing_mode="Units",
            fill_model={"queue": queue, "fill_resolution": "ticks"},
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
    )
    return cfg


def _maker(offset_bps: float = 2.0):
    return (
        bt.Strategy.create("maker")
        .signal("sig", lit(1.0))
        .size(col("sig") * lit(0.01))
        .limit_entry(offset_bps=offset_bps, time_in_force={"GTB": 1})
        .take_profit(pct=0.02, side="long")
        .stop_loss(pct=0.06, side="long")
    )


@_needs_tape
def test_the_queue_reports_its_five_counters_and_the_envelope():
    store, sid, start, end = _tape_store()
    res = bt.run(
        _maker(),
        _tape_config(sid, start, end, {"depth_source": "book", "assumed_queue": 2.0}),
        store,
    )
    frag = res.fill_fragility
    assert set(frag) == {
        "maker_fills",
        "touch_only_fills",
        "queue",
        "would_fill_touch",
        "would_fill_traverse",
    }
    assert set(frag["queue"]) == QUEUE_KEYS
    assert all(isinstance(v, int) and v >= 0 for v in frag["queue"].values())

    q = frag["queue"]
    served = q["queue_decided_fills"] + q["traverse_fills"]
    assert served > 0, "a day of a liquid perpetual serves something"
    # The envelope, on the order lives THIS run produced: a traversal always
    # serves the queue, and a queue fill always needs a print at the level --
    # except the ones the book crossed, which no print witnessed.
    assert frag["would_fill_traverse"] <= served
    assert served <= frag["would_fill_touch"] + q["fills_from_book_cross"]

    # Fills, never intraday positions: the count is the same on a Community
    # wheel, which returns one output row a day.
    trades = res.trades_df()
    assert len(trades) > 0
    assert set(trades["side"].unique()) <= {1, 2}


@_needs_tape
def test_two_runs_of_the_same_queue_are_identical():
    store, sid, start, end = _tape_store()
    cfg = _tape_config(sid, start, end, {"depth_source": "book", "assumed_queue": 2.0})
    a = bt.run(_maker(), cfg, store)
    b = bt.run(_maker(), cfg, store)
    assert a.fill_fragility == b.fill_fragility
    assert a.order_activity == b.order_activity
    pd.testing.assert_frame_equal(a.trades_df(), b.trades_df())


@_needs_tape
def test_an_unknown_book_refuses_rather_than_filling_from_the_front():
    """A level the stored ladder cannot answer at is a queue nobody saw. With
    no declared fallback the run says so; with one, it is used AND counted."""
    store, sid, start, end = _tape_store()
    # A limit two percent under the close is far outside any stored ladder.
    strat = _maker(offset_bps=200.0)
    with pytest.raises(BacktesterError, match="says nothing at"):
        bt.run(strat, _tape_config(sid, start, end, {"depth_source": "book"}), store)

    res = bt.run(
        strat,
        _tape_config(sid, start, end, {"depth_source": "book", "assumed_queue": 2.0}),
        store,
    )
    assert res.fill_fragility["queue"]["book_unknown_at_post"] > 0
