"""``execution.latency``: the shape it takes, what it refuses, what it reports.

A latency is the round trip between a decision and the market: the order is
visible ``order`` after it was placed, the cancellation takes effect ``cancel``
after it was decided, and between the two the stale quote can still be served.

Two halves, like every setting in this layer:

* the shape and the refusals, which need no tape and no licence and are what a
  wrong config hits first: a duration is normalised to nanoseconds, an unknown
  key is named, a negative one is named, and a latency asked for where there
  are no prints is named as such;
* the counters and the behaviour, which need a store holding a tape, since the
  whole model is "which print was the first one at or after this instant". A
  tape enters a store only through ``bt.ingest_trades``, which fetches a venue's
  archive over the network, so that half skips by name unless
  ``MANIFOLDBT_TEST_TAPE_STORE`` points at a store that already holds one, as
  ``<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD>``.

The semantics themselves are pinned by the engine's own tests: the two
durations on their own, and the loop that consumes them.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.config import latency as normalise_latency  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.exceptions import BacktesterError  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

LATENCY_KEYS = {"stale_fills", "overlapping_live_orders"}


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


def _config(last_ns, latency=None, **fill):
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
            fill_model=dict(fill) or None,
            latency=latency,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def _strategy():
    return (
        bt.Strategy.create("latence")
        .signal("sig", lit(1.0))
        .size(col("sig"))
        .limit_entry(price=99.5)
    )


# ---------------------------------------------------------------------------
# The shape
# ---------------------------------------------------------------------------


def test_a_duration_becomes_nanoseconds():
    """The engine counts nanoseconds; the user writes an interval."""
    assert normalise_latency(
        {"order": Interval.millis(20), "cancel": Interval.seconds(1)}
    ) == {"order": 20_000_000, "cancel": 1_000_000_000}


def test_each_half_is_optional_and_defaults_to_zero():
    assert normalise_latency({"order": Interval.millis(5)}) == {"order": 5_000_000}
    assert normalise_latency({}) == {}
    assert normalise_latency(None) is None


def test_a_bare_number_is_read_as_nanoseconds():
    assert normalise_latency({"order": 0, "cancel": 250}) == {
        "order": 0,
        "cancel": 250,
    }


@pytest.mark.parametrize(
    "value, message",
    [
        ({"ordre": 1}, "unknown latency key"),
        ({"order": -1}, "zero or more"),
        ({"cancel": -1}, "zero or more"),
        ({"order": "Trades"}, "not a duration"),
        ({"order": "20ms"}, "takes a duration"),
        (20, "takes a dict"),
    ],
)
def test_a_malformed_latency_is_named(value, message):
    """A typo in a duration that makes the P&L is reported as a typo."""
    with pytest.raises((TypeError, ValueError), match=message):
        normalise_latency(value)


def test_the_config_carries_it_only_when_asked():
    """A config that does not mention the round trip serialises exactly as it
    always did, which is what keeps every run before this setting unchanged."""
    assert "latency" not in bt.ExecutionConfig().to_json_dict()
    exec_cfg = bt.ExecutionConfig(latency={"order": Interval.millis(20)})
    assert exec_cfg.to_json_dict()["latency"] == {"order": 20_000_000}


# ---------------------------------------------------------------------------
# What refuses, and why
# ---------------------------------------------------------------------------


def test_a_latency_without_a_tape_is_refused_by_name(tmp_path):
    """A bar has a high and a low, not an order of prints: there is no first
    print at or after an instant to answer with."""
    store, last_ns = _store(tmp_path)
    config = _config(last_ns, {"order": Interval.millis(20)})
    with pytest.raises(BacktesterError, match="execution.latency needs a tape"):
        bt.run(_strategy(), config, store)


def test_a_zero_latency_is_refused_without_a_tape_too(tmp_path):
    """Naming a setting the run cannot honour is told, rather than let through
    because this particular value happened to be harmless."""
    store, last_ns = _store(tmp_path)
    config = _config(last_ns, {"order": 0, "cancel": 0})
    with pytest.raises(BacktesterError, match="execution.latency needs a tape"):
        bt.run(_strategy(), config, store)


def test_the_lite_paths_refuse_the_latency_by_name(tmp_path):
    store, last_ns = _store(tmp_path)
    config = _config(last_ns, {"order": Interval.millis(20)})
    with pytest.raises(BacktesterError, match="lite paths"):
        bt.run_batch_lite([_strategy()], config, store)


def test_a_run_without_a_latency_reports_the_three_counters_it_always_did(tmp_path):
    store, last_ns = _store(tmp_path)
    res = bt.run(_strategy(), _config(last_ns), store)
    activity = res.order_activity
    assert activity is not None
    assert set(activity) == {"orders_posted", "requotes", "expired_unfilled"}


# ---------------------------------------------------------------------------
# End to end, on a store that holds a tape
# ---------------------------------------------------------------------------

_TAPE = os.environ.get("MANIFOLDBT_TEST_TAPE_STORE", "")
_needs_tape = pytest.mark.skipif(
    not _TAPE,
    reason="set MANIFOLDBT_TEST_TAPE_STORE=<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD> "
    "to run the latency end to end (a tape enters a store only through "
    "bt.ingest_trades, which needs the network), and hold the tick-layer key",
)


def _tape_store():
    root, meta, sid, day = _TAPE.split("|")
    store = bt.DataStore(
        data_root=root, metadata_db=meta, arrow_dir=os.path.join(root, "mega")
    )
    start = int(bt.date_to_ns(day))
    return store, int(sid), start, start + 86_400_000_000_000


def _tape_config(sid, start, end, latency):
    return bt.BacktestConfig(
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
            fill_model={"fill_resolution": "ticks"},
            latency=latency,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
    )


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
def test_a_zero_latency_is_the_run_without_one():
    """The criterion the whole setting is judged on, on real prints."""
    store, sid, start, end = _tape_store()
    absent = bt.run(_maker(), _tape_config(sid, start, end, None), store)
    zero = bt.run(
        _maker(), _tape_config(sid, start, end, {"order": 0, "cancel": 0}), store
    )
    pd.testing.assert_frame_equal(absent.trades_df(), zero.trades_df())
    assert absent.fill_fragility == zero.fill_fragility
    assert absent.order_activity == zero.order_activity
    assert absent.tape_resolution == zero.tape_resolution
    assert (
        set(zero.order_activity) == {"orders_posted", "requotes", "expired_unfilled"}
    ), "a latency of zero is no latency, down to the keys it reports"


@_needs_tape
def test_a_latency_reports_its_two_counters():
    store, sid, start, end = _tape_store()
    res = bt.run(
        _maker(),
        _tape_config(
            sid, start, end, {"order": Interval.millis(20), "cancel": Interval.millis(20)}
        ),
        store,
    )
    activity = res.order_activity
    assert LATENCY_KEYS <= set(activity)
    assert all(isinstance(activity[k], int) and activity[k] >= 0 for k in LATENCY_KEYS)
    assert activity["overlapping_live_orders"] >= activity["stale_fills"], (
        "a stale fill needs a stale order, and every stale order posted beside "
        "a fresh one is counted as an overlap"
    )
    trades = res.trades_df()
    assert len(trades) > 0
    assert set(trades["side"].unique()) <= {1, 2}


@_needs_tape
def test_two_runs_of_the_same_latency_are_identical():
    store, sid, start, end = _tape_store()
    cfg = _tape_config(
        sid, start, end, {"order": Interval.millis(100), "cancel": Interval.millis(100)}
    )
    a = bt.run(_maker(), cfg, store)
    b = bt.run(_maker(), cfg, store)
    assert a.fill_fragility == b.fill_fragility
    assert a.order_activity == b.order_activity
    pd.testing.assert_frame_equal(a.trades_df(), b.trades_df())


@_needs_tape
def test_a_slower_order_serves_fewer_quotes():
    """The quote is in the book for less of its life, so less of it is served.

    The CANCEL half is held at zero here, and on purpose: a cancel that lands
    late leaves the quote standing, which serves MORE, and the two effects pull
    in opposite directions. This isolates the one under test.

    Asserted as an ordering rather than a number: the level, the day and the
    venue decide the size of the gap, the mechanism decides its sign.
    """
    store, sid, start, end = _tape_store()

    def entries(order_ns):
        res = bt.run(
            _maker(),
            _tape_config(sid, start, end, {"order": order_ns, "cancel": 0}),
            store,
        )
        trades = res.trades_df()
        return int((trades["side"] == 1).sum())

    assert entries(2_000_000_000) <= entries(20_000_000)


@_needs_tape
def test_a_slower_cancel_leaves_more_quotes_standing():
    """The other half, and the other direction: a quote that cannot be
    withdrawn is exposed for longer, so it is served more often -- by the flow
    the strategy had just decided it no longer wanted to meet."""
    store, sid, start, end = _tape_store()

    def stale(cancel_ns):
        res = bt.run(
            _maker(),
            _tape_config(sid, start, end, {"order": 0, "cancel": cancel_ns}),
            store,
        )
        return res.order_activity["stale_fills"]

    assert stale(2_000_000_000) >= stale(20_000_000)
