"""``result.resting_orders_df()``: one row per quote the run posted.

The journal is written wherever a resting order's life ends, so the first
thing a test has to pin is that no door forgets it: the number of rows and
``order_activity["orders_posted"]`` are the same number, counted in two
completely different places. Everything else -- the outcomes, the channel, the
two cancel instants -- hangs off that.

None of this needs a tape. A resting order rests on a bar grid too, and the
journal is written by the same loop; what a tape changes is which of the
outcomes actually happen. The end-to-end half that does need one is in
``test_fill_marks.py``'s store, behind the same environment variable.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

SYMBOL_ID = 1

#: Every outcome the column may carry. A value outside this set is a bug in
#: the engine or a name that changed without the doc changing with it.
OUTCOMES = {"filled", "partial", "expired", "cancelled_requote",
            "stale_filled", "cancelled"}

COLUMNS = [
    "symbol_id", "posted_at", "visible_at", "side", "level",
    "level_before_snap", "size", "queue_ahead_at_post", "book_known_at_post",
    "outcome", "filled_at", "filled_qty", "fill_channel",
    "cancel_decided_at", "cancel_effective_at",
]


def _store(tmp_path):
    """A few hours of one-minute bars that oscillate enough to fill a quote."""
    n = 240
    close = 100.0 + 3.0 * np.sin(np.arange(n) / 7.0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC"),
        "open": close,
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
        "volume": np.full(n, 1_000.0),
    })
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=SYMBOL_ID, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    return store, int(df["timestamp"].iloc[-1].value)


def _config(last_ns, **execution):
    return bt.BacktestConfig(
        universe=[SYMBOL_ID],
        time_range_start=0,
        time_range_end=last_ns + 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000.0,
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        execution=bt.ExecutionConfig(
            signal_delay=1,
            allow_short=False,
            position_sizing_mode="Units",
            **execution,
        ),
    )


def _maker():
    """A quote a quarter below the close, replaced every three bars."""
    return (
        bt.Strategy.create("maker")
        .signal("quote", col("close") - lit(0.25))
        .size(lit(1.0))
        .limit_entry(signal="quote", time_in_force={"GTB": 3})
        .take_profit(pct=0.5)
        .stop_loss(pct=1.0)
    )


def _taker():
    return bt.Strategy.create("taker").signal("p", lit(1.0)).size(col("p"))


@pytest.fixture(scope="module")
def quoted(tmp_path_factory):
    """One run of the maker, reused: the journal is read a dozen ways."""
    tmp = tmp_path_factory.mktemp("resting")
    store, last = _store(tmp)
    return bt.run(_maker(), _config(last, tick_size=0.05), store), store, last


# ---------------------------------------------------------------------------
# The count, which is the whole guarantee
# ---------------------------------------------------------------------------


def test_the_journal_holds_exactly_the_quotes_the_counter_counted(quoted):
    result, _store, _last = quoted
    df = result.resting_orders_df()
    assert df is not None
    posted = result.order_activity["orders_posted"]
    assert posted > 10, "this run posted too few quotes to say anything"
    assert len(df) == posted


def test_a_run_of_market_orders_carries_no_journal_at_all(quoted, same_metrics):
    """And is otherwise the run it always was."""
    _result, store, last = quoted
    plain = bt.run(_taker(), _config(last), store)
    assert plain.resting_orders_df() is None
    assert plain.order_activity["orders_posted"] == 0
    # The field is new, and a run that produces none must be untouched by it.
    twin = bt.run(_taker(), _config(last), store)
    assert same_metrics(plain.metrics, twin.metrics)
    assert plain.trades_df().equals(twin.trades_df())


def test_two_runs_write_the_same_journal(quoted):
    _result, store, last = quoted
    cfg = _config(last, tick_size=0.05)
    a = bt.run(_maker(), cfg, store).resting_orders_df()
    b = bt.run(_maker(), cfg, store).resting_orders_df()
    assert a.equals(b)


# ---------------------------------------------------------------------------
# The columns
# ---------------------------------------------------------------------------


def test_the_frame_has_the_columns_the_docstring_names(quoted):
    result, _store, _last = quoted
    df = result.resting_orders_df()
    assert list(df.columns) == COLUMNS
    for stamp in ("posted_at", "visible_at", "filled_at",
                  "cancel_decided_at", "cancel_effective_at"):
        assert str(df[stamp].dtype) == "datetime64[ns, UTC]", stamp
    assert set(df["side"].unique()) <= {1, -1}
    assert set(df["outcome"].unique()) <= OUTCOMES
    assert set(df["fill_channel"].dropna().unique()) <= {
        "queue", "traverse", "book_cross"}


def test_without_a_latency_a_quote_is_visible_the_instant_it_is_posted(quoted):
    result, _store, _last = quoted
    df = result.resting_orders_df()
    assert (df["visible_at"] == df["posted_at"]).all()


def test_a_quote_that_filled_says_when_and_how_much(quoted):
    result, _store, _last = quoted
    df = result.resting_orders_df()
    served = df[df["outcome"].isin({"filled", "partial", "stale_filled"})]
    assert len(served) > 0, "no quote was served on this run"
    assert served["filled_at"].notna().all()
    assert (served["filled_qty"] > 0).all()
    assert (served["filled_at"] >= served["visible_at"]).all()
    # And one that did not says nothing rather than saying zero.
    idle = df[~df["outcome"].isin({"filled", "partial", "stale_filled"})]
    assert idle["filled_at"].isna().all()
    assert (idle["filled_qty"] == 0).all()


def test_a_filled_quote_lands_on_a_row_of_the_trade_log(quoted):
    """The two are written from the same instant, so the join is exact."""
    result, _store, _last = quoted
    df = result.resting_orders_df()
    trades = result.trades_df()
    served = df.dropna(subset=["filled_at"])
    booked = set(trades["execution_timestamp"])
    assert set(served["filled_at"]) <= booked


def test_the_price_grid_shows_up_as_a_level_that_moved(quoted):
    result, _store, _last = quoted
    df = result.resting_orders_df()
    moved = df["level_before_snap"].notna()
    assert int(moved.sum()) == result.order_activity.get("levels_snapped", 0)
    if moved.any():
        # A bid is snapped DOWN, so the level posted is at or below the one
        # the strategy asked for.
        bids = df[moved & (df["side"] == 1)]
        assert (bids["level"] <= bids["level_before_snap"] + 1e-12).all()


def test_a_run_without_a_queue_model_says_it_read_no_ladder(quoted):
    result, _store, _last = quoted
    df = result.resting_orders_df()
    assert df["queue_ahead_at_post"].isna().all()
    assert df["book_known_at_post"].isna().all()
    assert df["fill_channel"].isna().all()


# ---------------------------------------------------------------------------
# What a cancellation records
# ---------------------------------------------------------------------------


def test_a_cancellation_stamps_both_instants_and_they_match_without_latency(quoted):
    result, _store, _last = quoted
    df = result.resting_orders_df()
    cancelled = df[df["cancel_decided_at"].notna()]
    assert len(cancelled) > 0, "no quote was ever withdrawn on this run"
    assert cancelled["cancel_effective_at"].notna().all()
    assert (cancelled["cancel_effective_at"]
            == cancelled["cancel_decided_at"]).all()
    assert (cancelled["cancel_decided_at"] >= cancelled["posted_at"]).all()


def test_an_expiry_and_a_requote_are_different_words(quoted):
    result, _store, _last = quoted
    df = result.resting_orders_df()
    expired = int((df["outcome"] == "expired").sum())
    assert expired == result.order_activity["expired_unfilled"]
