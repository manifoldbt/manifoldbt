"""The tape charts: what a maker's inventory, quotes and fills cost it.

Same split as `test_fill_marks.py`, and the same reason. Everything here runs
on a SYNTHETIC result -- a few hours of one-minute bars imported from a
DataFrame, with a resting entry order so the run keeps its quote counters and
its journal -- because a tape enters a store only through `bt.ingest_trades`,
which fetches a venue's archive over the network. A tape changes what is IN
the columns, not the code that draws them.

Two charts read `fill_marks_df()`, which a run on bars is refused. They get a
frame built from THIS run's own fills, wrapped around the real result so that
the equity, the trade log and the journal they also read are the real ones:
the assertion that the four terms of an attribution add up to the run's P&L is
only worth making against a real P&L.

Every chart is rendered to a PNG and its size checked, which is the only thing
a test can say about a picture, plus the shape of what it built.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from manifoldbt.result import QuoteResult  # noqa: E402

plot = pytest.importorskip("manifoldbt.plot")

SYMBOL_ID = 1
CHARTS = ("inventory", "pnl_attribution", "hourly", "order_life",
          "marks_by_channel", "quotes")


# ---------------------------------------------------------------------------
# One synthetic run, shared
# ---------------------------------------------------------------------------


def _store(tmp_path):
    """A day of one-minute bars that oscillate enough to serve a quote."""
    n = 1_440
    t = np.arange(n)
    close = 100.0 + 3.0 * np.sin(t / 7.0) + 0.004 * t
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
        fees=bt.FeeConfig(maker_fee_bps=0.0, taker_fee_bps=4.0,
                          default_fill_type="Taker"),
        slippage=Slippage.none(),
        execution=bt.ExecutionConfig(
            signal_delay=1,
            allow_short=False,
            position_sizing_mode="Units",
            **execution,
        ),
    )


def _maker():
    return (
        bt.Strategy.create("maker")
        .signal("quote", col("close") - lit(0.25))
        .size(lit(1.0))
        .limit_entry(signal="quote", time_in_force={"GTB": 3})
        .take_profit(pct=0.5)
        .stop_loss(pct=1.0)
    )


class _WithMarks:
    """A real result, plus the marks a run on bars is not allowed to take.

    Everything but ``fill_marks_df`` is delegated, so the equity curve, the
    trade log and the journal a chart reads beside the marks are this run's
    own. Only the four mark columns are invented, and they are invented from
    this run's own fills.
    """

    def __init__(self, result, marks):
        self._result = result
        self._marks = marks

    def __getattr__(self, name):
        return getattr(self._result, name)

    def fill_marks_df(self, backend="auto"):
        return self._marks


def _marks_for(result):
    """Marks in the shape `fill_marks_df()` returns, from a run's own fills.

    The reference at the fill sits a tick on the good side of the price, and
    each horizon walks a little further against the position: an attribution
    of these is a small captured half-spread against a growing adverse drift,
    which is the shape the real thing has.
    """
    trades = result.trades_df()
    side = np.where(trades["side"].to_numpy() == 1, 1, -1).astype(np.int8)
    price = trades["fill_price"].to_numpy(dtype=np.float64)
    drift = -np.linspace(0.002, 0.02, len(trades))
    frame = pd.DataFrame({
        "timestamp": trades["execution_timestamp"],
        "anchor": trades["execution_timestamp"],
        "symbol_id": trades["symbol_id"],
        "side": side,
        "price": price,
        "qty": trades["quantity"].to_numpy(dtype=np.float64),
        "position_after": np.zeros(len(trades)),
        "ref_at_fill": price + side * 0.01,
        "ref_source": "book",
    })
    for i, horizon in enumerate(("100ms", "1s", "10s"), start=1):
        frame[f"mark_{horizon}"] = frame["ref_at_fill"] + side * drift * i
        frame[f"source_{horizon}"] = "book"
    frame["book_half_spread_bps"] = 0.5
    return frame


class _NoJournalRaw:
    """What the engine answers for a run of a strategy that quotes with `.quote`.

    The counters are there; ``resting_orders_detail()`` is ``None``, because the
    journal of resting orders is only kept by the limit-order strategies of the
    main engine. This is the one answer these tests need from the engine, so it
    is the only one imitated.
    """

    order_activity = {"orders_posted": 40, "requotes": 6, "expired_unfilled": 25}
    fill_fragility = {
        "maker_fills": 9,
        "queue": {"queue_decided_fills": 5, "traverse_fills": 4,
                  "fills_from_book_cross": 1, "partial_fills": 0},
    }

    def resting_orders_detail(self):
        return None


class _QuoteRun(QuoteResult):
    """A ``QuoteResult`` without a real run behind it.

    A real run of a ``.quote`` strategy needs the order-level simulation and a
    book, which a synthetic store of bars cannot give and which this suite does
    not build. The class IS ``QuoteResult``, so what is under test is real: the
    ``isinstance`` the charts ask, and ``resting_orders_df()`` reading the
    engine's ``None``. Only the raw result and the marks are stood in for.
    """

    def __init__(self, marks=None):
        super().__init__(_NoJournalRaw(), None)
        self._marks = marks

    def fill_marks_df(self, backend="auto"):
        return self._marks


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The maker, its marks, and a second day to pool it with."""
    tmp = tmp_path_factory.mktemp("tape_plots")
    store, last = _store(tmp)
    result = bt.run(_maker(), _config(last, tick_size=0.05), store)
    assert result.trade_count > 20, "this run booked too few fills to draw"
    return _WithMarks(result, _marks_for(result))


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    """A run of market orders: no quotes, no journal, no marks."""
    tmp = tmp_path_factory.mktemp("tape_plots_bare")
    store, last = _store(tmp)
    plain = bt.Strategy.create("taker").signal("p", lit(1.0)).size(col("p"))
    return bt.run(plain, _config(last), store)


def _png(fig_call, path, **kwargs):
    """Render a chart to a PNG and return its pixel size."""
    pytest.importorskip("kaleido")
    fig_call(save=path, **kwargs)
    png = path.read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return (int.from_bytes(png[16:20], "big"),
            int.from_bytes(png[20:24], "big"))


# ---------------------------------------------------------------------------
# The names
# ---------------------------------------------------------------------------


def test_every_chart_is_exported():
    for name in CHARTS:
        assert name in plot.__all__, name
        assert callable(getattr(plot, name)), name


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_inventory_draws_the_position_the_price_and_both_sides(run):
    fig = plot.inventory(run, target=1.0, show=False)
    names = [t.name for t in fig.data]
    assert "Close" in names and "Position" in names
    assert {"Buy", "Sell"} <= set(names)
    # Two panels, and the position is the one on the second.
    position = next(t for t in fig.data if t.name == "Position")
    assert position.yaxis == "y2"
    assert position.line.shape == "hv", "a position does not interpolate"
    # The target, plus the zero line. One dotted line and not two: this run
    # never went short, and a line at -1.0 on a chart whose position never
    # leaves the positive half is a line about nothing.
    assert len(fig.layout.shapes) == 2


def test_inventory_renders_to_a_png_of_the_size_it_was_asked_for(run, tmp_path):
    out = tmp_path / "inventory.png"
    size = _png(lambda **kw: plot.inventory(run, figsize=(14, 7), **kw), out)
    assert out.stat().st_size > 5_000
    assert size == (1750, 875), f"got {size}"


def test_inventory_refuses_a_result_with_no_positions():
    class _Empty:
        positions = None

        @property
        def manifest(self):
            return {}

    with pytest.raises(Exception):
        plot.inventory(_Empty(), show=False)


# ---------------------------------------------------------------------------
# pnl_attribution
# ---------------------------------------------------------------------------


def test_the_four_terms_add_up_to_the_run_s_own_pnl(run):
    from manifoldbt._convert import equity_with_dates
    from manifoldbt.plot.pnl_attribution import attribution

    parts = attribution(run, "1s")
    _dates, equity = equity_with_dates(run)
    assert parts["total"] == pytest.approx(float(equity[-1] - equity[0]))
    assert (parts["half_spread"] + parts["adverse"] + parts["fees"]
            + parts["inventory"]) == pytest.approx(parts["total"])
    # And the three measured ones are what the columns say, not a residual.
    df = run.fill_marks_df()
    side = df["side"].to_numpy(dtype=float)
    qty = df["qty"].to_numpy(dtype=float)
    expected = float(np.nansum(
        side * (df["ref_at_fill"].to_numpy() - df["price"].to_numpy()) * qty))
    assert parts["half_spread"] == pytest.approx(expected)
    assert parts["fees"] == pytest.approx(-run.trades_df()["fees"].sum())
    # The marks were built to drift against the position, so the term is a cost.
    assert parts["adverse"] < 0


def test_the_waterfall_closes_on_the_total_it_drew(run):
    fig = plot.pnl_attribution(run, horizon="1s", show=False)
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert list(bar.measure) == ["relative"] * 4 + ["total"]
    assert sum(bar.y[:4]) == pytest.approx(bar.y[4])
    assert "+1s" in bar.x[1]


def test_the_horizon_has_to_be_one_of_the_three(run):
    with pytest.raises(ValueError, match="horizon must be one of"):
        plot.pnl_attribution(run, horizon="1m", show=False)


def test_pnl_attribution_refuses_a_run_with_no_marks(bare):
    with pytest.raises(ValueError, match="no fill marks"):
        plot.pnl_attribution(bare, show=False)


def test_pnl_attribution_renders_to_a_png(run, tmp_path):
    out = tmp_path / "attribution.png"
    size = _png(lambda **kw: plot.pnl_attribution(run, figsize=(11, 5.5), **kw), out)
    assert out.stat().st_size > 5_000
    assert size == (1375, 687), f"got {size}"


# ---------------------------------------------------------------------------
# hourly
# ---------------------------------------------------------------------------


def test_hourly_puts_four_panels_on_one_hour_axis(run):
    fig = plot.hourly(run, show=False)
    # Buy, sell, exposure, markout, fees.
    assert len(fig.data) == 5
    for trace in fig.data:
        assert len(trace.x) == 24
    assert fig.data[0].x[0] == "00" and fig.data[0].x[-1] == "23"


def test_hourly_pools_a_list_of_runs_and_says_how_many_days(run):
    one = plot.hourly(run, show=False)
    two = plot.hourly([run, run], show=False)
    fills_one = np.nansum(one.data[0].y) + np.nansum(one.data[1].y)
    fills_two = np.nansum(two.data[0].y) + np.nansum(two.data[1].y)
    assert fills_two == pytest.approx(2 * fills_one)
    # The same day twice is still one day: the note counts days, not runs.
    note = [a.text for a in two.layout.annotations if "fills over" in (a.text or "")]
    assert note and "one UTC day" in note[0]


def test_hourly_refuses_a_run_that_booked_no_fill():
    class _NoFills:
        def __getattr__(self, name):
            raise AttributeError(name)

    with pytest.raises(Exception):
        plot.hourly(_NoFills(), show=False)


def test_hourly_renders_to_a_png(run, tmp_path):
    out = tmp_path / "hourly.png"
    size = _png(lambda **kw: plot.hourly(run, figsize=(13, 10), **kw), out)
    assert out.stat().st_size > 5_000
    assert size == (1625, 1250), f"got {size}"


# ---------------------------------------------------------------------------
# order_life
# ---------------------------------------------------------------------------


def test_order_life_draws_the_counters_and_the_two_distributions(run):
    fig = plot.order_life(run, show=False)
    kinds = [t.type for t in fig.data]
    assert kinds.count("bar") == 2, "one bar trace per panel"
    # The journal adds the lives and the queue at post.
    assert kinds.count("histogram") >= 2
    titles = [a.text for a in fig.layout.annotations]
    assert any("Per quote" in t for t in titles)
    assert any("How long a quote stood" in t for t in titles)


def test_order_life_refuses_a_run_that_posted_no_quote(bare):
    with pytest.raises(ValueError, match="posted no resting order"):
        plot.order_life(bare, show=False)


def test_order_life_says_what_a_quote_run_does_not_carry(run):
    """A `.quote` run draws its counters, and the note names where the rest is."""
    fig = plot.order_life(_QuoteRun(), show=False)
    kinds = [t.type for t in fig.data]
    assert kinds.count("bar") == 2 and "histogram" not in kinds
    notes = [a.text for a in fig.layout.annotations if "orders_df()" in (a.text or "")]
    assert len(notes) == 1
    assert "fills_df()" in notes[0] and ".quote" in notes[0]
    # A run that kept the journal draws the panels and has nothing to excuse.
    full = plot.order_life(run, show=False)
    assert not [a for a in full.layout.annotations if "orders_df()" in (a.text or "")]


def test_order_life_renders_to_a_png(run, tmp_path):
    out = tmp_path / "order_life.png"
    size = _png(lambda **kw: plot.order_life(run, figsize=(13, 10), **kw), out)
    assert out.stat().st_size > 5_000
    assert size == (1625, 1250), f"got {size}"


# ---------------------------------------------------------------------------
# marks_by_channel
# ---------------------------------------------------------------------------


def test_marks_by_channel_refuses_a_run_with_no_journal(bare):
    with pytest.raises(ValueError, match="no fill marks"):
        plot.marks_by_channel(bare, show=False)


def test_marks_by_channel_sends_a_quote_run_to_its_own_tables(run):
    """Not "posted no resting order": that run posted quotes, and says where."""
    with pytest.raises(ValueError, match=r"quotes with \.quote") as err:
        plot.marks_by_channel(_QuoteRun(run.fill_marks_df()), show=False)
    message = str(err.value)
    assert "fills_df()" in message and "orders_df()" in message
    assert "posted no resting order" not in message


def test_marks_by_channel_refuses_a_run_whose_quotes_carry_no_channel(run):
    """No queue model, so nothing decided a channel and none is invented."""
    with pytest.raises(ValueError, match="no channel"):
        plot.marks_by_channel(run, show=False)


def test_the_join_keeps_one_row_per_marked_fill(run):
    from manifoldbt.plot.marks_by_channel import fills_by_channel

    joined = fills_by_channel(run)
    assert len(joined) <= len(run.fill_marks_df())
    assert {"fill_channel", "queue_ahead_at_post", "book_known_at_post"} \
        <= set(joined.columns)
    # Pooling a list of runs concatenates rather than merging them.
    assert len(fills_by_channel([run, run])) == 2 * len(joined)


# ---------------------------------------------------------------------------
# quotes
# ---------------------------------------------------------------------------


def test_quotes_draws_one_segment_per_order_over_the_window(run):
    journal = run.resting_orders_df()
    start = journal["posted_at"].min()
    end = journal["posted_at"].max()
    fig = plot.quotes(run, start=start, end=end, show=False)
    lines = [t for t in fig.data if t.mode == "lines" and t.name != "Close"]
    assert lines, "no quote was drawn"
    # Three points per segment: the two ends and the gap that breaks the line.
    for trace in lines:
        assert len(trace.x) % 3 == 0
        assert np.isnat(np.datetime64(trace.x[2]))
    # The x values stay on the date axis; an object array turns them into
    # integers and plotly then draws them off the side of every window.
    assert all("datetime64" in str(np.asarray(t.x).dtype) for t in lines)


def test_quotes_refuses_a_window_where_nothing_was_alive(run):
    with pytest.raises(ValueError, match="no quote was alive"):
        plot.quotes(run, start="2019-01-01", end="2019-01-02", show=False)


def test_quotes_refuses_a_run_that_posted_no_quote(bare):
    with pytest.raises(ValueError, match="posted no resting order"):
        plot.quotes(bare, show=False)


def test_quotes_sends_a_quote_run_to_its_own_tables():
    with pytest.raises(ValueError, match=r"quotes with \.quote") as err:
        plot.quotes(_QuoteRun(), show=False)
    message = str(err.value)
    assert "orders_df()" in message and "fills_df()" in message
    assert "posted no resting order" not in message


def test_quotes_renders_to_a_png(run, tmp_path):
    out = tmp_path / "quotes.png"
    journal = run.resting_orders_df()
    size = _png(
        lambda **kw: plot.quotes(
            run, start=journal["posted_at"].min(),
            end=journal["posted_at"].min() + pd.Timedelta(minutes=30),
            figsize=(14, 6.5), **kw),
        out,
    )
    assert out.stat().st_size > 5_000
    assert size == (1750, 812), f"got {size}"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_tearsheet_grows_a_tape_section_only_when_there_is_one(run, bare):
    with_tape = plot.tearsheet(run, show=False)
    assert "ON THE TAPE" in with_tape
    without = plot.tearsheet(bare, show=False)
    assert "ON THE TAPE" not in without
