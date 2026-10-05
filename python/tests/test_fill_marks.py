"""``execution.fill_marks``: what the market did after each fill.

Three halves, and the split is the one `test_fill_resolution.py` already makes.

The half that runs everywhere needs no tape: the JSON shape, the named refusal,
and the proof that a run which does not ask for the marks is the run it always
was. Those are the ones a wrong config hits first.

The chart runs everywhere too, because it reads nothing but the aggregates
dict: it is rendered to a PNG and its size checked, which is the only thing a
test can say about a picture.

The end-to-end half needs a store holding a tape, and a tape enters a store
only through ``bt.ingest_trades``, which fetches a venue's archive over the
network. So it skips by name unless ``MANIFOLDBT_TEST_TAPE_STORE`` points at a
store that already holds one, as ``<root>|<meta.sqlite>|<symbol_id>|<day>`` --
the same variable ``test_trade_clock.py`` reads. The per-fill semantics
themselves are pinned by the engine's own tests, unit and end to end.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.exceptions import BacktesterError  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

SYMBOL_ID = 1
HORIZONS = ("100ms", "1s", "10s")


# ---------------------------------------------------------------------------
# The name and its JSON
# ---------------------------------------------------------------------------


def test_a_config_that_says_nothing_serialises_exactly_as_before():
    cfg = bt.ExecutionConfig()
    assert cfg.fill_marks is False
    assert "fill_marks" not in cfg.to_json_dict()
    # And saying False out loud is saying nothing.
    explicit = bt.ExecutionConfig(fill_marks=False)
    assert explicit.to_json_dict() == cfg.to_json_dict()


def test_asking_for_the_marks_shows_up_in_the_json():
    cfg = bt.ExecutionConfig(fill_marks=True, fill_resolution="ticks")
    payload = cfg.to_json_dict()
    assert payload["fill_marks"] is True
    assert payload["fill_model"]["fill_resolution"] == "ticks"


# ---------------------------------------------------------------------------
# The refusal, and the run that does not ask
# ---------------------------------------------------------------------------


def _bar_store(tmp_path):
    n = 12
    close = 100.0 + np.arange(n) % 3
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000.0),
        }
    )
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df,
        symbol="TEST",
        symbol_id=SYMBOL_ID,
        interval="1m",
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
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            allow_short=False,
            position_sizing_mode="Units",
            **execution,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def _strategy():
    return bt.Strategy.create("marques").signal("pos", lit(1.0)).size(col("pos"))


def test_marks_on_bars_are_refused_by_name(tmp_path):
    store, last_ns = _bar_store(tmp_path)
    with pytest.raises(BacktesterError, match="fill marks need a tape"):
        bt.run(_strategy(), _config(last_ns, fill_marks=True), store)


def test_a_run_that_does_not_ask_is_the_run_it_always_was(tmp_path, same_metrics):
    store, last_ns = _bar_store(tmp_path)
    plain = bt.run(_strategy(), _config(last_ns), store)
    explicit = bt.run(_strategy(), _config(last_ns, fill_marks=False), store)

    assert plain.fill_marks is None
    assert explicit.fill_marks is None
    assert plain.fill_marks_df() is None
    # The same run: the field is absent from the config, so it is absent from
    # the manifest too, and the config hash the manifest carries is unchanged.
    # (run_id and timestamp differ between any two runs and are dropped.)
    assert plain.trades_df().equals(explicit.trades_df())
    assert same_metrics(plain.metrics, explicit.metrics)
    drop = {"run_id", "timestamp"}
    assert {k: v for k, v in plain.manifest.items() if k not in drop} == {
        k: v for k, v in explicit.manifest.items() if k not in drop
    }


def test_the_accessors_exist_and_the_chart_is_exported():
    assert hasattr(bt.ExecutionConfig(), "fill_marks")
    plot = pytest.importorskip("manifoldbt.plot")
    assert "fill_marks" in plot.__all__
    assert callable(plot.fill_marks)


# ---------------------------------------------------------------------------
# The chart
# ---------------------------------------------------------------------------


class _StubResult:
    """Just enough of a result for the chart: it reads the aggregates only."""

    def __init__(self, marks):
        self.fill_marks = marks


def _aggregates():
    return {
        "anchor": "bar_close",
        "horizons": [
            {"horizon": "100ms", "horizon_ns": 100_000_000, "mean_bps": -0.61,
             "median_bps": -0.02, "stderr_bps": 0.04, "marked_fills": 1586},
            {"horizon": "1s", "horizon_ns": 1_000_000_000, "mean_bps": -1.23,
             "median_bps": -0.11, "stderr_bps": 0.09, "marked_fills": 1586},
            {"horizon": "10s", "horizon_ns": 10_000_000_000, "mean_bps": -1.41,
             "median_bps": -0.18, "stderr_bps": 0.22, "marked_fills": 1584},
        ],
        "half_spread_captured_bps": 0.0062,
        "half_spread_fills": 1586,
        "book_half_spread_bps": 0.0064,
        "fills_total": 1586,
        "bootstrap_draws": 1000,
        "bootstrap_seed": 20260907,
    }


def test_the_chart_refuses_a_run_that_carries_no_marks():
    plot = pytest.importorskip("manifoldbt.plot")
    with pytest.raises(ValueError, match="no fill marks"):
        plot.fill_marks(_StubResult(None))


def test_the_chart_draws_one_bar_per_horizon_and_two_reference_lines():
    plot = pytest.importorskip("manifoldbt.plot")
    fig = plot.fill_marks(_StubResult(_aggregates()), show=False)
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert list(bar.x) == list(HORIZONS)
    assert list(bar.y) == [-0.61, -1.23, -1.41]
    assert list(bar.error_y.array) == [0.04, 0.09, 0.22]
    # Zero, the captured half-spread, and the book's own half-spread.
    assert len(fig.layout.shapes) == 3


def test_the_chart_renders_to_a_png_of_the_size_it_was_asked_for(tmp_path):
    plot = pytest.importorskip("manifoldbt.plot")
    pytest.importorskip("kaleido")
    out = tmp_path / "fill_marks.png"
    plot.fill_marks(_StubResult(_aggregates()), figsize=(10, 5), save=out)
    assert out.exists() and out.stat().st_size > 5_000

    png = out.read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # The IHDR chunk carries the pixel size: figsize is in inches at 80 px per
    # inch, and finalize() exports at the default 150 dpi, i.e. scale 150/96.
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    assert (width, height) == (1250, 625), f"got {width}x{height}"


# ---------------------------------------------------------------------------
# End to end, on a store that holds a tape
# ---------------------------------------------------------------------------

_TAPE = os.environ.get("MANIFOLDBT_TEST_TAPE_STORE", "")
_needs_tape = pytest.mark.skipif(
    not _TAPE,
    reason="set MANIFOLDBT_TEST_TAPE_STORE=<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD> "
    "to mark real fills (a tape enters a store only through bt.ingest_trades, "
    "which needs the network)",
)


def _tape_store():
    root, meta, sid, day = _TAPE.split("|")
    store = bt.DataStore(
        data_root=root, metadata_db=meta, arrow_dir=os.path.join(root, "mega")
    )
    start = int(bt.date_to_ns(day))
    return store, int(sid), start, start + 86_400_000_000_000


def _tape_config(sid, start, end):
    cfg = bt.BacktestConfig(
        universe=[sid],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.seconds(1),
        initial_capital=100_000.0,
        # Forced daily output from the first day, the way a run without a
        # licence sees it: a test asserting on intraday rows goes red there.
        output_resolution=Interval.days(1),
        slippage=Slippage.none(),
        fees=bt.FeeConfig.zero(),
        warmup_bars=60,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            allow_short=False,
            position_sizing_mode="Units",
            fill_resolution="ticks",
            fill_marks=True,
        ),
    )
    return cfg


def _crosser():
    """Long above a slow mean, flat below: enough fills to mark."""
    from manifoldbt.expr import when

    fast = col("close").rolling_mean(10)
    slow = col("close").rolling_mean(60)
    return (
        bt.Strategy.create("croisement")
        .signal("pos", when(fast > slow, lit(1.0), lit(0.0)))
        .size(col("pos"))
    )


@_needs_tape
def test_a_real_run_marks_its_fills_and_the_frame_has_the_shape():
    store, sid, start, end = _tape_store()
    res = bt.run(_crosser(), _tape_config(sid, start, end), store)

    marks = res.fill_marks
    assert marks is not None
    assert marks["anchor"] == "bar_close"
    assert [h["horizon"] for h in marks["horizons"]] == list(HORIZONS)
    assert marks["bootstrap_draws"] == 1000
    assert marks["fills_total"] == res.trade_count
    # Not a vacuous pass: a day with no fill would satisfy every shape below.
    assert marks["fills_total"] > 10, "no fill to mark on this day"

    df = res.fill_marks_df()
    assert len(df) == marks["fills_total"]
    expected = {
        "timestamp", "anchor", "symbol_id", "side", "price", "qty",
        "position_after", "ref_at_fill", "ref_source", "book_half_spread_bps",
    } | {f"mark_{h}" for h in HORIZONS} | {f"source_{h}" for h in HORIZONS}
    assert expected <= set(df.columns)
    assert set(df["side"].unique()) <= {1, -1}
    # The anchor is never before the fill, and never at it on a bar clock.
    assert (df["anchor"] > df["timestamp"]).all()
    # Every source is one of the two series, or nothing at all.
    for h in HORIZONS:
        assert set(df[f"source_{h}"].dropna().unique()) <= {"book", "tape"}


@_needs_tape
def test_two_runs_mark_the_same_fills_the_same_way():
    store, sid, start, end = _tape_store()
    cfg = _tape_config(sid, start, end)
    a = bt.run(_crosser(), cfg, store).fill_marks_df()
    b = bt.run(_crosser(), cfg, store).fill_marks_df()
    assert a.equals(b)
