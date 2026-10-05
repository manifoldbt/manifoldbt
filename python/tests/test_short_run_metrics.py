"""A run shorter than two days reports no metric taken from daily returns.

One day is one daily return: nothing to annualise. The engine used to fall
back to the returns of each bar, annualised by the bars in a year, which gave
a one-day maker at 100 ms a Sharpe of -1 000 and a CAGR of -99.7 % for a day
that lost 1.6 %. Those fields are now NaN, `warnings` says why, and what does
not annualise (total return, drawdown, trades) is still reported.

The rule has to hold on every path (`run`, the lite and full sweeps, the
batches) and every consumer that ranks has to survive the NaN: `best()`,
`worst()`, the walk-forward selection, the reprs.
"""
import math
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, param, when  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from manifoldbt.indicators import close as close_px, sma  # noqa: E402
from manifoldbt.sweep import SweepResult  # noqa: E402

CAPITAL = 100_000.0
DAY_NS = 86_400_000_000_000
START = pd.Timestamp("2021-03-01", tz="UTC")

WITHHELD = (
    "cagr", "volatility", "sharpe", "sortino", "calmar", "skewness", "kurtosis",
    "tail_ratio", "omega_ratio", "ulcer_index", "best_day", "worst_day",
    "avg_daily_return", "pct_positive_days", "tstat_sharpe", "alpha", "beta",
    "tstat_alpha",
)
REPORTED = ("total_return", "max_drawdown", "max_drawdown_duration_days")


def _bars(rows, seed=11):
    """A one-minute random walk from midnight, volatile enough to trade."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 6e-4, rows)))
    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = rng.uniform(0.2, 1.8, rows) * 3e-4 * close
    return pd.DataFrame({
        "timestamp": pd.date_range(START, periods=rows, freq="1min"),
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": np.full(rows, 1_000.0),
    })


def _store(tmp_path, rows):
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    return bt.import_dataframe(
        _bars(rows), symbol="SHORT", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )


def _config(last_bar_ns):
    """A range whose last bar is the one stamped `last_bar_ns` (the next bar
    is a minute later, the end one nanosecond past it)."""
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=int(START.value),
        time_range_end=last_bar_ns + 1,
        bar_interval=Interval.minutes(1),
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=0, execution_price="AtClose", max_position_pct=1.0,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def _fixed():
    sized = when(col("fast") > col("slow"), lit(1.0), lit(0.0))
    return (bt.Strategy.create("fixed")
            .signal("fast", sma(close_px, 5))
            .signal("slow", sma(close_px, 20))
            .size(sized))


def _swept():
    sized = when(col("fast") > col("slow"), lit(1.0), lit(0.0))
    return (bt.Strategy.create("swept")
            .signal("fast", sma(close_px, param("fast")))
            .signal("slow", sma(close_px, 20))
            .size(sized))


def _is_short(metrics, warnings):
    return (all(math.isnan(metrics[k]) for k in WITHHELD)
            and all(math.isfinite(metrics[k]) for k in REPORTED)
            and any("less than two days" in w for w in warnings))


def test_one_day_withholds_the_annualised_metrics_on_every_path(tmp_path):
    # 00:00 to 23:59: one UTC day.
    store = _store(tmp_path, 1_440)
    config = _config(int(START.value) + DAY_NS - 60_000_000_000)

    res = bt.run(_fixed(), config, store)
    m = res.metrics
    assert m["total_return"] != 0.0, "the strategy must trade, or nothing is proved"
    assert _is_short(m, res.warnings), (m, res.warnings)
    assert "trade_stats" in m and m["trade_stats"]["total_trades"] > 0

    lite = bt.run_sweep_lite(_swept(), {"fast": [5, 7]}, config, store)
    full = bt.run_sweep(_swept(), {"fast": [5, 7]}, config, store)
    for r in list(lite) + list(full):
        assert _is_short(r.metrics, r.warnings), (r.metrics, r.warnings)
    # Same run, same reported numbers, to the bit: lite and full agree with run.
    assert lite[0].metrics["total_return"] == m["total_return"]
    assert full[0].metrics["total_return"] == m["total_return"]
    # The lite loop accumulates the drawdown as (peak - equity) / peak: one
    # ULP from run()'s equity / peak - 1, as on every run length.
    assert lite[0].metrics["max_drawdown"] == pytest.approx(m["max_drawdown"], rel=1e-12)

    cols = bt.sweep_columns(lite, ["sharpe", "total_return"])
    assert np.isnan(cols["sharpe"]).all()
    assert np.isfinite(cols["total_return"]).all()

    # A summary that says n/a, not "nan" or "+nan%".
    text = res.summary()
    assert "nan" not in text.lower()
    assert "n/a" in text


def test_a_last_bar_at_the_next_midnight_is_still_one_day(tmp_path):
    # 00:00 to the next 00:00 inclusive: 1 441 bars, the last one alone in
    # "day two". It closes day one; the run is one day.
    store = _store(tmp_path, 1_441)
    config = _config(int(START.value) + DAY_NS)
    res = bt.run(_fixed(), config, store)
    # The data does reach day two: the daily series holds its few seconds...
    assert len(res.daily_returns_series(backend="pandas")) == 2
    # ...and the metrics do not annualise them.
    assert _is_short(res.metrics, res.warnings), res.metrics


def test_one_bar_past_midnight_is_two_days(tmp_path):
    store = _store(tmp_path, 1_442)
    config = _config(int(START.value) + DAY_NS + 60_000_000_000)
    res = bt.run(_fixed(), config, store)
    m = res.metrics
    assert math.isfinite(m["sharpe"]) and math.isfinite(m["cagr"]), m
    assert not any("less than two days" in w for w in res.warnings)
    # And the lite path agrees with run() on the daily statistics, as it
    # always did past one day.
    lite = bt.run_sweep_lite(_swept(), {"fast": [5]}, config, store)[0].metrics
    for k in ("sharpe", "volatility", "sortino", "cagr", "total_return"):
        assert lite[k] == pytest.approx(m[k], rel=1e-9, abs=1e-12), k


def test_best_and_worst_skip_nan_and_say_when_all_are(tmp_path):
    class _Fake:
        def __init__(self, metrics):
            self.metrics = metrics

    nan = float("nan")
    sweep = SweepResult([
        _Fake({"sharpe": nan, "total_return": 0.1}),
        _Fake({"sharpe": -1.0, "total_return": 0.2}),
        _Fake({"sharpe": 2.0, "total_return": -0.1}),
        _Fake({"sharpe": nan, "total_return": 0.0}),
    ])
    # A NaN first in line used to keep the lead: NaN > x and NaN < x are False.
    assert sweep.best("sharpe").metrics["sharpe"] == 2.0
    assert sweep.worst("sharpe").metrics["sharpe"] == -1.0
    assert sweep.best("total_return").metrics["total_return"] == 0.2

    all_nan = SweepResult([_Fake({"sharpe": nan}), _Fake({"sharpe": nan})])
    with pytest.raises(ValueError, match="NaN in every result"):
        all_nan.best("sharpe")
    with pytest.raises(ValueError, match="not found"):
        all_nan.best("nope")


def test_a_short_sweep_ranks_on_what_it_reports(tmp_path):
    store = _store(tmp_path, 1_440)
    config = _config(int(START.value) + DAY_NS - 60_000_000_000)
    sweep = bt.run_sweep(_swept(), {"fast": [3, 5, 7]}, config, store)
    with pytest.raises(ValueError, match="NaN in every result"):
        sweep.best("sharpe")
    best = sweep.best("total_return")
    assert best.metrics["total_return"] == max(r.metrics["total_return"] for r in sweep)
    df = sweep.to_df(backend="pandas")
    assert df["sharpe"].isna().all()
    # The reprs read NaN as no value, not as a span.
    assert "nan" not in repr(bt.run_sweep_lite(_swept(), {"fast": [3, 5]}, config, store)).lower()


def _wf_config(metric, train_days, test_days):
    return {
        "geometry": "pardo",
        "train": {"length": Interval.days(train_days)},
        "test": {"length": Interval.days(test_days)},
        "optimize_metric": metric,
        "param_grid": {"fast": [3, 5]},
    }


@pytest.mark.skipif(bt.license_info()[0] != "Pro",
                    reason="walk-forward: skipped on a Community wheel")
def test_walk_forward_on_windows_under_two_days(tmp_path):
    # Four days of minutes; one-day training windows, one-day tests.
    store = _store(tmp_path, 4 * 1_440)
    config = _config(int(START.value) + 4 * DAY_NS - 60_000_000_000)
    # Every combination NaN on the training window: refused by name, not
    # resolved by taking the first combination.
    with pytest.raises(ValueError, match="NaN for every"):
        bt.run_walk_forward(_swept(), _wf_config("sharpe", 1, 1), config, store)
    # A metric that does not annualise selects as usual; the one-day tests
    # report NaN where they have nothing, and no efficiency.
    r = bt.run_walk_forward(_swept(), _wf_config("total_return", 1, 1), config, store)
    assert r["n_folds"] >= 1
    for f in r["folds"]:
        assert math.isnan(f["oos_metrics"]["sharpe"])
        assert math.isfinite(f["oos_metrics"]["total_return"])
        assert f["wfe"] is None


def test_the_research_plots_never_mark_a_nan_as_the_best():
    pytest.importorskip("plotly")
    from manifoldbt.plot.research import _argmax_ignoring_nan, heatmap_2d

    nan = float("nan")
    # np.argmax returns the FIRST NaN: the old marker landed on it.
    grid = np.array([[nan, 1.0], [3.0, nan]])
    assert int(np.argmax(grid)) == 0
    assert tuple(int(i) for i in _argmax_ignoring_nan(grid)) == (1, 0)
    assert _argmax_ignoring_nan(np.full((2, 2), nan)) is None

    sweep = {"x_param": "a", "y_param": "b", "metric": "sharpe",
             "x_values": [1, 2], "y_values": [10, 20]}
    # Zones without drift rank the grid itself, NaN included.
    fig = heatmap_2d({**sweep, "metric_grid": grid.tolist()},
                     zones=True, drift=0, show=False)
    assert "most robust: 3.000" in fig.layout.title.text
    # Every cell NaN, the case of a sweep of runs shorter than two days:
    # drawn, and no best claimed.
    for zones in (None, True):
        fig = heatmap_2d({**sweep, "metric_grid": [[nan, nan], [nan, nan]]},
                         zones=zones, show=False)
        assert "plateau" not in fig.layout.title.text
        assert "robust" not in fig.layout.title.text
