"""A ``lead()`` in a strategy is look-ahead by construction, and no re-run
can see it.

``detect_lookahead`` re-runs the strategy over shorter windows and compares
trades. A signal built from ``close.lead(1)`` reads the same T+1 in every
window, so the trades agree and both re-run sub-tests report PASS, next to a
Sharpe in the hundreds. The docstring used to claim the opposite. The static
sub-test walks the expressions instead, and names the signal.

The walk lives in the binary with the re-runs (Pro, like the rest of the
detector), so everything here that reaches it is Pro-only; the mode check
alone is Python-side and runs on any tier.
"""
import os

import pytest

import manifoldbt as bt
from manifoldbt.diagnostics import detect_lookahead
from manifoldbt.indicators import close

pro_only = pytest.mark.skipif(
    bt.license_info()[0] != "Pro",
    reason="requires Pro: the detector is a Pro feature",
)


def _leaky():
    return (
        bt.Strategy.create("leaky")
        .signal("up", close.lead(1) > close)
        .size(bt.when(bt.col("up"), 1.0, 0.0))
    )


def _clean():
    return (
        bt.Strategy.create("clean")
        .signal("up", close > close.lag(1))
        .size(bt.when(bt.col("up"), 1.0, 0.0))
    )


def _static(strategy):
    """The static sub-test alone: no config, no store."""
    result = detect_lookahead(strategy, None, None, mode="static")
    assert [r.method for r in result.reports] == ["static"]
    return result.reports[0]


def test_unknown_mode_is_rejected_not_passed(monkeypatch):
    """An unknown mode used to run nothing and report PASS over no sub-test."""
    monkeypatch.setattr(bt, "_is_pro", lambda: True)
    with pytest.raises(ValueError, match="mode"):
        detect_lookahead(_leaky(), None, None, mode="bogus")


@pro_only
def test_lead_reads_are_found_wherever_they_sit():
    strategy = (
        bt.Strategy.create("many")
        .signal("next", close.lead(1))
        .signal("nested", bt.when(close.lead(5) > close, 1.0, 0.0))
        .signal("coarse", bt.tf("1h").close.lead(2))
        .signal("lagged", close.lag(3))
        .signal("noop", close.lead(0))
        .size(bt.col("next"))
    )
    report = _static(strategy)
    assert not report.passed
    # tf("1h").close is the column "1h.close" once serialized.
    assert {(d["field"], d["base"]) for d in report.details} == {
        ("signal 'next'", "close.lead(1)"),
        ("signal 'nested'", "close.lead(5)"),
        ("signal 'coarse'", "1h.close.lead(2)"),
    }
    assert report.mismatched == 3


@pro_only
def test_lead_in_position_sizing_is_found_too():
    strategy = bt.Strategy.create("sz").signal("s", close).size(close.lead(3))
    report = _static(strategy)
    assert [(d["field"], d["base"]) for d in report.details] == [
        ("position sizing", "close.lead(3)"),
    ]


@pro_only
def test_lead_period_from_a_parameter_is_reported_by_name():
    strategy = (
        bt.Strategy.create("p")
        .signal("x", close.lead(bt.param("n", default=2)))
        .size(bt.lit(1.0))
    )
    report = _static(strategy)
    assert [d["base"] for d in report.details] == ["close.lead(n)"]
    assert "param 'n'" in report.details[0]["extended"]


@pro_only
def test_clean_strategy_has_no_static_finding():
    report = _static(_clean())
    assert report.passed and report.details == []
    report.assert_clean()


@pro_only
def test_static_report_names_the_signal_and_fails():
    report = _static(_leaky())
    assert not report.passed
    assert report.mismatched == 1
    text = str(report)
    assert "signal 'up'" in text and "close.lead(1)" in text and "1 bar ahead" in text
    with pytest.raises(AssertionError, match="static"):
        report.assert_clean()


def test_rerun_modes_still_need_a_store(monkeypatch):
    monkeypatch.setattr(bt, "_is_pro", lambda: True)
    with pytest.raises(ValueError, match="static"):
        detect_lookahead(_leaky(), None, None, mode="extension")


# -- End to end, on the engine ------------------------------------------------

def _daily_bars(rows=400, seed=5):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    close_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, rows)))
    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close_px[:-1]
    wick = rng.uniform(0.2, 1.0, rows) * 0.005 * close_px
    return pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=rows, freq="1D", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close_px) + wick,
        "low": np.minimum(open_, close_px) - wick,
        "close": close_px,
        "volume": np.full(rows, 1_000.0),
    })


def _store(frame, tmp_path):
    root = str(tmp_path)
    return bt.import_dataframe(
        frame, symbol="SYNTH", symbol_id=1, interval="1d",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )


def _config(frame):
    from manifoldbt.helpers import Interval, Slippage
    pd = pytest.importorskip("pandas")

    ts = pd.DatetimeIndex(frame["timestamp"])
    one_day_ns = 86_400_000_000_000
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=int(ts[0].value),
        time_range_end=int(ts[-1].value) + one_day_ns,
        bar_interval=Interval.days(1),
        output_resolution=Interval.days(1),
        initial_capital=10_000.0,
        execution=bt.ExecutionConfig(
            signal_delay=0,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            allow_fractional=True,
            position_sizing_mode="FractionOfEquity",
        ),
        slippage=Slippage.fixed_bps(0),
    )


@pro_only
def test_detector_fails_a_lead_whatever_the_reruns_say(tmp_path):
    """The reported case: ``close.lead(1)`` must come back FAIL from ``all``.

    The re-runs are not asserted either way. Whether they notice the missing
    T+1 on the last bar of a window is an accident of the data; what the
    verdict must not depend on is that accident.
    """
    frame = _daily_bars()
    store = _store(frame, tmp_path)
    config = _config(frame)

    leaky = detect_lookahead(_leaky(), config, store, mode="all")
    by_method = {r.method: r for r in leaky.reports}
    assert set(by_method) == {"static", "extension", "truncation"}
    assert not by_method["static"].passed
    assert not leaky.passed
    with pytest.raises(AssertionError, match="static"):
        leaky.assert_clean()
    # The re-runs did compare trades: the leak trades a lot, so a verdict
    # over zero trades would mean the split fell outside the data.
    assert by_method["extension"].total_trades_overlap > 0
    assert by_method["truncation"].total_trades_overlap > 0

    clean = detect_lookahead(_clean(), config, store, mode="all")
    assert clean.passed, str(clean)
    assert sum(r.total_trades_overlap for r in clean.reports) > 0
