"""Chart-level regressions found by rendering a tearsheet and looking at it.

Both defects below were invisible to file size, tag presence and trace counts.
They only showed up on screen, so they are pinned here at the figure-spec
level, which is cheap enough to run in CI without a browser.
"""
import os

import pytest

import manifoldbt as bt
from manifoldbt import run_with_parquet

pytest.importorskip("plotly")

backtest_plots = pytest.importorskip("manifoldbt.plot.backtest")


@pytest.fixture
def backtest_result(golden_buy_hold_dir):
    strategy = bt.Strategy(
        name="chart_probe",
        signals={"signal": bt.lit(1.0)},
        position_sizing=bt.col("signal"),
    )
    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=4_000_000_000,
        bar_interval={"Days": 1},
        initial_capital=1000.0,
        currency="USD",
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            allow_fractional=True,
            skip_gap_bars=False,
            position_sizing_mode="Units",
        ),
        fees=bt.FeeConfig(),
        slippage={"FixedBps": {"bps": 0.0}},
        data_version="golden_v1",
        rng_seed=7,
    )
    return run_with_parquet(
        strategy.to_json(),
        config.to_json(),
        os.path.join(golden_buy_hold_dir, "bars_1m.parquet"),
        "golden_v1",
    )


def test_monthly_returns_year_axis_is_categorical(backtest_result):
    """Year rows are labels, not a number line.

    The labels are strings already, but with no explicit axis type plotly
    infers a linear scale and interpolates between them: a single-year
    backtest rendered ticks at 2,022.6 / 2,022.8 / 2023 / 2,023.2 / 2,023.4.
    """
    fig = backtest_plots.monthly_returns(backtest_result)
    assert fig.layout.yaxis.type == "category"


def test_returns_histogram_has_no_unnamed_legend_entry(monkeypatch):
    """No trace may reach the legend without a name.

    The histogram bars carried no name, so plotly labelled them "trace 0" and
    gave them a single colour swatch even though the bars are green or red by
    sign. The legend is there for the Normal overlay only.

    Returns are injected rather than backtested: the golden fixture spans less
    than two UTC days, so ``daily_returns_array`` comes back empty and the
    chart short-circuits before building any trace. Asserting over that empty
    figure passes no matter what the code does.
    """
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    monkeypatch.setattr(
        backtest_plots, "daily_returns_array", lambda _result: rng.normal(0, 0.01, 500)
    )

    fig = backtest_plots.returns_histogram(object())

    # Guard against the vacuous version of this test.
    assert len(fig.data) >= 2, "expected the bars plus the Normal overlay"

    legend_names = {
        trace.name for trace in fig.data if trace.showlegend is not False
    }
    assert legend_names, "no trace reaches the legend, the check would be vacuous"
    assert None not in legend_names, "an unnamed trace renders as 'trace 0'"
    assert not any(
        (name or "").startswith("trace ") for name in legend_names
    ), f"auto-generated trace label in the legend: {legend_names}"


# ── Adaptive axis / hover formats ────────────────────────────────────────────
# These formats used to be hardcoded, and the failures were invisible in any
# assertion on figure structure: a two-month backtest labelled every date tick
# "May 2025", a -0.9% drawdown labelled every value tick "0%", and a 10-BTC
# equity hovered as "$10". Pure functions now decide them, so the contract is
# testable without rendering.


def _dates(days):
    np = pytest.importorskip("numpy")
    return np.arange("2025-01-01", np.timedelta64(days, "D") + np.datetime64("2025-01-01"),
                     dtype="datetime64[D]").astype("datetime64[ns]")


def test_date_tickformat_follows_the_span():
    from manifoldbt._convert import date_tickformat

    assert date_tickformat(_dates(2)) == "%d %b %H:%M"
    assert date_tickformat(_dates(61)) == "%d %b", "a two-month window must show days"
    assert date_tickformat(_dates(365 * 4)) == "%b %Y", "long spans keep the historical format"


def test_money_hovertemplate_is_currency_and_magnitude_aware():
    np = pytest.importorskip("numpy")
    from manifoldbt._convert import money_hovertemplate

    btc = money_hovertemplate(np.array([10.0, 10.04]), "BTC")
    assert "BTC" in btc and "%{y:,.4f}" in btc, btc
    assert "$" not in btc, "a BTC equity must not hover in dollars"

    usd = money_hovertemplate(np.array([10_000.0, 21_313.0]), "USD")
    assert "$%{y:,.0f}" in usd, usd

    eur = money_hovertemplate(np.array([500.0]), "EUR")
    assert "\u20ac" in eur and "%{y:,.2f}" in eur, eur


def test_percent_tickformat_keeps_small_drawdowns_legible():
    from manifoldbt._convert import percent_tickformat

    assert percent_tickformat(-0.35) == ".0%"
    assert percent_tickformat(-0.02) == ".1%"
    assert percent_tickformat(-0.0037) == ".2%", "a -0.37% max dd must not read as 0%"


def test_run_currency_reads_the_manifest_and_survives_its_absence():
    from manifoldbt._convert import run_currency

    class WithManifest:
        manifest = {"config": {"currency": "BTC"}}

    assert run_currency(WithManifest()) == "BTC"
    assert run_currency(object()) == "USD", "no manifest must fall back, not raise"
