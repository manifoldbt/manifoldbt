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
