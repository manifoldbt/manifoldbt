"""Tearsheet rendering, in particular the offline (``plotlyjs="inline"``) report.

The inline path was written against ``plotly.io.get_plotlyjs``, which does not
exist and never has, so every offline report raised AttributeError. Nothing
exercised it, so nothing caught it. These tests render both modes for real.
"""
import os

import pytest

import manifoldbt as bt
from manifoldbt import run_with_parquet

pytest.importorskip("plotly")

tearsheet = pytest.importorskip("manifoldbt.plot.tearsheet").tearsheet

# Roughly the size of the embedded plotly runtime; the point is to tell an
# actually-embedded bundle from a one-line CDN script tag, not to pin a version.
_RUNTIME_BYTES = 2_000_000
_CDN_TAG = '<script src="https://cdn.plot.ly'


@pytest.fixture
def backtest_result(golden_buy_hold_dir):
    strategy = bt.Strategy(
        name="tearsheet_probe",
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


def test_inline_report_embeds_the_runtime(backtest_result, tmp_path):
    out = tmp_path / "inline.html"
    tearsheet(backtest_result, save=str(out), show=False, plotlyjs="inline")

    body = out.read_text(encoding="utf-8")
    assert _CDN_TAG not in body, "inline report still loads plotly from the CDN"
    assert out.stat().st_size > _RUNTIME_BYTES, (
        f"inline report is only {out.stat().st_size} bytes, "
        "the plotly runtime does not look embedded"
    )


def test_cdn_report_links_the_runtime_instead_of_embedding_it(
    backtest_result, tmp_path
):
    out = tmp_path / "cdn.html"
    tearsheet(backtest_result, save=str(out), show=False, plotlyjs="cdn")

    body = out.read_text(encoding="utf-8")
    assert _CDN_TAG in body
    assert out.stat().st_size < _RUNTIME_BYTES


def test_get_plotlyjs_comes_from_plotly_offline():
    """Pin the import location that the inline path depends on."""
    import plotly.io as pio
    from plotly.offline import get_plotlyjs

    assert not hasattr(pio, "get_plotlyjs"), (
        "plotly.io grew a get_plotlyjs; the tearsheet import can be simplified"
    )
    assert len(get_plotlyjs()) > _RUNTIME_BYTES
