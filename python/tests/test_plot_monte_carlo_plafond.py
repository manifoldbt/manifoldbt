"""``plot.monte_carlo`` caps its paths on the licence flag, not on the tier name.

The engine's Monte Carlo consults ``monte_carlo_unlimited``; the chart used to
compare the tier string to ``"Pro"``. Both agree today, and would part the day
the flag changes tier, as ``gpu_sweep`` once did. The chart now asks the same
question of the native layer (``_grant_couvre``), like the GPU gates.
"""
import importlib

import numpy as np
import pytest

pytest.importorskip("plotly")

import manifoldbt as bt  # noqa: E402

research = importlib.import_module("manifoldbt.plot.research")


@pytest.fixture
def chart(monkeypatch):
    rets = np.random.default_rng(0).normal(0.0, 0.01, 60)
    equity = 10_000.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    monkeypatch.setattr(research, "daily_returns_array", lambda _r: rets)
    monkeypatch.setattr(research, "equity_with_dates", lambda _r: (None, list(equity)))
    asked = []

    def draw(granted, n):
        def grant(feature):
            asked.append(feature)
            return granted

        monkeypatch.setattr(bt, "_grant_couvre", grant)
        fig = research.monte_carlo(object(), n_simulations=n, n_sample_paths=0, seed=1, show=False)
        return fig.layout.title.text

    return draw, asked


def test_without_the_flag_the_chart_stops_at_1000_paths(chart):
    draw, asked = chart
    assert "1,000 paths" in draw(False, 5_000)
    assert asked == ["monte_carlo_unlimited"]


def test_with_the_flag_the_chart_draws_every_path(chart):
    draw, _ = chart
    assert "5,000 paths" in draw(True, 5_000)


def test_under_the_ceiling_the_licence_is_not_consulted(chart):
    draw, asked = chart
    assert "500 paths" in draw(False, 500)
    assert asked == []
