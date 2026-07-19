"""Plotting module for manifoldbt (requires plotly).

Install with::

    pip install manifoldbt[plot]

Quick start::

    import manifoldbt as bt

    result = bt.run(strategy, config, store)
    bt.plot.tearsheet(result)              # full-page dashboard
    bt.plot.equity(result)                 # single chart, opens on its own

Every chart is interactive (crosshair, hover, zoom) and **shows itself by
default**: plotting is what you asked for, so no ``show=`` is needed. Charts
open in a native window (``pip install manifoldbt[window]``; falls back to a
browser tab, which you can also force with ``show="browser"``).

Three cases opt out of showing automatically, because showing would be
wrong: passing ``save=`` (you asked for a file, not a window), running
under pytest/CI (a window there blocks the run), and running inside a
notebook, where the cell already renders the returned Figure and showing
would print a second copy of the same chart.

Pass an explicit ``show=True`` to override any of them, or ``show=False``
to get the Figure back silently and compose it yourself. To place a chart
in the middle of a notebook cell, where there is no trailing expression for
Jupyter to display, call IPython's ``display(fig)`` on the returned figure.

``save=".html"`` writes a responsive interactive page. Static ``save=".png"``
is optional and needs ``pip install manifoldbt[png]`` (pulls a headless Chromium).
"""
try:
    import plotly  # noqa: F401
except ImportError:
    raise ImportError(
        "plotly is required for the plotting module. "
        "Install it with: pip install manifoldbt[plot]"
    ) from None

# Backtest result charts
from manifoldbt.plot.backtest import (
    annual_returns,
    benchmark_equity,
    drawdown,
    equity,
    monthly_returns,
    returns_histogram,
    rolling_sharpe,
    rolling_volatility,
    summary,
    var_chart,
)

# Candlestick / indicator chart
from manifoldbt.plot.chart import chart

# Research charts
from manifoldbt.plot.research import (
    correlation_matrix,
    heatmap_2d,
    monte_carlo,
    stability,
    stochastic_paths,
    surface_3d,
    walk_forward,
)

# Composite layouts
from manifoldbt.plot.tearsheet import research_report, tearsheet

# Window display (multi-window, matplotlib-style)
from manifoldbt.plot._window import show

# Theme
from manifoldbt.plot._theme import THEME, apply_theme

__all__ = [
    # Backtest result plots
    "chart",
    "summary",
    "equity",
    "benchmark_equity",
    "drawdown",
    "monthly_returns",
    "annual_returns",
    "returns_histogram",
    "var_chart",
    "rolling_sharpe",
    "rolling_volatility",
    # Research plots
    "heatmap_2d",
    "surface_3d",
    "walk_forward",
    "stability",
    "correlation_matrix",
    "monte_carlo",
    "stochastic_paths",
    # Composites
    "tearsheet",
    "research_report",
    # Window display
    "show",
    # Theme
    "THEME",
    "apply_theme",
]
