"""Clean dark theme — modern, readable, quant-oriented (plotly template)."""
from __future__ import annotations

from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Color palette — neutral dark, no decorative colors
# ---------------------------------------------------------------------------
WHITE = "#e8e6e3"
GRAY = "#8a8a8a"
DARK_GRAY = "#555555"
ACCENT = "#60a5fa"       # Neutral blue — primary data line
ACCENT_ALT = "#a78bfa"   # Subtle purple — secondary series
GREEN = "#22c55e"        # Positive only
RED = "#ef4444"          # Negative only
ORANGE = "#f59e0b"       # OOS / warning

BG_FIGURE = "#0c0c0f"
BG_AXES = "#111116"
BORDER = "#1e1e24"
GRID_RGBA = (1.0, 1.0, 1.0, 0.04)
GRID_COLOR = "rgba(255,255,255,0.045)"

SERIES_COLORS = [ACCENT, ACCENT_ALT, "#2dd4bf", ORANGE, RED, GREEN, "#f472b6", WHITE]

FONT_FAMILY = "Inter, system-ui, Segoe UI, Arial, sans-serif"
MONO_FAMILY = "SF Mono, Fira Code, Cascadia Code, Consolas, monospace"

# ---------------------------------------------------------------------------
# Colorscales (plotly format) — same stops as the old matplotlib colormaps
# ---------------------------------------------------------------------------
CS_DIVERGING = [[0.0, "#b91c1c"], [0.5, "#262626"], [1.0, "#15803d"]]
CS_SEQUENTIAL = [[0.0, "#b91c1c"], [0.5, "#d97706"], [1.0, "#15803d"]]
CS_CORRELATION = [[0.0, "#b91c1c"], [0.5, "#262626"], [1.0, "#1d4ed8"]]

# ---------------------------------------------------------------------------
# Layout defaults (also exported as THEME for backward compatibility)
# ---------------------------------------------------------------------------
THEME: dict = {
    "paper_bgcolor": BG_FIGURE,
    "plot_bgcolor": BG_AXES,
    "font": {"family": FONT_FAMILY, "color": GRAY, "size": 12},
    "title": {"font": {"color": WHITE, "size": 15}, "x": 0.01, "xanchor": "left"},
    "margin": {"l": 64, "r": 24, "t": 48, "b": 36},
    "hovermode": "x",
    "colorway": SERIES_COLORS,
    "hoverlabel": {
        "bgcolor": "#1a1a20",
        "bordercolor": BORDER,
        "font": {"family": MONO_FAMILY, "color": WHITE, "size": 12},
    },
    "legend": {
        "bgcolor": "rgba(17,17,22,0.6)",
        "bordercolor": BORDER,
        "borderwidth": 1,
        "font": {"color": GRAY, "size": 11},
    },
}

_AXIS = {
    "color": GRAY,
    "gridcolor": GRID_COLOR,
    "linecolor": BORDER,
    "zerolinecolor": GRID_COLOR,
    "ticks": "",
    "showspikes": True,
    "spikemode": "across",
    "spikethickness": 1,
    "spikedash": "dot",
    "spikecolor": GRAY,
}

_SCENE_AXIS = {
    "backgroundcolor": BG_AXES,
    "gridcolor": "rgba(255,255,255,0.08)",
    "color": GRAY,
    "showbackground": True,
    "zerolinecolor": "rgba(255,255,255,0.08)",
}


def _build_template():
    """Build the manifoldbt plotly template."""
    import plotly.graph_objects as go

    layout = dict(THEME)
    layout["xaxis"] = dict(_AXIS)
    layout["yaxis"] = dict(_AXIS)
    layout["scene"] = {
        "xaxis": dict(_SCENE_AXIS),
        "yaxis": dict(_SCENE_AXIS),
        "zaxis": dict(_SCENE_AXIS),
        "bgcolor": BG_FIGURE,
    }
    return go.layout.Template(layout=layout)


_REGISTERED = False


def apply_theme() -> None:
    """Register the manifoldbt template and set it as plotly's default."""
    global _REGISTERED
    import plotly.io as pio

    pio.templates["manifoldbt"] = _build_template()
    pio.templates.default = "manifoldbt"
    _REGISTERED = True


def _ensure_theme() -> None:
    if not _REGISTERED:
        apply_theme()


@contextmanager
def theme_context():
    """Backward-compatible context manager: ensures the theme is registered.

    With plotly the theme is a global template rather than a temporary
    rc-context, so this simply guarantees registration.
    """
    _ensure_theme()
    yield
