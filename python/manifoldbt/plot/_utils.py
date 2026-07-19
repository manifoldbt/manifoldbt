"""Shared plotting utilities (plotly)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from manifoldbt.plot._theme import WHITE, _ensure_theme

_RESPONSIVE_CSS = (
    "<style>html,body{height:100%;margin:0;background:#0c0c0f;overflow:hidden}"
    ".plotly-graph-div{width:100vw!important;height:100vh!important}</style>"
)

_IMAGE_EXTS = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp"}


def new_figure(
    figsize: Tuple[float, float] = (12, 4),
    title: Optional[str] = None,
):
    """Return a themed plotly Figure sized from a matplotlib-style figsize."""
    import plotly.graph_objects as go

    _ensure_theme()
    fig = go.Figure()
    fig.update_layout(width=int(figsize[0] * 80), height=int(figsize[1] * 80))
    if title:
        fig.update_layout(title_text=title)
    return fig


def _in_notebook() -> bool:
    """True inside a Jupyter/IPython kernel (not a plain terminal REPL)."""
    try:
        from IPython import get_ipython  # type: ignore
    except ImportError:
        return False
    try:
        ip = get_ipython()
    except Exception:
        return False
    return ip is not None and hasattr(ip, "kernel")


def _in_test_or_ci() -> bool:
    """True under pytest or on a CI runner.

    A test that builds a figure must not queue a window: show() runs from
    the atexit hook and blocks on the window process, so one bare plot call
    in a test suite hangs the whole run until a human closes it.
    """
    import os
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"))


def resolve_show(show: "bool | str | None",
                 save: Optional[Union[str, Path]]) -> "bool | str":
    """Resolve the ``show=None`` auto default.

    A chart you asked for is a chart you want to see, so the default shows
    it. Three cases opt out, because showing there would be wrong:

    - ``save`` was given: you asked for a file, not a window.
    - pytest/CI: show() runs from atexit and blocks on the window process.
    - a notebook: the cell already renders the returned Figure. Calling
      show() here too would emit a SECOND copy of the same chart, so the
      notebook path stays silent and lets the cell do the rendering.

    Explicit ``True``/``False``/``"browser"`` always wins. There is no
    "render it inline" value to pass, because that is what the notebook
    already does with the returned Figure; to place a chart mid-cell, call
    IPython's ``display(fig)``.
    """
    if show is not None:
        return show
    if save is not None:
        return False
    if _in_test_or_ci():
        return False
    return False if _in_notebook() else True


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal fraction as a percentage string."""
    return f"{value * 100:+.{decimals}f}%"


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a number as currency."""
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")
    return f"{symbol}{value:,.2f}"


def finalize(
    fig,
    *,
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    window_size: Optional[Tuple[int, int]] = None,
) -> "object":
    """Optionally save and/or display the figure, then return it.

    ``save`` routes on extension: ``.html`` writes a responsive interactive
    page; image extensions (.png/.svg/.pdf/...) go through kaleido.
    ``show``: ``None`` (default) shows the chart unless ``save`` was given or
    we are in a notebook (see :func:`resolve_show`); ``True`` (or ``"window"``)
    opens a native window (needs pywebview, else falls back to a browser tab);
    ``"browser"`` forces a browser tab; ``False`` returns the figure silently.
    ``dpi`` is kept for backward compatibility and maps to an export scale.
    """
    show = resolve_show(show, save)
    if _in_notebook():
        # new_figure() sets a pixel width sized for a window (1120px by
        # default). A notebook cell is narrower than that, so the chart
        # overflowed its output area: the right edge and the modebar were
        # pushed out of view. Drop the fixed width and let it track the cell,
        # keeping the height so the cell still has a definite size.
        fig.update_layout(width=None, autosize=True)
    if save is not None:
        path = Path(save)
        ext = path.suffix.lower()
        if ext in _IMAGE_EXTS:
            try:
                scale = max(1.0, dpi / 96.0)
                fig.write_image(str(path), scale=scale)
            except Exception as exc:  # kaleido missing or export failure
                raise RuntimeError(
                    f"Static image export to {ext} is optional and needs kaleido. "
                    "Install it with: pip install manifoldbt[png]  "
                    "(the default is the interactive chart - save to .html)"
                ) from exc
        else:
            write_responsive_html(fig, path)
    if show == "inline" and _in_notebook():
        # No-op on purpose. Rendering in the cell IS the notebook default, so
        # calling show() here would emit a second copy of the chart the cell
        # is already going to render. To place a chart mid-cell, where there
        # is no trailing expression, use IPython's display(fig).
        pass
    elif show in ("browser", "inline"):
        fig.show()
    elif show:  # True or "window" -> native window (browser tab fallback)
        from manifoldbt.plot._window import open_in_window
        title = "Chart"
        try:
            t = fig.layout.title.text
            if t:
                title = t.split("<br>")[0].strip() or title
        except Exception:
            pass
        open_in_window(fig, title=title, size=window_size or (1280, 720))
    return fig


def write_responsive_html(fig, path: Union[str, Path]) -> None:
    """Write a self-adjusting full-window HTML page for *fig*.

    Strips any fixed width/height so the plot fills (and resizes with) the
    window; a clean <title> replaces the browser's filename fallback.
    """
    # A fixed layout size would override plotly's responsive resizing.
    fig.update_layout(width=None, height=None, autosize=True)
    title = "Chart"
    try:
        t = fig.layout.title.text
        if t:
            title = t.split("<br>")[0].strip() or title
    except Exception:
        pass

    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="100%",
        config={"displayModeBar": False, "responsive": True},
    )
    head = "<head>" + _RESPONSIVE_CSS + f"<title>{title}</title>"
    html = html.replace("<head>", head, 1)
    Path(path).write_text(html, encoding="utf-8")


def chart_div(fig, *, height: Optional[int] = None) -> str:
    """Return an embeddable div (no plotly.js) for report composition."""
    if height is not None:
        fig.update_layout(height=height)
    fig.update_layout(width=None, autosize=True)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        default_width="100%",
        default_height=f"{height}px" if height else "100%",
        config={"displayModeBar": False, "responsive": True},
    )


def auto_title(result, fallback: str) -> str:
    """Build a title from result manifest strategy_name, or use fallback."""
    try:
        manifest = result.manifest
        if isinstance(manifest, dict) and "strategy_name" in manifest:
            return manifest["strategy_name"]
    except Exception:
        pass
    return fallback
