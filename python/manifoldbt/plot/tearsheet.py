"""Composite tearsheet — HTML strategy report."""
from __future__ import annotations

import base64
import io
import tempfile
import webbrowser
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from manifoldbt.plot._theme import (
    BG_AXES,
    BG_FIGURE,
    DARK_GRAY,
    GRAY,
    GREEN,
    RED,
    WHITE,
    theme_context,
)
from manifoldbt.plot._convert import equity_with_dates, positions_arrays
from manifoldbt.plot._utils import auto_title, format_pct
from manifoldbt.plot.backtest import (
    annual_returns,
    drawdown,
    equity,
    monthly_returns,
    returns_histogram,
    rolling_sharpe,
    rolling_volatility,
    summary,
    var_chart,
)


def _fig_to_base64(fig: Figure, dpi: int = 150) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_chart(chart_fn, result, figsize=(12, 4), dpi=150, **kwargs) -> str:
    """Call a chart function on a fresh figure/axes and return base64 PNG."""
    with theme_context():
        fig, ax = plt.subplots(figsize=figsize)
        chart_fn(result, ax=ax, **kwargs)
        fig.tight_layout()
        return _fig_to_base64(fig, dpi=dpi)


def _render_summary_b64(result, figsize=(12, 6), dpi=150) -> str:
    """Render the summary chart (equity+benchmark+trades+margin) to base64."""
    with theme_context():
        fig = summary(result, figsize=figsize)
        return _fig_to_base64(fig, dpi=dpi)


def _render_exposure_b64(result, figsize=(12, 4), dpi=150) -> str:
    """Render the exposure chart to base64 PNG."""
    with theme_context():
        fig, ax = plt.subplots(figsize=figsize)
        _render_exposure(ax, result)
        _set_title(ax, "Capital Exposure")
        _format_dates(ax)
        fig.tight_layout()
        return _fig_to_base64(fig, dpi=dpi)


_CSS = f"""
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: {BG_FIGURE};
    color: {WHITE};
    font-family: "SF Mono", "Fira Code", "JetBrains Mono", "Cascadia Code", Consolas, monospace;
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}}
.container {{
    max-width: 1800px;
    margin: 0 auto;
    padding: 24px 32px;
}}
.header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 16px 0 12px 0;
    border-bottom: 1px solid #1e1e24;
    margin-bottom: 20px;
}}
.header h1 {{
    font-size: 18px;
    font-weight: 700;
    color: {WHITE};
    letter-spacing: 0.5px;
}}
.header .dates {{
    font-size: 13px;
    color: {GRAY};
}}
/* Main layout: metrics left + charts right */
.main-grid {{
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 16px;
    margin-bottom: 16px;
}}
.metrics-panel {{
    background: {BG_AXES};
    border: 1px solid #1e1e24;
    border-radius: 4px;
    padding: 16px 20px;
}}
.charts-stack {{
    display: flex;
    flex-direction: column;
    gap: 12px;
}}
.charts-stack img {{
    width: 100%;
    display: block;
    border-radius: 4px;
    border: 1px solid #1e1e24;
}}
.section-label {{
    font-size: 10px;
    font-weight: 700;
    color: {DARK_GRAY};
    letter-spacing: 1.5px;
    margin-bottom: 4px;
    margin-top: 10px;
}}
.section-label:first-child {{
    margin-top: 0;
}}
.metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 1px 0;
    font-size: 12px;
}}
.metric-label {{
    color: {GRAY};
}}
.metric-dots {{
    flex: 1;
    border-bottom: 1px dotted #2a2a2a;
    margin: 0 6px;
    min-width: 10px;
    position: relative;
    top: -3px;
}}
.metric-value {{
    color: {GRAY};
    font-weight: 500;
    white-space: nowrap;
}}
.chart-row {{
    margin-bottom: 12px;
}}
.chart-row img {{
    width: 100%;
    display: block;
    border-radius: 4px;
    border: 1px solid #1e1e24;
}}
.chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 12px;
}}
.chart-grid img {{
    width: 100%;
    display: block;
    border-radius: 4px;
    border: 1px solid #1e1e24;
}}
.chart-grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 12px;
}}
.chart-grid-3 img {{
    width: 100%;
    display: block;
    border-radius: 4px;
    border: 1px solid #1e1e24;
}}
"""


def tearsheet(
    result,
    *,
    benchmark=None,
    title: Optional[str] = None,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> str:
    """Strategy report — self-contained HTML page.

    Returns the HTML string. Opens in browser when ``show=True``,
    writes to disk when ``save`` is given.
    """
    _ = benchmark  # reserved for future benchmark overlay support
    strategy_name = title or auto_title(result, "Backtest")
    metrics = result.metrics if hasattr(result, "metrics") else {}
    ts = metrics.get("trade_stats", {})
    dates, _ = equity_with_dates(result)

    date_start = str(dates[0])[:10] if len(dates) > 0 else "?"
    date_end = str(dates[-1])[:10] if len(dates) > 0 else "?"

    # ── Generate charts as base64 PNGs ────────────────────────────
    # Right column: summary chart (equity + benchmark + trades + margin)
    chart_summary = _render_summary_b64(result, figsize=(12, 6), dpi=dpi)
    chart_dd = _render_chart(drawdown, result, figsize=(12, 2.5), dpi=dpi)
    # Left column chart
    chart_annual = _render_chart(annual_returns, result, figsize=(5, 4), dpi=dpi)
    # Full width grids (2 per row)
    chart_monthly = _render_chart(monthly_returns, result, figsize=(8, 4), dpi=dpi)
    chart_hist = _render_chart(returns_histogram, result, figsize=(8, 4), dpi=dpi)
    chart_sharpe = _render_chart(rolling_sharpe, result, figsize=(8, 3.5), dpi=dpi)
    chart_vol = _render_chart(rolling_volatility, result, figsize=(8, 3.5), dpi=dpi)
    chart_var = _render_chart(var_chart, result, figsize=(8, 4), dpi=dpi)

    # ── Metrics ───────────────────────────────────────────────────
    ret = metrics.get("total_return", 0)
    _ = ret  # used below in metrics_html

    def _m(label, value, cls=""):
        esc_v = escape(str(value))
        cls_attr = f' class="metric-value {cls}"' if cls else ' class="metric-value"'
        return (
            f'<div class="metric-row">'
            f'<span class="metric-label">{escape(label)}</span>'
            f'<span class="metric-dots"></span>'
            f'<span{cls_attr}>{esc_v}</span>'
            f'</div>'
        )

    def _section(label):
        return f'<div class="section-label">{escape(label)}</div>'

    metrics_html = (
        _section("RETURNS")
        + _m("Total Return", format_pct(ret))
        + _m("CAGR", format_pct(metrics.get("cagr", 0)))
        + _m("Max Drawdown", format_pct(metrics.get("max_drawdown", 0)))
        + _m("Volatility", format_pct(metrics.get("volatility", 0)))
        + _m("Best Day", format_pct(metrics.get("best_day", 0)))
        + _m("Worst Day", format_pct(metrics.get("worst_day", 0)))
        + _m("% Pos Days", f"{metrics.get('pct_positive_days', 0):.1%}")
        + _section("RATIOS")
        + _m("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
        + _m("Sortino", f"{metrics.get('sortino', 0):.2f}")
        + _m("Calmar", f"{metrics.get('calmar', 0):.2f}")
        + _section("TRADING")
        + _m("Trades", f"{ts.get('total_trades', metrics.get('total_trades', 0))}")
        + _m("Win Rate", f"{ts.get('win_rate', metrics.get('win_rate', 0)):.1%}")
        + _m("Profit Factor", f"{ts.get('profit_factor', metrics.get('profit_factor', 0)):.2f}")
        + _m("Avg Hold", _fmt_hold_time(ts.get("avg_holding_seconds", 0)))
        + _m("Fees", f"{ts.get('total_fees', 0):.2f}")
    )

    # ── Assemble HTML ─────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(strategy_name)} — Tearsheet</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>{escape(strategy_name)}</h1>
  <span class="dates">{escape(date_start)} &rarr; {escape(date_end)}</span>
</div>

<div class="main-grid">
  <div>
    <div class="metrics-panel" style="margin-bottom:12px;">{metrics_html}</div>
    <img src="data:image/png;base64,{chart_annual}" alt="Annual Returns" style="width:100%; border-radius:4px; border:1px solid #1e1e24;">
  </div>
  <div class="charts-stack">
    <img src="data:image/png;base64,{chart_summary}" alt="Equity + Benchmark + Trades + Margin">
    <img src="data:image/png;base64,{chart_dd}" alt="Drawdown">
  </div>
</div>

<div class="chart-grid">
  <img src="data:image/png;base64,{chart_monthly}" alt="Monthly Returns">
  <img src="data:image/png;base64,{chart_hist}" alt="Returns Distribution">
</div>

<div class="chart-grid">
  <img src="data:image/png;base64,{chart_sharpe}" alt="Rolling Sharpe">
  <img src="data:image/png;base64,{chart_vol}" alt="Rolling Volatility">
</div>

<div class="chart-grid">
  <img src="data:image/png;base64,{chart_var}" alt="Value at Risk">
</div>

</div>
</body>
</html>"""

    # ── Save / Show ───────────────────────────────────────────────
    if save is not None:
        Path(save).write_text(html, encoding="utf-8")

    if show:
        # Write the report HTML
        if save is not None:
            report_path = Path(save).resolve()
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(html)
            tmp.close()
            report_path = Path(tmp.name).resolve()

        # Create a launcher HTML that opens the report in a 1600x850 window
        report_uri = report_path.as_uri()
        launcher_html = f"""<!DOCTYPE html><html><head><script>
        var w = window.open("{report_uri}", "_blank",
            "width=1600,height=850,menubar=no,toolbar=no,location=no,status=no");
        if (!w) window.location = "{report_uri}";
        else window.close();
        </script></head><body></body></html>"""

        launcher = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        )
        launcher.write(launcher_html)
        launcher.close()
        webbrowser.open(Path(launcher.name).resolve().as_uri())

    return html


def research_report(
    sweep_result: Optional[Dict[str, Any]] = None,
    wf_result: Optional[Dict[str, Any]] = None,
    stability_result: Optional[Dict[str, Any]] = None,
    *,
    title: str = "Research Report",
    figsize: tuple = (14, 6),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> List[Figure]:
    """Research report — one figure per analysis."""
    from manifoldbt.plot.research import (
        heatmap_2d,
        stability,
        walk_forward,
    )

    figs = []
    with theme_context():
        if sweep_result is not None:
            fig, ax = plt.subplots(figsize=figsize)
            heatmap_2d(sweep_result, ax=ax)
            figs.append(fig)
        if wf_result is not None:
            fig, ax = plt.subplots(figsize=figsize)
            walk_forward(wf_result, ax=ax)
            figs.append(fig)
        if stability_result is not None:
            fig, ax = plt.subplots(figsize=figsize)
            stability(stability_result, ax=ax)
            figs.append(fig)

    if not figs:
        raise ValueError("At least one result (sweep, wf, or stability) required.")

    if save is not None:
        path = Path(save)
        stem, suffix = path.stem, path.suffix or ".png"
        for i, f in enumerate(figs):
            out = path.parent / f"{stem}_{i + 1}{suffix}"
            f.savefig(str(out), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return figs


# ── Internal ─────────────────────────────────────────────────────────────────


def _set_title(ax, text):
    """Set title left-aligned, clearing any existing title from sub-functions."""
    ax.set_title("", loc="center")  # clear default
    ax.set_title(text, fontsize=9, loc="left", color=GRAY)


def _format_dates(ax):
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_ha("center")
    except Exception:
        pass


def _fix_rolling_xaxis(ax, result):
    try:
        dates, _ = equity_with_dates(result)
        from manifoldbt.plot._convert import daily_returns_array
        rets = daily_returns_array(result)
        n_rets = len(rets)
        aligned = dates[len(dates) - n_rets:] if len(dates) > n_rets else dates

        for line in ax.get_lines():
            xdata = line.get_xdata()
            n = len(xdata)
            if n <= 1:
                continue
            if isinstance(xdata[0], (int, float, np.integer, np.floating)):
                x0, x1 = float(xdata[0]), float(xdata[1])
                if abs(x1 - x0 - 1.0) < 0.01 and n <= len(aligned):
                    line.set_xdata(aligned[:n])

        _format_dates(ax)
        ax.relim()
        ax.autoscale_view()
    except Exception:
        pass


def _render_metrics_table(ax, metrics, ts):
    """Render metrics as single-column dotted-leader list with sections."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG_AXES)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK_GRAY)
        spine.set_linewidth(0.5)

    ret = metrics.get("total_return", 0)
    ret_color = GREEN if ret > 0 else RED if ret < 0 else GRAY
    W = 30  # total width for dotted leader alignment

    def _line(label, value):
        dots = "·" * max(1, W - len(label) - len(str(value)))
        return f"{label} {dots} {value}"

    # Build sections
    sections = [
        ("RETURNS", GRAY, [
            (_line("Total Return", format_pct(ret)), ret_color),
            (_line("CAGR", format_pct(metrics.get("cagr", 0))), GRAY),
            (_line("Max Drawdown", format_pct(metrics.get("max_drawdown", 0))), RED),
            (_line("Volatility", format_pct(metrics.get("volatility", 0))), GRAY),
            (_line("Best Day", format_pct(metrics.get("best_day", 0))), GRAY),
            (_line("Worst Day", format_pct(metrics.get("worst_day", 0))), GRAY),
        ]),
        ("RATIOS", GRAY, [
            (_line("Sharpe", f"{metrics.get('sharpe', 0):.2f}"), GRAY),
            (_line("Sortino", f"{metrics.get('sortino', 0):.2f}"), GRAY),
            (_line("Calmar", f"{metrics.get('calmar', 0):.2f}"), GRAY),
        ]),
        ("TRADING", GRAY, [
            (_line("Trades", f"{ts.get('total_trades', metrics.get('total_trades', 0))}"), GRAY),
            (_line("Win Rate", f"{ts.get('win_rate', metrics.get('win_rate', 0)):.1%}"), GRAY),
            (_line("Profit Factor", f"{ts.get('profit_factor', metrics.get('profit_factor', 0)):.2f}"), GRAY),
            (_line("Round Trips", f"{ts.get('round_trips', 0)}"), GRAY),
            (_line("Avg Hold", _fmt_hold_time(ts.get("avg_holding_seconds", 0))), GRAY),
            (_line("Fees", f"{ts.get('total_fees', 0):.2f}"), GRAY),
        ]),
    ]

    # Count total lines for spacing
    total = sum(1 + len(items) + 1 for _, _, items in sections)  # header + items + gap
    y = 0.97
    dy = 0.92 / total

    for section_name, section_color, items in sections:
        # Section header
        ax.text(0.06, y, section_name, fontsize=7, fontweight="bold",
                color=DARK_GRAY, transform=ax.transAxes, va="top",
                family="monospace")
        y -= dy * 1.2

        # Items
        for text, color in items:
            ax.text(0.06, y, text, fontsize=9, color=color,
                    transform=ax.transAxes, va="top", family="monospace")
            y -= dy

        # Gap between sections
        y -= dy * 0.5


def _fmt_hold_time(seconds):
    """Format holding time in human-readable units."""
    if seconds <= 0:
        return "—"
    days = seconds / 86400
    if days >= 365:
        return f"{days / 365:.1f}y"
    if days >= 30:
        return f"{days / 30:.1f}mo"
    if days >= 1:
        return f"{days:.0f}d"
    hours = seconds / 3600
    return f"{hours:.0f}h"


def _render_exposure(ax, result):
    try:
        pa = positions_arrays(result)
        pos_ts = pa["timestamp"]
        pos_cap = pa["capital"]
        pos_eq = pa["equity"]

        unique_ts, first_idx = np.unique(pos_ts, return_index=True)
        first_idx.sort()
        cap = pos_cap[first_idx]
        eq_arr = pos_eq[first_idx]
        used = np.where(eq_arr > 0, (1.0 - cap / eq_arr) * 100, 0.0)
        used = np.clip(used, 0, None)
        used_dates = unique_ts.astype("datetime64[ns]")

        ax.fill_between(used_dates, 0, used,
                         color=GREEN, alpha=0.10, edgecolor="none")
        ax.plot(used_dates, used, color=GREEN, linewidth=0.7, alpha=0.8)
        ax.axhline(0, color=DARK_GRAY, linewidth=0.4)
        ax.set_ylabel("Exposure %", fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, "No position data",
                transform=ax.transAxes, ha="center", va="center",
                color=DARK_GRAY, fontsize=9)
