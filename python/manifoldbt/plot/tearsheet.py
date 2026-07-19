"""Composite tearsheet - HTML strategy report with interactive plotly charts."""
from __future__ import annotations

import tempfile
import webbrowser
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from manifoldbt.plot._theme import (
    BG_AXES,
    BG_FIGURE,
    DARK_GRAY,
    GRAY,
    WHITE,
    theme_context,
)
from manifoldbt.plot._convert import equity_with_dates
from manifoldbt.plot._utils import auto_title, chart_div, format_pct, resolve_show
from manifoldbt.plot.backtest import (
    annual_returns,
    drawdown,
    monthly_returns,
    returns_histogram,
    rolling_sharpe,
    rolling_volatility,
    summary,
    var_chart,
)

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
.chart-cell {{
    background: {BG_FIGURE};
    border: 1px solid #1e1e24;
    border-radius: 4px;
    overflow: hidden;
}}
.chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 12px;
}}
.plotly-graph-div {{ width: 100% !important; }}
"""


def _div(fig, height: int) -> str:
    """Wrap a plotly figure div in a bordered cell."""
    return f'<div class="chart-cell">{chart_div(fig, height=height)}</div>'


def tearsheet(
    result,
    *,
    benchmark=None,
    title: Optional[str] = None,
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    plotlyjs: str = "cdn",
) -> str:
    """Strategy report - self-contained HTML page with interactive charts.

    Returns the HTML string. Opens in browser when ``show=True``,
    writes to disk when ``save`` is given.

    Args:
        plotlyjs: ``"cdn"`` (small file, needs network on open) or
            ``"inline"`` (fully offline report, ~4.4 MB heavier).
    """
    _ = benchmark  # reserved for future benchmark overlay support
    _ = dpi  # kept for backward compatibility (was the PNG export dpi)
    strategy_name = title or auto_title(result, "Backtest")
    metrics = result.metrics if hasattr(result, "metrics") else {}
    ts = metrics.get("trade_stats", {})
    dates, _vals = equity_with_dates(result)

    date_start = str(dates[0])[:10] if len(dates) > 0 else "?"
    date_end = str(dates[-1])[:10] if len(dates) > 0 else "?"

    # ── Generate interactive chart divs ────────────────────────────
    with theme_context():
        # show=False on every panel: these are embedded as divs in the page
        # below, so the auto-show default would open 8 stray windows.
        div_summary = _div(summary(result, show=False), height=520)
        div_dd = _div(drawdown(result, show=False), height=210)
        div_annual = _div(annual_returns(result, show=False), height=330)
        div_monthly = _div(monthly_returns(result, show=False), height=340)
        div_hist = _div(returns_histogram(result, show=False), height=340)
        div_sharpe = _div(rolling_sharpe(result, show=False), height=300)
        div_vol = _div(rolling_volatility(result, show=False), height=300)
        div_var = _div(var_chart(result, show=False), height=340)

    # ── Metrics ───────────────────────────────────────────────────
    ret = metrics.get("total_return", 0)

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

    # ── plotly.js include ─────────────────────────────────────────
    if plotlyjs == "inline":
        # get_plotlyjs lives in plotly.offline. plotly.io has never exposed it
        # (checked on 5.24 and 6.9), so the old plotly.io lookup raised
        # AttributeError on every inline report rather than embedding anything.
        from plotly.offline import get_plotlyjs
        plotly_js_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_js_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'

    # ── Assemble HTML ─────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(strategy_name)} · Tearsheet</title>
<style>{_CSS}</style>
{plotly_js_tag}
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
    {div_annual}
  </div>
  <div class="charts-stack">
    {div_summary}
    {div_dd}
  </div>
</div>

<div class="chart-grid">
  {div_monthly}
  {div_hist}
</div>

<div class="chart-grid">
  {div_sharpe}
  {div_vol}
</div>

<div class="chart-grid">
  {div_var}
</div>

</div>
</body>
</html>"""

    # ── Save / Show ───────────────────────────────────────────────
    if save is not None:
        Path(save).write_text(html, encoding="utf-8")

    # A report is an HTML page, not a Figure: it always opens in a browser
    # tab, so "inline" resolves to the same thing here.
    if resolve_show(show, save):
        if save is not None:
            report_path = Path(save).resolve()
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(html)
            tmp.close()
            report_path = Path(tmp.name).resolve()
        webbrowser.open(report_path.as_uri())

    return html


def research_report(
    sweep_result: Optional[Dict[str, Any]] = None,
    wf_result: Optional[Dict[str, Any]] = None,
    stability_result: Optional[Dict[str, Any]] = None,
    *,
    title: str = "Research Report",
    figsize: tuple = (14, 6),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> List[Any]:
    """Research report - one figure per analysis (plotly Figures)."""
    from manifoldbt.plot.research import (
        heatmap_2d,
        stability,
        walk_forward,
    )

    _ = title
    figs = []
    with theme_context():
        # show=False: this function does its own showing at the end.
        if sweep_result is not None:
            figs.append(heatmap_2d(sweep_result, figsize=figsize, show=False))
        if wf_result is not None:
            figs.append(walk_forward(wf_result, figsize=figsize, show=False))
        if stability_result is not None:
            figs.append(stability(stability_result, figsize=figsize, show=False))

    if not figs:
        raise ValueError("At least one result (sweep, wf, or stability) required.")

    if save is not None:
        path = Path(save)
        stem, suffix = path.stem, path.suffix or ".html"
        for i, f in enumerate(figs):
            out = path.parent / f"{stem}_{i + 1}{suffix}"
            if suffix.lower() == ".html":
                from manifoldbt.plot._utils import write_responsive_html
                write_responsive_html(f, out)
            else:
                scale = max(1.0, dpi / 96.0)
                f.write_image(str(out), scale=scale)
    if resolve_show(show, save):
        for f in figs:
            f.show()

    return figs


# ── Internal ─────────────────────────────────────────────────────────────────


def _fmt_hold_time(seconds):
    """Format holding time in human-readable units."""
    if seconds <= 0:
        return "-"
    days = seconds / 86400
    if days >= 365:
        return f"{days / 365:.1f}y"
    if days >= 30:
        return f"{days / 30:.1f}mo"
    if days >= 1:
        return f"{days:.0f}d"
    hours = seconds / 3600
    return f"{hours:.0f}h"
