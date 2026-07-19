"""Charts for BacktestResult visualization (plotly)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from manifoldbt.plot._theme import (
    ACCENT,
    ACCENT_ALT,
    DARK_GRAY,
    GRAY,
    GREEN,
    ORANGE,
    RED,
    WHITE,
    theme_context,
)
from manifoldbt.plot._convert import (
    daily_returns_array,
    equity_with_dates,
    positions_arrays,
    trades_arrays,
    _ts_to_int64,
)
from manifoldbt.plot._decimate import maybe_decimate
from manifoldbt.plot._utils import finalize, format_pct, new_figure


def _rgba(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _area_traces(x, y, baseline: float, color: str, *, width: float = 1.5,
                 name: Optional[str] = None, hovertemplate: Optional[str] = None):
    """Line + fill-to-baseline traces, with a vertical gradient when supported."""
    base = go.Scatter(
        x=x, y=np.full(len(x), baseline), mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    )
    kwargs = dict(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=width),
        fill="tonexty",
        name=name, showlegend=name is not None,
        hovertemplate=hovertemplate,
    )
    try:
        line_trace = go.Scatter(
            fillgradient=dict(
                type="vertical",
                colorscale=[[0.0, _rgba(color, 0.0)], [1.0, _rgba(color, 0.22)]],
            ),
            **kwargs,
        )
    except (ValueError, TypeError):  # plotly too old for fillgradient
        line_trace = go.Scatter(fillcolor=_rgba(color, 0.07), **kwargs)
    return [base, line_trace]


# ── Summary (the essential chart) ────────────────────────────────────────────


def summary(
    result,
    *,
    figsize: Tuple[float, float] = (14, 8),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """The essential chart: TWR equity + buy-and-hold benchmark, trade activity.

    Top panel:  TWR-normalized equity curve vs buy-and-hold (close price).
    Middle panel: daily trade count as a bar chart.
    Bottom panel: used margin percentage.
    Metrics displayed in the title line.
    """
    with theme_context():
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.6, 0.2, 0.2],
        )

        dates, eq_vals = equity_with_dates(result)
        metrics = result.metrics if hasattr(result, "metrics") else {}

        # ── TWR equity (normalized to 100) ────────────────────────
        twr_full = eq_vals / eq_vals[0] * 100
        d_dates, twr = maybe_decimate(dates, twr_full)
        fig.add_trace(go.Scatter(
            x=d_dates, y=twr, mode="lines", name="Strategy",
            line=dict(color=ACCENT, width=1.0),
            hovertemplate="%{x|%d %b %Y}  %{y:.1f}<extra>Strategy</extra>",
        ), row=1, col=1)

        # Faint green/red fill vs the 100 baseline
        for clip_lo, clip_hi, color in ((100.0, None, GREEN), (None, 100.0, RED)):
            clipped = np.clip(twr, clip_lo, clip_hi)
            fig.add_trace(go.Scatter(
                x=d_dates, y=np.full(len(d_dates), 100.0), mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=d_dates, y=clipped, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=_rgba(color, 0.04),
                hoverinfo="skip", showlegend=False,
            ), row=1, col=1)

        # ── Benchmark: buy-and-hold from close prices ─────────────
        positions = result.positions
        close_col = positions.column("close")
        close_raw = close_col.to_numpy(zero_copy_only=False) if hasattr(close_col, "to_numpy") else np.array(close_col.to_pylist())
        ts_ns = _ts_to_int64(positions.column("timestamp"))
        _, unique_idx = np.unique(ts_ns, return_index=True)
        unique_idx.sort()
        close_vals = close_raw[unique_idx].astype(np.float64)

        if len(close_vals) > 0 and close_vals[0] > 0:
            benchmark_raw = close_vals / close_vals[0] * 100

            # Vol-adjusted benchmark: scale to same volatility as strategy
            strat_rets = np.diff(twr_full) / twr_full[:-1]
            bench_rets = np.diff(benchmark_raw) / benchmark_raw[:-1]
            strat_vol = np.nanstd(strat_rets)
            bench_vol = np.nanstd(bench_rets)
            if bench_vol > 1e-12:
                adj_rets = bench_rets * (strat_vol / bench_vol)
                benchmark = np.empty_like(benchmark_raw)
                benchmark[0] = 100.0
                benchmark[1:] = 100.0 * np.cumprod(1.0 + adj_rets)
            else:
                benchmark = benchmark_raw

            b_dates, b_vals = maybe_decimate(dates[: len(benchmark)], benchmark)
            fig.add_trace(go.Scatter(
                x=b_dates, y=b_vals, mode="lines", name="Buy & Hold (vol-adj)",
                line=dict(color=GRAY, width=1.0), opacity=0.7,
                hovertemplate="%{x|%d %b %Y}  %{y:.1f}<extra>Buy & Hold</extra>",
            ), row=1, col=1)

        fig.add_hline(y=100, line_color=DARK_GRAY, line_width=0.4, row=1, col=1)
        twr_min, twr_max = float(np.nanmin(twr_full)), float(np.nanmax(twr_full))
        twr_range = max(twr_max - twr_min, 0.1)
        fig.update_yaxes(title_text="TWR (base 100)",
                         range=[twr_min - twr_range * 0.15, twr_max + twr_range * 0.15],
                         row=1, col=1)

        # Header metrics
        ret = metrics.get("total_return", 0)
        sharpe = metrics.get("sharpe", 0)
        mdd = metrics.get("max_drawdown", 0)
        n_trades = metrics.get("total_trades", result.trade_count)
        title = (
            f"Return {ret * 100:+.1f}%"
            f"    Sharpe {sharpe:.2f}"
            f"    Max DD {mdd * 100:.1f}%"
            f"    Trades {n_trades:,}"
        )
        fig.update_layout(title_text=title)

        # ── Adaptive smoothing window ──────────────────────────────
        smooth_label = ""
        if len(dates) >= 2:
            bar_ns = int(dates[1]) - int(dates[0])
            total_ns = int(dates[-1]) - int(dates[0])
            day_ns = 24 * 3_600_000_000_000
            target_ns = min(7 * day_ns, max(day_ns, int(total_ns * 0.05)))
            smooth_window = max(1, target_ns // max(bar_ns, 1))
            smooth_window = min(smooth_window, len(dates))
            smooth_days = round(target_ns / day_ns)
            smooth_label = f" ({smooth_days}d)" if smooth_days >= 1 else ""

        # ── Trade activity (daily trade count) ─────────────────────
        try:
            ta = trades_arrays(result)
            trade_ts = ta.get("execution_timestamp", np.array([], dtype="datetime64[ns]"))
            if len(trade_ts) > 0 and len(dates) >= 2:
                trade_days = trade_ts.astype("datetime64[D]")
                unique_days, day_counts = np.unique(trade_days, return_counts=True)
                day_dates = unique_days.astype("datetime64[ns]")

                fig.add_trace(go.Bar(
                    x=day_dates, y=day_counts, name="Trades/day",
                    marker_color=_rgba(ACCENT_ALT, 0.4), marker_line_width=0,
                    showlegend=False,
                    hovertemplate="%{x|%d %b %Y}  %{y} trades<extra></extra>",
                ), row=2, col=1)

                # Rolling 7-day average overlay
                eq_days = dates.astype("datetime64[D]")
                unique_eq_days = np.unique(eq_days)
                daily_on_grid = np.zeros(len(unique_eq_days), dtype=np.float64)
                day_map = {d: c for d, c in zip(unique_days, day_counts)}
                for i, d in enumerate(unique_eq_days):
                    daily_on_grid[i] = day_map.get(d, 0)
                win = min(7, len(daily_on_grid))
                if win > 1:
                    kernel = np.ones(win) / win
                    smoothed = np.convolve(daily_on_grid, kernel, mode="same")
                    fig.add_trace(go.Scatter(
                        x=unique_eq_days.astype("datetime64[ns]"), y=smoothed,
                        mode="lines", line=dict(color=ACCENT_ALT, width=1.0),
                        opacity=0.8, showlegend=False, hoverinfo="skip",
                    ), row=2, col=1)
        except Exception:
            pass
        fig.update_yaxes(title_text="Trades/day", row=2, col=1)

        # ── Used margin % (daily) ──────────────────────────────
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

            # Resample to daily (end-of-day snapshot)
            days = used_dates.astype("datetime64[D]")
            unique_days, _ = np.unique(days, return_index=True)
            day_last = np.searchsorted(days, unique_days, side="right") - 1
            daily_used = used[day_last]
            daily_dates = unique_days.astype("datetime64[ns]")

            fig.add_trace(go.Scatter(
                x=daily_dates, y=daily_used, mode="lines",
                line=dict(color=GREEN, width=0.7), opacity=0.8,
                fill="tozeroy", fillcolor=_rgba(GREEN, 0.10),
                showlegend=False,
                hovertemplate="%{x|%d %b %Y}  %{y:.1f}%<extra>Margin</extra>",
            ), row=3, col=1)
        except Exception:
            pass
        fig.update_yaxes(title_text=f"Margin %{smooth_label}", row=3, col=1)

        fig.update_layout(
            width=int(figsize[0] * 80), height=int(figsize[1] * 80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            bargap=0.0,
        )
        return finalize(fig, show=show, save=save)


# ── Equity Curve ─────────────────────────────────────────────────────────────


def equity(
    result,
    *,
    ax=None,
    color: str = ACCENT,
    title: str = "Equity Curve",
    figsize: Tuple[float, float] = (14, 5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Plot the portfolio equity curve over time.

    ``ax`` is accepted for backward compatibility and ignored (plotly backend).
    """
    with theme_context():
        fig = new_figure(figsize, title)
        dates, values = equity_with_dates(result)
        dates, values = maybe_decimate(dates, values)
        fig.add_traces(_area_traces(
            dates, values, float(values.min()), color, width=1.5,
            hovertemplate="%{x|%d %b %Y}   $%{y:,.0f}<extra></extra>",
        ))
        fig.update_yaxes(title_text="Equity")
        fig.update_xaxes(tickformat="%b %Y")
        return finalize(fig, show=show, save=save)


# ── Benchmark Overlay ────────────────────────────────────────────────────────


def benchmark_equity(
    result,
    benchmark: np.ndarray,
    *,
    ax=None,
    strategy_color: str = ACCENT,
    benchmark_color: str = DARK_GRAY,
    normalize: bool = True,
    labels: Tuple[str, str] = ("Strategy", "Buy & Hold"),
    title: str = "Strategy vs Benchmark",
    figsize: Tuple[float, float] = (14, 5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Overlay strategy equity and a benchmark, both normalized to 100."""
    with theme_context():
        fig = new_figure(figsize, title)
        dates, strat_eq = equity_with_dates(result)
        bench = np.asarray(benchmark, dtype=np.float64)
        n = min(len(strat_eq), len(bench))
        strat_eq, bench, dates = strat_eq[:n], bench[:n], dates[:n]

        if normalize and strat_eq[0] != 0 and bench[0] != 0:
            strat_eq = strat_eq / strat_eq[0] * 100
            bench = bench / bench[0] * 100

        d1, s1 = maybe_decimate(dates, strat_eq)
        d2, b1 = maybe_decimate(dates, bench)
        fig.add_trace(go.Scatter(
            x=d1, y=s1, mode="lines", name=labels[0],
            line=dict(color=strategy_color, width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=d2, y=b1, mode="lines", name=labels[1],
            line=dict(color=benchmark_color, width=1.0),
        ))
        fig.update_yaxes(title_text="Normalized" if normalize else "Equity")
        fig.update_xaxes(tickformat="%b %Y")
        fig.update_layout(legend=dict(x=0.01, y=0.99))
        return finalize(fig, show=show, save=save)


# ── Drawdown / Underwater ────────────────────────────────────────────────────


def drawdown(
    result,
    *,
    ax=None,
    color: str = RED,
    title: str = "Drawdown",
    figsize: Tuple[float, float] = (14, 3),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Plot the drawdown as a filled area chart."""
    with theme_context():
        fig = new_figure(figsize, title)
        dates, values = equity_with_dates(result)
        running_max = np.maximum.accumulate(values)
        dd = (values - running_max) / running_max
        dates, dd = maybe_decimate(dates, dd)

        fig.add_trace(go.Scatter(
            x=dates, y=dd, mode="lines",
            line=dict(color=color, width=0.9),
            fill="tozeroy", fillcolor=_rgba(color, 0.25),
            hovertemplate="%{x|%d %b %Y}   %{y:.1%}<extra></extra>",
        ))
        dd_min = float(dd.min()) if len(dd) else -0.01
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%",
                         range=[dd_min * 1.08, 0])
        fig.update_xaxes(tickformat="%b %Y")
        return finalize(fig, show=show, save=save)


# ── Monthly Returns Heatmap ──────────────────────────────────────────────────


def monthly_returns(
    result,
    *,
    ax=None,
    annotate: bool = True,
    title: str = "Monthly Returns (%)",
    figsize: Tuple[float, float] = (12, 5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Monthly returns heatmap (year rows x month columns + annual)."""
    from manifoldbt.plot._theme import CS_DIVERGING

    with theme_context():
        dates, values = equity_with_dates(result)
        ts = dates.astype("datetime64[M]")
        months = np.unique(ts)
        month_returns = {}
        for m in months:
            idx = np.nonzero(ts == m)[0]
            if len(idx) >= 2:
                month_returns[m] = values[idx[-1]] / values[idx[0]] - 1.0

        years = sorted({int(m.astype("datetime64[Y]").astype(int)) + 1970 for m in months})
        grid = np.full((len(years), 13), np.nan)

        for m, ret in month_returns.items():
            y = int(m.astype("datetime64[Y]").astype(int)) + 1970
            mo = int(m.astype("datetime64[M]").astype(int)) % 12
            grid[years.index(y), mo] = ret

        for yi in range(len(years)):
            row = grid[yi, :12]
            valid = row[~np.isnan(row)]
            if len(valid) > 0:
                grid[yi, 12] = np.prod(1.0 + valid) - 1.0

        abs_max = max(np.nanmax(np.abs(grid)), 0.01)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "YTD"]

        text = np.where(np.isnan(grid), "", np.vectorize(lambda v: f"{v * 100:+.1f}" if not np.isnan(v) else "")(grid))

        fig = new_figure(figsize, title)
        fig.add_trace(go.Heatmap(
            z=grid * 100, x=month_labels, y=[str(y) for y in years],
            colorscale=CS_DIVERGING, zmin=-abs_max * 100, zmax=abs_max * 100,
            text=text if annotate else None,
            texttemplate="%{text}" if annotate else None,
            textfont=dict(size=10),
            hovertemplate="%{y} %{x}: %{z:+.2f}%<extra></extra>",
            colorbar=dict(ticksuffix="%", outlinewidth=0, thickness=12),
            hoverongaps=False,
        ))
        # The year labels are strings, but without an explicit type plotly
        # reads them as numbers and interpolates: a single-year backtest drew
        # ticks at 2,022.6 / 2,022.8 / 2023 / 2,023.2 instead of one "2023" row.
        fig.update_yaxes(type="category", autorange="reversed")
        fig.update_xaxes(side="bottom", showspikes=False)
        fig.update_yaxes(showspikes=False)
        fig.update_layout(hovermode="closest")
        return finalize(fig, show=show, save=save)


# ── Annual Returns ───────────────────────────────────────────────────────────


def annual_returns(
    result,
    *,
    ax=None,
    title: str = "Annual Returns",
    figsize: Tuple[float, float] = (10, 4),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Annual returns bar chart with green/red conditional coloring."""
    with theme_context():
        dates, values = equity_with_dates(result)
        years_arr = dates.astype("datetime64[Y]").astype(int) + 1970
        unique_years = sorted(set(years_arr))
        ann_rets = []
        for y in unique_years:
            idx = np.nonzero(years_arr == y)[0]
            ann_rets.append(values[idx[-1]] / values[idx[0]] - 1.0 if len(idx) >= 2 else 0.0)

        fig = new_figure(figsize, title)
        colors = [GREEN if r >= 0 else RED for r in ann_rets]
        fig.add_trace(go.Bar(
            x=[str(y) for y in unique_years], y=ann_rets,
            marker_color=colors, opacity=0.85, marker_line_width=0,
            width=0.5,
            text=[format_pct(r) for r in ann_rets],
            textposition="outside", textfont=dict(color=GRAY, size=11),
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.5)
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(showspikes=False, type="category")
        fig.update_layout(hovermode="closest")
        return finalize(fig, show=show, save=save)


# ── Returns Histogram ────────────────────────────────────────────────────────


def returns_histogram(
    result,
    *,
    ax=None,
    bins: int = 100,
    title: str = "Returns Distribution",
    figsize: Tuple[float, float] = (12, 5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Histogram of daily returns with green/red coloring by sign."""
    with theme_context():
        fig = new_figure(figsize, title)
        rets = daily_returns_array(result)
        if len(rets) == 0:
            fig.update_layout(title_text=title + " (no data)")
            return finalize(fig, show=show, save=save)

        # Clip x-axis to P1-P99 range to avoid empty space from outliers
        p1, p99 = np.percentile(rets, [1, 99])
        margin = (p99 - p1) * 0.3
        xlim = (p1 - margin, p99 + margin)

        counts, bin_edges = np.histogram(rets, bins=bins, range=xlim)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bw = bin_edges[1] - bin_edges[0]
        colors = [GREEN if left >= 0 else RED for left in bin_edges[:-1]]

        fig.add_trace(go.Bar(
            x=centers, y=counts, width=bw,
            marker_color=colors, opacity=0.7, marker_line_width=0,
            hovertemplate="%{x:.2%}: %{y}<extra></extra>",
            # Kept out of the legend: the bars are green or red by sign, so a
            # single swatch misrepresents them, and unnamed it showed up as
            # "trace 0". The legend exists for the Normal overlay only.
            showlegend=False,
        ))
        fig.add_vline(x=0, line_color=DARK_GRAY, line_width=0.8, line_dash="dash")

        # Normal fit (pure numpy)
        mu, sigma = rets.mean(), rets.std()
        if sigma > 0:
            x = np.linspace(xlim[0], xlim[1], 200)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            fig.add_trace(go.Scatter(
                x=x, y=pdf * len(rets) * bw, mode="lines", name="Normal",
                line=dict(color=ACCENT, width=1.0), opacity=0.7,
                hoverinfo="skip",
            ))
            fig.update_layout(legend=dict(x=0.99, y=0.99, xanchor="right"))

        fig.update_xaxes(title_text="Daily Return", tickformat=".1%",
                         range=list(xlim))
        fig.update_yaxes(title_text="Frequency")
        fig.update_layout(hovermode="closest", bargap=0.05)
        return finalize(fig, show=show, save=save)


# ── Value at Risk ────────────────────────────────────────────────────────────


def var_chart(
    result,
    *,
    ax=None,
    confidence: float = 0.05,
    bins: int = 120,
    title: str = "Value at Risk",
    figsize: Tuple[float, float] = (12, 5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Returns histogram with VaR and CVaR lines at 5% and 1% levels."""
    with theme_context():
        fig = new_figure(figsize, title)
        rets = daily_returns_array(result)
        if len(rets) == 0:
            fig.update_layout(title_text=title + " (no data)")
            return finalize(fig, show=show, save=save)

        rets_pct = rets * 100

        # VaR/CVaR at 5% and 1%
        var_5 = float(np.percentile(rets, 5))
        cvar_5 = float(rets[rets <= var_5].mean()) if np.any(rets <= var_5) else var_5
        var_1 = float(np.percentile(rets, 1))
        cvar_1 = float(rets[rets <= var_1].mean()) if np.any(rets <= var_1) else var_1

        counts, bin_edges = np.histogram(rets_pct, bins=bins)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bw = bin_edges[1] - bin_edges[0]
        colors = []
        for left in bin_edges[:-1]:
            if left < var_1 * 100:
                colors.append(_rgba(RED, 0.5))
            elif left < var_5 * 100:
                colors.append(_rgba(ORANGE, 0.4))
            else:
                colors.append(_rgba(ACCENT, 0.5))

        fig.add_trace(go.Bar(
            x=centers, y=counts, width=bw, marker_color=colors,
            marker_line_width=0, showlegend=False,
            hovertemplate="%{x:.2f}%: %{y}<extra></extra>",
        ))

        # VaR/CVaR lines with legend proxies
        for val, color, dash, label in (
            (var_5, ORANGE, None, f"VaR 5%: {format_pct(var_5)}"),
            (cvar_5, ORANGE, "dash", f"CVaR 5%: {format_pct(cvar_5)}"),
            (var_1, RED, None, f"VaR 1%: {format_pct(var_1)}"),
            (cvar_1, RED, "dash", f"CVaR 1%: {format_pct(cvar_1)}"),
        ):
            fig.add_vline(x=val * 100, line_color=color, line_width=0.8,
                          line_dash=dash, opacity=0.8 if dash is None else 0.5)
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines", name=label,
                line=dict(color=color, width=1.2, dash=dash),
            ))

        fig.update_xaxes(title_text="Daily Return (%)")
        fig.update_yaxes(title_text="Frequency")
        fig.update_layout(hovermode="closest", bargap=0.05,
                          legend=dict(x=0.99, y=0.99, xanchor="right"))
        return finalize(fig, show=show, save=save)


# ── Rolling Sharpe ───────────────────────────────────────────────────────────


def rolling_sharpe(
    result,
    *,
    windows: Optional[List[int]] = None,
    ax=None,
    title: str = "Rolling Sharpe",
    trading_days_per_year: float = 365.25,
    figsize: Tuple[float, float] = (14, 4),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Rolling annualized Sharpe ratio."""
    if windows is None:
        windows = [126, 252]
    colors = [ACCENT, ACCENT_ALT, GREEN, RED]

    with theme_context():
        fig = new_figure(figsize, title)
        rets = daily_returns_array(result)

        for i, w in enumerate(windows):
            if len(rets) < w:
                continue
            rm = _rolling(rets, w, np.mean)
            rs = _rolling(rets, w, np.std)
            with np.errstate(divide="ignore", invalid="ignore"):
                sharpe = np.where(rs > 0, rm / rs * np.sqrt(trading_days_per_year), 0.0)
            fig.add_trace(go.Scatter(
                y=sharpe, mode="lines", name=f"{w}d",
                line=dict(color=colors[i % len(colors)], width=1.0),
                hovertemplate="day %{x}: %{y:.2f}<extra>" + f"{w}d" + "</extra>",
            ))

        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.5, line_dash="dash")
        fig.update_yaxes(title_text="Sharpe")
        fig.update_layout(legend=dict(x=0.01, y=0.99))
        return finalize(fig, show=show, save=save)


# ── Rolling Volatility ──────────────────────────────────────────────────────


def rolling_volatility(
    result,
    *,
    windows: Optional[List[int]] = None,
    ax=None,
    title: str = "Rolling Volatility",
    trading_days_per_year: float = 365.25,
    figsize: Tuple[float, float] = (14, 4),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Rolling annualized volatility."""
    if windows is None:
        windows = [126, 252]
    colors = [ACCENT, ACCENT_ALT, GREEN, RED]

    with theme_context():
        fig = new_figure(figsize, title)
        rets = daily_returns_array(result)

        for i, w in enumerate(windows):
            if len(rets) < w:
                continue
            rs = _rolling(rets, w, np.std)
            vol = rs * np.sqrt(trading_days_per_year)
            fig.add_trace(go.Scatter(
                y=vol, mode="lines", name=f"{w}d",
                line=dict(color=colors[i % len(colors)], width=1.0),
                hovertemplate="day %{x}: %{y:.1%}<extra>" + f"{w}d" + "</extra>",
            ))

        fig.update_yaxes(title_text="Volatility", tickformat=".0%")
        fig.update_layout(legend=dict(x=0.01, y=0.99))
        return finalize(fig, show=show, save=save)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _rolling(arr: np.ndarray, window: int, func) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(window - 1, len(arr)):
        out[i] = func(arr[i - window + 1 : i + 1])
    return out
