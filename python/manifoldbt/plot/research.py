"""Charts for research analysis results (sweep, walk-forward, stability) - plotly."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from manifoldbt.plot._theme import (
    ACCENT,
    BORDER,
    CS_CORRELATION,
    CS_SEQUENTIAL,
    DARK_GRAY,
    GRAY,
    MONO_FAMILY,
    ORANGE,
    WHITE,
    theme_context,
)
from manifoldbt.plot._convert import daily_returns_array, equity_with_dates
from manifoldbt.plot._utils import finalize, new_figure


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _extract_val(v):
    """Extract numeric values from ScalarValue dicts like {'Float64': 1.23}."""
    if isinstance(v, dict):
        for val in v.values():
            return val
    return v


def _grid_window_size(nx: int, ny: int, plot: int = 720, cbar: int = 160,
                      top: int = 70) -> tuple:
    """Window size matching the grid aspect (square grid -> square-ish window)."""
    if nx >= ny:
        pw, ph = plot, plot * ny / max(nx, 1)
    else:
        pw, ph = plot * nx / max(ny, 1), plot
    return (int(pw + cbar), int(ph + top))


def _moving_average_1d(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Edge-replicated moving average of window 2*radius+1 along axis (numpy)."""
    if radius < 1:
        return a
    pad = [(radius, radius) if ax == axis else (0, 0) for ax in range(a.ndim)]
    padded = np.pad(a, pad, mode="edge")
    cumsum = np.cumsum(padded, axis=axis)
    zero = np.zeros_like(np.take(cumsum, [0], axis=axis))
    cumsum = np.concatenate([zero, cumsum], axis=axis)
    n = a.shape[axis]
    width = 2 * radius + 1
    upper = np.take(cumsum, np.arange(width, width + n), axis=axis)
    lower = np.take(cumsum, np.arange(0, n), axis=axis)
    return (upper - lower) / width


def _box_blur_2d(a: np.ndarray, sigma_y: float, sigma_x: float, passes: int = 3) -> np.ndarray:
    """Separable box blur that approximates a Gaussian (central-limit theorem),
    pure numpy. A scipy-free fallback for _plateau_best."""
    out = a.astype(float)
    ry, rx = max(1, int(round(sigma_y))), max(1, int(round(sigma_x)))
    for _ in range(passes):
        out = _moving_average_1d(out, ry, axis=0)
        out = _moving_average_1d(out, rx, axis=1)
    return out


def _plateau_best(grid: np.ndarray):
    """Plateau-optimal cell: a blur finds the center of the best stable region,
    not a lucky spike (overfit-resistant). sigma = ~5% of each axis. Uses
    scipy's Gaussian filter when installed, else a pure-numpy box blur so the
    plotting extra needs no scipy."""
    filled = np.nan_to_num(grid, nan=np.nanmin(grid))
    sigma_y = max(1.0, grid.shape[0] * 0.05)
    sigma_x = max(1.0, grid.shape[1] * 0.05)
    try:
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(filled, sigma=(sigma_y, sigma_x))
    except ImportError:
        smoothed = _box_blur_2d(filled, sigma_y, sigma_x)
    return np.unravel_index(np.argmax(smoothed), smoothed.shape)


def _stats_annotation(fig, text: str) -> None:
    """Monospace stats box in the top-right corner."""
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        xanchor="right", yanchor="top", align="left",
        text=text.replace("\n", "<br>"), showarrow=False,
        font=dict(family=MONO_FAMILY, size=11, color=GRAY),
        bgcolor="rgba(17,17,22,0.9)", bordercolor=BORDER, borderwidth=1,
        borderpad=6,
    )


# ── 2D Parameter Sweep Heatmap ──────────────────────────────────────────────


def heatmap_2d(
    sweep_result: Dict[str, Any],
    *,
    ax=None,
    annotate: bool = True,
    fmt: str = ".3f",
    highlight_best: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """2D parameter sweep heatmap from ``run_sweep_2d()`` result.

    Expected keys: metric_grid, x_values, y_values, x_param, y_param, metric.
    """
    with theme_context():
        grid = np.array(sweep_result["metric_grid"], dtype=np.float64)
        x_vals = [_extract_val(v) for v in sweep_result["x_values"]]
        y_vals = [_extract_val(v) for v in sweep_result["y_values"]]
        x_param = sweep_result.get("x_param", "x")
        y_param = sweep_result.get("y_param", "y")
        metric = sweep_result.get("metric", "metric")
        nx, ny = len(x_vals), len(y_vals)

        text = None
        if annotate and nx * ny <= 100:
            text = np.vectorize(lambda v: "" if np.isnan(v) else f"{v:{fmt}}")(grid)

        fig = new_figure(figsize)
        fig.add_trace(go.Heatmap(
            z=grid, x=x_vals, y=y_vals,
            colorscale=CS_SEQUENTIAL,
            text=text, texttemplate="%{text}" if text is not None else None,
            textfont=dict(size=9),
            hovertemplate=(
                f"{x_param} %{{x}}<br>{y_param} %{{y}}<br>"
                f"{metric} %{{z:{fmt}}}<extra></extra>"
            ),
            colorbar=dict(outlinewidth=0, thickness=12),
            hoverongaps=False,
        ))

        best_label = None
        if highlight_best:
            best_idx = _plateau_best(grid)
            best_val = grid[best_idx]
            best_x = x_vals[best_idx[1]]
            best_y = y_vals[best_idx[0]]

            # Cell outline around the plateau-best combo
            dx = (x_vals[1] - x_vals[0]) / 2 if nx > 1 else 0.5
            dy = (y_vals[1] - y_vals[0]) / 2 if ny > 1 else 0.5
            fig.add_shape(
                type="rect",
                x0=best_x - dx, x1=best_x + dx, y0=best_y - dy, y1=best_y + dy,
                line=dict(color="white", width=2.5),
            )
            best_label = f"best: {best_val:{fmt}} ({x_param}={best_x:.0f}, {y_param}={best_y:.0f})"

        combos = nx * ny
        main_title = title or f"{metric} · Parameter Sweep ({combos:,} combos)"
        if best_label:
            main_title = f"{main_title}<br><span style='font-size:11px;color:{GRAY}'>{best_label}</span>"
        fig.update_layout(title_text=main_title, hovermode="closest")
        fig.update_xaxes(title_text=x_param, showspikes=False, constrain="domain")
        fig.update_yaxes(title_text=y_param, showspikes=False)

        # Square cells: lock the y/x pixel ratio to the data spacing so the grid
        # keeps its true aspect (a 100x100 sweep is a square), even on resize.
        if nx > 1 and ny > 1:
            dx = (float(x_vals[-1]) - float(x_vals[0])) / (nx - 1)
            dy = (float(y_vals[-1]) - float(y_vals[0])) / (ny - 1)
            if dx > 0 and dy > 0:
                fig.update_yaxes(scaleanchor="x", scaleratio=dx / dy,
                                 constrain="domain")
        return finalize(fig, show=show, save=save,
                        window_size=_grid_window_size(nx, ny))


# ── 3D Surface Plot ─────────────────────────────────────────────────────────


def surface_3d(
    sweep_result: Dict[str, Any],
    *,
    highlight_best: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    elev: float = 30,
    azim: float = -45,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """3D surface plot from a 2D parameter sweep result.

    Same input format as ``heatmap_2d``. ``elev``/``azim`` are kept for
    backward compatibility and mapped to the plotly camera.
    """
    with theme_context():
        grid = np.array(sweep_result["metric_grid"], dtype=np.float64)
        x_vals = np.array([_extract_val(v) for v in sweep_result["x_values"]], dtype=np.float64)
        y_vals = np.array([_extract_val(v) for v in sweep_result["y_values"]], dtype=np.float64)
        x_param = sweep_result.get("x_param", "x")
        y_param = sweep_result.get("y_param", "y")
        metric = sweep_result.get("metric", "metric")

        fig = new_figure(figsize)
        fig.add_trace(go.Surface(
            x=x_vals, y=y_vals, z=grid,
            colorscale=CS_SEQUENTIAL, opacity=0.98,
            colorbar=dict(title=dict(text=metric, side="right"),
                          outlinewidth=0, thickness=13, len=0.6),
            lighting=dict(ambient=0.75, diffuse=0.5, roughness=0.9, specular=0.1),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True,
                                 width=1)),
            hovertemplate=(
                f"{x_param} %{{x:.2f}}<br>{y_param} %{{y:.2f}}<br>"
                f"{metric} %{{z:.3f}}<extra></extra>"
            ),
        ))

        best_label = None
        if highlight_best:
            best_idx = _plateau_best(grid)
            best_val = grid[best_idx]
            bx = x_vals[best_idx[1]]
            by = y_vals[best_idx[0]]
            fig.add_trace(go.Scatter3d(
                x=[bx], y=[by], z=[best_val], mode="markers",
                marker=dict(color="white", size=6,
                            line=dict(color="black", width=2)),
                name="best", showlegend=False,
                hovertemplate=f"best {metric} %{{z:.3f}}<extra></extra>",
            ))
            best_label = f"best: {best_val:.3f} ({x_param}={bx:.0f}, {y_param}={by:.0f})"

        # Map matplotlib elev/azim to a plotly camera eye position
        r = 1.9
        elev_rad = np.deg2rad(elev)
        azim_rad = np.deg2rad(azim)
        eye = dict(
            x=r * np.cos(elev_rad) * np.cos(azim_rad),
            y=r * np.cos(elev_rad) * np.sin(azim_rad),
            z=r * np.sin(elev_rad),
        )

        combos = len(x_vals) * len(y_vals)
        main_title = title or f"{metric} · Surface ({combos:,} combos)"
        if best_label:
            main_title = f"{main_title}<br><span style='font-size:11px;color:{GRAY}'>{best_label}</span>"

        fig.update_layout(
            title_text=main_title,
            scene=dict(
                xaxis_title=x_param, yaxis_title=y_param, zaxis_title=metric,
                aspectmode="manual",
                aspectratio=dict(x=1.25, y=1.25, z=0.85),
                camera=dict(eye=eye),
            ),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        return finalize(fig, show=show, save=save)


# ── Walk-Forward Analysis ────────────────────────────────────────────────────


def walk_forward(
    wf_result: Dict[str, Any],
    *,
    mode: str = "auto",
    full_result=None,
    ax=None,
    is_color: str = ACCENT,
    oos_color: str = ORANGE,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Walk-forward analysis chart.

    Args:
        mode: ``"auto"`` (equity curves if available, bars otherwise),
              ``"equity"`` (force equity curves), ``"bars"`` (force bar chart),
              ``"stitched"`` (stitched OOS vs full backtest).
        full_result: BacktestResult from ``bt.run()`` on the full period
              (no WFO). Used by ``"stitched"`` mode as the baseline.
              If not provided, stitched mode only shows the OOS curve.
    """
    folds = wf_result["folds"]
    has_equity = any(len(f.get("is_equity", [])) > 0 for f in folds)

    if mode == "auto":
        mode = "equity" if has_equity else "bars"

    if mode == "equity":
        return _walk_forward_equity(wf_result, folds, is_color=is_color,
                                    oos_color=oos_color, title=title, figsize=figsize,
                                    show=show, save=save)
    elif mode == "stitched":
        return _walk_forward_stitched(wf_result, folds, full_result=full_result,
                                      is_color=is_color,
                                      oos_color=oos_color, title=title, figsize=figsize,
                                      show=show, save=save)
    else:
        return _walk_forward_bars(wf_result, folds, is_color=is_color,
                                  oos_color=oos_color, title=title, figsize=figsize,
                                  show=show, save=save)


def _fold_metric(fold, key, optimize_metric):
    val = fold.get(key)
    if isinstance(val, dict):
        return val.get(optimize_metric, val.get("sharpe", 0))
    return val if val is not None else 0


def _walk_forward_equity(wf_result, folds, *, is_color, oos_color, title, figsize, show, save):
    """Equity curve per fold: IS (blue) + OOS (orange) side by side."""
    optimize_metric = wf_result.get("optimize_metric", "sharpe")
    n = len(folds)

    with theme_context():
        fig = make_subplots(
            rows=1, cols=n, horizontal_spacing=0.02,
            subplot_titles=[
                f"Fold {f.get('fold_index', f.get('fold', i)) + 1}"
                for i, f in enumerate(folds)
            ],
        )

        for i, fold in enumerate(folds):
            col = i + 1
            is_eq = fold.get("is_equity", [])
            oos_eq = fold.get("oos_equity", [])

            if is_eq:
                fig.add_trace(go.Scatter(
                    y=is_eq, mode="lines",
                    line=dict(color=is_color, width=1.2), opacity=0.8,
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=col)

            if oos_eq:
                fig.add_trace(go.Scatter(
                    x=list(range(len(is_eq), len(is_eq) + len(oos_eq))), y=oos_eq,
                    mode="lines", line=dict(color=oos_color, width=1.2), opacity=0.8,
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=col)

            if is_eq and oos_eq:
                fig.add_vline(x=len(is_eq), line_color=DARK_GRAY, line_width=0.8,
                              line_dash="dash", row=1, col=col)

            is_m = (_fold_metric(fold, "is_metrics", optimize_metric)
                    or _fold_metric(fold, "is_metric", optimize_metric))
            oos_m = (_fold_metric(fold, "oos_metrics", optimize_metric)
                     or _fold_metric(fold, "oos_metric", optimize_metric))
            fig.add_annotation(
                x=0.04, y=0.96, xref=f"x{col if col > 1 else ''} domain",
                yref=f"y{col if col > 1 else ''} domain",
                xanchor="left", yanchor="top", showarrow=False, align="left",
                text=(f"<span style='color:{is_color}'>IS: {is_m:.2f}</span><br>"
                      f"<span style='color:{oos_color}'>OOS: {oos_m:.2f}</span>"),
                font=dict(family=MONO_FAMILY, size=10),
            )
            if i > 0:
                fig.update_yaxes(showticklabels=False, row=1, col=col)

        fig.update_layout(
            title_text=title or f"Walk-Forward Analysis ({optimize_metric})",
            width=int(figsize[0] * 80), height=int(figsize[1] * 80),
        )
        return finalize(fig, show=show, save=save)


def _walk_forward_bars(wf_result, folds, *, is_color, oos_color, title, figsize, show, save):
    """Grouped bar chart: IS vs OOS metric per fold."""
    optimize_metric = wf_result.get("optimize_metric", "sharpe")

    with theme_context():
        fig = new_figure(figsize)

        is_vals = [(_fold_metric(f, "is_metrics", optimize_metric)
                    or _fold_metric(f, "is_metric", optimize_metric)) for f in folds]
        oos_vals = [(_fold_metric(f, "oos_metrics", optimize_metric)
                     or _fold_metric(f, "oos_metric", optimize_metric)) for f in folds]
        labels = [f"Fold {f.get('fold_index', f.get('fold', i)) + 1}"
                  for i, f in enumerate(folds)]

        fig.add_trace(go.Bar(
            x=labels, y=is_vals, name="In-Sample",
            marker_color=is_color, opacity=0.65, marker_line_width=0,
            text=[f"{v:.2f}" if v != 0 else "" for v in is_vals],
            textposition="outside", textfont=dict(size=10, color=is_color),
        ))
        fig.add_trace(go.Bar(
            x=labels, y=oos_vals, name="Out-of-Sample",
            marker_color=oos_color, opacity=0.65, marker_line_width=0,
            text=[f"{v:.2f}" if v != 0 else "" for v in oos_vals],
            textposition="outside", textfont=dict(size=10, color=oos_color),
        ))
        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.5, line_dash="dash")
        fig.update_layout(
            title_text=title or f"Walk-Forward Analysis ({optimize_metric})",
            barmode="group", hovermode="closest",
            legend=dict(x=0.99, y=0.99, xanchor="right"),
        )
        fig.update_yaxes(title_text=optimize_metric.capitalize())
        fig.update_xaxes(showspikes=False, type="category")
        return finalize(fig, show=show, save=save)


def _walk_forward_stitched(wf_result, folds, *, full_result=None, is_color, oos_color, title, figsize, show, save):
    """Stitched OOS equity vs full backtest.

    - Orange: OOS segments from each fold, chained end-to-end.
      This is the TRUE out-of-sample performance of the WFO strategy.
    - Blue: full backtest with default params over the same period (no WFO).

    If orange ~ blue: no overfitting, WFO adds little.
    If blue >> orange: full backtest is overfitted.
    If orange >> blue: WFO optimization adds real value.
    """
    with theme_context():
        fig = new_figure(figsize)

        # 1. Stitch OOS segments: chain so each starts where previous ended
        stitched = []
        current_val = None
        fold_boundaries = []
        for fold in folds:
            oos_eq = fold.get("oos_equity", [])
            if not oos_eq:
                continue
            oos = np.array(oos_eq, dtype=float)
            if current_val is None:
                stitched.extend(oos.tolist())
                current_val = oos[-1]
            else:
                scale = current_val / oos[0] if oos[0] != 0 else 1.0
                scaled = oos * scale
                stitched.extend(scaled.tolist())
                current_val = scaled[-1]
            fold_boundaries.append(len(stitched))

        if not stitched:
            fig.update_layout(title_text="No OOS equity data available")
            return finalize(fig, show=show, save=save)

        stitched = np.array(stitched)
        x = np.arange(len(stitched))

        # 2. Full backtest equity (if provided)
        if full_result is not None:
            full_eq = np.array(full_result.equity_curve)
            if len(full_eq) > 0:
                indices = np.linspace(0, len(full_eq) - 1, len(stitched), dtype=int)
                full_resampled = full_eq[indices].astype(float)
                if full_resampled[0] != 0:
                    full_resampled = full_resampled * (stitched[0] / full_resampled[0])
                fig.add_trace(go.Scatter(
                    x=x, y=full_resampled, mode="lines",
                    name="Full backtest (default params)",
                    line=dict(color=is_color, width=0.8), opacity=0.4,
                ))

        # 3. Plot stitched OOS on top
        fig.add_trace(go.Scatter(
            x=x, y=stitched, mode="lines",
            name="Walk-forward (stitched OOS)",
            line=dict(color=oos_color, width=1.0), opacity=0.85,
        ))

        # Fold boundaries
        for b in fold_boundaries[:-1]:
            fig.add_vline(x=b, line_color=DARK_GRAY, line_width=0.5,
                          line_dash="dash", opacity=0.3)

        fig.update_layout(
            title_text=title or "Walk-Forward: Stitched OOS vs Full Backtest",
            legend=dict(x=0.01, y=0.99),
        )
        fig.update_xaxes(title_text="Bars")
        fig.update_yaxes(title_text="Equity")
        return finalize(fig, show=show, save=save)


# ── Parameter Stability ─────────────────────────────────────────────────────


def stability(
    stability_result: Dict[str, Any],
    *,
    ax=None,
    line_color: str = ACCENT,
    band_color: str = ACCENT,
    band_alpha: float = 0.15,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Parameter stability chart with mean +/- std shaded bands.

    Expected keys: values, metric_values, mean_metric, std_metric,
                   param_name, metric, stability_score.
    """
    with theme_context():
        fig = new_figure(figsize)

        param_vals = np.array(stability_result["values"], dtype=np.float64)
        metric_vals = np.array(stability_result["metric_values"], dtype=np.float64)
        mean = stability_result["mean_metric"]
        std = stability_result["std_metric"]
        param_name = stability_result.get("param_name", "parameter")
        metric_name = stability_result.get("metric", "metric")
        score = stability_result.get("stability_score", None)

        # ±1σ band
        fig.add_trace(go.Scatter(
            x=param_vals, y=np.full(len(param_vals), mean - std), mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=param_vals, y=np.full(len(param_vals), mean + std), mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor=_rgba(band_color, band_alpha),
            name=f"±1σ: {std:.3f}", hoverinfo="skip",
        ))
        fig.add_hline(y=mean, line_color=band_color, line_width=1.0,
                      line_dash="dash")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=f"Mean: {mean:.3f}",
            line=dict(color=band_color, width=1.0, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=param_vals, y=metric_vals, mode="lines+markers",
            line=dict(color=line_color, width=1.8),
            marker=dict(size=6, color=line_color),
            name=metric_name, showlegend=False,
            hovertemplate=f"{param_name} %{{x}}: %{{y:.3f}}<extra></extra>",
        ))

        t = title or f"{metric_name} Stability"
        if score is not None:
            t += f"  (score: {score:.2f})"
        fig.update_layout(title_text=t,
                          legend=dict(x=0.99, y=0.99, xanchor="right"))
        fig.update_xaxes(title_text=param_name)
        fig.update_yaxes(title_text=metric_name)
        return finalize(fig, show=show, save=save)


# ── Correlation Matrix ───────────────────────────────────────────────────────


def correlation_matrix(
    symbols: List[str],
    matrix: List[List[float]],
    *,
    ax=None,
    annotate: bool = True,
    title: str = "Correlation Matrix",
    figsize: Tuple[float, float] = (8, 7),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Symbol correlation matrix heatmap."""
    with theme_context():
        mat = np.array(matrix, dtype=np.float64)

        fig = new_figure(figsize, title)
        fig.add_trace(go.Heatmap(
            z=mat, x=symbols, y=symbols,
            colorscale=CS_CORRELATION, zmin=-1, zmax=1,
            text=np.round(mat, 2) if annotate else None,
            texttemplate="%{text:.2f}" if annotate else None,
            textfont=dict(size=10),
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(outlinewidth=0, thickness=12),
        ))
        fig.update_yaxes(autorange="reversed", showspikes=False,
                         scaleanchor="x", scaleratio=1, constrain="domain")
        fig.update_xaxes(tickangle=45, showspikes=False, constrain="domain")
        fig.update_layout(hovermode="closest")
        n = len(symbols)
        return finalize(fig, show=show, save=save,
                        window_size=_grid_window_size(n, n))


# ── Fan chart internals (Monte Carlo + stochastic) ──────────────────────────


def _batched_paths_trace(x, paths: np.ndarray, n_sample_paths: int, color: str):
    """All faded sample paths as ONE trace (None-separated) for performance."""
    k = min(n_sample_paths, paths.shape[0])
    if k <= 0:
        return None
    n = paths.shape[1]
    xs = np.empty((k, n + 1), dtype=np.float64)
    ys = np.empty((k, n + 1), dtype=np.float64)
    xs[:, :n] = np.asarray(x, dtype=np.float64)
    ys[:, :n] = paths[:k]
    xs[:, n] = np.nan
    ys[:, n] = np.nan
    return go.Scatter(
        x=xs.ravel(), y=ys.ravel(), mode="lines",
        line=dict(color=color, width=0.3), opacity=0.06,
        hoverinfo="skip", showlegend=False, connectgaps=False,
    )


def _fan_bands(fig, x, pct_lines, percentiles, band_color):
    """Fill between symmetric percentile bands."""
    for lo, hi in [(0, -1), (1, -2)]:
        if lo < len(percentiles) and abs(hi) <= len(percentiles):
            fig.add_trace(go.Scatter(
                x=x, y=pct_lines[percentiles[lo]], mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x, y=pct_lines[percentiles[hi]], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor=_rgba(band_color, 0.08),
                hoverinfo="skip", showlegend=False,
            ))


# ── Monte Carlo Fan ──────────────────────────────────────────────────────────


def monte_carlo(
    result,
    *,
    n_simulations: int = 1000,
    method: str = "bootstrap",
    percentiles: Optional[List[int]] = None,
    n_sample_paths: int = 50,
    ax=None,
    median_color: str = ACCENT,
    band_color: str = ACCENT,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    seed: Optional[int] = None,
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Monte Carlo fan chart with percentile bands, sample paths, and risk stats.

    Args:
        result: BacktestResult from ``bt.run()``.
        n_simulations: Number of simulated paths.
        method: ``"bootstrap"`` (sample with replacement, default) for tail risk
            estimation, or ``"permutation"`` (shuffle without replacement) for
            path-dependency testing.
        percentiles: Percentile levels for bands. Default ``[5, 25, 50, 75, 95]``.
        n_sample_paths: Number of individual paths to draw (faded). 0 to disable.
        seed: Random seed for reproducibility.
    """
    # Cap to 1000 sims for Community
    try:
        from manifoldbt import _license_info, _warn_pro
        tier, _ = _license_info()
        if tier != "Pro" and n_simulations > 1000:
            _warn_pro(f"Monte Carlo capped to 1,000 sims (requested {n_simulations:,})")
            n_simulations = 1000
    except Exception:
        if n_simulations > 1000:
            n_simulations = 1000

    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    if title is None:
        method_label = "bootstrap" if method == "bootstrap" else "permutation"
        title = f"Monte Carlo - {n_simulations:,} paths ({method_label})"

    with theme_context():
        fig = new_figure(figsize, title)

        rets = daily_returns_array(result)
        _, orig_equity = equity_with_dates(result)

        if len(rets) < 2:
            fig.update_layout(title_text=title + " (insufficient data)")
            return finalize(fig, show=show, save=save)

        rng = np.random.default_rng(seed)
        initial = orig_equity[0] if len(orig_equity) > 0 else 1.0
        n_days = len(rets)

        # Generate simulated paths
        paths = np.zeros((n_simulations, n_days + 1))
        paths[:, 0] = initial
        for i in range(n_simulations):
            if method == "permutation":
                sampled = rng.permutation(rets)
            else:  # bootstrap (default)
                sampled = rng.choice(rets, size=n_days, replace=True)
            paths[i, 1:] = initial * np.cumprod(1.0 + sampled)

        x = np.arange(n_days + 1)
        pct_lines = {pct: np.percentile(paths, pct, axis=0) for pct in percentiles}

        sample_trace = _batched_paths_trace(x, paths, n_sample_paths, band_color)
        if sample_trace is not None:
            fig.add_trace(sample_trace)
        _fan_bands(fig, x, pct_lines, percentiles, band_color)

        # Original equity (dashed), resampled to MC daily resolution
        if len(orig_equity) > n_days * 2:
            indices = np.linspace(0, len(orig_equity) - 1, n_days + 1, dtype=int)
            orig_resampled = np.array(orig_equity)[indices]
        else:
            orig_resampled = np.array(orig_equity[:n_days + 1])
        fig.add_trace(go.Scatter(
            x=np.arange(len(orig_resampled)), y=orig_resampled, mode="lines",
            name="Original", line=dict(color="#e8e9ed", width=0.8, dash="dash"),
            opacity=0.4,
        ))

        running_peak = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_peak) / running_peak
        max_dd_per_path = drawdowns.min(axis=1) * 100

        if method == "bootstrap":
            for pct in percentiles:
                ret_pct = (pct_lines[pct][-1] / initial - 1) * 100
                if pct == 50:
                    fig.add_trace(go.Scatter(
                        x=x, y=pct_lines[pct], mode="lines",
                        name=f"P{pct} (median): {ret_pct:+.1f}%",
                        line=dict(color=median_color, width=2),
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=x, y=pct_lines[pct], mode="lines",
                        name=f"P{pct}: {ret_pct:+.1f}%",
                        line=dict(color=band_color, width=0.5), opacity=0.4,
                    ))

            dd_p5 = np.percentile(max_dd_per_path, 5)
            dd_p50 = np.percentile(max_dd_per_path, 50)
            p_ruin = np.mean((paths[:, -1] / initial - 1) < -0.5) * 100
            _stats_annotation(fig, (
                f"P(ruin) = {p_ruin:.2f}%\n"
                f"Max DD (P5): {dd_p5:.1f}%\n"
                f"Max DD (median): {dd_p50:.1f}%"
            ))
        else:
            # Permutation: skill vs luck via drawdown rank
            fig.add_trace(go.Scatter(
                x=x, y=pct_lines[50], mode="lines", name="Median path",
                line=dict(color=median_color, width=2),
            ))
            for pct in percentiles:
                if pct != 50:
                    fig.add_trace(go.Scatter(
                        x=x, y=pct_lines[pct], mode="lines", showlegend=False,
                        line=dict(color=band_color, width=0.5), opacity=0.4,
                    ))

            orig_eq = np.array(orig_resampled)
            orig_peak = np.maximum.accumulate(orig_eq)
            orig_max_dd = ((orig_eq - orig_peak) / orig_peak).min() * 100

            dd_p5 = np.percentile(max_dd_per_path, 5)
            dd_p50 = np.percentile(max_dd_per_path, 50)
            dd_p95 = np.percentile(max_dd_per_path, 95)
            dd_rank = np.mean(max_dd_per_path <= orig_max_dd) * 100
            _stats_annotation(fig, (
                f"Realized max DD:  {orig_max_dd:.1f}%\n"
                f"Permuted DD P5:   {dd_p5:.1f}%\n"
                f"Permuted DD P50:  {dd_p50:.1f}%\n"
                f"Permuted DD P95:  {dd_p95:.1f}%\n"
                f"DD rank:          {dd_rank:.0f}th percentile"
            ))

        fig.update_xaxes(title_text="Days")
        fig.update_yaxes(title_text="Equity")
        fig.update_layout(legend=dict(x=0.01, y=0.99, font=dict(size=10)))
        return finalize(fig, show=show, save=save)


# ── Stochastic Simulation Paths ───────────────────────────────────────────


def stochastic_paths(
    result: Dict[str, Any],
    *,
    percentiles: Optional[List[int]] = None,
    n_sample_paths: int = 50,
    ax=None,
    median_color: str = ACCENT,
    band_color: str = ACCENT,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    show: bool = False,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Fan chart for stochastic simulation paths with percentile bands.

    Args:
        result: Dict returned by ``mbt.run_stochastic(..., store_paths=True)``.
            Must contain ``paths`` (flat Arrow array) and ``paths_n_steps``.
        percentiles: Percentile levels for bands. Default ``[5, 25, 50, 75, 95]``.
        n_sample_paths: Number of individual paths to draw (faded). 0 to disable.
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    paths_raw = result.get("paths")
    n_steps = result.get("paths_n_steps")
    n_paths = result.get("n_paths", 0)
    model_name = result.get("model_name", "stochastic")

    if paths_raw is None or n_steps is None:
        raise ValueError(
            "result has no paths data. Run with store_paths=True."
        )

    # Reshape flat Arrow/numpy array -> (n_paths, n_steps+1)
    flat = np.asarray(paths_raw, dtype=np.float64)
    paths = flat.reshape((n_paths, n_steps))

    if title is None:
        title = f"Stochastic simulation - {model_name} ({n_paths:,} paths)"

    with theme_context():
        fig = new_figure(figsize, title)

        x = np.arange(paths.shape[1])
        pct_lines = {pct: np.percentile(paths, pct, axis=0) for pct in percentiles}

        sample_trace = _batched_paths_trace(x, paths, n_sample_paths, band_color)
        if sample_trace is not None:
            fig.add_trace(sample_trace)
        _fan_bands(fig, x, pct_lines, percentiles, band_color)

        s0 = paths[0, 0] if paths.shape[1] > 0 else 100.0
        for pct in percentiles:
            final = pct_lines[pct][-1]
            ret_pct = (final / s0 - 1) * 100
            if pct == 50:
                fig.add_trace(go.Scatter(
                    x=x, y=pct_lines[pct], mode="lines",
                    name=f"P{pct} (median): {ret_pct:+.1f}%",
                    line=dict(color=median_color, width=2),
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=x, y=pct_lines[pct], mode="lines",
                    name=f"P{pct}: {ret_pct:+.1f}%",
                    line=dict(color=band_color, width=0.5), opacity=0.4,
                ))

        # Stats box
        final_prices = paths[:, -1]
        running_peak = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_peak) / running_peak
        max_dd_per_path = drawdowns.min(axis=1) * 100

        dd_p5 = np.percentile(max_dd_per_path, 5)
        dd_p50 = np.percentile(max_dd_per_path, 50)
        mean_ret = (np.mean(final_prices) / s0 - 1) * 100
        _stats_annotation(fig, (
            f"Mean return: {mean_ret:+.1f}%\n"
            f"Max DD (P5): {dd_p5:.1f}%\n"
            f"Max DD (P50): {dd_p50:.1f}%"
        ))

        fig.update_xaxes(title_text="Time steps")
        fig.update_yaxes(title_text="Price")
        fig.update_layout(legend=dict(x=0.01, y=0.99, font=dict(size=10)))
        return finalize(fig, show=show, save=save)
