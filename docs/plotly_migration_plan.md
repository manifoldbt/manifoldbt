# Plan: replace matplotlib with plotly in `manifoldbt.plot`

Branch: `feat/plot-plotly-backend`

## Why

Decision from the plotting benchmark (research/plotting_bench/): plotly is the
single interactive renderer going forward. It has a native Python API (already
an optional dependency and already used by `chart(interactive=True)`), covers
2D and 3D, has no watermark or attribution constraint (MIT), and its rendered
output is markedly more modern than the current matplotlib charts. Every chart
gains crosshair, hover tooltips, wheel-zoom and pan for free, and the tearsheet
upgrades from static base64 PNGs to fully interactive embedded charts.

## Scope

`crates/bt-python/python/manifoldbt/plot/` (2,922 lines, 19 public functions)
plus `sweep.py:plot_metric` and packaging metadata. The Rust side is untouched.

## Public API contract (kept)

Every public function keeps its name, module, required arguments and data
semantics. What changes:

| Aspect | Before | After |
|---|---|---|
| Return type | matplotlib `Figure` | plotly `go.Figure` |
| `show=True` | `plt.show()` window | browser tab (plotly `fig.show()`) |
| `save=` | `.png` via Agg | `.html` (interactive, responsive) or `.png/.svg/.pdf` via kaleido |
| `ax=` param | draw into given Axes | accepted, ignored (deprecation note in docstring) |
| `figsize=` | inches | accepted, mapped to pixels (x80) for the default layout size |
| Theme | rcParams dict | plotly template registered as `manifoldbt` |

Composition changes: `tearsheet()` no longer renders sub-charts through `ax=`;
it embeds each chart's interactive div directly (see below).

## File-by-file

1. `_theme.py` -- keep the palette constants (they are imported across the
   module and by user code). Replace the rcParams THEME with a plotly layout
   template (`go.layout.Template`) using the same colors, fonts and grid alpha.
   `apply_theme()` registers it and sets it as default; `theme_context()` kept
   as a no-op context manager for backcompat. Colorscales `bt_diverging`,
   `bt_sequential`, `bt_correlation` become plain colorscale lists.

2. `_utils.py` -- `finalize(fig, show, save)` routes: `.html` via `write_html`
   (responsive full-window CSS, `displayModeBar: False`), image extensions via
   `write_image` with a clear error if kaleido is missing, `show` via
   `fig.show()`. `get_or_create_ax` replaced by `new_figure(figsize, title)`.
   `format_pct`, `format_currency`, `auto_title` unchanged.

3. `_decimate.py` (new) -- min/max per pixel-column decimation (pure numpy,
   from research/plotting_bench/decimate.py, measured: 1.16 ms at 1M points,
   exact on extremes). Applied to equity/drawdown/benchmark series above
   ~20k points so saved HTML stays light at 1m resolution.

4. `backtest.py` -- port all 10 functions to plotly. Equity gets the gradient
   fill + crosshair look validated in research/plotting_bench/equity/.
   `monthly_returns` becomes `go.Heatmap` with annotations, `annual_returns`
   a colored bar, histograms are prebinned with numpy then drawn as `go.Bar`
   so per-bin green/red coloring is preserved, `summary` is a 3-row
   `make_subplots` with shared x. Rolling charts keep index x (as today).

5. `chart.py` -- `_chart_interactive` (already plotly) becomes the only path;
   `_draw_candles` and the matplotlib branch are deleted. `interactive=` kept
   and ignored. The `n_bars` default can later be raised now that candles are
   vectorized, out of scope here.

6. `research.py` -- `heatmap_2d` and `correlation_matrix` become `go.Heatmap`;
   `surface_3d` becomes `go.Surface` (camera/lighting tuned in
   research/plotting_bench/equity/plot_surface_plotly.py); `walk_forward`
   keeps its three modes on `make_subplots`; `stability` line + band;
   `monte_carlo` / `stochastic_paths` keep their simulation logic (including
   the Community 1,000-sim cap) and render the fan with one batched trace for
   sample paths (None-separated) plus percentile fills and a stats annotation.

7. `tearsheet.py` -- the report keeps its layout and CSS but each chart slot
   embeds `fig.to_html(full_html=False, include_plotlyjs=False)` instead of a
   base64 PNG; plotly.js is included once (param `plotlyjs="cdn"|"inline"`,
   default cdn; inline gives a fully offline report at +4.4 MB).
   `research_report` returns plotly figures and saves `.html` per figure.

8. Packaging and stragglers -- `pyproject.toml`: `plot = ["plotly>=5.0"]`
   (kaleido documented for static export, not forced: it ships Chromium,
   portability-first). `all`/`dev` extras updated. `plot/__init__.py` import
   guard checks plotly. `sweep.py:plot_metric` rewritten to delegate to
   `plot.heatmap_2d` / a plotly bar.

## Testing

Smoke script (scratchpad) renders every public function against a real
backtest (RSI long-only, 10 perps, 2021-2026) and the real 156k sweep grid,
saving `.html` + `.png` for each; PNGs eyeballed before commit. Existing
pytest suite run to catch import regressions.

## Out of scope

Window mode (`--app`) helper, decimation inside the Rust core, raising the
candlestick `n_bars` default, removing matplotlib from the `dev` extra while
other tooling still uses it (bench scripts).
