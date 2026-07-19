"""SweepResult — ergonomic wrapper for parameter sweep results."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

from manifoldbt.dataframe import results_to_df
from manifoldbt.result import Result


class SweepResult:
    """Results from a parameter sweep with DataFrame and analysis shortcuts.

    Wraps a list of ``BacktestResult`` objects returned by ``run_sweep()``
    and provides easy access to metrics, comparison, and plotting.

    Example::

        sweep = bt.run_sweep(strategy, {"fast": [10, 20, 30]}, config, store)
        print(len(sweep))              # 3
        df = sweep.to_df()             # DataFrame with metrics per combo
        best = sweep.best("sharpe")    # Result with highest Sharpe
        sweep.plot_metric("sharpe")    # bar/heatmap chart
    """

    __slots__ = ("_results", "_param_grid")

    def __init__(
        self,
        results: Sequence[Any],
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        self._results = [
            r if isinstance(r, Result) else Result(r)
            for r in results
        ]
        self._param_grid = param_grid or {}

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, idx: int) -> Result:
        return self._results[idx]

    def __iter__(self) -> Iterator[Result]:
        return iter(self._results)

    def to_df(self, backend: str = "auto") -> Any:
        """All results as a DataFrame with metrics and parameter columns.

        Args:
            backend: ``"pandas"``, ``"polars"``, or ``"auto"``.

        Returns:
            DataFrame with one row per parameter combination.
            Parameter columns are prefixed with ``param_``.
        """
        return results_to_df(self._results, self._param_grid, backend=backend)

    def best(self, metric: str = "sharpe") -> Result:
        """Return the Result with the highest value for *metric*.

        Args:
            metric: Metric name (e.g. ``"sharpe"``, ``"total_return"``, ``"sortino"``).
        """
        return self._extremum(metric, maximize=True)

    def worst(self, metric: str = "sharpe") -> Result:
        """Return the Result with the lowest value for *metric*.

        Args:
            metric: Metric name.
        """
        return self._extremum(metric, maximize=False)

    def _extremum(self, metric: str, maximize: bool) -> Result:
        best_val = None
        best_result = None
        for r in self._results:
            m = r.metrics
            val = m.get(metric) if isinstance(m, dict) else None
            # Check nested trade_stats
            if val is None and isinstance(m, dict):
                ts = m.get("trade_stats")
                if isinstance(ts, dict):
                    val = ts.get(metric)
            if val is None:
                continue
            if best_val is None or (val > best_val if maximize else val < best_val):
                best_val = val
                best_result = r
        if best_result is None:
            raise ValueError(f"Metric {metric!r} not found in any result")
        return best_result

    def plot_metric(self, metric: str = "sharpe", **kwargs: Any) -> Any:
        """Plot a metric across sweep results (plotly).

        For 2-parameter sweeps, delegates to ``bt.plot.heatmap_2d``.
        For 1-parameter sweeps, produces a bar chart.

        Args:
            metric: Metric to visualize.
            **kwargs: ``figsize``, ``show``, ``save`` forwarded to the plot.
        """
        from manifoldbt.plot._theme import ACCENT, theme_context
        from manifoldbt.plot._utils import finalize, new_figure

        df = self.to_df(backend="pandas")
        param_cols = [c for c in df.columns if c.startswith("param_")]
        # None = the auto default (show, unless save= or a notebook); both
        # branches below hand it to finalize(), which resolves it.
        show = kwargs.pop("show", None)
        save = kwargs.pop("save", None)

        if len(param_cols) == 2:
            from manifoldbt.plot.research import heatmap_2d

            x_col, y_col = param_cols[0], param_cols[1]
            pivot = df.pivot_table(index=y_col, columns=x_col, values=metric)
            sweep_result = {
                "metric_grid": pivot.values.tolist(),
                "x_values": list(pivot.columns),
                "y_values": list(pivot.index),
                "x_param": x_col.replace("param_", ""),
                "y_param": y_col.replace("param_", ""),
                "metric": metric,
            }
            return heatmap_2d(sweep_result, show=show, save=save, **kwargs)

        import plotly.graph_objects as go

        with theme_context():
            if len(param_cols) == 1:
                p_col = param_cols[0]
                fig = new_figure(kwargs.pop("figsize", (10, 5)),
                                 f"{metric} by {p_col.replace('param_', '')}")
                fig.add_trace(go.Bar(
                    x=[str(v) for v in df[p_col].values], y=df[metric].values,
                    marker_color=ACCENT, marker_line_width=0,
                ))
                fig.update_xaxes(title_text=p_col.replace("param_", ""),
                                 type="category", showspikes=False)
            else:
                fig = new_figure(kwargs.pop("figsize", (10, 5)),
                                 f"{metric} across sweep")
                fig.add_trace(go.Bar(
                    x=list(range(len(df))), y=df[metric].values,
                    marker_color=ACCENT, marker_line_width=0,
                ))
                fig.update_xaxes(title_text="run", showspikes=False)
            fig.update_yaxes(title_text=metric)
            fig.update_layout(hovermode="closest")
            return finalize(fig, show=show, save=save)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={len(v)} vals" for k, v in self._param_grid.items())
        return f"SweepResult({len(self)} runs, {params})"
