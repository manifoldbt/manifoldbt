"""SweepResult — ergonomic wrapper for parameter sweep results."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

from manifoldbt.dataframe import results_to_df
from manifoldbt.result import Result


def _axis_name(column: str) -> str:
    """The axis a labelled sweep column names, without its prefix."""
    return column.split("_", 1)[1] if "_" in column else column


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

    __slots__ = ("_results", "_param_grid", "_execution_grid")

    def __init__(
        self,
        results: Sequence[Any],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        execution_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        self._results = [
            r if isinstance(r, Result) else Result(r)
            for r in results
        ]
        self._param_grid = param_grid or {}
        self._execution_grid = execution_grid or {}

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
            DataFrame with one row per combination, in the order the engine
            ran them: axes sorted by parameter name, last axis varying fastest
            (not the dict's insertion order), and the execution axes slower
            than all of them. Parameter columns are prefixed with ``param_``,
            execution columns with ``exec_``, and both are read from each run's
            own manifest -- so a row's label is the combination that produced
            it, whatever order the results came back in.
        """
        return results_to_df(
            self._results,
            self._param_grid,
            backend=backend,
            execution_grid=self._execution_grid,
        )

    def best(self, metric: str = "sharpe") -> Result:
        """Return the Result with the highest value for *metric*.

        A NaN is no value: a run shorter than two days reports its Sharpe,
        Sortino, CAGR and other daily statistics as NaN, and such a result is
        never returned as the best (nor as the worst). When every result holds
        NaN, a ``ValueError`` says so.

        Args:
            metric: Metric name (e.g. ``"sharpe"``, ``"total_return"``, ``"sortino"``).
        """
        return self._extremum(metric, maximize=True)

    def worst(self, metric: str = "sharpe") -> Result:
        """Return the Result with the lowest value for *metric*.

        Args:
            metric: Metric name. NaN values are skipped, as in :meth:`best`.
        """
        return self._extremum(metric, maximize=False)

    def _extremum(self, metric: str, maximize: bool) -> Result:
        best_val = None
        best_result = None
        found = False
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
            found = True
            # NaN compares false both ways: a NaN first in line used to keep
            # the lead against every number after it.
            if val != val:
                continue
            if best_val is None or (val > best_val if maximize else val < best_val):
                best_val = val
                best_result = r
        if not found:
            raise ValueError(f"Metric {metric!r} not found in any result")
        if best_result is None:
            raise ValueError(
                f"Metric {metric!r} is NaN in every result, so none is the best. "
                "A run shorter than two days reports no metric taken from daily "
                "returns (sharpe, sortino, volatility, cagr, calmar, ...); see "
                "each result's warnings. Rank on total_return or max_drawdown, "
                "or run over at least two days."
            )
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
        # Execution axes are axes too: a surface of service against a queue
        # depth is the whole point of sweeping one, and reading only param_
        # columns would draw it as a bar chart of unlabelled runs.
        param_cols = [c for c in df.columns if c.startswith(("param_", "exec_"))]
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
                "x_param": _axis_name(x_col),
                "y_param": _axis_name(y_col),
                "metric": metric,
            }
            return heatmap_2d(sweep_result, show=show, save=save, **kwargs)

        import plotly.graph_objects as go

        with theme_context():
            if len(param_cols) == 1:
                p_col = param_cols[0]
                fig = new_figure(kwargs.pop("figsize", (10, 5)),
                                 f"{metric} by {_axis_name(p_col)}")
                fig.add_trace(go.Bar(
                    x=[str(v) for v in df[p_col].values], y=df[metric].values,
                    marker_color=ACCENT, marker_line_width=0,
                ))
                fig.update_xaxes(title_text=_axis_name(p_col),
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
        axes = [f"{k}={len(v)} vals" for k, v in self._param_grid.items()]
        axes += [f"{k}={len(v)} vals" for k, v in self._execution_grid.items()]
        return f"SweepResult({len(self)} runs, {', '.join(axes)})"
