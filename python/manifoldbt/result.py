"""Rich Result wrapper for BacktestResult with DataFrame, plotting, and Jupyter support."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from manifoldbt.dataframe import arrow_to_df, arrow_to_series, results_to_df


class Result:
    """Ergonomic wrapper around the Rust ``BacktestResult``.

    Provides DataFrame conversion, pretty summaries, plotting shortcuts,
    and Jupyter rich display while delegating all raw attribute access
    to the underlying Rust object for full backward compatibility.

    Example::

        result = bt.run(strategy, config, store)
        print(result.summary())
        df = result.trades_df()
        result.plot()
    """

    __slots__ = ("_raw", "_per_strategy")

    def __init__(self, raw: Any) -> None:
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_per_strategy", None)

    # ------------------------------------------------------------------
    # Backward-compatible delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    @property
    def raw(self) -> Any:
        """Access the underlying Rust ``BacktestResult`` directly."""
        return self._raw

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def equity_df(self, backend: str = "auto") -> Any:
        """Equity curve as a DataFrame with ``timestamp`` and ``equity`` columns.

        Args:
            backend: ``"pandas"``, ``"polars"``, or ``"auto"``.
        """
        from manifoldbt._convert import equity_with_dates

        dates, values = equity_with_dates(self._raw)

        from manifoldbt.dataframe import _resolve_backend
        backend = _resolve_backend(backend)

        if backend == "pandas":
            import pandas as pd
            return pd.DataFrame({"timestamp": dates, "equity": values})
        if backend == "polars":
            import polars as pl
            return pl.DataFrame({"timestamp": dates.astype("datetime64[ms]"), "equity": values})
        return {"timestamp": dates, "equity": values}

    def trades_df(self, backend: str = "auto") -> Any:
        """Trades as a DataFrame with all trade fields.

        Args:
            backend: ``"pandas"``, ``"polars"``, or ``"auto"``.
        """
        return arrow_to_df(self._raw.trades, backend=backend)

    def round_trips_df(self, backend: str = "auto") -> Any:
        """Round trips as a DataFrame, one row per entry-to-flat cycle.

        Rebuilt from the fill log with the engine's own pairing rules, so
        the row count, win rate and expectancy match ``trade_stats``.
        Columns: symbol_id, entry_timestamp, exit_timestamp, side (1 long,
        2 short), entry_price, exit_price, quantity, fees, pnl, return_pct,
        exit_reason (-1 when still open), holding_seconds, is_open,
        entry_row, exit_row.

        Args:
            backend: ``"pandas"``, ``"polars"``, or ``"auto"``.
        """
        from manifoldbt._trades import round_trips
        from manifoldbt.dataframe import _resolve_backend

        cols = round_trips(self._raw)
        backend = _resolve_backend(backend)
        if backend == "pandas":
            import pandas as pd
            return pd.DataFrame(cols)
        if backend == "polars":
            import polars as pl
            return pl.DataFrame({
                k: (v.astype("datetime64[ms]") if v.dtype.kind == "M" else v)
                for k, v in cols.items()
            })
        return cols

    def positions_df(self, backend: str = "auto") -> Any:
        """Position trace as a DataFrame.

        Args:
            backend: ``"pandas"``, ``"polars"``, or ``"auto"``.
        """
        return arrow_to_df(self._raw.positions, backend=backend)

    def daily_returns_series(self, backend: str = "auto") -> Any:
        """Daily returns as a Series.

        Args:
            backend: ``"pandas"``, ``"polars"``, or ``"auto"``.
        """
        return arrow_to_series(self._raw.daily_returns, name="daily_return", backend=backend)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Pretty-printed performance summary as a formatted string."""
        m = self._raw.metrics
        if not isinstance(m, dict):
            return str(m)

        name = self._raw.manifest.get("strategy_name", "backtest") if isinstance(self._raw.manifest, dict) else "backtest"

        lines = [
            f"Strategy: {name}",
            "-" * 40,
        ]

        _fmt = [
            ("Total Return", "total_return", _pct),
            ("CAGR", "cagr", _pct),
            ("Volatility", "volatility", _pct),
            ("Sharpe", "sharpe", _f2),
            ("Sortino", "sortino", _f2),
            ("Calmar", "calmar", _f2),
            ("Max Drawdown", "max_drawdown", _pct),
            ("Best Day", "best_day", _pct),
            ("Worst Day", "worst_day", _pct),
            ("% Positive Days", "pct_positive_days", _pct),
        ]

        for label, key, fmt in _fmt:
            val = m.get(key)
            if val is not None:
                lines.append(f"  {label:<20s} {fmt(val):>12s}")

        # Trade stats
        ts = m.get("trade_stats")
        if isinstance(ts, dict):
            lines.append("")
            lines.append("  Trades")
            lines.append("  " + "-" * 38)
            _ts_fmt = [
                ("Total", "total_trades", _int),
                ("Win Rate", "win_rate", _pct),
                ("Profit Factor", "profit_factor", _f2),
                ("Expectancy", "expectancy", _f2),
                ("Avg Win", "avg_win", _f4),
                ("Avg Loss", "avg_loss", _f4),
                ("Total Fees", "total_fees", _f2),
            ]
            for label, key, fmt in _ts_fmt:
                val = ts.get(key)
                if val is not None:
                    lines.append(f"    {label:<18s} {fmt(val):>12s}")

            # Signal quality metrics (MAE/MFE)
            sq = ts.get("signal_quality")
            if isinstance(sq, dict):
                lines.append("")
                lines.append("  Signal Quality")
                lines.append("  " + "-" * 38)
                _sq_fmt = [
                    ("Avg MAE", "avg_mae", _pct),
                    ("Avg MFE", "avg_mfe", _pct),
                    ("Edge Ratio", "edge_ratio", _f2),
                    ("Entry Efficiency", "avg_entry_efficiency", _pct),
                    ("Exit Efficiency", "avg_exit_efficiency", _pct),
                ]
                for label, key, fmt in _sq_fmt:
                    val = sq.get(key)
                    if val is not None:
                        lines.append(f"    {label:<18s} {fmt(val):>12s}")

        return "\n".join(lines)

    def profile_summary(self) -> str:
        """Pretty-printed timing breakdown of the backtest execution."""
        p = self._raw.profile
        if not isinstance(p, dict):
            return str(p)

        total_us = p.get("total_us", 0)
        phases = [
            ("Data loading", p.get("data_load_us", 0)),
            ("Alignment", p.get("align_us", 0)),
            ("Signal eval", p.get("signal_eval_us", 0)),
            ("Runtime prep", p.get("runtime_prep_us", 0)),
            ("Simulation", p.get("simulation_us", 0)),
            ("Output build", p.get("output_build_us", 0)),
        ]

        def _fmt_time(us: int) -> str:
            if us >= 1_000_000:
                return f"{us / 1_000_000:.2f}s "
            if us >= 1_000:
                return f"{us / 1_000:.1f}ms"
            return f"{us}us   "

        lines = [
            f"Profile (total: {_fmt_time(total_us)})",
            "-" * 44,
        ]
        for name, us in phases:
            pct = (us / total_us * 100) if total_us > 0 else 0
            bar = "#" * int(pct / 2.5)
            lines.append(f"  {name:<16s} {_fmt_time(us):>9s}  {pct:5.1f}%  {bar}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plotting (delegates to existing plot module)
    # ------------------------------------------------------------------

    def plot(self, kind: str = "tearsheet", **kwargs: Any) -> Any:
        """Plot backtest results.

        Args:
            kind: Chart type: ``"tearsheet"``, ``"equity"``, ``"drawdown"``,
                ``"monthly_returns"``, ``"summary"``, ``"annual_returns"``,
                ``"rolling_sharpe"``, ``"rolling_volatility"``,
                ``"returns_histogram"``, ``"trades"``, ``"trade_pnl"``.
            **kwargs: Forwarded to the underlying plot function.
        """
        from manifoldbt import plot

        dispatch = {
            "tearsheet": plot.tearsheet,
            "equity": plot.equity,
            "drawdown": plot.drawdown,
            "monthly_returns": plot.monthly_returns,
            "summary": plot.summary,
            "annual_returns": plot.annual_returns,
            "rolling_sharpe": plot.rolling_sharpe,
            "rolling_volatility": plot.rolling_volatility,
            "returns_histogram": plot.returns_histogram,
            "trades": plot.trades,
            "trade_pnl": plot.trade_pnl,
        }
        fn = dispatch.get(kind)
        if fn is None:
            raise ValueError(
                f"Unknown plot kind {kind!r}. "
                f"Available: {', '.join(sorted(dispatch))}"
            )
        return fn(self._raw, **kwargs)

    def plot_equity(self, **kwargs: Any) -> Any:
        """Shortcut for ``plot(kind="equity")``."""
        return self.plot("equity", **kwargs)

    def plot_drawdown(self, **kwargs: Any) -> Any:
        """Shortcut for ``plot(kind="drawdown")``."""
        return self.plot("drawdown", **kwargs)

    def plot_monthly_returns(self, **kwargs: Any) -> Any:
        """Shortcut for ``plot(kind="monthly_returns")``."""
        return self.plot("monthly_returns", **kwargs)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, *others: "Result", backend: str = "auto") -> Any:
        """Compare metrics across multiple results as a DataFrame.

        Args:
            *others: Other Result objects to compare with.
            backend: DataFrame backend.

        Returns:
            DataFrame with one row per result and all metrics as columns.
        """
        all_results = [self] + list(others)
        return results_to_df(all_results, backend=backend)

    # ------------------------------------------------------------------
    # Jupyter integration
    # ------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        m = self._raw.metrics
        if not isinstance(m, dict):
            return f"<pre>{self.summary()}</pre>"

        name = self._raw.manifest.get("strategy_name", "backtest") if isinstance(self._raw.manifest, dict) else "backtest"

        rows_html = []
        _fmt = [
            ("Total Return", "total_return", _pct),
            ("CAGR", "cagr", _pct),
            ("Sharpe", "sharpe", _f2),
            ("Sortino", "sortino", _f2),
            ("Max Drawdown", "max_drawdown", _pct),
            ("Volatility", "volatility", _pct),
            ("Calmar", "calmar", _f2),
        ]
        for label, key, fmt in _fmt:
            val = m.get(key)
            if val is not None:
                rows_html.append(f"<tr><td><b>{label}</b></td><td style='text-align:right'>{fmt(val)}</td></tr>")

        ts = m.get("trade_stats")
        if isinstance(ts, dict):
            for label, key, fmt in [("Trades", "total_trades", _int), ("Win Rate", "win_rate", _pct), ("Profit Factor", "profit_factor", _f2)]:
                val = ts.get(key)
                if val is not None:
                    rows_html.append(f"<tr><td><b>{label}</b></td><td style='text-align:right'>{fmt(val)}</td></tr>")

        return (
            f"<div style='font-family:monospace;max-width:400px'>"
            f"<h4 style='margin:0 0 8px 0'>{name}</h4>"
            f"<table style='border-collapse:collapse;width:100%'>"
            f"{''.join(rows_html)}"
            f"</table></div>"
        )

    def __repr__(self) -> str:
        m = self._raw.metrics
        sharpe = m.get("sharpe", "?") if isinstance(m, dict) else "?"
        ret = m.get("total_return", "?") if isinstance(m, dict) else "?"
        trades = self._raw.trade_count
        return f"Result(return={_pct(ret) if isinstance(ret, (int, float)) else ret}, sharpe={_f2(sharpe) if isinstance(sharpe, (int, float)) else sharpe}, trades={trades})"


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _pct(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return str(v)
    return f"{v:+.2%}" if v >= 0 else f"{v:.2%}"


def _f2(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return str(v)
    return f"{v:.2f}"


def _f4(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return str(v)
    return f"{v:.4f}"


def _int(v: Any) -> str:
    if isinstance(v, (int, float)):
        return str(int(v))
    return str(v)
