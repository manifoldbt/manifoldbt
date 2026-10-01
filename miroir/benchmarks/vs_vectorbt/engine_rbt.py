"""raptorbt adapter.

Public API only, and the same shape as the other two: everything that can be
prepared once is prepared before timing, and the timed closure is what a
raptorbt user writes - indicators, entry and exit arrays, one call to
``run_single_backtest``, then reading the headline numbers off the result.

Execution conventions, chosen to line up with the reference
-----------------------------------------------------------
* ``upon_bar_close=True`` fills at the close of the signal bar, which is
  manifoldbt's ``signal_delay=0`` plus ``execution_price="AtClose"``. Measured:
  with it off, the fill lands on that same bar's *open* rather than on the next
  bar, so it is not a delay switch, and there is no setting on this side that
  reproduces a one-bar delay.
* Sizing is left at the default, which takes the whole equity of the bar before
  the entry. Measured against ``position_sizes``: a constant 0.5 buys exactly
  half that equity, so the default is the full-equity fraction manifoldbt calls
  ``FractionOfEquity`` and vectorbt calls ``size_type="percent"``. The account is
  flat at that point, so the previous bar's equity and the equity at the fill
  are the same number and the three engines size identically. None of that is
  taken on trust: ``sma_cross`` comes back bit-identical to manifoldbt's final
  equity, which is what the gate actually checks.
* ``fees`` is a fraction of notional charged on each side, so 5 bps taker on
  both legs is ``fee_bps / 10_000`` (measured against a trade's own ``fees``
  field, which equals ``size * (entry + exit) * fee``).

Signals are levels, not transitions, exactly as on the vectorbt side: entries is
"the condition holds", exits is "it no longer holds". raptorbt opens when flat
and entries is true, which is the target-position semantics of the reference,
with one documented exception after a bracket exit (see ``workloads.py``).

Indicators come from raptorbt's own Rust implementations rather than from numpy,
because that is both what a user would write and what deserves to be timed.
``sma`` and ``rsi`` reproduce the reference to 3.3e-13 and to the last bit
respectively. ``ema`` does not, which is why the workload that needs one is
declared unsupported instead of being quietly run on different numbers.

Brackets, and the unit trap in them
-----------------------------------
``set_fixed_stop`` and ``set_fixed_target`` exist on ``BacktestConfig`` and on
``InstrumentConfig`` both. On 0.9.0 the two spellings agree to the last bit
(verified: same final equity, same round-trips, whether the bracket is set on
one, the other, or both), so the config-level one is used here as the simpler of
the two.

Their unit is a fraction, not a percent: 0.15 is a 15% stop, and the workload's
0.15% is ``0.0015``. Passing the percent number is not an error, it is a stop so
wide that it never triggers, which is a benchmark that quietly measures a
different strategy.

Version floor
-------------
0.9.0 renamed the config and result classes, dropping the ``Py`` prefix they
carried through 0.4.x. Rather than support both, the import fails loudly on an
older install: a silent fallback here would mean publishing a number from an
engine five minor versions behind the one named in the report.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import raptorbt as rbt

from workloads import CAPITAL, WORKLOADS

NAME = "raptorbt"

if not hasattr(rbt, "BacktestConfig"):        # 0.4.x and earlier
    raise ImportError(
        "raptorbt is too old for this harness: no BacktestConfig, which means "
        "a release before 0.9.0. Install the pinned version from "
        "requirements-lock.txt."
    )


def probe() -> Dict[str, Any]:
    # From the installed metadata, not from ``rbt.__version__``: that constant
    # is hand-maintained and has been wrong before (0.4.1 shipped declaring
    # 0.4.0), and a benchmark that misreports which version it measured is worse
    # than one that does not say.
    import importlib.metadata as md

    try:
        version = md.version("raptorbt")
    except Exception:
        version = getattr(rbt, "__version__", "unknown")
    return {"engine": NAME, "version": version}


def _level(key: str, close: np.ndarray) -> np.ndarray:
    """The entry condition, as a level, from raptorbt's own indicators."""
    p = WORKLOADS[key].params
    if key in ("sma_cross", "sma_cross_metrics", "bracket_sl_tp"):
        fast = np.asarray(rbt.sma(close, p["fast"]), dtype=np.float64)
        slow = np.asarray(rbt.sma(close, p["slow"]), dtype=np.float64)
        # A NaN comparison is already False; saying so explicitly keeps the
        # warmup out of the signal by construction rather than by numpy detail.
        return (fast > slow) & ~np.isnan(fast) & ~np.isnan(slow)
    raise KeyError("workload {!r} is not supported by raptorbt".format(key))


def _config(key: str):
    """Everything the engine is told about the run, including the bracket.

    Percentages are handed over as fractions (see the module docstring).
    """
    p = WORKLOADS[key].params
    config = rbt.BacktestConfig(
        initial_capital=CAPITAL,
        fees=float(p.get("fee_bps", 0.0)) / 10_000.0,
        slippage=0.0,
        upon_bar_close=True,
    )
    if "sl_pct" in p:
        config.set_fixed_stop(p["sl_pct"] / 100.0)
    if "tp_pct" in p:
        config.set_fixed_target(p["tp_pct"] / 100.0)
    return config


def prepare(key: str, df, workdir: str | None = None) -> Callable[[], Dict[str, Any]]:
    """Untimed setup; returns the closure the harness times."""
    p = WORKLOADS[key].params
    config = _config(key)
    wants_metrics = bool(p.get("metrics"))

    # Contiguous float64 columns, built once. The other adapters are handed
    # their data ready to use too, so nobody pays for marshalling inside the
    # measurement.
    timestamps = df["timestamp"].astype("int64").to_numpy()
    open_ = np.ascontiguousarray(df["open"].to_numpy(dtype=np.float64))
    high = np.ascontiguousarray(df["high"].to_numpy(dtype=np.float64))
    low = np.ascontiguousarray(df["low"].to_numpy(dtype=np.float64))
    close = np.ascontiguousarray(df["close"].to_numpy(dtype=np.float64))
    volume = np.ascontiguousarray(df["volume"].to_numpy(dtype=np.float64))

    def run() -> Dict[str, Any]:
        level = _level(key, close)
        result = rbt.run_single_backtest(
            timestamps, open_, high, low, close, volume,
            level, ~level,
            config=config,
        )
        m = result.metrics
        out = {
            "total_return": float(m.total_return_pct) / 100.0,
            "final_equity": float(m.end_value),
            "round_trips": int(m.total_closed_trades),
            # raptorbt books round-trips; ``total_trades`` is that same count
            # rather than a fill count, so there is no honest number to report.
            "fills": None,
            "total_fees": float(m.total_fees_paid),
        }
        if wants_metrics:
            # Computed inside the run whether or not anyone reads them, like the
            # reference and unlike vectorbt. That is the point of the workload
            # pair, and it means reading them costs nothing measurable here.
            out.update({
                # Handed over as a positive percentage; the harness works in
                # signed fractions of the account, as every other engine does.
                "max_drawdown": -float(m.max_drawdown_pct) / 100.0,
                "sharpe": float(m.sharpe_ratio),
                "sortino": float(m.sortino_ratio),
                # No volatility in the metrics object. Left absent rather than
                # recomputed off the equity curve: this column is meant to show
                # what the engine hands a user, and filling the gap in would
                # publish the harness's arithmetic as raptorbt's.
                "volatility": None,
            })
        return out

    return run


def diagnose(key: str, df, workdir: str | None = None) -> Dict[str, Any]:
    """Untimed measurement of how far the bracket divergence goes.

    The reference counts the round-trips it opens on an exit bar. The mirror
    image on this side is how many round-trips are missing: raptorbt never
    re-arms while the level holds, so its count is the reference's minus that
    same population. Reporting both makes the claim checkable instead of asking
    a reader to take the subtraction on faith.
    """
    note = WORKLOADS[key].notes.get(NAME)
    if note is None or note.status != "documented":
        return {}
    metrics = prepare(key, df, workdir)()
    return {
        "round_trips": metrics["round_trips"],
        "final_equity": metrics["final_equity"],
    }
