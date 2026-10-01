"""vectorbt adapter.

The timed region is what a vectorbt user writes: compute the indicators, derive
the signals, call ``Portfolio.from_signals``, read the headline metrics. The
DataFrame-to-Series conversion happens once, before timing, so neither engine
pays for data marshalling inside the measurement.

Indicator definitions are mirrored on manifoldbt's, not approximated
-------------------------------------------------------------------
* SMA: rolling mean, NaN for the first ``n-1`` bars. Identical by construction.
* EMA: ``alpha = 2/(span+1)``, seeded on the first observation, emitted from the
  first bar. That is exactly ``ewm(span=n, adjust=False)``.
* RSI: Wilder, seeded with the *simple* average of the first ``period`` deltas
  and emitted from bar ``period`` onwards. A plain ``ewm(alpha=1/period)`` over
  the whole delta series is a different indicator (it seeds on the first delta
  and emits from bar 1), which is why the seed is written explicitly here. The
  recursion itself still runs through pandas' C ``ewm``, so this is not a Python
  loop handicapping vectorbt.

Signals are fed as a *level*, not as transitions: ``entries`` is "the condition
holds", ``exits`` is "it no longer holds". With ``accumulate=False`` vectorbt
enters when flat and the level is true, which reproduces manifoldbt's
target-position semantics, including a re-entry after a bracket exit while the
condition is still true.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt

import data as data_mod
from vectorbt.portfolio.enums import StopExitPrice

from workloads import CAPITAL, FREQ, WORKLOADS

NAME = "vectorbt"


def probe() -> Dict[str, Any]:
    return {"engine": NAME, "version": vbt.__version__}


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI with manifoldbt's exact seeding (see module docstring)."""
    values = close.to_numpy(dtype=np.float64)
    delta = np.empty_like(values)
    delta[0] = np.nan
    delta[1:] = values[1:] - values[:-1]

    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    # Seed at index `period` with the simple mean of the first `period` deltas,
    # then let ewm(alpha=1/period) carry Wilder's recursion from there.
    gain[:period] = np.nan
    loss[:period] = np.nan
    gain[period] = np.nanmean(np.where(delta[1 : period + 1] > 0.0, delta[1 : period + 1], 0.0))
    loss[period] = np.nanmean(np.where(delta[1 : period + 1] < 0.0, -delta[1 : period + 1], 0.0))

    alpha = 1.0 / period
    avg_gain = pd.Series(gain, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    avg_loss = pd.Series(loss, index=close.index).ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 -> RSI is 100 by definition (matches the engine).
    return out.where(avg_loss != 0.0, 100.0)


def indicators(key: str, close: pd.Series) -> Dict[str, pd.Series]:
    """Indicator series for a workload. Used by the timed path and by the
    definition check, so the two can never drift apart."""
    p = WORKLOADS[key].params
    if key in ("sma_cross", "sma_cross_metrics", "bracket_sl_tp",
               "sma_cross_costs", "multi_asset"):
        return {
            "fast": close.rolling(p["fast"]).mean(),
            "slow": close.rolling(p["slow"]).mean(),
        }
    if key == "ema_rsi_fees":
        return {
            "fast": close.ewm(span=p["fast"], adjust=False).mean(),
            "slow": close.ewm(span=p["slow"], adjust=False).mean(),
            "rsi": _wilder_rsi(close, p["rsi_period"]),
        }
    raise KeyError(f"unknown workload {key!r}")


def _level(key: str, ind: Dict[str, pd.Series]) -> pd.Series:
    p = WORKLOADS[key].params
    if key in ("sma_cross", "sma_cross_metrics", "bracket_sl_tp",
               "sma_cross_costs", "multi_asset"):
        return (ind["fast"] > ind["slow"]).fillna(False)
    if key == "ema_rsi_fees":
        return (
            (ind["fast"] > ind["slow"])
            & (ind["rsi"] > p["rsi_lo"])
            & (ind["rsi"] < p["rsi_hi"])
        ).fillna(False)
    raise KeyError(f"unknown workload {key!r}")


def prepare(key: str, df, workdir: str | None = None) -> Callable[[], Dict[str, Any]]:
    """Untimed setup; returns the closure the harness times."""
    p = WORKLOADS[key].params
    index = pd.DatetimeIndex(df["timestamp"])
    close = pd.Series(df["close"].to_numpy(dtype=np.float64), index=index)
    open_ = pd.Series(df["open"].to_numpy(dtype=np.float64), index=index)
    high = pd.Series(df["high"].to_numpy(dtype=np.float64), index=index)
    low = pd.Series(df["low"].to_numpy(dtype=np.float64), index=index)

    if "units" in p:
        size, size_type = p["units"], "amount"
    else:
        size, size_type = p["alloc"], "percent"
    fees = float(p.get("fee_bps", 0.0)) / 10_000.0
    slippage = float(p.get("slippage_bps", 0.0)) / 10_000.0
    wants_metrics = bool(p.get("metrics"))
    assets = int(p.get("assets", 1))
    sl = p["sl_pct"] / 100.0 if "sl_pct" in p else None
    tp = p["tp_pct"] / 100.0 if "tp_pct" in p else None

    if assets > 1:
        # One book, not five. `from_signals` on a frame of columns builds five
        # independent portfolios unless it is told otherwise, and five separate
        # books is a different question from the one manifoldbt answers when it
        # walks a universe. `group_by` with `cash_sharing` is the spelling that
        # asks the same thing. With fixed-unit sizing the cash constraint never
        # binds, which is what lets the two agree at all: on a fraction of
        # equity they would also have to agree on which asset gets the cash
        # first, and that is policy rather than arithmetic.
        frames = data_mod.make_universe(len(df), assets)
        closes = pd.DataFrame(
            {"A%d" % sid: f["close"].to_numpy(dtype=np.float64)
             for sid, f in frames.items()},
            index=index,
        )

        def run_multi() -> Dict[str, Any]:
            fast = closes.rolling(p["fast"]).mean()
            slow = closes.rolling(p["slow"]).mean()
            level = (fast > slow).fillna(False)
            portfolio = vbt.Portfolio.from_signals(
                closes, entries=level, exits=~level,
                init_cash=CAPITAL, size=size, size_type=size_type,
                fees=fees, slippage=slippage, direction="longonly",
                accumulate=False, freq=FREQ,
                group_by=True, cash_sharing=True,
            )
            total_return = float(portfolio.total_return())
            trades = portfolio.trades
            return {
                "total_return": total_return,
                "final_equity": CAPITAL * (1.0 + total_return),
                "round_trips": int(trades.closed.count()),
                "fills": None,
                "total_fees": float(trades.records["entry_fees"].sum()
                                    + trades.records["exit_fees"].sum()),
            }

        return run_multi

    def run() -> Dict[str, Any]:
        level = _level(key, indicators(key, close))
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=level,
            exits=~level,
            open=open_,
            high=high,
            low=low,
            init_cash=CAPITAL,
            size=size,
            size_type=size_type,
            fees=fees,
            slippage=slippage,
            sl_stop=sl,
            tp_stop=tp,
            stop_exit_price=StopExitPrice.StopMarket,
            direction="longonly",
            accumulate=False,
            freq=FREQ,
        )
        total_return = float(portfolio.total_return())
        trades = portfolio.trades
        out = {
            "total_return": total_return,
            "final_equity": CAPITAL * (1.0 + total_return),
            "round_trips": int(trades.closed.count()),
            "fills": None,  # vectorbt books round-trips, not individual fills
            "total_fees": float(trades.records["entry_fees"].sum()
                                + trades.records["exit_fees"].sum()),
        }
        if wants_metrics:
            out.update(_summary(portfolio))
        return out

    return run


def _summary(portfolio) -> Dict[str, Any]:
    """The same performance summary manifoldbt returns from every run.

    Written out by hand rather than through ``pf.sharpe_ratio()`` and friends for
    two reasons. First, basis: manifoldbt computes its ratios on *daily* returns
    annualised by sqrt(365) and its drawdown at full bar resolution, while
    vectorbt's accessors annualise at the data's own frequency, so the native
    calls would return different numbers and the comparison would be timing two
    different computations. Second, speed: this version is measurably faster than
    a single native accessor on the same data, so vectorbt is credited with the
    quicker of the two paths available to it.

    The cost that dominates either way is materialising the equity curve, which
    ``from_signals`` defers until a risk metric asks for it.
    """
    equity = portfolio.value()
    drawdown = float((equity / equity.cummax() - 1.0).min())

    daily = equity.resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    annualiser = np.sqrt(365.0)
    deviation = returns.std(ddof=1)
    downside = returns[returns < 0].std(ddof=1)
    mean = returns.mean()
    return {
        "max_drawdown": drawdown,
        "sharpe": float(mean / deviation * annualiser),
        "sortino": float(mean / downside * annualiser),
        "volatility": float(deviation * annualiser),
    }
