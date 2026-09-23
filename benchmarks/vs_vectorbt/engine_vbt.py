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
import divergence
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


def _series(df):
    """The four price Series, marshalled once. Shared with ``diagnose`` so the
    measurement does not read a differently-built frame than the run."""
    index = pd.DatetimeIndex(df["timestamp"])
    return (
        pd.Series(df["close"].to_numpy(dtype=np.float64), index=index),
        pd.Series(df["open"].to_numpy(dtype=np.float64), index=index),
        pd.Series(df["high"].to_numpy(dtype=np.float64), index=index),
        pd.Series(df["low"].to_numpy(dtype=np.float64), index=index),
    )


def _sizing(key: str) -> Dict[str, Any]:
    """Workload params in ``from_signals`` spelling. Shared for the same reason
    ``indicators`` is: two callers, one definition, no way to drift."""
    p = WORKLOADS[key].params
    if "units" in p:
        size, size_type = p["units"], "amount"
    else:
        size, size_type = p["alloc"], "percent"
    return {
        "size": size,
        "size_type": size_type,
        "fees": float(p.get("fee_bps", 0.0)) / 10_000.0,
        "slippage": float(p.get("slippage_bps", 0.0)) / 10_000.0,
        "sl": p["sl_pct"] / 100.0 if "sl_pct" in p else None,
        "tp": p["tp_pct"] / 100.0 if "tp_pct" in p else None,
    }


def _book(close, open_, high, low, level, *, size, size_type, fees, slippage, sl, tp):
    """``from_signals`` in one place. The timed path and the untimed measurement
    call this, so a semantics change reaches both or neither."""
    return vbt.Portfolio.from_signals(
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


def prepare(key: str, df, workdir: str | None = None) -> Callable[[], Dict[str, Any]]:
    """Untimed setup; returns the closure the harness times."""
    p = WORKLOADS[key].params
    close, open_, high, low = _series(df)

    sizing = _sizing(key)
    size, size_type = sizing["size"], sizing["size_type"]
    fees, slippage = sizing["fees"], sizing["slippage"]
    wants_metrics = bool(p.get("metrics"))
    assets = int(p.get("assets", 1))

    if assets > 1:
        index = pd.DatetimeIndex(df["timestamp"])
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
        portfolio = _book(close, open_, high, low, level, **sizing)
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


def diagnose(key: str, df, workdir: str | None = None) -> Dict[str, Any]:
    """Untimed measurement of how far the bracket divergence goes.

    The reference counts the round-trips it opens on an exit bar. raptorbt's
    mirror image is a *missing* population — it does not re-arm, so its count is
    the reference's minus that population exactly. vectorbt's is not missing, it
    is *late*: one order per bar, so the re-entry the reference books at the
    close of the exit bar lands on the bar after, and is lost only when the
    entry level has gone false by the time vectorbt can act on it. Counting the
    population here would say vectorbt diverges an order of magnitude less than
    raptorbt, which is true of the count and false of the money.

    So what is counted is the delay: round-trips entered exactly one bar after
    the previous exit *while the entry level was already true at that exit bar*.
    The qualifier is not decoration. Without it the count also picks up ordinary
    re-entries, where the level went false on the exit bar and true again on the
    next one and both engines enter on the same bar for the same reason — 33 of
    them at 100,000 bars, and they are not divergence.

    Two halves, so the claim is checkable by subtraction rather than on faith:
    ``reentries_deferred`` plus the round-trip delta the harness already
    computes is the reference's own ``reentries_on_exit_bar``. Measured over the
    published matrix, exactly:

        100,000 bars     831 +    94 =    925
        1,000,000 bars  8,238 + 1,092 =  9,330
        10,000,000 bars 83,124 + 10,171 = 93,295

    The third row pairs a measured 83,124 with the reference count and delta from
    the committed CI run; a different manifoldbt build reports 93,281 at that size
    and the identity closes there too.

    ``reentries_on_exit_bar`` is reported here too and is expected to be zero.
    It is this workload's note — "vectorbt processes one order per bar and
    re-enters on the next bar instead" — written as a measurement that can fail,
    rather than as a sentence a reader has to believe.
    """
    note = WORKLOADS[key].notes.get(NAME)
    if note is None or note.status != "documented":
        return {}

    close, open_, high, low = _series(df)
    level = _level(key, indicators(key, close))
    portfolio = _book(close, open_, high, low, level, **_sizing(key))

    # Closed trades only, on both sides. A position still open on the last bar
    # is not a round-trip: it is not counted in `round_trips`, the sentence in
    # the report says "N of them" about that same closed population, and after
    # the one-line repair in `engine_mbt.diagnose` the reference does not count
    # its own unclosed final entry either. Counting it here instead made the
    # rendered line claim 1018 of 2695 closed round-trips at 120,000 bars when
    # only 1017 of them qualified.
    closed = portfolio.trades.closed
    entry_idx = np.asarray(closed.records["entry_idx"])
    exit_idx = np.asarray(closed.records["exit_idx"])

    total_return = float(portfolio.total_return())
    out = {
        "round_trips": int(closed.count()),
        "final_equity": CAPITAL * (1.0 + total_return),
    }
    out.update(divergence.reentry_counts(entry_idx, exit_idx,
                                         level.to_numpy(dtype=bool)))
    return out
