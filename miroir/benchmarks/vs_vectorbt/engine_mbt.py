"""manifoldbt adapter.

Public API only. The timed region is exactly what a user writes:
``bt.run(strategy, config, store)`` plus reading the headline metrics off the
result. Building the store from the DataFrame happens once, before timing, and
is excluded on both sides (vectorbt is likewise handed arrays it does not have
to load).

Execution conventions, chosen to line up with vectorbt rather than to flatter
either engine (same conventions as the parity suite shipped with the library):

* ``signal_delay=0`` and ``execution_price="AtClose"`` -> a market signal fills
  at the close of the signal bar, which is what ``from_signals`` does by default.
* ``warmup_bars=0`` -> the indicator's own NaN warmup is what suppresses early
  signals, identically on both sides.
* ``FractionOfEquity`` sizing is taken at the signal-bar close, which for a
  market entry equals the fill price, so vectorbt ``size_type="percent"`` is the
  matching mode.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict

import data as data_mod
import manifoldbt as bt
from manifoldbt.expr import col, lit, when
from manifoldbt.helpers import Interval, Slippage
from manifoldbt.indicators import close as close_px, ema, rsi, sma

from workloads import CAPITAL, WORKLOADS

NAME = "manifoldbt"


def probe() -> Dict[str, Any]:
    return {"engine": NAME, "version": bt.__version__}


def _config(df, *, sizing: str, fee_bps: float, slippage_bps: float = 0.0,
            universe=None) -> "bt.BacktestConfig":
    last_ns = int(df["timestamp"].iloc[-1].value)
    fees = (
        bt.FeeConfig.zero()
        if fee_bps == 0.0
        else bt.FeeConfig(maker_fee_bps=fee_bps, taker_fee_bps=fee_bps)
    )
    return bt.BacktestConfig(
        universe=universe or [1],
        time_range_start=0,
        # A day past the last bar: the range is inclusive of everything generated.
        time_range_end=last_ns + 86_400_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=0,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            position_sizing_mode=sizing,
        ),
        fees=fees,
        slippage=(Slippage.none() if slippage_bps == 0.0
                  else Slippage.fixed_bps(slippage_bps)),
        warmup_bars=0,
    )


def _strategy(key: str):
    p = WORKLOADS[key].params

    if key in ("sma_cross", "sma_cross_metrics"):
        return (
            bt.Strategy.create(key)
            .signal("fast", sma(close_px, p["fast"]))
            .signal("slow", sma(close_px, p["slow"]))
            .size(when(col("fast") > col("slow"), lit(p["alloc"]), lit(0.0)))
        )

    if key == "bracket_sl_tp":
        return (
            bt.Strategy.create("bracket_sl_tp")
            .signal("fast", sma(close_px, p["fast"]))
            .signal("slow", sma(close_px, p["slow"]))
            .size(when(col("fast") > col("slow"), lit(p["alloc"]), lit(0.0)))
            .stop_loss(pct=p["sl_pct"])
            .take_profit(pct=p["tp_pct"])
        )

    if key in ("sma_cross_costs", "multi_asset"):
        # The same crossover as the headline workload. What differs is the cost
        # model and the number of symbols the config points at, neither of which
        # is visible from the strategy: a universe is walked by the engine, not
        # spelled out per asset, which is the whole point of the comparison.
        return (
            bt.Strategy.create(key)
            .signal("fast", sma(close_px, p["fast"]))
            .signal("slow", sma(close_px, p["slow"]))
            .size(when(col("fast") > col("slow"), lit(p["units"]), lit(0.0)))
        )

    if key == "ema_rsi_fees":
        entry = (
            (col("fast") > col("slow"))
            & (col("rsi") > lit(p["rsi_lo"]))
            & (col("rsi") < lit(p["rsi_hi"]))
        )
        return (
            bt.Strategy.create("ema_rsi_fees")
            .signal("fast", ema(close_px, p["fast"]))
            .signal("slow", ema(close_px, p["slow"]))
            .signal("rsi", rsi(close_px, p["rsi_period"]))
            .signal("entry", entry)
            .size(when(col("entry"), lit(p["units"]), lit(0.0)))
        )

    raise KeyError(f"unknown workload {key!r}")


def prepare(key: str, df, workdir: str) -> Callable[[], Dict[str, Any]]:
    """Untimed setup; returns the closure the harness times."""
    p = WORKLOADS[key].params
    fee_bps = float(p.get("fee_bps", 0.0))
    sizing = "Units" if "units" in p else "FractionOfEquity"

    root = os.path.join(workdir, key)
    os.makedirs(root, exist_ok=True)
    data_root = os.path.join(root, "data")
    metadata_db = os.path.join(root, "metadata.sqlite")
    assets = int(p.get("assets", 1))
    if assets > 1:
        # The universe is derived from the same generator and the same base
        # seed, so the first symbol is the single-asset series bit for bit and
        # the whole set is reproducible from the digest already recorded.
        frames = data_mod.make_universe(len(df), assets)
        for symbol_id, frame in frames.items():
            store = bt.import_dataframe(
                frame, symbol="A%d" % symbol_id, symbol_id=symbol_id,
                interval="1m", data_root=data_root, metadata_db=metadata_db)
        universe = list(frames)
    else:
        store = bt.import_dataframe(
            df, symbol="BENCH", symbol_id=1, interval="1m",
            data_root=data_root, metadata_db=metadata_db)
        universe = [1]
    strategy = _strategy(key)
    config = _config(df, sizing=sizing, fee_bps=fee_bps,
                     slippage_bps=float(p.get("slippage_bps", 0.0)),
                     universe=universe)

    wants_metrics = bool(p.get("metrics"))

    def run() -> Dict[str, Any]:
        result = bt.run(strategy, config, store)
        m = result.metrics
        ts = m.get("trade_stats") or {}
        out = {
            "total_return": float(m["total_return"]),
            "final_equity": CAPITAL * (1.0 + float(m["total_return"])),
            "round_trips": int(ts.get("round_trips", 0)),
            "fills": int(ts.get("total_trades", 0)),
            "total_fees": float(ts.get("total_fees", 0.0)),
        }
        if wants_metrics:
            # Already computed by run(): reading them costs nothing measurable,
            # which is the whole point of the comparison.
            out.update({
                "max_drawdown": float(m["max_drawdown"]),
                "sharpe": float(m["sharpe"]),
                "sortino": float(m["sortino"]),
                "volatility": float(m["volatility"]),
            })
        return out

    return run


def diagnose(key: str, df, workdir: str) -> Dict[str, Any]:
    """Untimed measurement of *how much* a documented divergence actually bites.

    For the bracket workload this counts the round-trips whose entry lands on the
    same bar as the previous exit, which is precisely the population the other
    engines handle differently: vectorbt takes the next bar, raptorbt does not
    re-enter at all. Reporting the count turns "the engines differ" into a number
    a reader can weigh, and it is the same population for both of them.
    """
    if not any(note.status == "documented" for note in WORKLOADS[key].notes.values()):
        return {}

    p = WORKLOADS[key].params
    root = os.path.join(workdir, key + "_diag")
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df, symbol="BENCH", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "metadata.sqlite"),
    )
    result = bt.run(
        _strategy(key),
        _config(df, sizing="Units" if "units" in p else "FractionOfEquity",
                fee_bps=float(p.get("fee_bps", 0.0)),
                slippage_bps=float(p.get("slippage_bps", 0.0))),
        store,
    )
    trades = result.trades_df()
    ts = trades["execution_timestamp"].to_numpy()
    # Fills alternate entry, exit, entry, exit ... An entry that shares a bar
    # with the exit before it is one the other engine would defer by one bar.
    entries, exits = ts[0::2], ts[1::2]
    n = min(len(exits), len(entries) - 1)
    same_bar = int((exits[:n] == entries[1 : n + 1]).sum()) if n > 0 else 0
    stats = result.metrics.get("trade_stats") or {}
    round_trips = int(stats.get("round_trips", 0))
    return {
        "reentries_on_exit_bar": same_bar,
        "round_trips": round_trips,
        "share_of_round_trips": (same_bar / round_trips) if round_trips else 0.0,
        "sl_exits": int(stats.get("sl_exits", 0)),
        "tp_exits": int(stats.get("tp_exits", 0)),
    }
