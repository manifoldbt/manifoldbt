"""Tick-level backtesting: what a trade tape answers that OHLCV bars cannot.

A bar carries four prices, not the path between them. Two questions it can only
guess at, and a tape settles:

  * **Which came first.** When a stop and a take-profit both sit inside one
    candle, a bar backtest must assume an order. The tape knows.
  * **What price you actually got.** Fills land on prices that really traded,
    not on a close or a bar-edge approximation.

Everything here reads a Binance ``aggTrades`` CSV: a real dump from
data.binance.vision, or one from :func:`generate_tape` for a reproducible
example. Bar-level backtests are unaffected by this module; it is a separate
layer, not a change to :func:`manifoldbt.run`.

**This module is the research harness, not the product path.** The queue, the
market maker and the per-trade runner here are where those mechanics were
prototyped, on a CSV and outside the engine. They then became engine settings,
and that is where a strategy reaches for them: the same DSL, the same
:func:`manifoldbt.run`, the same result object, on a tape put in the store by
:func:`manifoldbt.ingest_trades` rather than named by a path.

  * ``fill_resolution="ticks"`` -- level orders (stop-loss, take-profit,
    trailing stop, limit and stop entries) resolved against the prints inside
    each bar instead of against its high and low, with
    ``result.tape_resolution`` counting what the tape decided;
  * ``bar_interval=Interval.trades()`` -- the trade clock: one simulation row
    per print;
  * ``fill_model={"queue": ...}`` -- the volume resting ahead of an order
    decides its fill, read from a book stored by
    :func:`manifoldbt.ingest_book`;
  * ``execution.latency`` -- the round trip between a decision and the book;
  * ``execution.fill_marks`` -- what the market did after each fill.

See "Backtesting on the Tape" in the strategy authoring guide for the reading
order, and ``examples/28_trade_clock_market_maker.py`` for all of it on one
day. The same refusal answers every one of these doors today, this module
included.

Three ways to run a strategy on a tape:

  * :func:`run_orderflow` - the built-in aggressor-imbalance strategy,
    vectorized in Rust.
  * :func:`run_market_maker` - passive two-sided quoting with a modelled queue.
  * :func:`run_strategy` - your own Python callable, once per trade.

Plus :func:`simulate_bracket` (and its batch form :func:`simulate_brackets`,
which reads the tape once and resolves every order in parallel, in Rust),
which answers the stop-versus-target question directly, :func:`sweep_orderflow_thr`, which sweeps the entry threshold with
the feature columns computed once, and :func:`tape_to_bars`, which aggregates
a tape into the engine's own bar schema so the same trades can be run through
the bar engine and questioned at tick resolution.

Example::

    import manifoldbt as bt

    bt.ticks.generate_tape("tape.csv", n_ticks=200_000)
    print(bt.ticks.tape_info("tape.csv"))
    print(bt.ticks.run_orderflow("tape.csv", enter_thr=0.35))

.. note::
   This layer is a Researcher feature: a Pro licence does not unlock it, and
   every function then raises ``PermissionError``. ``sweep_orderflow_thr``
   additionally fans out, so it is counted like a bar sweep: one threshold is
   one combination.

.. note::
   The queue in :func:`run_market_maker` is *modelled* from the trade tape: it
   starts every order behind a constant multiple of its own size, so a
   market-making result here is indicative rather than execution-grade. The
   engine's own ``fill_model={"queue": ...}`` reads the depth stored at the
   order's own level and instant instead, which is what a queue is.
"""
from __future__ import annotations

from ._native.ticks import (  # noqa: F401
    generate_tape,
    tape_to_bars,
    run_market_maker,
    run_orderflow,
    run_strategy,
    simulate_bracket,
    simulate_brackets,
    sweep_orderflow_thr,
    tape_info,
)

__all__ = [
    "generate_tape",
    "tape_info",
    "tape_to_bars",
    "run_orderflow",
    "sweep_orderflow_thr",
    "run_market_maker",
    "run_strategy",
    "simulate_bracket",
    "simulate_brackets",
]
