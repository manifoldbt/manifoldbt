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

The bar engine has its own door to a tape, and it is not this module:
``ExecutionConfig(fill_resolution="ticks")`` makes :func:`manifoldbt.run`
resolve level orders (stop-loss, take-profit, trailing stop, limit and stop
entries) against the trades stored for the symbol -- put there by
:func:`manifoldbt.ingest_trades`, not by a CSV path -- instead of against each
bar's high and low, and ``result.tape_resolution`` counts what the tape
decided. Same tier, different entry point.

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
   No licence currently sold enables this layer: every function raises
   ``PermissionError`` today. ``sweep_orderflow_thr`` additionally fans out,
   so it is counted like a bar sweep: one threshold is one combination.

.. note::
   The queue in :func:`run_market_maker` is *modelled* from the trade tape, not
   read from an order book. Order-book depth (L2) is not part of this layer, so
   market-making results are indicative rather than execution-grade.
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
