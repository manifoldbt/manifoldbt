"""Entry orders — resting an entry at a price instead of taking the close.

By default an entry takes a market fill on the execution bar. This example runs
the same signal four ways so the difference is visible in one place:

    market            fill at the execution bar's close
    limit             wait for a pullback, fill passively (maker, no slippage)
    stop              wait for a breakout, fill through the level (taker + gap)
    limit on a signal rest on a level the DSL computes (here: 1 ATR below close)

Then the same idea on the way out: a stop two ATR from the entry, whose
distance the DSL computes per trade instead of a single percentage.

Demonstrates:
  - market, limit, stop and signal-priced entries, side by side
  - passive fills (maker, no slippage) against aggressive ones
  - an entry resting on a level the DSL computes
  - a stop whose distance comes from a signal, frozen at the entry

Data: shared store — real market data from `data/` (see examples/README.md)

Usage:
    python examples/20_entry_orders.py
"""

from time import perf_counter

import manifoldbt as mbt
from manifoldbt.indicators import atr, close, ema
from manifoldbt.helpers import Interval, Slippage, time_range

from _bootstrap import open_store

# -- Signal -------------------------------------------------------------------
fast = ema(close, 12)
slow = ema(close, 50)
trend = mbt.when(fast > slow, 1.0, 0.0)

# The level a signal-priced entry rests on: one ATR below the close.
pullback = close - atr(14)

# The DISTANCE a signal-priced stop measures, as a percentage of the price:
# two ATR. On this market it runs from roughly 1% to 5% depending on the day,
# which no single `pct` can be.
stop_dist = mbt.lit(2.0) * atr(14) / close * mbt.lit(100.0)


def build(name: str, entry, exit_order=None) -> "mbt.Strategy":
    """The same strategy every time; only the orders change."""
    s = (
        mbt.Strategy.create(name)
        .signal("fast", fast)
        .signal("slow", slow)
        .signal("pullback", pullback)
        .signal("stop_dist", stop_dist)
        .size(trend)
    )
    s = exit_order(s) if exit_order else s.stop_loss(pct=3.0)
    return entry(s) if entry else s


VARIANTS = {
    # Market: no entry order at all. The fast kernel stays available.
    "market": None,
    # Passive: 25 bps below the signal close. Unfilled after 5 bars it expires,
    # and while the signal still asks for the position it is placed again at the
    # level of the bar that reposts it.
    "limit -25bps": lambda s: s.limit_entry(offset_bps=25, time_in_force={"GTB": 5}),
    # Breakout: 25 bps above. Crosses the book, and a gap through it fills at the open.
    "stop +25bps": lambda s: s.stop_entry(offset_bps=-25, time_in_force={"GTB": 5}),
    # Signal-priced: rest on whatever the DSL computed, here close - atr(14).
    "limit @ close-ATR": lambda s: s.limit_entry(signal="pullback", time_in_force={"GTB": 5}),
}

# -- Config -------------------------------------------------------------------
start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(4),
    initial_capital=10_000,
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,
    # A resting order is placed from one bar's signal and gated against the
    # next, so its level has to come from a bar that has already closed. The
    # engine refuses `signal_delay=0` here rather than filling at a price that
    # did not exist when the order was placed.
    execution=mbt.ExecutionConfig(signal_delay=1),
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    store = open_store()

    print(f"{'entry':<20} {'trades':>7} {'return':>9} {'sharpe':>8} {'elapsed':>9}")
    print("-" * 56)

    for label, entry in VARIANTS.items():
        strategy = build(label.replace(" ", "_"), entry)
        t0 = perf_counter()
        result = mbt.run(strategy, config, store)
        elapsed = perf_counter() - t0

        m = result.metrics
        print(
            f"{label:<20} {result.trades.num_rows:>7} "
            f"{m['total_return']:>8.1%} {m['sharpe']:>8.2f} {elapsed:>8.2f}s"
        )

        # A resting entry can simply never fill. That failure mode looks like a
        # clean backtest, so the engine reports it rather than staying silent.
        for w in result.warnings:
            if "unfilled" in w:
                print(f"{'':<20} ! {w}")

    # -- The same market entry, stopped two ways ------------------------------
    # `signal=` reads the distance on the bar whose signal opened the trade and
    # freezes it for that trade, so a volatile entry gets room and a quiet one
    # does not. `brackets_not_armed` counts the trades whose distance the
    # series could not give (a warm-up hole, typically): it must read 0.
    print()
    print(f"{'stop':<20} {'trades':>7} {'return':>9} {'sharpe':>8} {'sl exits':>9}")
    print("-" * 56)

    for label, exit_order in {
        "3% flat": lambda s: s.stop_loss(pct=3.0),
        "2 ATR (signal)": lambda s: s.stop_loss(signal="stop_dist"),
    }.items():
        result = mbt.run(
            build(label.replace(" ", "_"), None, exit_order), config, store)
        m = result.metrics
        print(
            f"{label:<20} {result.trades.num_rows:>7} "
            f"{m['total_return']:>8.1%} {m['sharpe']:>8.2f} "
            f"{m['trade_stats']['sl_exits']:>9}"
        )
        if result.brackets_not_armed:
            print(f"{'':<20} ! {result.brackets_not_armed} trade(s) with no bracket")
