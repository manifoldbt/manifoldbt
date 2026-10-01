"""Writing your own indicators — the ones the library does not ship.

Demonstrates:
  - an indicator as a plain function returning an `Expr`
  - `scan` for stateful indicators no rolling window can express
  - `param(...)` to make a custom indicator sweepable

Data: shared store — real market data from `data/` (see examples/README.md)

Usage:
    python examples/19_custom_indicators.py

────────────────────────────────────────────────────────────────────────────
THE MENTAL MODEL
────────────────────────────────────────────────────────────────────────────
An indicator here is NOTHING but a Python function returning an `Expr`. An
`Expr` is a *node in a computation graph*: writing `(high + low) / 2` touches
no data — it describes an operation. The whole graph is then compiled and
evaluated **in Rust**, in one vectorised pass. That is why your own indicators
run at the speed of the built-in ones: they end up in the same engine.

The whole `manifoldbt.indicators` library is written this way (`sma` ==
`source.rolling_mean(period)`). So "adding an indicator" means "writing a
function that composes `Expr`s". Three levels, from the common to the rare.
"""

from time import perf_counter

import manifoldbt as mbt
# Base columns (already Exprs) plus a few helpers.
from manifoldbt.indicators import open, high, low, close, volume, sma, rsi, ema
# Low-level bricks: lit (constant), col (column by name), when (if/else),
# scan/s (recursive state), param (sweepable parameter).
from manifoldbt.expr import lit, col, when, scan, s, param
from manifoldbt.helpers import time_range, Slippage, Interval

from _bootstrap import open_store


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1 — COMPOSING PRIMITIVES  (99% of cases)
# ═══════════════════════════════════════════════════════════════════════════
# Combine columns + operators (+ - * /, > < >= & | ~) + Expr methods
# (rolling_mean/std/min/max/median, ewm_mean, zscore, pct_change, diff, lag,
#  rsi, linreg_*, cross_above/below, cumsum, rank, ...). Every call returns
# an Expr, so everything chains.

def awesome_oscillator(fast=5, slow=34):
    """Awesome Oscillator (Bill Williams) — NOT in the library.

        AO = SMA(median price, 5) − SMA(median price, 34),  median = (H+L)/2

    Momentum: positive means buying pressure, negative means selling.
    """
    median_price = (high + low) / 2          # Expr: an operation on 2 columns
    return sma(median_price, fast) - sma(median_price, slow)   # the result Expr


def dist_to_ma_pct(period=20):
    """Distance from price to its moving average, in % — NOT in the library.

    Negative means the price sits BELOW its average (oversold), which makes it
    a natural building block for mean reversion. One line of composition.
    """
    ma = sma(close, period)
    return (close - ma) / ma * 100.0


def intraday_range_pct():
    """Bar range as a % of the close — NOT in the library.

    An instant volatility proxy. Shows that OHLC columns mix freely.
    """
    return (high - low) / close * 100.0


def rsi_zscore(period=14, lookback=365):
    """Standardised RSI: how extreme the RSI is against ITS OWN history.

    Composes a built-in indicator (rsi) with rolling statistics — the same
    pattern used in strategies/rsi_dynamic_alloc.py.
    """
    r = rsi(close, period)
    return (r - r.rolling_mean(lookback)) / r.rolling_std(lookback)


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 2 — `scan`: STATEFUL / RECURSIVE INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
# When today's value depends on YESTERDAY's (recursion) and no rolling window
# suffices, reach for `scan`. It runs as a small scalar VM, entirely in Rust
# (no Python callback per bar).
#
#   scan(state=..., update=..., output=...)
#     • state  : state variables and their initial value (first row)
#     • update : expressions evaluated on every bar, IN ORDER
#                - s.prev("x") = value of "x" on the previous bar
#                - s.var("k")  = value computed earlier WITHIN THE SAME step
#                - an update name matching a state name rewrites that state
#     • output : which variable to emit as the result
#
# Proof that it is enough: the shipped Kalman and GARCH are written with scan
# ALONE (see manifoldbt/indicators.py).

def up_streak():
    """Count of consecutive UP bars — NOT in the library, and impossible with
    a plain rolling window (it needs a counter that resets).

        streak = previous streak + 1  if close > close(-1),  else 0
    """
    is_up = close > close.lag(1)             # boolean Expr (1.0 / 0.0) per bar
    return scan(
        state={"n": lit(0.0)},               # counter seeded at 0
        update={
            # if is_up: prev(n) + 1  else: 0
            "n": when(is_up, s.prev("n") + lit(1.0), lit(0.0)),
        },
        output="n",
    )


def ema_from_scratch(alpha=0.1):
    """A hand-rolled EMA via scan — purely to show the mechanism.
    (EMA is built in: `ema(close, span)`. This one is pedagogical.)

        ema = alpha * close + (1 - alpha) * previous ema
    """
    return scan(
        state={"ema": close},                # seeded with the first close
        update={"ema": lit(alpha) * close + lit(1.0 - alpha) * s.prev("ema")},
        output="ema",
    )


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 3 — THE LIMITS (WORTH KNOWING)
# ═══════════════════════════════════════════════════════════════════════════
# • NO Python callback per bar: `scan` runs in Rust, and you cannot inject a
#   Python function called on every candle (it would be slow). As long as the
#   logic expresses in Expr + when + scan, it works.
# • A GENUINELY new indicator, not expressible that way, needs a new `Expr`
#   variant and its Rust kernel — the contributor path, not the user path.
# • External data (hashrate, funding, sentiment…): `mbt.register_exo(...)`,
#   then `exo("name")` returns an Expr usable like any other column.


# ═══════════════════════════════════════════════════════════════════════════
# BONUS — MAKING YOUR INDICATOR SWEEPABLE
# ═══════════════════════════════════════════════════════════════════════════
# Periods accept `param(...)` in place of an integer. The engine then
# recompiles once per combination and sweeps the grid in parallel, without
# changing a line of the indicator:
#
#   ao = awesome_oscillator(fast=param("fast"), slow=param("slow"))
#   # then, with the grid passed separately (the indicator is unchanged):
#   #   batch = mbt.run_sweep_lite(
#   #       strategy,
#   #       {"fast": [3, 5, 8], "slow": [21, 34, 55]},
#   #       config, store,
#   #   )
#
# (see examples/08_sweep_2d_heatmap.py for the full sweep.)


# ═══════════════════════════════════════════════════════════════════════════
# PUTTING A CUSTOM INDICATOR IN A STRATEGY AND BACKTESTING IT
# ═══════════════════════════════════════════════════════════════════════════
# Using `dist_to_ma_pct` (mean reversion): long when the price sits well below
# its average, out when it has caught up.

dist = dist_to_ma_pct(period=48)             # our custom indicator
streak = up_streak()                         # a second one, exposed too

signal = when(dist < -5.0, 1.0,              # >5% below the MA -> buy the dip
         when(dist > 0.0, 0.0))              # back at the MA -> exit, else hold

strategy = (
    mbt.Strategy.create("custom_indicator_demo")
    .signal("dist_to_ma_%", dist)            # .signal() exposes it in the report
    .signal("up_streak", streak)
    .size(signal)
    .describe("Mean reversion driven by a custom indicator (distance to the MA)")
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2026-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(1),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(allow_short=False, max_position_pct=1.0),
    fees=mbt.FeeConfig.zero(),               # fee-free, for the example
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,                          # >= the longest window used
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    store = open_store()

    t0 = perf_counter()
    result = mbt.run(strategy, config, store)
    print(result.summary())
    print(f"\nElapsed: {perf_counter() - t0:.2f}s")
