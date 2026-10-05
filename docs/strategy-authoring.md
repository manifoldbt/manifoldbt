# Strategy Authoring Guide

> **manifoldbt** — Python DSL for Declarative Strategy Definition

This guide describes how to define trading strategies using the manifoldbt Python DSL. A strategy is declared once in Python; the whole backtest then runs in the Rust engine.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Indicators](#indicators)
3. [Signals & Sizing](#signals--sizing)
4. [Parameters & Sweeps](#parameters--sweeps)
5. [Backtest Configuration](#backtest-configuration)
6. [Execution Model](#execution-model)
7. [Backtesting on the Tape](#backtesting-on-the-tape)
8. [The Trade Clock](#the-trade-clock)
9. [Order Book Columns](#order-book-columns)
10. [Queue Model](#queue-model)
11. [Latency](#latency)
12. [Execution Models](#execution-models)
13. [A Service Rule of Your Own](#a-service-rule-of-your-own)
14. [Marking a Fill](#marking-a-fill)
15. [Calibrating Against Live Fills](#calibrating-against-live-fills)
16. [Sweeping Execution Parameters](#sweeping-execution-parameters)
17. [Orders, One by One](#orders-one-by-one)
18. [Quoting in the DSL](#quoting-in-the-dsl)
19. [A Market Stored Order by Order](#a-market-stored-order-by-order)
20. [Fee & Slippage Models](#fee--slippage-models)
21. [Orders (SL/TP/Trailing)](#orders-sltptrailing)
22. [Entry Orders](#entry-orders)
23. [Cross-Asset References](#cross-asset-references)
24. [Dataset Auto-Resolution](#dataset-auto-resolution)
25. [Diagnostics](#diagnostics)
26. [Profiling](#profiling)
27. [Complete Examples](#complete-examples)
28. [Indicator Reference](#indicator-reference)

---

## Quick Start

```python
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Indicators
fast = ema(close, 12)
slow = ema(close, 50)

# -- Strategy
strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.5, 0.0))
    .stop_loss(pct=3.0)
)

# -- Config
start, end = time_range("2022-01-01", "2025-01-01")
config = mbt.BacktestConfig(
    universe=[1],
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(12),
    initial_capital=10_000,
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,
)

# -- Run
store = mbt.DataStore(data_root="data", metadata_db="metadata/metadata.sqlite")
result = mbt.run(strategy, config, store)
print(result.summary())
```

---

## Indicators

All indicators are available from `manifoldbt.indicators`. They return `Expr` objects that compose into the expression graph — no data is touched at definition time.

```python
from manifoldbt.indicators import (
    close, open, high, low, volume,  # price columns
    ema, sma, dema, tema, wma, hma, kama,  # moving averages
    rsi, roc, momentum, macd,  # momentum
    bollinger_bands, atr, natr, keltner_channels,  # volatility
    stoch_k, williams_r, cci, adx,  # oscillators
    obv, vwap, mfi,  # volume
    kalman, garch,  # filters
)
```

### Usage

```python
fast = ema(close, 12)          # EMA with span 12
slow = sma(close, 50)          # SMA with window 50
strength = rsi(close, 14)      # RSI with period 14
upper, mid, lower = bollinger_bands(close, period=20, num_std=2.0)
```

### Method chaining

Column expressions (`close`, `high`, etc.) support method chaining:

```python
zscore = close.zscore(60)           # rolling z-score
slope = close.linreg_slope(20)      # linear regression slope
smoothed = close.ewm_mean(12)       # EMA
lagged = close.lag(5)               # 5-bar lag
ret = close.pct_change(1)           # 1-bar return
```

---

### Windows in time

A window is normally a **bar count**: `close.rolling_mean(30)` averages the last
thirty rows. On a grid with holes that is not what it looks like. On an irregular
grid (thin sessions, halted hours, sub-minute bars that print only when
something traded), thirty bars can span thirty seconds on a busy minute and
four minutes on a quiet one -- the same expression measures a different thing
depending on the flow.

Pass an `Interval` instead of an integer and the window is read on the
timestamps:

```python
from manifoldbt.helpers import Interval
# Interval.millis / seconds / minutes / hours / days. `millis` is a duration
# here: no store holds sub-second bars, so a bar run refuses it as `bar_interval`
# by name. Only a strategy that quotes takes it there, as its wake-up clock.

mid = close.rolling_mean(Interval.seconds(30))    # 30 SECONDS, not 30 bars
vol = close.rolling_std(Interval.minutes(5))
lo, hi = close.rolling_min(Interval.seconds(10)), close.rolling_max(Interval.seconds(10))
z = close.zscore(Interval.seconds(30))
```

**Semantics.** The window is `(t - d, t]`: closed on the right, open on the
left, the same convention as `pandas.rolling("30s")`. A row exactly `d` old is
already out. While the series holds less than `d` of history -- that is, while
`t - t_first < d` -- the answer would come from a truncated window, so the output
is NaN, exactly as a bar-count window is NaN over its first `window - 1` rows.
Past that point the values equal `pandas.rolling("30s")` row for row.

A NaN anywhere in the window yields NaN for `rolling_mean`, `rolling_sum`,
`rolling_std`, `zscore` and the pair statistics, while `rolling_min` and
`rolling_max` simply ignore the NaN rows. That is the same rule as the
bar-count versions, so switching a window from bars to seconds changes the
window and nothing else. Timestamps must be non-decreasing; they are on every
grid the engine builds.

**Where a duration is accepted.**

| Operator | Bar count | Duration |
|----------|-----------|----------|
| `rolling_mean`, `rolling_sum`, `rolling_std`, `rolling_min`, `rolling_max` | yes | yes |
| `zscore`, `rolling_var` | yes | yes |
| `rolling_corr`, `rolling_cov`, `rolling_beta` | yes | yes |
| `count_over` | yes | yes |
| `ewm_mean` | `span=` | `halflife=` |
| everything else (`rsi`, `atr`, `rolling_median`, `lag`, pivots, ...) | yes | **refused by name** |

An operator with no time-based implementation refuses a duration and says so,
rather than reading it as a bar count.

**`ewm_mean(halflife=...)`.** `ewm_mean(span=20)` counts bars and is the plain
recurrence. `ewm_mean(halflife=Interval.seconds(2))` decays with the time
elapsed between two rows: a row `dt` old weighs `0.5 ** (dt / halflife)`. It
matches `pandas.ewm(halflife="2s", times=...).mean()`, the weighted-average
form, which is the only one pandas offers on an irregular axis. Having no
window it has no warmup either -- the value exists from the first row, like the
span form.

**Two operators that only exist in time.**

```python
from manifoldbt import indicators as ind

quiet_for = ind.time_since(volume > 1000)        # SECONDS since it was last true
printed   = ind.count_over(Interval.seconds(30)) # bars the last 30 s actually printed
```

`time_since(cond)` is the twin of `bars_since(cond)`: seconds since the last row
where the condition was true, `0.0` on a true row, NaN until it first is.
`count_over(Interval.seconds(30))` -- a duration with no condition -- counts the
**rows** in the window, which is how a strategy reads how gappy its own grid is.
`count_over(cond, Interval.seconds(30))` still counts the true ones.

**Durations are literal.** `param()` sweeps a bar count, not a duration:
`rolling_mean(param("w"))` works, a swept duration does not exist yet. A sweep
axis in seconds is a later step.

**Not on GPU.** A duration window evicts on a bound that moves with the
timestamps, where every GPU kernel here indexes by a constant row offset. A
sweep whose expressions carry one is refused by name (`gpu-sweep-unsupported`)
and has to run with `device="cpu"`; it is not silently downgraded, because that
would return numbers computed over a different window from the CPU's.

---

## Signals & Sizing

### Strategy builder

```python
strategy = (
    mbt.Strategy.create("my_strategy")
    .signal("fast", fast)           # named signal
    .signal("slow", slow)           # signals form a DAG
    .size(signal_expr)              # position sizing expression
    .describe("Strategy description")
)
```

### `mbt.when()` — conditional logic

```python
# Long when fast > slow, flat otherwise
signal = mbt.when(fast > slow, 0.5, 0.0)

# Nested: long / short / flat
signal = mbt.when(fast > slow, 0.25,
         mbt.when(fast < slow, -0.25, 0.0))

# Hold current position (omit 3rd arg or use NaN)
signal = mbt.when(rsi(close, 14) < 30, 1.0)  # buy oversold, hold otherwise
```

### `mbt.hold()` holds the POSITION, not the value

`mbt.hold()` is NaN, and the engine reads NaN at the top of the position
expression as *leave the position where it is*. It holds the position the
simulator currently carries, not the last value of the expression around it.
Two consequences worth knowing before you write a regime filter:

- after a stop-loss or take-profit fired, `hold()` keeps the position **flat**.
  It does not replay the last signal value and buy back in.
- under a filter, `hold()` holds whatever the filter last wrote:

```python
state = mbt.when(imb >= thr, 1.0,
        mbt.when(imb <= -thr, -1.0, mbt.hold()))
pos   = mbt.when(regime_open, state, 0.0)      # reads as a trap
```

While the filter is closed the target is `0.0`, so the position goes flat. When
it reopens on a bar where `state` falls through to `hold()`, the engine holds
that flat position: exposure only returns on the **next** threshold crossing.
Measured on a flow strategy, a filter open 55% of the time left 0.7% exposure.
Nothing warns, and routing `state` through a named `.signal()` changes nothing.

Both behaviours are legitimate. Pick the one you meant:

| Intent | Write |
|--------|-------|
| Exit while the regime is hostile, re-enter only on a **new** signal | keep `hold()` inside the `when`: `mbt.when(regime_open, <...hold()...>, 0.0)` |
| Persistent state that you **mask** | build the state as a series, then mask it (below) |

```python
# .ffill() carries the last non-NaN value of the SERIES. The omitted false
# branch of when() is already NaN, so this is the whole idiom:
state = mbt.when(imb >= thr, 1.0,
        mbt.when(imb <= -thr, -1.0)).ffill()
pos   = mbt.when(regime_open, state, 0.0)      # back on the bar the filter reopens
```

Masking a `ffill` state does not destroy it. `cond.value_when(value)` writes the
same series with an explicit trigger; with no filter and nothing else moving the
position, the three spellings give the same trades.

### Arithmetic on expressions

```python
trend = fast - slow
spread = close / (pair_close + mbt.lit(1e-12))   # mbt.lit() for constants in arithmetic
signal = -spread_z * mbt.lit(0.05)                # negation + scaling
dev2 = (close - sma(close, 20)) ** 2              # a literal integer power
```

> **Note:** `mbt.lit()` is needed for constants in arithmetic (`close + mbt.lit(1e-12)`). Numbers auto-coerce inside `mbt.when()`.

An expression is a series the engine evaluates later, bar by bar, so it has
no truth value in Python. Python's `max()` and `min()`, `if`, `and`, `or`,
`not`, `in` and a chained `a < b < c` raise a `TypeError` on one instead of
picking an operand. Write `max_val(a, b)` / `min_val(a, b)` (from
`manifoldbt.indicators`), `(a > 0) & (b < 1)`, `|` and `~`,
`(a < b) & (b < c)`, and `mbt.when(cond, x, y)` for an if/else. `a != b` is
`~(a == b)`. `x ** n` takes a literal integer `n` from -8 to 8, 0 excepted,
and is written as products: `x ** 2` is exactly `x * x`, `x ** -2` is
`1 / (x * x)`; for a square root use `sqrt(x)`.

### Rounding to a price grid

A venue quotes on a grid. BTCUSDT moves in units of `0.1`, so `120087.3` is a
price and `120087.304348` is not -- and an order at a price the grid does not
carry finds no depth in the book, which is what a queue model needs to stand
behind. A computed level lands anywhere, so put it back on the grid:

```python
from manifoldbt.expr import col, lit

fair = col("bid") * lit(0.5) + col("ask") * lit(0.5)
bid  = (fair - lit(0.5) * col("spread")).floor_to(0.1)   # a bid rounds DOWN
ask  = (fair + lit(0.5) * col("spread")).ceil_to(0.1)    # an ask rounds UP
mid  = fair.round_to(0.1)                                # nearest, half UP
```

| Method | Answer |
|---|---|
| `.round_to(step)` | nearest multiple; an exact half goes **up** (towards `+inf`) |
| `.floor_to(step)` | largest multiple at or below -- the passive side of a **bid** |
| `.ceil_to(step)` | smallest multiple at or above -- the passive side of an **ask** |

`step` is a strictly positive **literal**, never a `param()`: it is read once,
at compile time, as the decimal it is written as (`0.1` is one tenth, not the
binary double just above it), which is what a tick size is. A step that is not
a positive number is refused by name at compile time.

**The answer is the double the literal denotes.** On a `0.1` grid, `100.34`
rounds to exactly `100.3` -- the same double `100.3` produces. The obvious
spelling does not: `round(x / step) * step` gives `100.30000000000001`, one ulp
away, and to a book lookup one ulp is a different price level. NaN stays NaN
and an infinity stays itself.

**Half up, not half to even.** `0.25` on a `0.1` grid is `0.3` here, where
`numpy.round(0.25, 1)` is `0.2`. The convention is fixed and does not depend on
the value's parity.

**Snapping twice is snapping once.** A level already on the grid comes back
unchanged, by any of the three and in any order -- so re-snapping a resting
level never walks it a tick further away.

The three are accepted in the level of an entry order, including a level that
reads `position()`. They have **no GPU kernel**: a sweep whose expressions carry
one is refused by name (`gpu-sweep-unsupported`) and runs with `device="cpu"`.
Snapping exactly needs integer arithmetic a kernel has no type for, and a GPU
answer one tick away from the CPU's would be worse than no GPU answer.

### Sizing modes

| Mode                         | Meaning                                              |
|------------------------------|------------------------------------------------------|
| `FractionOfEquity` (default) | `1.0` = allocate 100% of current equity              |
| `FractionOfInitialCapital`   | `1.0` = allocate 100% of initial capital (no compounding) |
| `Units`                      | `1.0` = hold exactly 1 unit (share/contract/coin)    |

```python
execution=mbt.ExecutionConfig(position_sizing_mode="FractionOfInitialCapital")
```

### Special values

| Value  | Behavior                                |
|--------|-----------------------------------------|
| `1.0`  | Full long position                      |
| `0.0`  | Flat (close position)                   |
| `-0.5` | Short 50% (requires `allow_short=True`) |
| `NaN`  | Hold the current POSITION unchanged (`mbt.hold()`) |

---

## Parameters & Sweeps

Use `mbt.param()` to define sweepable parameters in indicator periods:

```python
fast = ema(close, mbt.param("fast", default=12))
slow = ema(close, mbt.param("slow", default=50))

strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.25, -0.25))
)
```

Parameters are auto-collected from expressions — no `.param()` needed on the Strategy.

A default belongs to the strategy whose expressions declare it: variants built
in one session (`slow=20`, then `slow=41`) each run with their own. Within one
strategy, give the default once and write `mbt.param("slow")` at the other uses;
two different defaults for one name are refused when the strategy is
serialised, unless `.param("slow", default=...)` on the strategy settles it.

### Sweep execution

```python
# Full sweep (returns Result per combo)
sweep = mbt.run_sweep(strategy, {"fast": [5, 12, 20], "slow": [50, 100]}, config, store)
best = sweep.best("sharpe")

# Lite sweep (metrics only, much faster for large grids)
batch = mbt.run_sweep_lite(strategy, {"fast": range(5, 100), "slow": range(10, 500)}, config, store)
```

Grids this size need Pro; Community is capped, and the engine tells you where
you stand when you hit it. Single `bt.run()` calls are never gated.

`run_sweep_lite` is optimized for large parameter grids (100k+ combos):
- Cartesian product expansion in Rust (no Python loop)
- Signals and position sizing compiled as one graph, so what they share is
  computed once
- Shared indicator cache (EMA(12) computed once, reused across combos)
- Bars and higher-timeframe columns resampled once per sweep, not per combo
- Metrics only — no Arrow output, and signals nothing reads are never
  materialised

The execution settings sweep too, on their own grid beside this one:
`execution_grid={"fill_model.queue.assumed_queue": [1, 2, 5]}`. They make the
P&L of a maker and none of them is knowable from a backtest, so they belong on
an axis rather than in a decision -- see [Sweeping Execution
Parameters](#sweeping-execution-parameters).

A strategy that quotes (`Strategy.quote`) sweeps through the same two calls,
with the same grids and the same order, on the order-level simulation: see
[Quoting in the DSL](#sweeps).

Add `device="cuda"` on a machine with an NVIDIA GPU and a CUDA build. It pays
off on large grids, where it is typically an order of magnitude faster;
`device="auto"` (the default) picks between CPU and GPU for you, since the GPU
loses on small ones. Results are identical either way. A strategy the GPU
cannot take falls back to the CPU and says which setting caused it.

---

## Backtest Configuration

```python
config = mbt.BacktestConfig(
    universe=[1, 2],                       # symbol IDs
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(4),        # signal evaluation resolution
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(...),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,                        # bars to skip for indicator warmup
    accuracy=False,                        # True = simulate on 1-min bars
    book_levels=None,                      # order book levels read per side (None: all)
)
```

### Bar intervals

```python
Interval.seconds(1)    # 1-second (Pro)
Interval.minutes(1)    # 1-min
Interval.minutes(15)   # 15-min
Interval.hours(1)      # 1-hour
Interval.hours(4)      # 4-hour
Interval.hours(12)     # 12-hour
Interval.days(1)       # daily
Interval.trades()      # one row per trade -- see The Trade Clock
```

Anything below one minute is a Pro step; Community simulates at one minute at
the finest. `bar_interval` is one minute when left out.

### Accuracy mode

```python
config = mbt.BacktestConfig(
    bar_interval=Interval.hours(4),   # signals on 4h
    accuracy=True,                    # simulation on 1-min bars
    ...
)
```

When `accuracy=True`, the engine loads `bars_1m` and runs in hybrid mode: signals evaluated on `bar_interval`, simulation tick-by-tick on 1-min bars. Use for precise SL/TP fill detection. ~60x slower than normal mode.

---

## Execution Model

```python
mbt.ExecutionConfig(
    signal_delay=0,                    # bars between signal and execution
    execution_price="AtClose",         # AtClose, AtOpen, AtVwap, MidPrice,
                                       # or ExecutionPrice.custom(name)
    max_position_pct=0.5,              # max position as fraction of equity
    allow_short=True,                  # allow short positions
    allow_fractional=True,             # allow fractional units
    position_sizing_mode="FractionOfEquity",
    pyramiding=False,                  # True = signal is delta, not target
)
```

### Filling at a computed level

`ExecutionPrice.custom(name)` accepts a bar column (`"vwap"`, ...) **or the
name of any signal the strategy defines**, so a market fill can land on a level
the DSL computes instead of the bar's close. The canonical use is a band
strategy on native fine bars: the entry level is known before the bar starts,
and the touch bar itself proves the level traded (it sits between open and
high), yet a close fill would be systematically on the wrong side of it.

```python
from manifoldbt.indicators import close, high, low, open, sma
sma_20 = sma(close, 20)
band_up, band_dn = sma_20 * 1.012, sma_20 * 0.992
exec_level = mbt.when(high >= band_up,
                      mbt.when(open >= band_up, open, band_up),   # gapped through
             mbt.when(low <= band_dn,
                      mbt.when(open <= band_dn, open, band_dn),
             close))
strategy = strategy.signal("exec_level", exec_level)
config.execution.execution_price = mbt.ExecutionPrice.custom("exec_level")
```

One series covers entry AND exit fills. The rules that keep it honest:

- the series is read at the order's **signal row**, never ahead of it;
- a fill outside the execution bar's `[low, high]` range draws a warning;
- a row with no value (warm-up) falls back to the close, with a warning;
- a name that is neither a column nor a signal is rejected before the run;
- a bar column always wins over a same-named signal (warned about).

**A custom execution price keeps the fast kernel, and the GPU.** Sweeping one is
as fast as sweeping a plain `AtClose` strategy, and `device="cuda"` accepts it:
the level is evaluated in the kernel beside the position sizing, so correct
fills no longer cost throughput. Results are identical on both devices.

Two conditions, and the sweep says so when either fails: the name must resolve
to a **signal** rather than a bar column (a column is read per execution row, a
different rule), and the strategy must have no exit orders, whose entry-bar
re-check needs the general loop. `run()` is unaffected either way.

### Filling at the open

`execution_price="AtOpen"` fills at the **open of the execution bar** -- the
signal row shifted by `signal_delay`. With `signal_delay=1` that is the honest
intraday convention: the bar closes, the signal is computed, and the order
reaches the market at the next bar's opening print. Sizing, equity and the
duplicate-signal check keep reading the execution bar's close, exactly as under
`AtClose`; only the fill price moves.

**AtOpen keeps the fast kernel, and the GPU.** Sweeping it costs the same as
sweeping `AtClose` (measured within 5% on a one-second store), and
`device="cuda"` accepts it with no fallback. Results are identical on both
devices, to the bit.

Two combinations stay on the general loop, and the sweep names them: an
AtOpen fill combined with **exit orders** (a fill that lands mid-bar can be
stopped out by the rest of that same bar, which the fast kernel does not
replay), and a **multi-asset** universe. `run()` is unaffected either way.

### Signal delay

| Value | Behavior                                                        |
|-------|-----------------------------------------------------------------|
| `0`   | **Default.** Fill at the close of the signal bar                |
| `1`   | Fill on the next bar (t+1)                                      |
| `2+`  | Fill N bars after the signal                                    |

`0` models a decision taken on the bar's own close and filled at that close, the
market-on-close convention, and it is what vectorbt's `from_signals` does. It is
the right default for coarse bars, where one bar of delay would mean pricing a
full day of latency into a decision that in reality reaches the market in
seconds.

Raise it when a bar is short enough that one bar is a plausible
decision-to-fill latency: on 1s or sub-second bars, `signal_delay=1` *is* the
realistic setting, and `0` assumes an infinitely fast round trip. The engine
does not infer this from `bar_interval`, so it is on you to set it.

It is a delay in EVENTS, and the round trip it stands in for is a delay in
TIME. On a tape the two are different questions, and
[`execution.latency`](#latency) asks the second one.

---

## Backtesting on the Tape

Everything above runs on bars: a fixed grid, four prices per step, and a fill
that some convention has to place somewhere inside a step. A market maker cannot
be written that way. Its whole result lives in the difference between "the price
reached my level" and "the volume resting in front of me was consumed", and a
candle cannot tell those apart.

This chapter and the four that follow are that difference, together with three
settings that live beside their bar-grid neighbours further down (durations,
windows in time, `position()`). All of it is **one layer with two doors and no
third one**: the **store** holds what the venue published, the **config** says
what the engine does with it. There is no second strategy API and no second
language. The same DSL, the same `mbt.run`, the same result object.

### What comes in through the store

| Call | What it holds | Read by |
|---|---|---|
| `bt.ingest_trades(...)` | the **tape**: one row per print, with its timestamp, price, quantity and aggressor side | the trade clock, `fill_resolution="ticks"`, the queue, latency, the marks |
| `bt.ingest_book(...)` | the **book**: the first `levels` prices and sizes on each side, at every instant one of them changed | the quote columns, the queue's depth, the marks' mid |

Both are stored per symbol and per UTC day, like every other dataset. Neither is
a strategy setting: a run reads what is on disk, and says so by name when what
it needs is not there.

The book comes from Bybit's public archive, one file per day, as deep as Bybit
recorded that day (checked on BTCUSDT, ETHUSDT and BTCUSD; a symbol listed
later starts later):

| `category` | Symbols | Depth |
|---|---|---|
| `"linear"` | USDT perps (`BTCUSDT`) | 500 levels up to 2025-08-20, 200 from 2025-08-21; BTCUSDT goes back to 2023-01-18 |
| `"inverse"` | coin-margined perps (`BTCUSD`), sizes in contracts | 500 levels up to 2025-08-20, 200 from 2025-08-21 |
| `"spot"` | `BTCUSDT` | 200 levels, from about 2025-04-30 |

There is no option archive. `levels` is 200 for a new symbol, the depth every
day is published at; `levels=500` holds on a range published that deep, and a
range that crosses into a shallower day is refused before anything is
downloaded, with the day to end it before. A store keeps one depth and one
market per symbol: later days follow the depth the symbol is stored at.

Depth costs disk and load time, and a perp day is busier than a spot one. BTCUSDT
linear on 2025-08-20, stored as changes: 266 MB at 500 levels, 158 MB at 200
from the same archive; loaded with its tape, about 0.85 s and 0.55 s.

The tape takes the same `category`: `"spot"` (the default) reads the spot
archive, `"linear"` and `"inverse"` read Bybit's derivatives archive, which
starts on the day a contract was listed (2019-10-01 for BTCUSD, 2020-03-25 for
BTCUSDT). A perp studied on the tape takes both from the same category and
under the same `symbol_id`:

```python
bt.ingest_trades("bybit", "XRPUSDT", symbol_id=1, start="2023-03-01",
                 end="2023-03-01", category="linear")
bt.ingest_book("bybit", "XRPUSDT", symbol_id=1, start="2023-03-01",
               end="2023-03-01", category="linear")
```

One symbol is one market, tape and book together: each stored day records its
category, and a store holding the spot tape or book of a symbol refuses its
linear one before anything is downloaded, as it refuses a second id for the same
ticker. Asking for an inverse contract as `"linear"` is refused with the right
word. Inverse sizes are in contracts on the tape as in the book.

Bybit stamps a derivative trade in seconds with a decimal fraction, converted to
nanoseconds exactly. The fraction is coarser than the engine's nanosecond, and
how coarse depends on the day:

| Days | Stamp of a derivative trade |
|---|---|
| up to 2021-12-06 | a microsecond (inverse) or a tenth of a millisecond (linear); the file lists the day newest first and is read back in time order |
| 2021-12-07 to 2022-12-20 | the whole second |
| from 2022-12-21 | a tenth of a millisecond |

Spot trades are stamped to the millisecond. So prints share timestamps: more than
half of XRPUSDT's on 2023-03-01, every print of a second on a whole-second day.
They keep the order the venue matched them in, the trade clock steps through
them one by one at the same instant, and anything shorter than the stamp (a
latency, a `time_in_force`, a window in time) cannot tell them apart. On the
whole-second days, a latency under a second is not something the tape can
measure. Bybit identifies a derivative trade by a UUID, which the tape's integer
`trade_id` cannot hold: it reads back null.

A book is stored in one of two shapes, one per symbol, and every reader sees
the same states whichever it is:

| `book_format` | On disk | When it pays |
|---|---|---|
| `"deltas"` (default for a new symbol) | `{provider}/book_delta/{symbol}/`: each row holds only the levels that changed at that instant, compressed. A day of BTCUSDT 200 levels deep is about 44 MB | deep books: loads in a fraction of a second at any depth, and the engine replays the changes as it reads. Read ten levels deep or less, the book is laid out whole in memory as it loads |
| `"ladder"` | `{provider}/book/{symbol}/`: the whole top-`levels` ladder on every row. The same day is about 2.7 GB | thin books read row by row; the only shape earlier versions of manifoldbt read |

`bt.convert_book_to_deltas("BTCUSDT", data_root=..., metadata_db=...)` rewrites
a book stored as ladders day by day, and replaces a day only once its changes
replay into exactly the stored ladders, every price and size bit for bit.

### What comes in through the config

| Setting | What it changes | Where |
|---|---|---|
| `fill_model.fill_resolution="ticks"` | keeps the bar grid; settles level orders on the prints inside each bar | [Execution Model](#execution-model) |
| `bar_interval=Interval.trades()` | replaces the grid: one simulation row per print | [The Trade Clock](#the-trade-clock) |
| `time_in_force=Interval.seconds(1)` | an order lives for a duration instead of a number of events | [Time in force](#time-in-force) |
| a window written `Interval.seconds(30)` | a rolling window measured on the clock, not on the row count | [Windows in time](#windows-in-time) |
| `position()` inside an order's level | the quote steps away as the inventory grows | [Reading the position from a level](#reading-the-position-from-a-level) |
| `fill_model.queue` | the queue in front of a resting order decides the fill, in place of a price convention | [Queue Model](#queue-model) |
| `fill_model.custom` | a service rule you wrote decides how much of each print reaches a resting order | [A Service Rule of Your Own](#a-service-rule-of-your-own) |
| `execution.latency` | the round trip between a decision and the book | [Latency](#latency) |
| `execution.fill_marks` | what the market did in the seconds after each fill | [Marking a Fill](#marking-a-fill) |
| `fill_model.queue.model`, `fill_model.adverse` | a NAMED assumption about the service and its cost, in place of the default one | [Execution Models](#execution-models) |
| `execution_grid=` on a sweep | any of the above, on an axis instead of in a decision | [Sweeping Execution Parameters](#sweeping-execution-parameters) |

### The order to switch them on

Each step is readable on its own, and every one of them moves the numbers. Turn
them on one at a time and keep the run before as the reference.

1. **The tape.** `fill_resolution="ticks"` on the bars you already run: same
   strategy, same grid, level orders now settled by the prints instead of by a
   high and a low. `result.tape_resolution` says how much of the run the tape
   decided.
2. **The book.** No result moves yet. `bid`, `ask`, `spread` and the two depths
   stop being null, and the marks gain a mid to measure against instead of the
   last print.
3. **The clock.** `bar_interval=Interval.trades()`: one row per print, nothing
   aggregated. This is the step that changes the shape of a run rather than its
   numbers, so read what it refuses before spending a day on it.
4. **Durations.** `time_in_force=Interval.seconds(1)`, and windows written as an
   `Interval`. On a tape a count of events is not a length of time, and a maker
   means the second, not the print.
5. **The queue.** `fill_model.queue`: a fill stops being a convention and
   becomes a volume calculation. Expect fewer fills than `touch` gave and
   slightly more than `traverse`.
6. **The latency.** `execution.latency`: the quote reaches the book late and
   leaves it late. Service falls, adverse selection rises.
7. **The marks.** `execution.fill_marks`: the number that says whether any of
   the above was worth doing. On ten BTCUSDT days of a one-sided maker, the
   captured half-spread is 0.006 bps against 1.2 bps of adverse drift one second
   later, two hundred times more. That order of magnitude is the reason this
   layer exists, and it is visible from step 1.
8. **The named models.** `fill_model.queue.model`, `fill_model.adverse`: the
   assumptions of steps 5 and 6, replaced one at a time by a named one and swept
   as a cost axis. See [Execution Models](#execution-models).
9. **The check.** `bt.reconcile`: the fills a broker really granted over the
   same days, marked by the same code and counted the same way. It is what says
   whether any of steps 5 to 8 was set anywhere near right, and it is the only
   thing that measures the part of the adverse selection that exists because
   your quote was there. See [Calibrating Against Live
   Fills](#calibrating-against-live-fills).

### What holds for every step

**One loop, and it refuses by name.** All of it runs on the general simulation
loop. The fast kernel, the CUDA sweep and the two lite drivers
(`run_batch_lite`, `run_sweep_lite`) implement none of it, and each **refuses by
name** instead of falling back: `fast_path_blocker` names the setting,
`run_sweep_lite_gpu` answers `gpu-sweep-unsupported: ...` before it even opens a
CUDA context, and the lite drivers answer before loading a single day. There is
no silent downgrade anywhere in this layer, and that is deliberate: a downgrade
would answer a different question with numbers that look like an answer to
yours. Each section lists what it refuses, and the reason is always the same
one -- the setting is a claim about volume, order or time, and the path that
refused it does not walk a tape.

**A day, not a month.** The clock keeps positions and equity per event, and a
liquid perpetual prints two to three million events a day. One day of BTCUSDT is
comfortable on a laptop; a month of it is not, and neither is a sweep over one.
That is why `output_resolution` defaults to **one second** under this clock
rather than to one row per event: ask for `Interval.trades()` explicitly when
you want the per-event curve, and expect the memory bill that comes with it.
Sizing a study in days is the constraint to design around.

**Anything counted "in bars" counts events.** `signal_delay`, `warmup_bars`,
`{"GTB": n}` and every window given as an integer count rows, and a row is a
print. Thirty prints is a fraction of a second on a busy minute and several
minutes on a quiet one. When you mean time, write time: [Windows in
time](#windows-in-time) and [Time in force](#time-in-force).

**Walk-forward is not adapted to this clock.** It cuts its folds using the
signal interval as a duration, and an event is not a duration: a geometry given
in bars comes out as windows of zero length. Nothing stops you from calling it,
and nothing should be read from what it returns.

**Two runs give the same bits.** The simulation is one ordered walk over the
events, in the order the venue sent them, prints sharing a timestamp never
re-sorted. The floating-point additions happen in that order and in the same
order every time: nothing is reassociated from one run to the next, and no
reduction is split across a thread count that could vary. So two runs of the
same config against the same store return the same equity curve, the same fill
prices and the same counters, byte for byte -- pinned on ten BTCUSDT days by
comparing bytes, not by comparing within a tolerance. `output_resolution`
changes what is handed back, never what was computed.

### One word per thing

| Word | What it means here |
|---|---|
| **event** | one simulation row: a bar on a time grid, a print on the trade clock |
| **print** | one trade on the tape, as the venue published it |
| **resting order** | an order standing in the book at a level: a limit entry, an armed stop-limit, a take-profit |
| **requote** | cancelling a resting order and posting another; never a move, and always the back of a fresh queue |
| **queue-decided fill** | a fill granted because the volume ahead of the order at its level was consumed |
| **book cross** | the opposite best price crossing the level, serving the order with no print at it |
| **stale fill** | a fill taken by a quote after its cancellation was decided and before it took effect |

---

## The Trade Clock

`bar_interval=Interval.trades()` replaces the time grid with the tape: **one
simulation row per print**, in the order the venue published them. Nothing is
aggregated, so nothing is guessed about what happened inside a bar -- there is
no inside.

```python
config = mbt.BacktestConfig(
    universe=[1],
    time_range_start=start, time_range_end=end,
    bar_interval=Interval.trades(),
    execution=mbt.ExecutionConfig(signal_delay=1),
)
```

The tape comes from the store (`bt.ingest_trades`), like every other input, and
the clock is a setting rather than a second API. See
[Backtesting on the Tape](#backtesting-on-the-tape) for the two doors and the
order to open them in.

### The row

| Column | Under the trade clock |
|---|---|
| `timestamp` | the trade's exchange timestamp, nanoseconds |
| `open` `high` `low` `close` `vwap` | all five equal the trade's price |
| `volume` | the trade's quantity |
| `side` | `+1.0` when the aggressor bought, `-1.0` when it sold |
| `buy_volume` / `sell_volume` | the quantity on the aggressor's side, `0.0` on the other |
| `trade_count` | `1` |
| `bid` `ask` `spread` | the top of book standing at or before the trade, null when no book is stored |
| `depth_at_best_bid` `depth_at_best_ask` | the size resting at each best, same rule |

`side`, `buy_volume` and `sell_volume` are the flow columns: `col("side")`,
`col("buy_volume").rolling_sum(Interval.seconds(34))` and so on read exactly
what the tape says, with no bucketing in between. They are consistent with the
same columns on bars rebuilt by `bt.bars_from_trades`, which sum them over a
bucket -- a bucket of one here.

The quote columns come from the stored book (`bt.ingest_book`), joined **as of**
each print: the state left by every update stamped at or before it, with a book
update and a trade sharing a timestamp ordered book-then-trade. Before the
book's first row nothing is known and the five columns are null; a book is never
guessed backwards. They exist whether or not a book was stored -- what a
strategy can name must not depend on what the store happens to hold -- so a run
with no book reads nulls rather than failing to compile.

Timestamps repeat: a venue can print several trades at the same nanosecond, and
the tape order is then the order the venue sent them. That order is preserved
end to end, never re-sorted. A venue stamping coarser than the nanosecond makes
it the rule rather than the exception: Bybit's derivative trades carry a tenth
of a millisecond, and the whole second in 2022 (see [What comes in through the
store](#what-comes-in-through-the-store)).

### Events, not bars

Every count the engine expresses "in bars" counts **events** under this clock:

- `signal_delay` -- a decision taken at event `i` reaches the market at event
  `i + signal_delay`;
- `warmup_bars`;
- `time_in_force={"GTB": n}` -- an entry order rests for `n` events, and is
  posted again after that while the target still holds, so a constant target
  reaches its level whatever `n` is;
- any window given as an integer: `rolling_sum(34)` is thirty-four prints.

A window given as a duration still counts time: `rolling_sum(Interval.seconds(34))`
reads the prints of the last thirty-four seconds, however many that is. On a
tape those are the two different questions, and they are meant to be asked
separately. `time_since` and `count_over(Interval)` read the clock, not the row
number.

So does an order's lifetime, when it is given as one:
`time_in_force=Interval.seconds(1)` cancels the order on the first print stamped
a second or more after the one that posted it, however many prints that is. That
is what a maker means by "my quote is one second old"; `{"GTB": 30}` on a tape
is thirty prints, a fraction of a second at one moment and a minute at another.
See [Time in force](#time-in-force).

### How an order fills

A **market order** decided at event `i` fills at the price of event
`i + signal_delay`. With the default `signal_delay=1` that is the next print
after the one the decision saw; `signal_delay=0` fills on the deciding trade
itself, which assumes an infinitely fast round trip. `execution_price` still
selects a column, but `AtClose`, `AtOpen` and `AtVwap` name the same number: a
trade has one price.

A **level order** -- a `limit_entry`, a `stop_entry`, and every bracket leg --
rests and is resolved **trade by trade** over the events that follow, by the
same resolver that serves `fill_resolution="ticks"` inside a bar:

- a **maker** level (a limit entry, the resting leg of a stop-limit, a
  take-profit) obeys `passive_fill`. Under `"touch"` (default) a print *at* the
  level fills it, and that fill is counted in `result.fill_fragility` because a
  print at a level proves volume traded there, not that the queue in front was
  consumed. Under `"traverse"` only a print *through* the level fills it. Set
  [`fill_model.queue`](#queue-model) and the queue answers the question instead
  of the convention bounding it.
- an **aggressive** level (stop-loss, trailing stop, a stop entry, the arming
  leg of a stop-limit, market-if-touched) fills on the first print at or beyond
  it, **at that print's price**, not at the level. A stop at 98 whose first
  print below it is 96.5 fills at 96.5.

Bracket legs are re-checked at **every** event the position is open. When one
print satisfies both the stop and the take-profit, the stop wins, exactly as on
bars. An order filled at event `i` is checked against events `i+1` onward and
never against event `i` itself: the print that filled it cannot also close it.

Fees and slippage are unchanged, and are charged on the resolved fill price.
`fill_model.max_participation_rate` keeps its meaning and gains a sharper one:
a bar's volume is now a single print's quantity, so a cap of `0.1` says an order
may take a tenth of each trade it meets, one trade at a time.

### What comes out

Positions and equity are tracked **per event**; `output_resolution` then decides
what is returned, and under this clock it defaults to **one second** rather than
to one row per event ([a day, not a month](#what-holds-for-every-step)). Ask for
the per-event curve explicitly:

```python
config.output_resolution = Interval.trades()   # one output row per event
```

The metrics resample by timestamp, so irregular steps of a few milliseconds and
holes of several seconds are handled the same way a gappy bar grid is. Trade
counters count **fills**, as they already do everywhere else.

`result.tape_resolution` is filled in on this clock too, and its counters count
**events**: `bars_on_tape` is the number of (symbol, event) pairs where a level
order was live and the prints decided it. `bars_fallen_back` is always zero
here -- every row carries its own print, so there is never a bar rule to fall
back on. `result.fill_fragility` keeps its meaning exactly: how many maker fills
were granted on a print *at* the level rather than through it, and
`result.order_activity` says how many quotes it took to get them (see
[Requoting](#requoting-is-cancelling-and-posting-again)).

### What refuses, and why

The trade clock runs on the general loop only. Each of these refuses by name
rather than falling back quietly:

- the fast path, the CUDA sweep and the lite paths, as
  [everywhere in this layer](#what-holds-for-every-step). The two lite drivers
  answer before they read anything: a day of a liquid perpetual is millions of
  prints to load for an answer already known.
- **one symbol per run**. Two tapes have no common clock: aligning them would
  forward-fill one symbol's last trade onto the other's timestamps and invent
  prints that never happened, with volumes counted twice.
- `resample_to`, `precise`, `extra_timeframes`, `signal_source` and
  `execution_source` all build a second series beside the tape, and there is no
  answer to "which of the two did this order see".

## Order Book Columns

A book stored with `bt.ingest_book` is readable by the strategy itself, level by
level, on bars as well as on the trade clock. Each column is named after what it
reads and the level it reads it at, and `bt.book` builds the names:

```python
import manifoldbt as bt

imb = bt.book.imbalance(5)                  # col("book_imbalance_5")
thin_ask = bt.book.ask_depth(3) < 2.0       # col("ask_depth_3") < 2.0

strat = (
    bt.Strategy.create("imbalance")
    .signal("long", bt.when((imb > 0.3) & thin_ask, 1.0, 0.0))
    .size(bt.col("long"))
)
```

| Column | `bt.book` | Reads |
|---|---|---|
| `bid_price_<k>` `ask_price_<k>` | `bid_price(k)` `ask_price(k)` | the price of level `k`, 1 = best |
| `bid_size_<k>` `ask_size_<k>` | `bid_size(k)` `ask_size(k)` | the size resting at level `k` |
| `bid_depth_<k>` `ask_depth_<k>` | `bid_depth(k)` `ask_depth(k)` | the size resting over levels 1 to `k` |
| `book_imbalance_<k>` | `imbalance(k)` | `(bid_depth_k - ask_depth_k) / (bid_depth_k + ask_depth_k)`, in [-1, 1] |

Nothing in the config names them. The engine sees which ones the strategy reads,
loads the stored book once, only as deep as the deepest `k` asked, and adds the
columns to the bars before any signal is evaluated. From there they are bar
columns like `close`, and a sweep or a lite sweep reads them exactly as a single
run does. A strategy that names none loads no book at all.

### How deep a run reads

By default a run reads **every level stored**. `BacktestConfig.book_levels`
bounds it: the run then reads that many levels per side, or the stored depth
when the book is shallower. The bound holds for everything in the run that reads
the book: these columns, the [queue](#queue-model) ahead of a resting order, the
quotes of the trade clock, the fill marks, and the book a quoting strategy reads
at each wake-up.

A bound is for speed. On a BTCUSDT day stored 200 levels deep, a maker quoting
three levels a side every 200 ms (a store opened afresh for each run):

| Stored as | Every level (default) | `book_levels=25` |
|---|---|---|
| whole ladders (`book_format="ladder"`) | 0.75 to 1.3 s | about 0.55 s |
| changes (`book_format="deltas"`, the default) | about 0.6 s | about 0.6 s |

The same day stored ten levels deep runs in about 0.35 s. Bound it only when the
strategy reads no further than the bound, and the book is stored as ladders.

```python
config = bt.BacktestConfig(..., book_levels=25)   # bt.book.bid_size(40) is refused
```

### When a row reads the book

A row reads the ladder standing **at the instant its own close is known**, never
later:

- on bars, one nanosecond before the next bar opens: the book and the `close`
  describe the same instant. A gap in the bars (a weekend, a halt) does not
  stretch the window, so a Friday bar never reads Monday's book;
- on resampled bars (`bar_interval` coarser than the stored bars, or
  `resample_to`), at the close of the last stored bar of the group, like the
  group's `close`;
- on [the trade clock](#the-trade-clock), at the print itself, with a book
  update and a trade sharing a millisecond ordered book first: the same rule
  as the `bid` and `ask` columns of that clock.

### Where the book says nothing

The value is null (NaN in a DataFrame) rather than a guess:

- before the book's first stored update;
- on a day the store holds no book for, and before the first update of a day
  whose previous day is not stored either: the standing ladder would be hours
  or days old;
- at a level that does not exist at that instant, when a side is thinner than
  `k`;
- for `book_imbalance_<k>` when both sides are empty.

`bid_depth_<k>` and `ask_depth_<k>` sum the levels that exist and are `0.0` on
an empty side.

A null flows through the expression like a NaN close would: a comparison on it
is null, `when()` on a null condition is null, and a null at the top of the
sizing **holds the current position** (see [Special values](#special-values)).
A strategy in a position when the book goes unknown stays in it until the book
says something again.

### What refuses, and why

- **no stored book** for a symbol of the universe over the run: the columns
  would be null from end to end and the strategy would silently never trade.
  The error names the column and points at `bt.ingest_book`;
- **a level deeper than the run reads**: `bid_price_30` under
  `book_levels=25` is refused by name before anything is read, and the message
  says to raise `book_levels`;
- **a level deeper than the stored book**: `bid_size_50` on a book ingested
  with `levels=20` is refused rather than read as zero, and a depth is never a
  floor passed off as a total;
- the **CUDA sweep**, which loads no book: it refuses by name, and a lite sweep
  asked for the GPU runs on the CPU instead.

## Queue Model

A resting order is not served because the market reached its price. It is served
because everything resting ahead of it at that price went first. Ignoring that is
the classic way a market-making backtest lies: it pockets a half-spread it would
never have captured. `fill_model={"queue": {...}}` replaces the price rule
(`passive_fill`) with the queue rule, for the two channels where an order really
rests.

```python
config.execution.fill_model = {
    "fill_resolution": "ticks",      # required: a queue is served print by print
    "queue": {
        "depth_source": "book",      # "book" (default), or "assumed"
        "assumed_queue": 2.0,        # in MULTIPLES OF THE ORDER SIZE
        "cancel_ahead_rate": 0.0,    # fraction of the queue cancelling per second
    },
}
```

### The rule, in ten lines

1. The queue decides every **passive** order and nothing else: a `limit_entry`,
   the resting leg of a `stop_limit_entry` once its breakout has armed it, and a
   `take_profit`. A stop-loss, a trailing stop, a `stop_entry` and a
   `market_if_touched` cross the book, so they keep being served by the first
   print at or beyond their level.
2. At the instant it is posted, the order joins a queue of `ahead` units: the
   depth stored at **its own level** as of that instant (`depth_source="book"`),
   or `assumed_queue` times its own size (`"assumed"`).
3. That instant is the event the order was posted on, which is the first event it
   can fill on: the close of the bar the strategy read under `signal_delay=1` on a
   time grid, and the print that posted it under the trade clock. It is the same
   instant a duration `time_in_force` counts from.
4. Requoting is cancelling and posting again, so a replacement order reads the
   depth again, at its own instant, and starts at the back of that queue — even
   when its level has not moved. A take-profit rebuilt on a new average entry is
   a requote too.
5. When the stored book says nothing at that level and instant — before its first
   update, on a side standing empty, or deeper than the deepest level stored or
   read (`book_levels` when it is set: see
   [How deep a run reads](#how-deep-a-run-reads)) — the queue is **unknown**: `assumed_queue` takes over as the declared fallback
   when one was given, and the run is refused by name when none was. It is never
   read as zero, which would put the order at the front of a queue nobody saw.
   **A level off the tick grid finds no depth in the book**, so it lands in this
   case every single time: set `execution.tick_size` (see
   [Entry Orders](#entry-orders)) or snap the level in the expression with
   `.floor_to` / `.ceil_to`, and read `book_unknown_at_post` to check. (`mbt.sim`
   and a strategy that quotes put every price on the symbol's tick themselves:
   see [What the venue does](#what-the-venue-does).) A level deeper than the
   levels read from the store is refused as such, the message naming how many
   levels were read and the deepest price among them.
6. A print AT the level from the side that consumes the order (a selling
   aggressor for a bid, a buying one for an ask) burns `ahead` first, and only
   what is left over fills the order. Partial fills are the normal case.
7. A print strictly THROUGH the level serves the whole rest of the order,
   whatever the queue and whoever the aggressor: on one and the same book, a
   print below a bid means everything resting at that bid was taken out first.
8. So does the best opposite price crossing the level, a **book cross**: the
   book went past the order without a print at it. The stored book is read at the instants the tape prints, plus once at
   the end of each event's window.
9. `cancel_ahead_rate` retires a fraction of the queue per second, as
   `exp(-rate * dt)` between two events: the cancellations in front that a
   level-2 book cannot show. `0.0`, the default, is the assumption that nobody
   ahead ever cancels, which is the most pessimistic one available.
10. A partial fill moves the position and arms the bracket exactly as a full fill
    does; the remainder keeps resting, with its queue, until it fills or its time
    in force runs out.

`assumed_queue` and `cancel_ahead_rate` are parameters that make the P&L. Sweep
them and publish them as cost axes; do not pick one and hide it. `assumed_queue`
is a multiple of the order size rather than an absolute quantity because an
absolute quantity has no scale from one symbol to the next: the same ten days
measured a median queue of 62 times the quoted size on BTCUSDT and 7 times on
XRPUSDT.

### One decision per event

The queue is walked print by print, and the walk of an event stops on the first
print that serves anything. Whatever that print gave is the fill; the rest of the
order stays at the front of its (now empty) queue and resumes on the next event.
Under the trade clock an event *is* one print, so nothing is skipped. On a
one-second grid the prints that follow a partial inside the same second are read
on the next second instead, which is the same one-fill-per-order-per-bar rule the
engine already applies to a level entry.

That is why the same day, the same decisions and the same one-second order life
fill more often under the trade clock than on one-second bars: over ten BTCUSDT
days, 28 485 entries against 15 543, from fewer quotes and with a third of the
partials (217 against 551). An order gets one decision per event, and the trade
clock gives it far more events.

A **take-profit** is the exception to the partial rule, because the bracket it
belongs to exits the whole position and cannot exit half of it: its queue
accumulates across prints and the exit fires only once the WHOLE size is served.
Partials before that are counted and it keeps waiting.

### What comes out

`result.fill_fragility` gains a `queue` object beside its two existing counters:

```python
result.fill_fragility
# One BTCUSDT day, a one-sided maker requoting every second:
# {'maker_fills': 1404, 'touch_only_fills': 0,
#  'queue': {'queue_decided_fills': 25, 'traverse_fills': 1379,
#            'partial_fills': 9, 'book_unknown_at_post': 778,
#            'fills_from_book_cross': 643},
#  'would_fill_touch': 7432, 'would_fill_traverse': 1329}
```

- `queue_decided_fills` — the volume ahead was consumed and the trade reached the
  order. This is what modelling the queue bought.
- `traverse_fills` — a print through the level, or the book crossing it, served
  the order. This is what the `traverse` convention already gave for free.
- `partial_fills` — orders whose life ended having filled part of their size.
- `book_unknown_at_post` — orders posted behind an `assumed_queue` because the
  stored book could not answer at their level and instant.
- `fills_from_book_cross` — of `traverse_fills`, the ones decided by a **book
  cross** rather than by a print through the level. Kept apart because an
  archive can carry a transient crossing the live sequence never had: subtract
  it if you do not believe yours.

`would_fill_touch` and `would_fill_traverse` sit beside it: how many of the same
order lives the `touch` and `traverse` conventions would have filled. They bound
the queue result without a second run:

```
would_fill_traverse  <=  queue_decided_fills + traverse_fills
                     <=  would_fill_touch + fills_from_book_cross
```

Both halves are the service rule read backwards. A print through the level
serves the queue whatever is ahead, so every traversal is a fill; and a queue
fill needs a print AT the level, which `touch` would have taken — except the
ones no print witnessed, which the book crossing decided.

They are counted on the lives *this* run produced, so they are an envelope of
this run and not of two other ones: `touch` fills sooner, so a touch run would
hold a position where this one is still quoting, and would not have posted the
same orders at all.

`result.order_activity` and `result.tape_resolution` keep their meanings.

### With the marks

`execution.fill_marks` and a queue read the same stored book, and the run loads
it once for both: the queue reads the depth at the level of a resting order, the
marks read the mid. Turning the marks on changes nothing the queue decides --
same fills, same counters, byte for byte -- and it answers the question the
counters raise. `queue_decided_fills` against `traverse_fills` says how an order
was served; the marks say what that service cost.

On two BTCUSDT days, joining each entry to the channel that served it, the two
are not the same trade:

| served by | fills | +100 ms | +1 s | +10 s |
|---|---|---|---|---|
| the queue (queue-decided) | 170 | -0.97 | -1.03 | -1.50 |
| a print through the level | 3 140 | -1.23 | -1.40 | -1.59 |

In basis points, negative being adverse. A queue-decided fill is the less
adversely selected of the two by **0.37 bps at one second** (± 0.17), and the gap
is gone by ten (+0.09 ± 0.36). The mechanism is the one the service rule states:
a traversal means the price went THROUGH the level, so the order was served
precisely because the market was leaving; a queue-decided fill means the flow
that consumed the level stopped there. A hundred and seventy queue-decided fills
is a small sample -- the sign holds on both days, the size of the gap is not
something to bank on.

### What refuses, and why

A queue is a claim about volume and time, and every setting below would make that
claim without the data behind it. Each is refused by name rather than approximated:

- `fill_resolution="bar"`. A bar has a high and a low, not an order of prints:
  there is no queue to burn. Set `fill_resolution="ticks"`, or run the trade
  clock, which resolves on prints by construction.
- **the fast path, the CUDA sweep and the lite paths**, as
  [everywhere in this layer](#what-holds-for-every-step). None of them walks a
  tape, so none of them can serve a queue.
- `max_participation_rate`. Both it and the queue cap a fill, and the queue is
  the finer of the two: how much of a print reaches the order is exactly what it
  computes.
- a **stored book that cannot answer**, under `depth_source="book"` with no
  `assumed_queue`. Ingest the book over those days (`bt.ingest_book`), raise
  `book_levels` when the order rests deeper than the run reads, or declare the
  fallback.

The book is loaded **per symbol**, so a multi-asset run is served: each symbol's
orders read their own ladder. A symbol whose book is missing is refused (or falls
back) on its first quote, by the rule above, rather than quietly filling from the
front of the queue.

---

## Latency

Every model above assumes the decision and the market are the same instant: the
strategy reads a print and its order is in the book on that same print. No maker
works that way. `execution.latency` puts the round trip back in, in time rather
than in events.

```python
from manifoldbt.helpers import Interval

config.execution.latency = {
    "order":  Interval.millis(20),   # decision -> the market sees the order
    "cancel": Interval.millis(20),   # cancel decided -> the quote is gone
}
```

Absent, or `{"order": 0, "cancel": 0}`, there is no latency and the run is the
run it always was, **byte for byte**. Both keys are optional and default to
zero, so `{"order": Interval.millis(20)}` models a venue that acknowledges a
cancel instantly, which no venue does.

The same dict takes two more keys, `response` (the venue's answer coming back)
and `feed` (the market reaching the strategy), which only a strategy that
quotes reads: see [Quoting in the DSL](#four-latencies). A bar run refuses
either one set above zero, by name, rather than run as if it were not there.

### Order latency

A decision taken on event `t` — the close of the bar, or the print, that posts
the order — reaches the market at `t + order`. From there:

- **only prints stamped at or after `t + order` can serve it.** A print at
  `t + order - 1 ns` does not, whatever its price;
- **the queue is read at `t + order`**, not at `t`. That is where the order
  joins the queue, so what stands in front of it is the depth at the instant it
  arrived, not the depth the strategy saw when it decided;
- a **duration `time_in_force` counts from `t + order`**: the lifetime is the
  time the order spends in the book, and `posted_at` means the instant it
  reached the market. A `{"GTB": n}` life keeps counting EVENTS from the
  decision, as it always has — so an order whose latency outlives its event
  count expires before the market ever sees it, which is exactly what happens
  when you requote faster than your own round trip;
- a **market order** decided at `t` fills at the first print stamped at or after
  `t + order`, **at that print's price**, plus the configured slippage. It does
  not fill at a bar's close, open or VWAP: under latency the bar's label is a
  decision instant, not a fill instant.

On a **bar clock**, `t` is the close of the bar that decided. An order visible
after the close of the next bar has no print left to fill on there, and slides
to the bar after: a 2 s latency on a 1 s grid means the order is first offered
prints two bars later, and a `{"GTB": 1}` order dies before that ever happens.

`signal_delay` is not latency and latency does not replace it. `signal_delay`
counts EVENTS between the row a signal is read on and the row the order is
placed on — the decision lag, the bar you were not allowed to look at yet.
Latency counts TIME between placing that order and the market seeing it. They
add: with `signal_delay=1` and `order=20ms` on a one-second grid, a signal read
on bar `i` produces an order posted at the close of bar `i` — the label bar
`i+1` carries — and the market sees it twenty milliseconds into bar `i+1`.

### Cancel latency, and two live orders

A cancellation decided at `t` — an expiry, a target that moved, a requote —
takes effect at `t + cancel`. Until then the stale quote is still standing, at
the level it was posted at, and **it can still be served. That fill is real and
the engine books it.**

The consequence is not a bug, it is the job:

> **After a requote, two orders are live for `cancel` nanoseconds** — the old
> one waiting to be cancelled and the new one waiting to be seen — **and both
> can fill.** The position then overshoots the target by one clip.

What the engine does with the overshoot: **it keeps it.** The position stands as
filled, and the strategy reduces it on the next event through its own target,
the same way it would flatten any other inventory. The engine does not net the
second fill out, does not refuse it, and does not silently shrink the new order
to make room. That is the honest form: the fill happened, the quote was in the
book, and a maker who cancels at 10 ms and requotes at 1 ms owns both sides of
that trade.

`result.order_activity` gains two counters, and only under latency:

```python
result.order_activity
# {'orders_posted': 17232, 'requotes': 17130, 'expired_unfilled': 17109,
#  'stale_fills': 84, 'overlapping_live_orders': 17130}
```

- `stale_fills` — fills booked by an order after its cancellation was decided
  and before it took effect. These are the fills a zero-latency backtest does
  not have.
- `overlapping_live_orders` — how many times a new order was posted while a
  cancelled one was still live. It is the exposure that produces the line above,
  counted whether or not anything filled.

**One stale quote at a time.** A strategy carries one resting entry order, and a
cancel latency lets the order it is replacing outlive it — no more than that.
Requote faster than your cancels are acknowledged and a real venue would hold a
third and a fourth quote; there is no ladder here to put them in, so the older
one goes when a newer cancellation arrives. That is why the overshoot is one
clip and not a compounding one: each extra live copy would be sized against a
position that does not yet include the copies still standing.

Cancel latency applies to the resting **entry** order, the channel a strategy
actually requotes on. A bracket leg is not cancelled by the strategy: it is
replaced with the position it protects.

### Brackets under latency

The two kinds of level part company, as they do everywhere else:

- an **aggressive** leg — a stop-loss, a trailing stop, a `stop_entry`, a
  `market_if_touched` — crosses the book. It is decided at the print `p` that
  reaches its level, sent then, and **executed at the first print stamped at or
  after `p + order`, at that print's price**. Between the two the price keeps
  moving, and it usually keeps moving the way that hurt: that gap is what a stop
  costs under latency, and a zero-latency backtest books it at the trigger for
  free. The execution print is normally at or beyond the level for that reason,
  but nothing forces it to be — the order was sent, and a sent market order
  fills where it lands;
- a **take-profit** rests, so it obeys the passive rule: it becomes visible
  `order` after the instant the bracket posted it, joins its queue there, and is
  served from there on.

### What it needs, and what it refuses

Latency needs **the tape**: the whole model is "which print was the first one at
or after this instant", and a bar's high and low do not carry instants. Set
`fill_model.fill_resolution="ticks"`, or run the trade clock. A run that asks
for latency without one is refused by name (`latency needs a tape`).

Refused by name, as [everywhere in this layer](#what-holds-for-every-step):
`fill_resolution="bar"`, the fast path, the CUDA sweep, the lite paths, and a
negative duration.

### What it costs

Ten BTCUSDT days, one-second bars rebuilt from the tape, a one-sided maker
requoting every bar behind the best bid, `fill_model.queue` on the stored
ten-level book, `fill_marks=True`. Same days, same decisions, same quote; only
the latency moves. The **cancel half is held at zero here**, because it pulls
the other way — a quote that cannot be withdrawn stands longer and fills more —
and this table isolates "the same quote, seen later by the market". Markouts are
in basis points over 15 000-odd entries; negative is adverse.

| order | quotes | entries | service rate | served by a traversal | half-spread captured | +1 s | +10 s |
|---|---|---|---|---|---|---|---|
| 0 | 133 509 | 15 543 | 11.6 % | 96.76 % | -0.521 | -0.638 | -0.716 |
| 20 ms | 135 072 | 15 224 | 11.3 % | 96.73 % | -0.550 | -0.674 | -0.754 |
| 100 ms | 137 441 | 15 026 | 10.9 % | 97.01 % | -0.555 | -0.688 | -0.771 |
| 500 ms | 153 559 | 13 023 | 8.5 % | 97.90 % | -0.549 | -0.710 | -0.804 |
| 2 s | 426 654 | 0 | 0 % | — | — | — | — |

Three things to read in it. The service rate falls: the quote is in the book for
less of its one-second life. The share of fills served by a print going THROUGH
the level rises, 96.8 % to 97.9 %: what is left is the flow that was leaving,
which is what adverse selection is made of. And the markout follows, 11 % worse
at one second between zero and half a second.

The last row is not a rounding error. A `{"GTB": 1}` quote on a one-second grid
lives one second; two seconds of latency mean it is cancelled before it ever
reaches the market, so it posts 426 654 quotes and fills none. That is the honest
answer, and it is the reason a duration `time_in_force` is the right one to use
on a tape.

With the cancel half switched on at the same values, the picture gains the other
half of the trade: 241 stale fills at 20 ms, 10 201 at 500 ms, 29 927 at 2 s, on
114 684 to 255 287 overlaps — a requote every second against a 20 ms
acknowledgement means the two quotes overlap essentially always and are served
together about one time in five hundred. The service rate then stops falling
(11.6 % at zero, 11.4 % at 20 ms, 12.3 % at 500 ms): a quote that cannot be
withdrawn is exposed for longer, and the two effects pull against each other.
The half-spread keeps degrading all the same, -0.521 to -0.594 bps.

---

## Execution Models

The queue and the latency are two answers to one question: what does it take for
a resting order to be served, and what does the service cost. Each of them
carries a number that decides whether a market maker is profitable, and not one
of those numbers is readable in a public archive.

This chapter is the menu of NAMED models that put those numbers on the table.
Every one of them is **off by default**: a run that names none of them is the
run the engine always did, byte for byte.

> **These parameters make the P&L. They are swept and they are published; they
> are not chosen.** A backtest that reports one number under one assumed queue
> and one assumed severity of adverse selection is reporting the assumption, not
> the market. Publish the surface: the result at each value of the axis, and the
> range over which the answer holds.

### The menu

| Setting | Model | Parameters | Default | What it assumes | Counter | Oracle |
|---|---|---|---|---|---|---|
| `fill_model.queue.model` | `"risk_adverse"` | none | **on**, the historical rule | every cancellation we cannot see happened BEHIND us, as long as the level still holds the volume ahead | `queue_decided_fills` | `hftbacktest`'s `RiskAdverseQueueModel` |
| `fill_model.queue.model` | `"power"` | `n` (finite, `> 0`) | off | a share `f(back)/(f(back)+f(front))` of every unexplained cancellation at the level happened behind us, `f(x) = x^n` | `prob_queue_decided_fills` | `hftbacktest`'s `PowerProbQueueModel` |
| `fill_model.queue.model` | `"log"` | none | off | the same, with `f(x) = log(1 + x)` | `prob_queue_decided_fills` | `hftbacktest`'s `LogProbQueueModel` |
| `fill_model.adverse.model` | `"conditional"` | `horizon` (`> 0`), `slope` (`>= 0`, per bp) | off | a print at the level serves the order only if the market then moves against it — **it reads the future** | `adverse_decided_fills`, `adverse_refused_fills` | none: a stress test, calibrated on a real fill log or on nothing |
| `fill_model.adverse.model` | `"snipe"` | `threshold_ticks` (`>= 0`), `extra_cancel_latency` (`>= 0`) | off | the venue is slowest exactly when the quote most needs withdrawing | `order_activity.sniped_fills` | none: a property of your connection to your venue |

Each model degenerates to the model it replaces at the zero of its own
parameter, **bit for bit**: `slope = 0` and `extra_cancel_latency = 0` are the
runs that never named them. That is what makes a sweep of one of them readable —
the first point of the axis is the run you already have.

### The probabilistic queues

```python
config.execution.fill_model = {
    "fill_resolution": "ticks",
    "queue": {"model": "power", "n": 3.0, "depth_source": "book"},
    # or {"model": "log", "depth_source": "book"}
}
```

`risk_adverse` refuses to move the volume ahead of an order when the level
shrinks with no print at it: a level-2 archive never says whose order left, and
assuming the cancellations were in front of us is assuming our own luck. Its one
bound is the level itself: nobody can be ahead of us for more than the level
holds, so a level shown below the volume ahead brings the queue down to what it
shows. These two answer with a stated probability instead. When the depth at our own level
falls by `chg` units and no print explains it,

```
prob      = f(back) / (f(back) + f(front))      the decrease came from BEHIND
est_front = front - (1 - prob) * chg + min(back - prob * chg, 0)
front     = min(est_front, new_depth)
```

with `front` the volume ahead, `back = prev_depth - front` the volume behind,
and `f(x) = x^n` (`power`) or `f(x) = log(1 + x)` (`log`). The depth is read
just before each print: the state stamped at a print's own instant may already
show it, and that print is counted once, as a trade, not a second time as a
decrease with nothing to explain it. The formula is
`hftbacktest`'s `ProbQueueModel`, reproduced rather than approximated so the two
engines can be compared fill by fill.

`n` steers how sharply the model believes the queue: near zero it shares every
cancellation half and half whatever the position; large, it gives almost all of
it to whichever side holds more volume. `log` is the same idea with a fixed,
gentler curvature and no parameter.

They need `depth_source="book"`. An assumed queue carries no depth at the level
to watch, so a probabilistic queue on one would silently be the historical rule
under another name; it is refused.

**The duel.** Three BTCUSDT days, the same 15 415, 16 745 and 17 232 quotes, the
same one-bar life, the same size, fed to `hftbacktest` through its own converter
for Bybit archives; neither engine reads the other's files. Agreement is per
quote, at zero latency:

| day | model | quotes | served here | served by hftbacktest | agreement |
|---|---|---|---|---|---|
| 2025-08-13 | `risk_adverse` | 15 415 | 1 653 | 1 654 | 99.994 % |
| 2025-08-13 | `power`, n=3 | 15 415 | 1 654 | 1 656 | 99.974 % |
| 2025-08-13 | `log` | 15 415 | 1 657 | 1 658 | 99.955 % |
| 2025-08-23 | `risk_adverse` | 16 745 | 733 | 732 | 99.994 % |
| 2025-08-23 | `power`, n=3 | 16 745 | 736 | 735 | 99.970 % |
| 2025-08-23 | `log` | 16 745 | 738 | 736 | 99.964 % |
| 2025-08-30 | `risk_adverse` | 17 232 | 740 | 740 | 100 % |
| 2025-08-30 | `power`, n=3 | 17 232 | 740 | 741 | 99.983 % |
| 2025-08-30 | `log` | 17 232 | 740 | 741 | 99.994 % |

Every `risk_adverse` disagreement was traced quote by quote on the raw event
stream, against a reference simulator that reproduces `hftbacktest`'s verdicts
exactly. Two remain over the three days, and neither is the rule. A print BELOW
the order's price by a buying aggressor serves it here and not in `hftbacktest`,
which reads only selling prints against a bid: an offer stood under the bid while
the order rested. A print stamped at the very instant the order leaves (the
next bar's label) is counted by the harness that drives `hftbacktest`; here the
order no longer rests at the next bar's open. The probabilistic rules keep three
to six a day, not traced yet.

### Conditional adverse selection

```python
from manifoldbt.helpers import Interval

config.execution.fill_model = {
    "fill_resolution": "ticks",
    "queue": {"depth_source": "book"},
    "adverse": {"model": "conditional",
                "horizon": Interval.millis(100),
                "slope": 0.5},
}
```

> **THIS MODEL READS THE FUTURE.** It looks `horizon` forward in the tape, past
> the print it is deciding on, and lets the fill through with a probability that
> depends on what it finds there. **It assumes the market knew.** No live
> strategy can be filled by this rule, no walk-forward validates it, and a
> result produced under it is not a result a venue could have given you.

It exists for one thing: to put a number on how much a maker's P&L depends on
the fills it gets when nothing happens next. Let `m` be the move of the
reference over `horizon` after the print, in basis points, signed so that
**positive is adverse**. The fill is granted when `u < exp(-slope * max(-m, 0))`,
with `u` a deterministic draw. An adverse or flat move is always served; a
favourable one is served less and less often as `slope` grows.

Only a print **at** the level is subject to it. A print through the level and a
book cross serve the order whatever the model thinks: on one and the same book
everything resting there was taken out.

```python
result.fill_fragility["queue"]
# {..., 'adverse_decided_fills': 812, 'adverse_refused_fills': 5031}
```

Sweep `slope` and read how fast the edge disappears as the benign fills are
taken away. A run whose two counters are wildly unbalanced is a run whose
`slope` is telling nobody anything.

### Sniping

```python
config.execution.tick_size = 0.1
config.execution.latency = {"order": Interval.millis(20),
                            "cancel": Interval.millis(20)}
config.execution.fill_model = {
    "fill_resolution": "ticks",
    "queue": {"depth_source": "book"},
    "adverse": {"model": "snipe",
                "threshold_ticks": 2.0,
                "extra_cancel_latency": Interval.millis(30)},
}
```

When the mid has moved against the side the order rests on by more than
`threshold_ticks` since the quote was posted, the cancellation takes
`cancel + extra_cancel_latency` instead of `cancel`. The quote stands longer, at
a level the market has left behind, and the fills it takes in that window are
the ones a sniper would have handed it.

It decides no fill. It changes one number — the instant a cancellation takes
effect — and everything downstream is the behaviour [Latency](#latency) already
documents: the stale fill is real, the position overshoots by one clip, the
engine keeps the overshoot. `result.order_activity["sniped_fills"]` is what the
model bought, out of `stale_fills`.

It needs `execution.latency` (there is nothing to lengthen otherwise),
`execution.tick_size` (its threshold counts in ticks) and `fill_model.queue`
(which is what loads the stored book its trigger reads). Each is refused by name.

### Two runs give the same bits

Every draw in this chapter is a **pure function** of `rng_seed`, the order and
the event: `u = f(rng_seed, order, event, stream)`. It is not a running
generator, so it does not depend on the order the events are visited in, on how
many orders the run has posted before, or on how a loop was parallelised — none
of which should move a fill. Two runs of the same config against the same store
return the same fills, the same counters and the same equity curve, byte for
byte, and adding a symbol to a run cannot move another symbol's fills.

### Writing your own

Every model above implements one interface, and so can yours. A resting order is
shown every event it could be served on and answers with a decision.

What it is shown: the order (side, level, size, remaining, post instant, age in
nanoseconds and in seconds), the volume ahead of it, the book (best bid and ask,
the depth at its level and at the best price, the mid at the post instant and
how far the mid has moved since, in ticks and in basis points, positive being
adverse), the event (the print — price, size, aggressor — or the fact that only
the book moved), whether the event went past the order rather than up to it, how
much of this print is eligible to consume it, the cancellation horizon if one is
decided, and the draw.

What it answers: how much fills now, what the volume ahead becomes, and which
channel served it. **Nobody but the model touches the volume ahead**, so the
volume it is shown is what the previous event left, before this print takes
anything off it. The historical rule, written in the interface's own words, is

```
served = min(max(print_qty_eligible - queue_ahead, 0), remaining)
ahead  = 0 if served > 0 else max(queue_ahead - print_qty_eligible, 0)
```

Two things are not negotiable. A **traversal and a book cross** are inherited by
every model: a print strictly through the level serves everything resting at it,
and so does the opposite best price crossing it. A model that served less than
the `traverse` convention would be measuring its own pessimism rather than the
market. And a rule that **cannot decide** — a NaN where a price was expected, a
division by zero — stops the run naming the event rather than serving an
arbitrary quantity.

### What refuses, and why

As [everywhere in this layer](#what-holds-for-every-step), each is refused by
name rather than approximated:

- **the fast path, the CUDA sweep and the lite paths.** None of them walks a
  tape, so none of them can serve any of this.
- `fill_model.adverse` **without** `fill_model.queue`: both models speak about an
  order RESTING in a queue, and `fill_model.queue` is also what loads the stored
  book they read.
- `"snipe"` without `execution.latency`, or without `execution.tick_size`.
- `"power"` / `"log"` under `depth_source="assumed"`.
- `"power"` without `n`, and `n` under any other model: a parameter that decides
  nothing must not sit in a config looking as though it does.
- a resting level that is not a finite price. It joins no queue and no print can
  reach it, so an order posted there would quote for the whole run and fill
  nothing.
- every parameter outside its domain, named with the key that carries it.

---

## A Service Rule of Your Own

[Execution Models](#execution-models) is a menu of named rules. This chapter is
the door out of it: the same interface, opened to a rule you write yourself.

A quant who has fitted a service curve on a year of real executions -- service
falling with the queue, service conditioned on how far the mid has moved since
the quote went in, a hazard with a slope they calibrated -- has a rule that is
in no menu and never will be. It goes here, in the same DSL the strategy is
written in, and it is driven by exactly the code `risk_adverse` is driven by:
the same [`RestingContext`](#execution-models), the same traversal floor, the
same counters.

```python
from manifoldbt.expr import col, lit, param, when
from manifoldbt.indicators import max_val, min_val

config.execution.fill_model = {
    "fill_resolution": "ticks",
    "queue": {"depth_source": "book", "assumed_queue": 2.0},
    "custom": {
        # Serve nothing until the mid has come `slope` ticks toward the level,
        # then serve what gets past the queue.
        "fill": when(
            col("mid_move_since_post_ticks") > param("slope"),
            min_val(
                max_val(col("print_qty_eligible") - col("queue_ahead"), lit(0.0)),
                col("order_size_left"),
            ),
            lit(0.0),
        ),
        "override_traverse": False,
    },
}
```

The rule is **data**: it serialises with the config, it sweeps through
`param()` like any other axis, it is refused by name when it says something the
engine cannot honour, and it is not a second language.

### What the rule decides, and what it does not

`custom` does not replace the queue. It replaces the queue's **service rule**,
so `fill_model.queue` must be set beside it and a run without one is refused by
name. The split is the whole design:

| The engine owns | The rule owns |
|---|---|
| the depth the order joins at the post instant | how much of each event reaches it |
| the traversal, and the book cross | (unless `override_traverse`) |
| | what is left of the queue afterwards |

That second column's last line is the interface's rule, not this chapter's: a
model owns the volume ahead, and the driver stores exactly what comes back. A
rule written here inherits the risk-adverse burn -- the print that went past the
order took its own size off the queue, whether or not the order was the one it
went to -- so an author writes a service rule and not a queue bookkeeping.

The rule reads `queue_ahead` as it stood **before** the current event burnt it.
That is what makes the default rule expressible exactly:

```python
fill = min_val(max_val(col("print_qty_eligible") - col("queue_ahead"), lit(0.0)),
               col("order_size_left"))
```

On two BTCUSDT days of a one-sided maker requoting every second, that rewrite
books the **same fills as `risk_adverse`, and the same equity curve byte for
byte** -- 3 607 fills on one day, 873 on the other; and a unit test puts the two
`decide` calls side by side over a grid of queues and print sizes and compares
the decisions field by field. If they parted anywhere, a rule an author
calibrated would be measured against a service the engine does not implement.

**The traversal is a fact, not a model.** A print strictly through the level, or
the best opposite price crossing it, means everything resting at that price went
first. The engine serves the order's whole remainder before the rule is ever
asked, and the fill is counted where it always was (`traverse_fills`,
`fills_from_book_cross`). An author who has measured otherwise -- an archive with
transient crossings, a venue with hidden size -- sets `"override_traverse": True`
and sees those events with `is_traverse = 1`, `print_qty_eligible` equal to the
order's whole remainder, and answers for them too. On the maker above,
traversals are 94 % of the fills, so a rule that does not take them over is only
choosing on the other 6 %.

### The fields a rule reads

They are read as columns, and they are the only columns a rule may name.

| Field | What it is |
|---|---|
| `queue_ahead` | volume resting ahead of the order, before this event burns any |
| `order_size_left` | quantity still to fill |
| `order_age` | seconds since the order reached the market (after `latency.order`) |
| `level` | the price the order rests at |
| `depth_at_level` | depth stored at that level, as of this event |
| `best_bid` `best_ask` `mid` | the stored book's top, as of this event |
| `mid_move_since_post_bps` | mid move since the post, **positive is adverse** -- down under a bid, up over an ask, which is also the direction that serves it |
| `mid_move_since_post_ticks` | the same move in ticks; needs `execution.tick_size` |
| `print_price` `print_qty` | the print; `NaN` and `0` on a book update |
| `print_aggressor` | `+1` the aggressor bought, `-1` it sold |
| `print_qty_eligible` | the most this event can serve, whatever the rule answers |
| `is_book_update` `is_traverse` | `1` / `0` |
| `u` | a uniform draw on `[0, 1)`, one per (order, event) |

Plus `lit()` and `param()`. Operators: arithmetic, comparisons (`> < ==`, and
the `>= <=` built from them), the boolean operators, `when()`, `abs`, `min_val`,
`max_val`, and `round_to` / `floor_to` / `ceil_to`.

The value is the quantity served, truncated to
`min(print_qty_eligible, order_size_left)`; `0` is "nothing here".

These are the interface's own fields, read through
[`RestingContext`](#execution-models) rather than through a second context of
this chapter's making: a rule written here and a model of the menu cannot
disagree about what "the mid" was.

`u` is what makes a probabilistic rule reproducible: it is a pure function of
the run's seed, the order and the event, so two runs of the same config draw the
same numbers, no run carries a generator state, and adding a second symbol
cannot move the first one's draws.

### What is refused, and why

Everything outside that grammar is refused **by name at compile time**, before a
day is read, and the two families that matter are refused for two different
reasons:

- **an operator with memory** -- a rolling window, a lag, `ffill`, an indicator.
  A rule is asked one question about one print and holds no history; the only
  history it gets is the one the context carries explicitly (`order_age`,
  `queue_ahead`, `mid_move_since_post_*`). An operator with memory would need a
  state per resting order per node, kept in step with requotes and partial
  fills, and a rule whose state silently resets on a requote is worse than no
  rule.
- **a strategy signal** -- any column that is not a field above. A signal is
  computed in batch, before the loop, on the simulation's rows; a resting order
  is served between them. There is no row to read, and reading the nearest one
  would be a lookahead nobody asked for.

`position()` is refused too: the position is what the fills produce, and a rule
that reads it while deciding one is a loop. Skew the quote instead, in the entry
order's level, where [`position()`](#reading-the-position-from-a-level) is read.

And a rule is refused before the walk when it reads something the run cannot
answer: the book (`mid`, `best_bid`, `best_ask`, `depth_at_level`,
`mid_move_since_post_*`) with no book stored, or `mid_move_since_post_ticks`
with no `execution.tick_size`.

Refused as everywhere in this layer: the fast path, the CUDA sweep and the lite
paths, each by name.

### NaN is loud

**A rule may be wrong. It must never be silent.**

Every field that cannot be answered is `NaN` rather than a plausible number, and
`NaN` propagates through the arithmetic **and through the comparisons** -- a
comparison against an unknown is unknown here, not false. A rule that answers
`NaN` stops the run, naming the event and listing which fields could have been
the unknown one. So does a negative quantity, and an infinity.

```
custom fill rule returned NaN at event 41207 (order resting at 63461.9 since
1750550400123456789 ns, queue ahead 0.24). A field it read had no value: ...
```

Two deliberate exits from that rule:

- `min_val` and `max_val` keep the semantics they have everywhere else in the
  DSL (they return the other operand when one side is `NaN`). They are how a
  rule says "and if that is unknown, use this instead", in writing.
- `when()` evaluates only the branch it takes, so a guarded rule
  (`when(col("is_book_update") > lit(0), lit(0), f(col("print_price")))`) never
  touches the field it guarded against.

### What comes out

A rule you wrote is one more model behind the shared interface, so its counters
land where every other model's do -- inside `fill_fragility["queue"]`, and only
when one ran:

```python
result.fill_fragility["queue"]
# {'queue_decided_fills': 55, 'traverse_fills': 2770, 'partial_fills': 24,
#  'book_unknown_at_post': 1809, 'fills_from_book_cross': 1157,
#  'custom_decided_fills': 55, 'custom_rule_events': 5156,
#  'custom_declined_events': 5101, 'custom_truncated_answers': 0}
```

- `custom_decided_fills` -- fills the rule had the last word on. It sits BESIDE
  the channels rather than replacing them: a rule still fills through a queue, a
  traversal or a book cross, and those keep their meaning.
- `custom_rule_events` -- events it was asked about. An event nothing could
  serve never reaches it.
- `custom_declined_events` -- of those, the ones it answered zero on.
- `custom_truncated_answers` -- answers the engine cut down to what the event
  could serve. **This is the one to read first**: a rule that is systematically
  truncated is not the rule its author thinks they wrote.

### Sweeping a rule

A `param()` inside the rule is an ordinary sweep axis, provided the strategy
declares it (`.param("slope", default=0.0)`) -- that is where a reader looks for
the axes of a run, and it is what makes the manifest label each row. It crosses
with [`execution_grid`](#sweeping-execution-parameters) like any other: the
`param()` moves what is INSIDE the rule, a path like
`fill_model.custom.override_traverse` moves the setting around it, and nothing
in the engine has to know the two belong to the same object.

Two BTCUSDT days, the maker above, the rule taking the traversals over, swept on
`slope` (how far the mid must have come toward the level) and `p` (the
probability that a qualifying print serves it, through `u`):

| slope | p | service rate | +1 s markout |
|---|---|---|---|
| -1.0 | 0.25 | 43.7 % | -0.869 |
| -1.0 | 0.50 | 48.2 % | -0.853 |
| -1.0 | 0.75 | 50.0 % | -0.851 |
| -1.0 | 1.00 | 50.6 % | -0.846 |
| 0.25 | 0.25 | 30.2 % | -0.976 |
| 0.25 | 0.50 | 35.8 % | -0.955 |
| 0.25 | 0.75 | 39.3 % | -0.946 |
| 0.25 | 1.00 | 42.7 % | -0.948 |

In basis points, negative being adverse; 2026-06-22, 6 300 to 9 500 quotes per
cell; the second day gives the same shape (19.7 % to 23.0 %, -0.621 to -0.605). Both axes move both numbers, and the slope moves them the way the service
rule predicts rather than the way one would hope: **demanding that the mid come
to you before you are served cuts the service rate by a third and makes the
selection 10 % worse**, because the fills that survive the filter are precisely
the ones the market walked into. That is the shape a cost axis is supposed to
have, and it is why these are swept and published rather than chosen once.

### A callable, for research

Fitting a rule is an iteration loop, and an author should not have to express a
half-formed idea in the DSL to look at it once:

```python
def rule(queue_ahead, order_size_left, order_age, level, depth_at_level,
         best_bid, best_ask, mid, mid_move_since_post_bps,
         mid_move_since_post_ticks, print_price, print_qty, print_aggressor,
         print_qty_eligible, is_book_update, is_traverse, u):
    left = print_qty_eligible - queue_ahead
    return min(left, order_size_left) if left > 0 else 0.0

config.execution.fill_model = {
    "fill_resolution": "ticks",
    "queue": {"depth_source": "book", "assumed_queue": 2.0},
    "python": rule,                                   # or {"fill": rule, "override_traverse": True}
}
```

The seventeen fields arrive as **positional** arguments, in the order of the
table above, and the callable returns the quantity served. It is checked the
same way a rule written in the DSL is.

Two things it is not. It is **not part of a config**, so the run it produced is
described only beside its code: `mbt.run` carries it, and every other driver
(`run_sweep`, `run_batch`, and the lite paths) refuses it by name rather than
dropping it and returning N results that look like answers. And it is **slow**:
the cost is the boundary itself.

### What it costs

One BTCUSDT day, 746 414 prints, seven interleaved repetitions, medians.

**What a run costs**, on the parity config -- `risk_adverse`, the same rule
rewritten in the DSL, and the same rule as a callable, all three booking the
same 3 607 fills, so all three walk the same order lives:

| | prints/s |
|---|---|
| `risk_adverse` | 4 140 000 |
| the same rule, written in the DSL | 3 820 000 |
| the same rule, as a callable | 3 180 000 |

**What one answer costs**, isolated: a quote given a minute of life, taking the
traversals over, answering zero so nothing fills -- 386 354 calls, and the three
rules walk exactly the same events:

| | ns per call |
|---|---|
| the Python boundary | 344 |
| ~89 more nodes of DSL | 537, so **~6 ns a node** |

A one-node rule written in the DSL costs less than this machine's run-to-run
noise; the boundary into Python costs about as much as sixty nodes of the
expression it replaces. Read the order rather than the absolute figures: the
same measurement on a loaded machine moves every number and none of the
comparisons.

---

## Marking a Fill

Fees and a fill count say what a strategy paid to trade. They do not say what
it paid to *be quoted*: a resting order is served, on average, by the flow that
was about to move the price against it. `execution.fill_marks=True` measures
that, and puts it beside the half-spread the quote earned.

```python
config.execution.fill_marks = True          # needs the tape (see below)
result = mbt.run(strategy, config, store)

result.fill_marks          # the aggregates, as a dict
result.fill_marks_df()     # one row per fill
mbt.plot.fill_marks(result)
```

**The convention**, in five lines, because a markout is only comparable when
everyone says how they took it:

- the **anchor** is the CLOSE of the bar the fill was booked on -- the first
  instant at which the fill is certainly done. A fill carries its bar's label,
  not the instant inside it, so nothing is ever read from before the fill. On
  the trade clock a bar is one print, so the anchor is the next print;
- the **reference** at an instant is the mid of the stored book as of it
  (`bt.ingest_book`), or the price of the last tape print at or before it when
  no book answers. Every mark says which of the two it read;
- the **marks** are the reference at the anchor and at +100 ms, +1 s and +10 s
  after it. An instant past the end of the data is `NaN`, never the last known
  value;
- the **sign** is `+1` on a buy and `-1` on a sell: the direction the fill takes
  the position in. A fill that closes a position is signed by the leg it
  executed, not by the flat it leaves, and `position_after` is given per fill so
  you can keep the opening fills alone;
- the **markout** at horizon `h` is `1e4 * sign * (reference - fill price) /
  fill price`, in basis points of the fill price. **Negative is adverse
  selection**: the market left in the direction that hurts.

`result.fill_marks` carries one entry per horizon -- `mean_bps`, `median_bps`,
`stderr_bps` (a thousand bootstrap resamples of the fills, seeded, so the same
fills give the same bar twice) and `marked_fills` -- plus two numbers to read
them against, when a book was stored: `half_spread_captured_bps`, the mark at
horizon zero, and `book_half_spread_bps`, the market's own half-spread at the
same instants, which is the most a quote resting at the touch could have earned.

```python
{'anchor': 'bar_close',
 'horizons': [{'horizon': '100ms', 'horizon_ns': 100000000, 'mean_bps': -0.61,
               'median_bps': -0.02, 'stderr_bps': 0.04, 'marked_fills': 1586},
              {'horizon': '1s',    'horizon_ns': 1000000000,  'mean_bps': -1.23, ...},
              {'horizon': '10s',   'horizon_ns': 10000000000, 'mean_bps': -1.41, ...}],
 'half_spread_captured_bps': 0.0062, 'half_spread_fills': 1586,
 'book_half_spread_bps': 0.0064, 'fills_total': 1586,
 'bootstrap_draws': 1000, 'bootstrap_seed': 20260907}
```

Those orders of magnitude are the point: a captured half-spread of six
thousandths of a basis point against more than a full basis point of drift at
one second. A maker on that book does not lose because its fills were modelled
loosely, it loses because a resting order is served when the price is leaving.

`fill_marks_df()` gives the detail: `timestamp` (the fill's row), `anchor`,
`symbol_id`, `side`, `price`, `qty`, `position_after`, `ref_at_fill` and
`ref_source`, then `mark_100ms` / `mark_1s` / `mark_10s` each with a
`source_100ms` / `source_1s` / `source_10s` beside it (`"book"`, `"tape"`, or
nothing at all past the end of the data), and `book_half_spread_bps`.

Two things it needs, and one it costs:

- **the tape.** A bar's high and low say a level was reached, not what stood at
  the touch a hundred milliseconds later, so a run on bars alone is refused by
  name. Set `fill_model.fill_resolution="ticks"` or run on the trade clock.
- **a book, to separate the two halves.** Without one the marks read the last
  print, which is the mid give or take a half-spread -- enough for the drift,
  not enough to say what the quote earned, so the half-spread fields stay `NaN`.
- **one ordered walk over the tape, after the loop.** The simulation never reads
  a mark: a run that does not ask for them is byte for byte the run it always
  was, which is why the setting is off by default.

Marks are bar-quantised on bars. The anchor sits up to one bar after the fill,
so on coarse bars the captured half-spread carries whatever the market did
inside that bar; the finer the rows, the closer it gets to the quote's own edge.

The marks of a run and the marks of a broker's own fills are the same
function called twice, which is what [`bt.reconcile`](#calibrating-against-live-fills)
is built on.

They compose with [`fill_model.queue`](#queue-model), and the pair is worth more
than either alone: the queue says how each fill was served, the marks say what
that service cost, and the two read one shared copy of the book. See
[With the marks](#with-the-marks).

---

## Calibrating Against Live Fills

Every setting in the four chapters above is an assumption. How long was the
queue in front of the order, how fast did the volume ahead of it cancel, how
late did the quote reach the book: each of them makes the P&L of a maker, and a
backtest cannot check any of them against itself. It can only be internally
consistent, and a consistent story about a queue is still a story.

The check is a journal of the fills a broker actually granted over the same
days. `bt.reconcile` marks both sides with the same code, against the same tape
and the same book, pairs them one to one, and says where they part company.

```python
result = mbt.run(strategy, config, store)     # execution.fill_marks = True

rec = mbt.reconcile(result, broker_fills, store=store, symbol_id=1,
                    tolerance=Interval.seconds(1))

print(rec.summary())      # one factual sentence per quantity
rec.horizons_df()         # markout of each side, and the paired difference
rec.by_hour_df()          # fills served, both sides, by hour of the UTC day
rec.by_day_df()
rec.pairs_df()            # one row per matched pair, with both sides' marks
mbt.plot.reconcile(rec)
```

`broker_fills` is a DataFrame with four columns: a UTC `timestamp`, a `side`, a
`price` and a `qty`, plus a numeric `symbol_id` when the run held more than one
symbol. The common broker spellings are read without a rename (`time`,
`quantity`, `avg_price`, `B`/`S`, `buy`/`sell`, ...); a side that reads as
neither a buy nor a sell is refused by name rather than taken for a sell, which
would flip the sign of every mark it touches. Any other column the export
carries -- `order_id`, `posted_at`, the level the order rested at -- is yours to
keep; nothing here reads it.

**The conventions are the marks'**, unchanged and not configurable: the same
anchor (the close of the simulation row the fill falls in), the same reference
(the stored book's mid, the last print otherwise), the same three horizons, the
same sign. The days are reloaded from the run's own manifest, so the reference
the journal is marked against is the reference the backtest was marked against,
byte for byte. That is what makes the first check below possible.

### What it answers

**Service.** How many of the posted quotes each side served, globally, by UTC
day and by hour of the day. `service["live_rate"]` and `service["sim_rate"]`
share one denominator -- the quotes the BACKTEST posted, from
`result.order_activity["orders_posted"]` -- because the two sides ran the same
decisions, so the same posting count prices both. This is the number a queue
model exists to get right, and the one it gets wrong first.

**Adverse selection.** The markout of each side at +100 ms, +1 s and +10 s, each
with its bootstrap standard error, and then the difference taken PAIRWISE over
the fills that matched. The paired form is the one to read: the two populations
do not hold the same fills, so `diff_mean_bps` mixes execution with composition,
while `paired_mean_bps` compares the same fill with itself and its error bar is
smaller by an order of magnitude.

**Where the two disagree.** The distribution of the price gap
(`price_gap_bps`, in basis points of the backtest price, **negative when the
real fill was worse**) and of the time gap over the paired fills, the share of
real fills with no backtest equivalent, and the share of backtest fills the
market never granted.

**What the quote earned.** The captured half-spread on both sides, against the
market's own half-spread at the same instants.

### Matching

One to one, on the same symbol and the same side, inside `tolerance`. A real
fill takes the closest free backtest fill in the window; ties go to the smaller
price gap and then to the earlier row, so the answer never depends on a thread
or a hash order. A backtest fill is taken at most once: two real fills a
millisecond apart cannot both be "the same fill" as one simulated one, and
pretending they are would hide exactly the over-service this measurement exists
to find. Nothing is matched across sides, however close.

A window wider than the strategy's own requote interval starts pairing a real
fill with the wrong quote, so `tolerance` is worth setting deliberately.
`price_tolerance` is optional and off by default: the price gap is a
MEASUREMENT here, and a tolerance narrower than that gap would hide it by
turning both fills into unmatched ones. Set it to the venue's tick when you
want two fills at different prices refused as different fills.

### Check it against itself first

The first thing to run is the run's own fill log:

```python
log = result.trades_df()
same = mbt.reconcile(result, log.rename(columns={"execution_timestamp": "timestamp",
                                                 "fill_price": "price",
                                                 "quantity": "qty"}),
                     store=store, symbol_id=1)
assert same.matched == same.sim_fills          # every fill paired with itself
assert all(h["paired_mean_bps"] == 0.0 for h in same.horizons)
```

Zero everywhere, exactly, by construction: the two sides went through one call
each into the same marking code over the same series. If that is not zero, the
store no longer holds the days the run covered, and nothing a real journal says
can be trusted either. `examples/29_reconcile_live_fills.py` runs this check,
then degrades the same log on purpose -- forty per cent of the fills dropped,
the survivors priced half a basis point worse -- and reads the degradation back
out of the measurement.

### The limit, said plainly

**Reactive adverse selection does not simulate. It measures, here.** A backtest
replays a tape that never saw your quotes: nobody in it pulled a bid because
your order appeared, and nobody leaned on you. Its adverse selection is
whatever the market was going to do anyway. A real journal's fills were served
by participants who could see the quote and react to it, and the paired
difference at each horizon is that reactive part -- the whole of it, including
the parts nobody has a model for. No queue depth, no cancellation rate and no
latency will produce it, because none of them describe a market that is
answering you.

That is why this measurement is the only thing that makes an execution model
credible, and why nothing in the output names a setting to change. What it
gives you is the target; the next section is how the choice of setting gets
made in the open.

---

## Sweeping Execution Parameters

**The execution parameters make the P&L. They are swept and published as a
surface. They are not chosen.**

A queue depth of two rather than five, a round trip of twenty milliseconds
rather than a hundred: on a maker those are not details around the edge of a
result, they are most of the result. None of them is knowable from a backtest,
and a backtest that fixes them reports one point of a surface as if it were the
answer. So they go on axes, beside the `param()` grid:

```python
sweep = mbt.run_sweep(
    strategy, {"edge_bps": [0.5, 1.0, 2.0]}, config, store,
    execution_grid={
        "fill_model.queue.assumed_queue":      [1, 2, 5],
        "latency.order":                       [Interval.millis(0),
                                                Interval.millis(20),
                                                Interval.millis(100)],
        "fill_model.queue.cancel_ahead_rate":  [0.0, 0.1],
    },
)

df = sweep.to_df()          # exec_* columns beside the param_* ones
```

Each key is a **dotted path into `config.execution`**, and that is all the
engine knows about it: the path is resolved generically against the config's
own shape, so a setting that exists can be swept, whatever it is.
`signal_delay`, `max_participation_rate`, `tick_size`, `passive_fill`,
`fill_model.queue.depth_source` are axes for the same reason
`fill_model.queue.assumed_queue` is.

Values are written the way the config takes them, including durations:
`Interval.millis(20)` is passed as nanoseconds for you.

### Reading the surface

`sweep.to_df()` gives one row per combination, labelled with `exec_<path>`
columns read from **the config each run actually executed under** (its own
manifest), so a row's label cannot drift from the run that produced it.

Six columns say what the execution did, and `bt.sweep_columns` serves them off a
full sweep:

```python
cols = mbt.sweep_columns(list(sweep), [
    "service_rate",               # maker fills over the quotes posted
    "queue_decided_fills",        # fills the queue granted, volume ahead consumed
    "stale_fills",                # fills taken after a cancel, before it took effect
    "adverse_1s_bps",             # mean markout at 1 s, negative = adverse
    "adverse_10s_bps",
    "half_spread_captured_bps",   # what the quote earned at the touch
])
```

The three markout columns need `execution.fill_marks=True`; each column is
`NaN` on a run that configured no model behind it (no queue, no latency, no
marks). They come from the counters and the marks of a FULL result, so asking a
LITE sweep for one is refused by name -- the lite drivers walk no tape.

A flat axis is worth a second look, and the engine has already removed one
explanation for it: a path that names no setting, or a value the model refuses,
fails **by name before a day is read**. So an axis that comes back flat is flat
because the setting does not move this strategy, not because it was silently
dropped. `fill_model.queue.assumed_queue` under `depth_source="book"` is the
common case: it is only the declared fallback for an instant the stored book
cannot answer, so a strategy whose level always sits on a quoted price will not
feel it at all. Under `depth_source="assumed"` the same axis moves service by a
factor of two.

### The order the combinations come in

Same rule as `param()`, for the same reason: **axes sorted by name, the last
axis varying fastest**, whatever order the dict was written in. A reshape on the
dict's insertion order silently transposes the surface.

The execution axes are the **slowest** axes of the sweep: the results are one
block of the whole parameter surface per execution combination. So with `P`
parameter combinations and `E` execution ones, result `i` ran under execution
combination `i // P` and parameter combination `i % P`, and an empty
`execution_grid` leaves the enumeration exactly as it was.

```python
from manifoldbt.dataframe import grid_combos, exec_grid_combos
exec_grid_combos(execution_grid)    # the execution combinations, in order
grid_combos(param_grid)             # the parameter ones, in order
```

### What it costs, and what refuses

The days are loaded and aligned ONCE for the whole sweep, at the depth the
deepest combination needs -- a grid where one combination reads the tape loads
the tape for all of them -- and every combination then simulates under its own
config. Measured on one UTC day of Bybit BTCUSDT spot (50,215 one-second bars,
1.4M prints, tick-resolved fills, a book-backed queue, a latency and the marks):
a single run is 130 ms, of which 71 ms is loading and aligning; the 3 x 3 x 2
grid above is 238 ms in total, **13.2 ms per combination**. The shared load is
the whole saving, and on a short run it is most of the bill.

All of it stays on the general simulation loop, and the refusals are the ones
the rest of this layer already had, applied per combination rather than once:

- `run_sweep_lite` and `run_batch_lite` walk no tape, so a combination
  configuring `fill_model.queue`, `execution.latency` or the trade clock is
  refused by name, exactly as a base config would be. Axes that need no tape
  (`signal_delay`, `max_participation_rate`) sweep there normally;
- the CUDA sweep takes one execution config, so `device="cuda"` with an
  `execution_grid` is refused rather than run against the base config;
  `device="auto"` simply stays on the CPU;
- `run_batch` takes the same `execution_grid`, with the same enumeration: one
  block of the whole batch per execution combination.

Two sweeps of the same grid over the same store return the same bits, on every
column, like every other run in this layer.

---

## Orders, One by One

`mbt.run` asks a strategy for a target position per row and turns it into
orders itself. `mbt.sim` does the opposite: it hands a Python function a
simulated venue, and the function sends its own orders, one by one, as it would
to an exchange. It reads the book and the tape as they reach it, posts, cancels
and replaces, and learns what happened to each order when the venue's answer
comes back. Four latencies separate it from the venue.

Every order resting in the book is served by the same queue model as
`fill_model={"queue": ...}` (see [Queue Model](#queue-model)): a strategy that
rests one order at a time gets the same fills through either door.

```python
import manifoldbt as mbt
from manifoldbt.helpers import Interval

def maker(sim):
    while sim.elapse(Interval.millis(100)):       # False once the data is over
        book = sim.book("BTCUSDT")                 # as it reached us
        if book.best_bid is None:
            continue
        if not sim.orders("BTCUSDT", side="buy", status="live"):
            sim.post("BTCUSDT", "buy", book.best_bid, 0.01, tif="GTX")

config = mbt.sim.Config(
    latency={"entry": Interval.millis(5), "response": Interval.millis(5)},
    queue={"depth_source": "book", "assumed_queue": 2.0},
)
res = mbt.sim.run(maker, config, store, symbols=["BTCUSDT"],
                  start="2025-08-13", end="2025-08-14")
res.fills_df()
```

The store needs the book and the tape of every symbol over the range
(`mbt.ingest_book`, `mbt.ingest_trades`, with the same `category` for a perp).
The function returns when it wants
the run to end; the clock never goes back. A complete maker is in
`examples/30_order_api_market_maker.py`.

`levels=` (on `mbt.sim.run` and `mbt.sim.Market.from_store`) bounds how many
levels per side of the stored book the venue and the strategy read: **every
stored level by default**, at most `levels` when it is given (the stored depth
when the book is shallower). Past it the book says nothing, as past the stored depth: `levels()`
stops there, and an order resting further out joins a queue the book cannot
measure (`assumed_queue`, or a refusal by name).

### The strategy's side

| Call | What it does |
|---|---|
| `sim.now()` | The strategy's local time, in nanoseconds |
| `sim.elapse(dt)` | Move the clock `dt` forward; `False` once it reaches the end of the data |
| `sim.wait(trade=, book=, order=, timeout=)` | Move the clock to the next event of the kinds named (any kind when none is), or `timeout` from now; returns why it woke |
| `sim.book(symbol)` | The book as the strategy sees it: `best_bid`, `best_ask`, `mid`, `spread`, `qty_at(p)`, `levels(n)`, `depth_through(p)`, `tick` |
| `sim.trades(symbol)` | The prints that reached the strategy since the last call, as `(ts_ns, price, qty, aggressor)` |
| `sim.orders(symbol, side=, status=)` | The strategy's orders as it knows them. `status="live"`: may still fill; `"resting"`: in the book, partially filled or not; or a status name |
| `sim.order(id)` | One order: `price`, `qty`, `filled`, `remaining`, `avg_price`, `status`, `queue_ahead`, `cancel_pending`, ... |
| `sim.position(symbol)`, `sim.cash()` | As the fills that reached the strategy say |
| `sim.post(symbol, side, price, qty, tif=)` | Send an order and get its id. `price=None` is a market order; a price is put on the tick grid first (see below) |
| `sim.cancel(id)` | Send a cancellation |
| `sim.replace(id, price=, qty=)` | Change the price, the total size, or both; a new price is put on the tick grid first |

Durations are an `Interval`, a `datetime.timedelta` or nanoseconds. A symbol is
its ticker or its index in `symbols`.

`tif` is `"GTC"` (the default), `"GTX"` (post-only), `"IOC"`, `"FOK"` or
`"GTD"` with `expire_at=` (a venue instant in nanoseconds).

### Four delays

```python
mbt.sim.Config(latency={
    "entry": Interval.millis(5),     # a post or a replacement reaches the venue
    "cancel": Interval.millis(5),    # a cancellation does ("entry" when left out)
    "response": Interval.millis(5),  # an acknowledgement or a fill comes back
    "feed": Interval.millis(2),      # a book state or a print reaches the strategy
})
```

Each key is optional and zero when left out.

- An order sent at `t` reaches the venue at `t + entry`. Until then it is in
  nobody's queue: a print at its price in the meantime does not serve it.
- A cancellation sent at `t` takes effect at `t + cancel`. Until then the order
  can still fill, **and that fill is real**. It is booked, and the venue then
  answers the cancellation with a refusal.
- What the venue does at `T` reaches the strategy at `T + response`: the order
  stays `in_flight` until its acknowledgement arrives, and `position()` moves
  when the fill arrives, not when it happens.
- A book state or a print stamped `T` is seen at `T + feed`: `sim.book()` shows
  the stored state as of `now - feed`.

`queue_ahead` on an order is the venue's estimate as of its last answer about
that order (acknowledgement, fill, replacement), delayed like any other answer.
On a market stored order by order it is exact and moves with the feed: see
[A Market Stored Order by Order](#a-market-stored-order-by-order).

### What the venue does

At one venue instant the market comes first: the prints stamped at that
instant, then the book state that shows them, then the actions that arrive at
that instant, in the order they were sent. So an order arriving at `T` joins the
queue as the state of `T` shows it, and the prints of `T` do not serve it.

**Every price is on the tick grid.** The venue quotes on the symbol's tick (the
store's `tick_size`, or the `tick_size` a market was built with), and an order's
price is put on it when the order is sent, before anything reads it:

- a price already on the grid is kept, bit for bit;
- a price within a millionth of a tick of a grid point is that point: it is the
  float error of arithmetic on grid prices, `116994.4 + 0.2` giving
  `116994.59999999999`;
- any other price moves to the passive side, a bid **down** and an ask **up**,
  as `.floor_to` / `.ceil_to` do, and is counted in `res.levels_snapped`;
- a buy below the first tick is refused.

The order carries that price from then on: `sim.order(id).price`, the journal
and the fills show it. Without it, a sell a hair under `116994.6` would read a
print AT `116994.6` as one THROUGH its price and be served whole, ahead of the
queue the same order posted at `116994.6` waits in. A symbol without a tick
(`tick_size=0`) has its prices put on the stored book's own price grid, eight
decimals, and nothing else moves.

**An order that would not execute on arrival rests.** Its queue is the size the
book shows at its price, plus the strategy's own earlier orders at that price
(first come, first served among them). Prints at its price burn that queue, and
what they leave past it serves the order, in part or in full. The queue can
never exceed the size the level shows. A print strictly through the order's
price, whichever side crossed, or an opposite best price at or through it,
serves the rest: the level was emptied.

**An order that would execute on arrival** takes the displayed opposite ladder,
level by level, within its limit, and the rest of it expires. The simulated
fills take nothing out of the stored book, so a rest left there would be
crossed at once by the very liquidity it just took. A `GTX` that would execute
is rejected; a `FOK` that cannot fill whole takes nothing; a market order takes
what the ladder offers.

**A replacement** at the same price with a smaller size keeps the order's place
in the queue; any other change sends it to the back of the queue at its new
price. A new price that would execute is refused, and the order stays as it
was.

**Fees** are the maker fee on a fill of a resting order and the taker fee on a
fill at arrival (`maker_fee_bps`, `taker_fee_bps`).

The simulated orders do not move the market: the stored book and tape are
replayed as they were.

### What comes out

| | |
|---|---|
| `res.orders_df()` | Every order as it ended |
| `res.fills_df()` | Every fill at the venue's instant, with its `channel`: `queue`, `traverse` (a print through the price), `book_cross`, `taker` |
| `res.events_df()` | The order journal: what the strategy sent (local time) and what the venue did (venue time) |
| `res.equity_df()` | Cash plus each position at its venue mid, at each wake-up |
| `res.metrics` | Orders, fills, volume, fees, P&L, maximum drawdown, final position |
| `res.fill_marks` | The markouts of [Marking a Fill](#marking-a-fill), with the horizons counted from the fill itself (`anchor == "fill"`) |
| `res.fill_fragility` | What the queue decided for the orders that rested, in the shape of [Queue Model](#queue-model): `maker_fills`, and under `queue` the `queue_decided_fills`, `traverse_fills`, `fills_from_book_cross`, `partial_fills` and `book_unknown_at_post` (orders posted behind `assumed_queue`); `would_fill_touch` / `would_fill_traverse` count the lives of an order at one level that a print reached, or went through. `touch_only_fills` is a bar convention and stays `0` |
| `res.levels_snapped` | Orders whose price was between two ticks and moved to the passive side |

### A market without a store

`mbt.sim.Market.from_ladders` builds a market from book states and prints
written out by hand, which is how to test a strategy on a scenario small enough
to follow:

```python
market = mbt.sim.Market.from_ladders(
    "BTCUSDT", tick_size=0.1,
    states=[(1_000_000, [(100.0, 2.0), (99.9, 9.0)], [(100.1, 5.0)])],
    trades=[(3_000_000, 100.0, 3.0, "sell")],
)
res = mbt.sim.run(strategy, markets=[market])
```

### Against another engine

On three BTCUSDT days, the quotes of the queue duel replayed through `mbt.sim`
give exactly the verdicts of `mbt.run`: 49,392 quotes, no difference.

A maker quoting both sides every second, written for `mbt.sim` and for
hftbacktest 2.4.4 (partial fills, `risk_adverse`), agrees quote by quote on
99.990 % to 99.998 % of 518,382 quotes, at every latency from 0 to 500 ms. The
differences come from two named rules:

- a print through our price whose aggressor is on our side (a buyer below our
  bid) serves the order here, because the level was emptied; hftbacktest reads
  only the prints of the other side;
- a book update filed before the print it already shows counts that print
  twice in hftbacktest's queue; here the prints of an instant come before the
  state that shows them.

### What refuses, and why

- A symbol the run does not hold, a side other than `"buy"`/`"sell"`, an unknown
  `tif`, a `"GTD"` without `expire_at`, a size or a price that is not finite and
  positive: refused when the call is made, by name.
- Cancelling or replacing an order the strategy knows to be finished, or a
  market order's price: refused. Cancelling an order whose cancellation is
  already on its way does nothing.
- A latency key other than the four, a negative latency, an unknown queue key:
  refused before the run starts.
- A symbol with no stored book over the range: refused by name ("ingest it
  first"). A range with no prints is accepted: the book alone can cross an
  order.

### What it costs

Everything between two wake-ups runs in Rust. A step with more than about a
thousand venue events to play releases the GIL for the rest of it, so other
Python threads run while the venues catch up; a shorter one keeps it, since
releasing it would cost more than the step. On a BTCUSDT day of 1.2 million
events (430,000 book states, 770,000 prints):

- the event loop alone processes 37 million events per second;
- a strategy that wakes every 100 ms and does nothing costs 0.12 µs per
  wake-up, 0.36 µs when it reads the book and its live orders;
- a maker that reads, posts and cancels at every wake-up runs the whole day in
  0.66 s, 0.77 µs per wake-up (864,000 wake-ups, 96,000 orders).

A `Book` reads the stored state in place, so its cost does not grow with the
depth of the book. What does grow with it is loading a day stored as whole
ladders: `levels=` bounds it to the levels the strategy needs. `sim.orders()` filtered on `"live"`, `"resting"`,
`"in_flight"` or `"partially_filled"` reads the live orders alone and costs the
same whatever the history; any other filter walks every order sent.

---

## Quoting in the DSL

`mbt.sim` hands the venue to a Python function, which decides at every wake-up.
When the decision fits in a few expressions -- where to stand, how much, and
whether to stand at all -- it can be written as **quotes** instead, in the same
DSL as the signals of `mbt.run`. The engine compiles them once and runs the
whole span on the venue of [Orders, One by One](#orders-one-by-one), without
calling back into Python, and a strategy written this way sweeps like any
other.

```python
import manifoldbt as mbt
from manifoldbt import book
from manifoldbt.helpers import Interval, time_range

CLIP, MAX_INVENTORY = 0.01, 0.05

inventory = mbt.position()
clips = mbt.round(inventory / CLIP)                     # one level back per clip held
bids, asks = book.bid_levels(), book.ask_levels()
book_ok = (bids > 0) & (asks > 0)
bid = book.bid_price_at(mbt.clip(clips, 0, bids - 1) + 1)
ask = book.ask_price_at(mbt.clip(-clips, 0, asks - 1) + 1)

strategy = (
    mbt.Strategy.create("maker")
    .quote("buy", mbt.when(book_ok, bid), CLIP, tif="GTX",
           enabled=inventory < MAX_INVENTORY - 1e-12)
    .quote("sell", mbt.when(book_ok, ask), CLIP, tif="GTX",
           enabled=inventory > -MAX_INVENTORY + 1e-12)
)

start, end = time_range("2025-08-13", "2025-08-14")
config = mbt.BacktestConfig(
    universe=[1], time_range_start=start, time_range_end=end,
    bar_interval=Interval.millis(100),                 # the wake-up clock
    initial_capital=100_000.0,
    fees=mbt.FeeConfig(maker_fee_bps=0.0, taker_fee_bps=4.0),
    execution=mbt.ExecutionConfig(
        latency={"order": Interval.millis(5), "cancel": Interval.millis(5),
                 "response": Interval.millis(5), "feed": Interval.millis(2)},
        fill_model={"queue": {"depth_source": "book", "assumed_queue": 2.0}},
    ),
)
res = mbt.run(strategy, config, store)                 # a QuoteResult
res.fills_df()
```

This is the maker of `examples/30_order_api_market_maker.py`, and on the same
days it gives the same order journal, fill for fill, as the function written
for `mbt.sim` (`examples/31_dsl_market_maker.py` is the complete script). The
store needs the book and the tape of the symbol over the range, as for
`mbt.sim`.

### What a quote is

A quote is an order the strategy wants to keep at the venue, written as three
expressions:

```python
strategy.quote(side, price, size, tif="GTX", enabled=None, cooldown=None)
```

- `side`: `"buy"` or `"sell"`;
- `price`: the limit, a number or an expression;
- `size`: the quantity, in units, a number or an expression;
- `tif`: `"GTX"` (post-only, the default: refused on arrival if it would
  cross), `"GTC"`, `"IOC"` or `"FOK"`;
- `enabled`: a condition; the quote stands while it holds, always when left
  out;
- `cooldown`: how long the quote posts nothing after it sent a cancellation,
  `Interval.millis(500)` or a number of nanoseconds; no pause when left out
  or zero (see [Ages, and the price of one's own order](#ages-and-the-price-of-ones-own-order)).

A strategy declares as many quotes as it wants, and each stands for at most one
order at a time. It takes no `.size()`, no bracket, no entry order and no
constraint: each quote carries its own size, and the inventory is bounded in
the quotes themselves (`enabled=`).

### One wake-up

At every wake-up all the quotes are evaluated first, on the state the wake-up
found; then they act, one after the other, **in the order they were
declared**:

1. `enabled` is null, or `price` or `size` is not a number (NaN, a null): the
   quote **holds** whatever it has, an order or nothing;
2. `enabled` is false, or `size` is zero or less: its live order is
   **cancelled** (unless a cancellation is already on its way) and the quote
   stands for nothing;
3. its live order is already at `price`: the order is **kept**, with its place
   in the queue. `price` is first put on the symbol's tick grid, as the post
   would put it (see [What the venue does](#what-the-venue-does)): a price a
   few ulps off a tick is that tick, and one between two ticks goes down for a
   buy and up for a sell. So a price computed two ways is still one price, and
   `book.ask_price_at(1) + 2 * 0.1` quotes the same as its `.round_to(0.1)`;
4. otherwise the live order is cancelled and a **new one is posted** at `price`
   for `size`, at the back of the queue.

A quote with a `cooldown` posts nothing for that long after it last sent a
cancellation, by rule 2 or by rule 4: under a cooldown, rule 4 cancels at once
and posts at the first wake-up past the pause, at the price wanted then.

An order that has filled, been cancelled or been refused frees its quote
without a cancellation. A price that is not finite and positive, or a size that
is infinite, stops the run, naming the quote and the instant: NaN is how a
quote says "hold", so any other non-number is a mistake.

Rule 4 cancels and posts at the same wake-up. Under a cancel latency the old
order can still fill until the cancellation lands, so for a while two orders of
one quote can be live. `order_activity` counts, **per quote**, the posts made
while the previous order of the same quote was still live as the strategy knows
it, its cancellation sent and the answer not back yet
(`overlapping_live_orders`: a requote under latency, or a quote withdrawn and
posted again before its cancellation came back), and the fills taken after the
cancellation was sent (`stale_fills`). The orders of the other quotes never
count, of the same side or not: a grid of three buys that requotes one level
counts one.

`when(cond, x)` without a third argument is NaN when the condition is false,
which is rule 1: the quote holds. `when(cond, x, 0.0)` as a size withdraws it
instead (rule 2). The two are different decisions, and the DSL keeps them
apart.

### The clock

`bar_interval` is the wake-up period: `Interval.millis(100)` wakes the
strategy ten times a second, and `Interval.seconds(1)` or coarser works too.
`Interval.millis` is accepted as a `bar_interval` here and nowhere else: no
store holds sub-second bars, and a strategy that quotes has none.

The first wake-up comes one period after the first instant the data covers,
and none comes at or after its end. `warmup_bars=k` makes the first `k`
wake-ups do nothing. The equity is sampled at every wake-up; the curves of the
result are then downsampled to `output_resolution` (one second when left out).

### Four latencies

The four delays of [Orders, One by One](#four-delays), under the names of
`execution.latency`:

```python
execution=mbt.ExecutionConfig(latency={
    "order": Interval.millis(5),     # a post reaches the venue
    "cancel": Interval.millis(5),    # a cancellation does
    "response": Interval.millis(5),  # an acknowledgement or a fill comes back
    "feed": Interval.millis(2),      # a book state or a print reaches the strategy
})
```

Each key is optional and **zero when left out**, `cancel` included: as for a
bar run, and unlike `mbt.sim.Config`, where `cancel` defaults to `entry`.

The state a wake-up reads is the strategy's, not the venue's: `position()`,
`cash()`, `last_fill_px()` move when a fill reaches the strategy (`response`
after the venue booked it), an order is live from the instant it is sent until
the answer that ends it arrives, and the book is the stored state as of
`now - feed`. The equity of the result marks the venue's position at the
venue's mid, as `mbt.sim` does.

`response` and `feed` are read by a strategy that quotes and by nothing else;
a bar run refuses them by name.

### What a quote reads

Two kinds of inputs meet in a quote.

**Computed before the run**, once, on the grid of wake-ups: the book columns of
[Order Book Columns](#order-book-columns) (`book.bid_price(k)`,
`book.imbalance(k)`, ...), read as the strategy sees the book at each wake-up;
the strategy's own signals (`.signal(name, expr)`, then `mbt.col(name)`), with
their indicators and rolling windows; parameters (`mbt.param`); and
`mbt.col("timestamp")`, the wake-up's instant.

**Read at the wake-up**, from the simulation:

| Expression | What it reads |
|---|---|
| `mbt.position()` | The position, in signed units |
| `mbt.cash()` | The cash: `initial_capital`, minus the buys, plus the sells, fees included |
| `mbt.live_qty(side)` | What is left to fill of the live orders of `side`; `0` when there is none |
| `mbt.order_age(side)` | Seconds since the newest live order of `side` was sent; NaN when there is none |
| `mbt.order_price(side)` | Price of the newest live order of `side`, on the tick grid; NaN when there is none |
| `mbt.last_fill_px(side)` | Price of the last fill of `side`; NaN before the first |
| `mbt.queue_ahead(side)` | Size ahead of the newest live order of `side`, as of the venue's last answer about it; NaN when there is none |
| `mbt.position_age()` | Seconds since the position left zero or changed sign; NaN while flat |
| `mbt.last_fill_age(side)` | Seconds since the last fill of `side` reached the strategy; without `side`, of either; NaN before the first |
| `mbt.last_cancel_age(side)` | Seconds since the strategy sent its last cancellation on `side`; NaN before the first |
| `book.bid_price_at(level)`, `book.ask_price_at(level)` | Price of a level, `1` = the best; NaN where the level does not exist |
| `book.bid_levels()`, `book.ask_levels()` | How many levels the side shows, `0` when it is empty |

`side` is `"bid"` (the buy orders) or `"ask"` (the sell orders). `level` may be
an expression, which may itself read the state:
`book.bid_price_at(mbt.clip(clips, 0, book.bid_levels() - 1) + 1)` steps back
one level per clip of inventory.

The book a quote reads, at the wake-up and in the batch, is the stored book,
every level of it unless `book_levels` bounds it (see
[How deep a run reads](#how-deep-a-run-reads)). Under a bound, `bid_levels()`
counts no further, `bid_price_at(k)` past it is NaN as past the stored depth, a
book column deeper than it is refused before the run, and an order resting
further out joins a queue the venue cannot measure (`assumed_queue`).

At the wake-up these combine with arithmetic, comparisons, `&`, `|`, `~`,
`mbt.when`, the element-wise functions of `manifoldbt.indicators` (`abs_val`,
`min_val`, `max_val`, `sqrt`, `log`, `exp`, ...), `.round_to(step)` /
`.floor_to(step)` / `.ceil_to(step)` for a price on a tick grid, and two
functions made for them:

- `mbt.round(x)`: the nearest integer, an exact half going to the **even**
  one, as Python's `round` does (`mbt.round(2.5) == 2`). On plain numbers it
  is the builtin;
- `mbt.clip(x, lo, hi)`: `x` brought into `[lo, hi]`; a NaN in any of the
  three gives NaN, and `lo > hi` gives `hi`.

A part of a quote that reads no state is computed before the run with the
rest of the batch, whatever operators it uses; only the part that reads the
state is evaluated at each wake-up, and an operator with memory (a rolling
window, a lag, an indicator) over the state is refused by name.

### Ages, and the price of one's own order

An operator with memory cannot look back at the state, so what a strategy
needs to remember about its own past, the state keeps for it: when the
position opened, when a fill came, when it last cancelled, where its order
stands. Every age is in **seconds, on the strategy's clock**, and counts from
the instant the strategy **knew**:

- `position_age()` and `last_fill_age(side)` count from the instant the fill
  **reached** the strategy, `response` after the venue booked it: the instant
  `position()` and `last_fill_px(side)` change. At the first wake-up that sees
  a new position, its age is already the time since that fill arrived, not
  zero;
- `position_age()` starts when the position leaves zero or changes sign in
  one fill, and runs on through the fills that add to it or trim it. A
  position within `1e-12` of zero is flat, and its age NaN;
- `last_cancel_age(side)` counts from the instant the strategy **sent** the
  cancellation, not from the venue's answer: the strategy knows when it
  cancelled without waiting, and a cancellation that arrives too late (the
  order filled first) was still sent. Every cancellation of the side counts,
  whichever quote sent it: a quote withdrawn (rule 2) and a quote moved to
  another price (rule 4, cancel then post) alike. An order that fills, is
  refused (a GTX that would cross) or expires (the rest of an IOC) was not
  cancelled;
- `order_price(side)` is the price of the newest live order of the side, the
  one `order_age(side)` and `queue_ahead(side)` read: with several quotes on
  one side, the one sent last. The price is the one on the tick grid, and an
  order has it from the instant it is sent until the answer that ends it.

Each is NaN until the event has happened, and **a comparison with NaN is
false**: `age >= d` is false before the first event, `~(age < d)` true.

**Pause after a cancellation.** `cooldown=` on the quote keeps it from posting
for that long after it sent a cancellation, counted from the sending on the
strategy's clock, as `last_cancel_age` counts:

```python
.quote("buy", book.bid_price_at(1), 0.01, enabled=..., cooldown=Interval.millis(500))
```

- the quote stands before its first cancellation;
- only the cancellations the quote itself sent start the pause: a withdrawal
  (rule 2) and the cancel of a requote (rule 4). A fill, a refused GTX or an
  expired IOC do not; neither do the cancellations of another quote of the
  same side;
- a requote under a cooldown is "cancel now, post after the pause": the old
  order is cancelled at the wake-up the price moves, the quote stands for
  nothing during the pause, and posts at the first wake-up past it, at the
  price wanted then. A cooldown longer than the cancellation takes to land
  (`cancel` and `response`) thus never leaves two orders of the quote live at
  once. A quote that should follow the price at every wake-up takes no
  cooldown;
- `cooldown` left out, or zero, is the quote without a pause, the run it
  always made.

The same pause can be written in `enabled`, as
`enabled=... & ~(mbt.last_cancel_age("bid") < 0.5)`, never `age >= 0.5`
(false before the first cancellation, so the quote would never stand). It is
the pause of the side rather than of the quote, and it is read before the
wake-up acts: a requote posts, and the next wake-up, inside the pause,
withdraws the order it just posted. For one quote a side that never requotes,
as below, the two give the same journal.

**Pull without reposting.** Rule 4 moves a quote whose price changed: cancel
and post at once. A quote that should leave the queue and wait instead stands
at its own order's price while it has one, so that it can only be kept
(rule 3) or pulled (rule 2):

```python
from manifoldbt.indicators import abs_val

pos = mbt.position()
bb, ba = book.bid_price_at(1), book.ask_price_at(1)
has = mbt.live_qty("bid") > 0
inv_ok = pos < 0.05 - 1e-12
pull = (~inv_ok | ~(bb > 0)
        | (abs_val(mbt.order_price("bid") - bb) > 1e-6)   # the best bid left it
        | (mbt.queue_ahead("bid") > 2.0))                  # too much ahead of it
strategy = mbt.Strategy.create("maker").quote(
    "buy", mbt.when(has, mbt.order_price("bid"), bb), 0.01, tif="GTX",
    enabled=(has & ~pull) | (~has & (ba > bb) & inv_ok),
    cooldown=Interval.millis(500))                         # 500 ms after a pull
```

A live order during the pause is the one whose cancellation is on its way:
the quote has already let it go, rule 2 has nothing more to send, and nothing
is posted until the pause is over.

**Hold for a bounded time.** A taker out after two seconds, without a timer
order of its own:

```python
pos = mbt.position()
old = mbt.position_age() >= 2.0 - 1e-6        # False while flat
out_long = (pos > 1e-9) & ((book.imbalance(3) < 0) | old)
out_short = (pos < -1e-9) & ((book.imbalance(3) > 0) | old)
taker = (
    mbt.Strategy.create("taker")
    .quote("sell", book.bid_price_at(1), pos, tif="IOC", enabled=out_long)
    .quote("buy", book.ask_price_at(1), -pos, tif="IOC", enabled=out_short)
    # ... the entries, enabled only while neither exit is
)
```

Both give, order for order and fill for fill, the journal of the same logic
written for `mbt.sim` with its own clocks and its own record of what it
cancelled when. The ages differ from such a clock only where the definitions
do: a holding time counted from the first wake-up that saw the position is
shorter than `position_age()` by the time between the fill's arrival and that
wake-up, which moves the exit only when the threshold falls in between.

### What comes out

`mbt.run` returns a `QuoteResult`: everything a `Result` has, read the same
way, and the venue's own record beside it.

| | |
|---|---|
| `res.metrics` | The metrics of every run, as returns on `initial_capital`, and what the run ended with in money: `final_equity` and `pnl`. On a run of one day, as above, the ones taken from daily returns (Sharpe, Sortino, volatility, CAGR, Calmar...) are NaN and `res.warnings` says why: see [Metrics Reference](#metrics-reference) |
| `res.trades_df()` | One row per fill |
| `res.equity_df()`, `res.positions_df()` | At `output_resolution` |
| `res.order_activity` | `orders_posted`, `requotes` (a post that replaced a live order of the same quote), `expired_unfilled` (orders cancelled with nothing filled), with a latency `stale_fills` and `overlapping_live_orders`, and when they are not zero `post_only_rejected` (GTX quotes refused on arrival because they would have crossed) and `levels_snapped` (quotes between two ticks, posted on the passive side; `res.warnings` says so too) |
| `res.fill_fragility` | What the queue decided, as for a bar run under [Queue Model](#queue-model): `maker_fills`, `queue` (`queue_decided_fills`, `traverse_fills`, `fills_from_book_cross`, `partial_fills`, `book_unknown_at_post`: read it to know how many quotes stood behind `assumed_queue`), `would_fill_touch`, `would_fill_traverse` (per life of an order at one level). `touch_only_fills` is a bar convention and stays `0`. A sweep's `service_rate` and `queue_decided_fills` columns read it |
| `res.fill_marks` | The markouts of [Marking a Fill](#marking-a-fill), counted from the fill itself (`anchor == "fill"`) |
| `res.orders_df()`, `res.fills_df()`, `res.events_df()` | The tables of `mbt.sim`: every order, every fill with its `channel`, the order journal. They speak the venue's words, those of the order API: `qty`, `fee`, `price`, `side` as `"buy"` or `"sell"`; `res.trades_df()` keeps the trade log's own, the same for every run: `quantity`, `fees`, `fill_price`, `side` as 1 or 2 |
| `res.sim` | The whole venue record, as `mbt.sim.run` returns it |
| `res.profile` | `data_load_us` is the load of the book and the tape, `signal_eval_us` the batch before the run, `simulation_us` the wake-ups; `total_us` is the whole of `mbt.run` in the engine, the venue's setting up, the markouts and the result included |

### Sweeps

A strategy that quotes sweeps like any other, through the same calls, in the
same order, with the same labels:

```python
spread = mbt.param("spread", default=1)
bid = book.bid_price_at(spread)
ask = book.ask_price_at(spread)
strategy = (
    mbt.Strategy.create("maker")
    .quote("buy", bid, 0.01, enabled=mbt.position() < 0.05)
    .quote("sell", ask, 0.01, enabled=mbt.position() > -0.05)
)

sweep = mbt.run_sweep(
    strategy, {"spread": [1, 2, 3]}, config, store,
    execution_grid={"latency.order": [Interval.millis(1), Interval.millis(10)],
                    "latency.cancel": [Interval.millis(1), Interval.millis(10)]},
)
df = sweep.to_df()                  # param_spread, exec_* and every metric
sweep[0].fills_df()                 # each combination is a QuoteResult
```

- **Each combination is the run `mbt.run` makes alone** of the strategy with
  those values as its defaults, under that execution config: the same orders,
  fills, order events and equity, to the bit, whatever `max_parallelism`.
- The book and the tape are loaded **once**; the combinations then run in
  parallel in the engine, each on its own copy of the venue.
- The parameters are the `mbt.param` the quotes and the signals read. The
  natural execution axes are `latency.order`, `latency.cancel`,
  `latency.response`, `latency.feed` and `fill_model.queue.*`; every combination
  is read by the rules of this chapter before a day is loaded, and one the
  simulation cannot honour is refused by name.
- The order is the one of [Sweeping Execution
  Parameters](#the-order-the-combinations-come-in): parameters sorted by name,
  the last fastest, and the execution axes slower than all of them.
- `mbt.run_sweep_lite` runs the same combinations and keeps the metrics only:
  no order journal, no markouts, no curves. Its metrics are **the full run's,
  to the bit** -- not the daily approximation of a lite bar sweep. It is the
  call for a large grid: a full combination keeps its whole journal.
- `mbt.run_batch` and `mbt.run_batch_lite` take a list of strategies that
  quote, one result each, with the same `execution_grid`. A batch holds
  strategies that quote or strategies that do not, never both.

On one BTCUSDT day at ten levels (2025-08-23: 864,000 wake-ups, about 55,000
orders per combination), a combination alone costs about 0.23 s. Twenty-four
combinations through `run_sweep_lite`, one thread per physical core:

| Threads | Combinations per second | Speed-up |
|---|---|---|
| 1 | 4.4 | 1.0 |
| 2 | 8.4 | 1.9 |
| 4 | 15.1 | 3.4 |
| 6 | 20.3 | 4.6 |

The combinations share nothing but the market data, so what keeps the speed-up
under the number of cores is the memory they all read: each walks the day's
book on its own. A full `run_sweep` keeps every combination's journal, about
50 MB on such a day: 1.6 GB for these twenty-four, where the lite sweep peaks at
0.35 GB on one thread and 1 GB on six, and keeps nothing once a combination is
done.

### What refuses, and why

Refused by name, before anything is loaded:

- **in the strategy**: `.size()`, a bracket or an entry order, a constraint,
  `tif="GTD"` (the quote itself decides when its order leaves), a negative
  `cooldown`; a signal that
  reads the state (a signal is a series computed before the run, when no state
  exists yet: read the state in the quote itself); a column the wake-up grid
  does not hold (`close`, `volume`: there are no bars); `tf(...)` (the
  wake-ups are the only grid); an operator across symbols;
- **in the config**: more than one symbol (quote each in its own run),
  `Interval.trades()` as the clock, a `signal_delay` (the latencies say when
  the strategy sees and acts), and every setting of the bar engine a quote has
  no use for, each named with its reason: `resample_to`, `accuracy`,
  `extra_timeframes`, `exo_data`, `position_sizing_mode`, `pyramiding`,
  `max_position_pct`, `allow_short=False`, `execution_price`,
  `execution.orders`, `tick_size` (the symbol's own tick, read from the
  store, applies; round in the expression with `.round_to(tick)` to choose the
  level yourself), `max_participation_rate`, `passive_fill`, a custom or
  adverse fill model, `slippage`, funding, borrow and minimum fees, a fill
  callable, among others;
- **in the calls**: `device="cuda"` and `precision="fp32"` (the order-level
  simulation runs on the CPU, in double precision), and a batch that mixes
  strategies that quote with strategies that do not. Walk-forward, stability,
  2-D sweeps and portfolios refuse a strategy that quotes.

### What it costs

Everything runs in the engine: the batch once before the run, then at each
wake-up one small program per expression, walked against the state, without a
call to Python. On three BTCUSDT days at ten levels (864,000 wake-ups, 55,000
to 130,000 orders), the maker at the top of this chapter costs **about 240 to
300 ns per wake-up**, the venue included. The same maker written for `mbt.sim`
as a Python function costs 1.7 to 1.9 µs per wake-up, six to seven times more,
and the difference is the Python. Measured on an i5-13600KF, one performance
core, with a release build of the wheel.

`res.profile` says where a run went: `data_load_us` is the load, `signal_eval_us`
the batch, `simulation_us` the wake-ups, and `total_us` all of it. A wake-up
evaluates every quote, so its cost grows with the number of quotes and with what
they read; the batch part does not, it is paid once.

On a book stored as whole ladders (`book_format="ladder"`), a run costs with the
depth it reads: on a BTCUSDT day stored 200 levels deep, a maker quoting three
levels a side every 200 ms ran in 0.75 to 1.3 s reading every level (the
default) and about 0.55 s with `book_levels=25`. Stored as changes (`book_format="deltas"`, the
default shape), the depth read barely changes the cost: about 0.6 s either way.

---

## A Market Stored Order by Order

A book stored by price says how much rests at each price. A feed stored order
by order (market by order, or level 3) says which orders rest there, in which
order they arrived, and every change to each of them. The place of an order in
its queue is then read off the data, not estimated: what [Queue Model](#queue-model)
has to assume about the cancellations it cannot see, this market shows.

```python
import manifoldbt as mbt

store = mbt.ingest_mbo("S120925-v50.txt.gz", {"AAPL": 1, "QQQ": 2},
                       date="2025-12-09")
res = mbt.sim.run(maker, mbt.sim.Config(), store, symbols=["AAPL"],
                  start="2025-12-09", end="2025-12-10")
```

`mbt.ingest_mbo` reads a day of Nasdaq TotalView-ITCH 5.0: the file Nasdaq
publishes (`.gz`, read as it inflates), or a stream of its messages kept as
they are (plain or ZSTD), such as an extract of some symbols. Every symbol
asked for is read in one pass. ITCH stamps nanoseconds since midnight in New
York: they are stored as instants (UTC), with New York's offset on the
session's date.

### What is stored

One Arrow file per symbol and UTC day, under `{exchange}/mbo/{symbol}/`, one
row per event of one order:

| `action` | What happened to the order |
|---|---|
| `A` | it entered the book (the new order of a replacement too, flagged) |
| `E` | it executed, at its own price |
| `C` | it executed at another price (a cross), printed or not |
| `X` | part of it was cancelled |
| `D` | it left the book: what was left of it |
| `U` | it was replaced: it left the book, and the order replacing it follows at the same instant, at the back of its queue |
| `P` | a print against an order the book does not show |
| `Q` | the print of a cross |

Every row carries the side, the price, the size and the instant the order
entered the book: the book by price, and the volume ahead of an order, are
sums over rows. A day reads on its own: it opens on the orders resting at
midnight UTC (the evening of a New York session is past it), and the orders
still resting when the session ends leave the book then. Each day is checked
as it is written (an event about an order that is not in the book, or taking
out more than is left of it, is refused and the day left as it was), and
ingesting a day again replaces what it covers.

On disk, a day of AAPL is 26 MB, of QQQ 133 MB (2.8 and 15.2 million events).

### Reading it

`mbt.sim.run`, `mbt.sim.Market.from_store` and `mbt.run` with
`Strategy.quote` read such a symbol from its events, whatever else the store
holds of it (`Market.by_order` says so). Its book by price and its tape are
read off them: every reading of the book of [Orders, One by One](#orders-one-by-one)
and [Quoting in the DSL](#quoting-in-the-dsl) (`bt.book.bid_price_at`,
`bid_levels`, `sim.book(...)`, the columns of the book) reads it unchanged,
one state per instant an event changed it. The tape holds the executions and
the prints that reached it; the prints of a cross are left out (no side took
them).

A busy stock's book goes thousands of levels deep, down to orders parked at a
hundredth of a cent and at the price cap: it is read **1000 levels deep** unless
`levels=` (or `BacktestConfig.book_levels`) says otherwise, as a book stored a
thousand levels deep.

### The exact queue

Under the exact queue, an order of ours that reaches the venue at `t0` stands
behind every order of its level that entered the book at or before `t0` (at
one venue instant the market comes first), and before every one that enters
after. Then, event by event:

1. an execution, a cancellation, a deletion or a replacement of an order
   **ahead** of ours takes its size out of the volume ahead;
2. an execution of an order **behind** ours means the market went through our
   place: ours is served, as much as that execution took, shared in their
   order among our orders it went through (a partial fill is the normal case);
3. an execution on our side at a **worse price** means the market went
   through our level: ours is served whole;
4. an order entering the **other side** at our price or through it would
   have met ours first: ours is served, as much as it brings, best price
   then first come among our orders it crosses (`book_cross`).

Our fills take nothing out of the market's book: its events are replayed as
they were. Prints against orders the book does not show (`P`) and crosses
serve nothing: they are not in the displayed queue. Everything else is the
venue of [Orders, One by One](#orders-one-by-one): the four latencies, an order
that would execute on arrival taking the displayed book, a post-only order
refused, a smaller size at the same price keeping its place and any other
replacement going to the back.

`queue_ahead` (`sim.order(id).queue_ahead`, `bt.queue_ahead(side)` in a
quote) is the volume ahead of the order exactly, ours included, as the feed
shows it at the strategy's clock: `feed` behind the venue, from the moment the
acknowledgement has told the strategy which order is its own.

### Naming the queue

| `queue` | On a market stored order by order | On a book stored by price |
|---|---|---|
| left out | the exact queue | `risk_adverse` reading the book |
| `{"model": "fifo"}` | the exact queue | refused by name |
| another model | that model, reading the book by price and the tape | that model |

A queue named in `bt.sim.Config(queue=...)` or in
`fill_model={"queue": ...}` is the queue of every market of the run: running a
model of the book by price on a market stored order by order measures what it
misses. `bt.run` on bars or on the trade clock refuses `"fifo"` by name.

### Checked against

On the day of 2025-12-09 (AAPL, SPY, QQQ, NVDA, TSLA):

- the events read back from the store are, one for one, those of an
  independent reconstruction of the messages (39.8 million), and the book by
  price is the same at every one of its 31.6 million instants;
- 250,000 orders placed at random near the best price get, from the venue,
  what a direct replay of the day in explicit FIFO queues gives them: the
  volume ahead when placed and 0 to 30 s later, the instant of service and its
  cause. The replay of that study knows neither rule 4 nor an order taking the
  book on arrival; with rule 4 added to it, the only differences left are the
  orders that crossed the spread when they arrived (0.2 to 1.7 % of them),
  which the venue executes as takers;
- a maker quoting one unit at the best bid and ask every 100 ms, moved when
  the best moves, inventory within five units, gives the same posts,
  cancellations, fills, inventory and cash as the same maker in that replay,
  to the unit (33,247 to 138,664 quotes a day). Without rule 4, the replay
  gives the study's own figures; rule 4 is worth from 14 % (TSLA) to 195 %
  (QQQ) more fills.

### Against hftbacktest

The same day, handed to hftbacktest 2.4.4 as its own order-by-order events
(`l3_fifo_queue_model`, `no_partial_fill_exchange`): an `A` row adds the order,
an `E` or `C` row is a fill followed by what is left of the order or its
deletion, an `X` row a modification or a deletion, `D` and `U` deletions; `P`
and `Q` rows, which move no order the book shows, are left out.

Every 100 ms, both quotes are cancelled and two new post-only quotes placed at
the best bid and the best ask, whatever became of the previous ones: the quotes
depend only on the book, the same in both engines, and each gets a verdict in
each. The latency is zero, or 50 µs of feed and 100 µs for an order, a
cancellation and a response.

| symbol | latency | quotes | served, same instant | served here earlier | served here only | served there only | never served |
|---|---|---|---|---|---|---|---|
| AAPL | 0 | 467,998 | 7,659 | 1,330 | 464 | 0 | 458,545 |
| AAPL | 50 / 100 µs | 467,998 | 7,662 | 1,330 | 466 | 0 | 458,540 |
| QQQ | 0 | 467,998 | 69,995 | 799 | 55 | 0 | 397,149 |
| QQQ | 50 / 100 µs | 467,998 | 70,017 | 802 | 53 | 0 | 397,126 |

hftbacktest never serves a quote this venue does not, and every difference is
rule 3. hftbacktest serves an order through its price only between the
market's best price, its own orders left out, and the price traded: an order of
ours left alone at its price, once the orders ahead of it executed, is better
than that best, and an execution a tick further does not reach it. It is served
later, when an order entering the other side crosses it (a median of 58 µs
later on AAPL, 47 µs on QQQ; 1.3 ms and 0.6 ms at the 90th percentile), or not
at all when none does before the quote is cancelled. The aggressor that paid a
worse price than ours met ours first: this venue serves it at that execution.

With 100 shares a quote, the same quotes are served at the same instants, and
1,704 of them on AAPL, 5,425 on QQQ, for another size: an execution behind
ours, or an order entering the other side, serves here as much as it takes;
hftbacktest's exchange has no partial fill and serves the whole order.

A maker whose quotes depend on its inventory parts from the other engine at
their first different fill. At the best bid and ask every 100 ms, a quote kept
while its price holds, a side withdrawn at five lots of inventory or when the
imbalance of the best level runs against it by more than 0.6, written with
`Strategy.quote` here and in numba for hftbacktest, zero latency, no fee, marked
at the close:

| symbol | lot | fills here | fills in hftbacktest | P&L here | P&L in hftbacktest | position at the close |
|---|---|---|---|---|---|---|
| AAPL | 1 | 7,146 | 6,952 | -47.32 | -46.55 | -4 and -4 |
| QQQ | 1 | 43,077 | 43,041 | -77.74 | -78.08 | -5 and -5 |
| AAPL | 100 | 14,266 (613,994 shares) | 6,952 (695,200) | -4,621.77 | -4,655.00 | -378 and -400 |
| QQQ | 100 | 73,180 (3,884,056 shares) | 43,041 (4,304,100) | -11,305.36 | -7,807.50 | -586 and -500 |

Their first different fill is rule 3 (the same quote, served there 83 µs later
on AAPL, 63 ms later on QQQ), except on QQQ with 100 shares: a partial fill.

### What refuses, and why

- An order resting past the levels read, on a side that shows as many as it
  can: the volume there is not read. Read more levels, or quote within them.
- `"fifo"` named on a book stored by price, or in a run on bars or on the
  trade clock.
- A range with no event of the symbol in it: ingest the day first.

### What it costs

A day of QQQ (15.2 million events): read from the store in 0.5 s, its book by
price and its tape laid out in 2.1 to 2.5 s (1000 levels); a day of AAPL in 0.1
and 0.4 s. That is on six performance cores; on one, loading the QQQ session
takes 4 s instead of 2.5. The venue reads the events only while an order of
ours rests, and only at the levels it can reach: the maker of the duel above,
100 shares a quote, runs the QQQ session in 3.0 s once loaded (0.6 s of quotes
computed, 2.4 s of venue), the AAPL session in 0.54 s.

## Fee & Slippage Models

### Fees

```python
mbt.FeeConfig.binance_perps()    # maker=2bps, taker=5bps, funding
mbt.FeeConfig.binance_spot()     # maker=10bps, taker=10bps
mbt.FeeConfig.zero()             # no fees (for development)

# Custom
mbt.FeeConfig(
    maker_fee_bps=2.0,
    taker_fee_bps=5.0,
    funding_rate_column="funding_rate",
    default_fill_type="Taker",
)
```

### Slippage

```python
Slippage.fixed_bps(2)     # 2 bps per trade (simplest)
Slippage.volume_impact(0.1, exponent=0.5)   # qty/volume model
Slippage.spread_based(0.5)                  # spread-based
```

---

## Orders (SL/TP/Trailing)

Each of the three takes `side="both"` (default), `"long"` or `"short"`: an
order armed on one side leaves the other bare, so a strategy that trades
both directions can stop its shorts and let its longs run. A distance swept
from a grid (`stop_loss` as a sweep parameter) keeps the side the strategy
set.

```python
strategy = (
    mbt.Strategy.create("my_strat")
    .signal(...)
    .size(...)
    .stop_loss(pct=3.0, side="short")   # 3% stop-loss, on the shorts only
    .take_profit(pct=5.0)               # 5% take-profit, both sides
    .trailing_stop(pct=2.0)             # 2% trailing stop, both sides
)
```

Orders travel with the strategy, so they apply wherever it runs: `run`,
`run_batch`, `run_sweep` and `run_portfolio`.

In a portfolio each leg is an independent run on `initial_capital * weight`
and the portfolio equity is the sum of the legs. Two consequences:
`max_position_pct` clamps on the leg's equity, not the portfolio's, and with
the default `FractionOfEquity` sizing each leg compounds on its own. Set
`position_sizing_mode="FractionOfInitialCapital"` when the portfolio must
equal the sum of its legs run separately.

### What each order accepts

| order | `pct` | `offset_bps` | `price` | `signal` | options |
|---|---|---|---|---|---|
| `limit_entry`, `stop_entry`, `market_if_touched` | no | yes | yes | yes | `time_in_force`, `size_at_fill_price` |
| `stop_limit_entry` | no | no | `stop`, `limit` | `stop_signal`, `limit_signal` | same |
| `stop_loss`, `take_profit` | yes | no | no | yes | `side` |
| `trailing_stop` | yes | no | no | yes | `use_high`, `side` |

### A distance that changes with the market

`pct=2.0` is one number for the whole run. `signal="name"` reads the distance
from a series the strategy computes, so a stop can be two ATR wide on a quiet
day and twice that on a violent one. The series holds a **percentage of the
price**, the same unit `pct` uses, and it is exclusive of `pct`.

```python
from manifoldbt.indicators import atr, close

# 2 ATR, expressed as a percentage of the price
stop_dist = mbt.lit(2.0) * atr(14) / close * mbt.lit(100.0)

strategy = (
    mbt.Strategy.create("atr_stop")
    .signal("stop_dist", stop_dist)      # named, so the order can reference it
    .size(...)
    .stop_loss(signal="stop_dist")
)
```

Three rules make that a well-defined order rather than a moving target:

**The distance is read on the signal bar.** The bar whose signal decided the
entry, the same row `limit_entry(signal=...)` reads its level at, so there is
no look-ahead: the distance is known before the fill happens.

**It is frozen for the life of the trade.** The level is computed once, from
the price the entry actually filled at (slippage included, exactly as `pct`
does it), and the trade keeps it until it closes. A trailing stop freezes its
trail *distance* the same way and then ratchets normally on the bar high (or
the close, with `use_high=False`). A distance that kept moving during the trade
would be a different order, not this one.

**A distance the series cannot give arms nothing, loudly.** If the series holds
NaN, zero or a negative number on the signal bar, no bracket is armed on that
trade at all, `result.brackets_not_armed` counts it, and `result.warnings`
names the order. A position you believe is protected and is not is the worst
failure this family has, so the engine refuses to invent a level.

```python
result = mbt.run(strategy, config, store)
print(result.brackets_not_armed)   # 0 when every trade got its bracket
```

A warm-up is the usual cause: `atr(14)` is NaN for its first 13 bars. Set
`warmup_bars` on the config, or gate the sizing on the indicator being ready.

A `param()` inside the series makes the distance sweepable like any other
expression, with nothing else to declare.

### Cost of a signal distance

A `signal=` distance is read per trade, which the fast kernels do not do: a
strategy that uses one runs on the general loop, and a sweep over it stays on
the CPU. `run_sweep` is always on the CPU and says nothing; asking for the GPU
with `run_sweep_lite(device="cuda")` warns with the reason ("an exit order takes
its distance from a signal") and runs on the CPU. A constant `pct` keeps the
fast path and the GPU.

---

## Entry Orders

By default an entry takes a market fill on the execution bar (see
[Execution Model](#execution-model)). Four order types let the entry rest at a
price instead:

| Builder method | Fills when | Fill price | Costs |
|---|---|---|---|
| `.limit_entry(...)` | price comes **to** the level | the level exactly | maker, no slippage |
| `.stop_entry(...)` | price breaks **through** the level | the level, or the open if the bar gapped through it | taker + slippage |
| `.market_if_touched(...)` | price comes **to** the level | the level | taker + slippage |
| `.stop_limit_entry(...)` | breaks through `stop`, then rests at `limit` | the limit | maker, no slippage |

### Where the level comes from

Every method takes exactly one of three price forms:

```python
.limit_entry(offset_bps=25)          # 25 bps below the signal close (above, for a sell)
.limit_entry(price=60_000)           # a fixed level
.limit_entry(signal="entry_px")      # a level this strategy computes
```

`signal=` is the general form: name any signal the strategy defines and the
order rests on that series, read on the signal bar.

```python
from manifoldbt.indicators import atr, close, ema

trend = ema(close, 50)
entry_px = close - atr(14)          # rest one ATR below the close

strategy = (
    mbt.Strategy.create("pullback_entry")
    .signal("trend", trend)
    .signal("entry_px", entry_px)   # named so the order can reference it
    .size(mbt.when(close > trend, 1.0, 0.0))
    .limit_entry(signal="entry_px", time_in_force={"GTB": 5})
    .stop_loss(pct=3.0)
)
```

### Resting on the venue's tick grid

A venue quotes on a grid, and a level off it finds no depth in the book -- the
queue model has nothing to place the order behind, and no venue would have
accepted the order at all. `execution.tick_size` brings every level an order
**rests** at onto that grid before posting it, in the direction of passivity: a
bid down, an ask up.

```python
execution=mbt.ExecutionConfig(tick_size=0.1)     # BTCUSDT
# result.order_activity["levels_snapped"] -> how many levels moved
```

Aggressive legs keep their level: a `stop_entry`, a `market_if_touched`, a
stop-loss and a take-profit all execute at the price of the print that triggered
them, not at a level standing in the ladder. A `stop_limit_entry`'s stop is a
trigger too, so only the limit it then rests at is snapped. Leaving it out is the
run the engine always did, byte for byte, and a level already on the grid is
never moved -- so snapping in the expression (`.floor_to(0.1)`) and setting
`tick_size` do not fight. The tick is **not** read from the store: an ingested
book carries its symbol's tick in its metadata, and wiring that through is its
own step.

### Time in force

| Policy | Written | Lives for |
|---|---|---|
| good til cancelled | `"GTC"` (default) | until filled, or until the target moves |
| good til bar | `{"GTB": n}` | `n` **events**: bars on a time grid, prints on the trade clock |
| good til time | `Interval.seconds(1)`, `Interval.millis(500)`, or `{"GTT": Interval.millis(500)}` | a **duration**, whatever the grid did in between |
| immediate or cancel | `"IOC"` | one event; `{"GTB": 1}` under another name |

**An entry order that expires unfilled is posted again on the next event, at that
event's level, for as long as the target holds and still differs from the position
you hold.** A target is a desired position, not a one-off event, so expiry is how
you requote: `{"GTB": 1}` on a target that never moves places a fresh order every
bar, each at the level that bar computes. Nothing changes for an order that
filled, nor for one cancelled because the target itself moved.

**A duration is not a count of events.** An order posted at instant `t` with
`time_in_force=Interval.seconds(1)` is cancelled on the first event stamped at or
after `t + 1s`, **before** that event's prints are offered to it, and the target
it served may post again on that same event. `t` is the timestamp of the event
the order was posted on — the first one it could fill on; with the default
`signal_delay=1` on a time grid, that is the close of the bar the strategy read.

On a regular grid a duration of one bar and `{"GTB": 1}` are the same order. They
part company on any grid with holes in it, for the reason
[Windows in time](#windows-in-time) gives: `{"GTB": 30}` is thirty rows, which is
thirty seconds on a busy minute and four minutes on a quiet one on one-second
bars rebuilt from a tape, and thirty prints under the trade clock -- a fraction
of a second at one moment and a minute at another. A maker that means "my quote
is one second old" writes `Interval.seconds(1)`.

A duration of zero or less is refused by name: an order that expires on the
instant it is posted never reaches the market.

### Requoting is cancelling and posting again

**An order never moves.** Changing the level of a resting order cancels it and
posts a different order, at the back of whatever queue there is, with its own
timestamp. That is what a venue does, and it is what the engine does: the level
of a resting order is read once, on the event that posted it, and held until the
order fills, expires, or is replaced because the target moved.

So a strategy requotes by letting its order expire. `Interval.seconds(1)` on a
market maker means: quote, wait a second, cancel, quote again at the level that
event computes — and lose the queue position each time, including when the new
level is identical to the old one. There is no "amend in place" that would keep
it. What losing it costs is what [`fill_model.queue`](#queue-model) prices.

`result.order_activity` counts what the orders did, next to `fill_fragility`,
which counts what the fills did:

```python
result.order_activity
# {'orders_posted': 5219, 'requotes': 5183, 'expired_unfilled': 5183}
```

- `orders_posted` — level entry orders posted, the first quote of a target and
  every repost. A market entry is not a quote and is not counted.
- `requotes` — of those, the ones that replaced a resting order, or
  followed one cancelled unfilled on that event or the one before. The first
  order posted for a target that had none is not a requote.
- `expired_unfilled` — level orders cancelled having filled nothing.

It is `None` on paths that do not track it (the portfolio runner), like
`fill_fragility`.

### Reading the position from a level

A market maker steps away from the market as its inventory grows. `position()`
is the quantity held, in signed units, at the instant the order is posted:

```python
from manifoldbt import position
from manifoldbt.expr import col, lit
from manifoldbt.helpers import Interval

quote = col("fair") - lit(0.5) * col("spread") - lit(2.0) * lit(tick) * position()

strategy = (
    mbt.Strategy.create("maker")
    .signal("quote", quote)
    .size(col("wants_to_quote") * lit(qty))
    .limit_entry(signal="quote", time_in_force=Interval.seconds(1))
    .take_profit(pct=0.02)
    .stop_loss(pct=0.06)
)
```

It is **not a column**. A column is computed for every bar before the simulation
starts; the position at bar `i` is the result of what the strategy did with bars
`0..i`, and only the simulation knows it. So the engine splits that one
expression: everything that does not mention the position is evaluated in batch
like any other signal, and the few nodes left are walked once per posted order,
with the position substituted.

Two consequences worth stating plainly:

- **The market data in the level is delayed like every other signal**
  (`signal_delay`); the inventory is not, because it is the strategy's own book,
  known without a round trip. What the round trip itself costs is
  [`execution.latency`](#latency), which is a separate setting and does not
  change how this level is read.
- **What is readable, and where.** `position()` is accepted in the level of an
  entry order declared **on the strategy** (`.limit_entry(signal=...)` and the
  stop-limit's limit), and there only through `+`, `-`, `*`, `/`, `abs`, `min`,
  `max` and the three price-grid operators (`round_to`, `floor_to`, `ceil_to`),
  combined with any sub-expression that does not itself read it.
  Everywhere else — a size, a plain signal, a comparison, a `when()`, or any
  operator with memory (a rolling window, a lag, an indicator) — it is refused
  **by name at compile time**, naming the expression that carried it. It is
  never read as something else.
- **Nothing else may read that level.** A level that reads the position is not
  a series, so `col("quote")` elsewhere in the strategy is refused too. Name the
  shared part (`.signal("fair_bid", ...)`) and let the level be that name plus
  the inventory term.

The refusals are not an oversight. A comparison in the batch evaluator carries a
null mask that a per-event scalar walk has no equivalent of (a comparison against
NaN is a null there, which `when()` reads as its false branch), and two
evaluators that must be kept in step by hand eventually disagree. Arithmetic has
no such subtlety, so the grammar stops there.

**The level is the order's, not the side's.** One entry order serves whichever
direction the target asks for, so the same expression prices the order that
*reduces* the position too — and there the skew pushes the level the wrong way,
which makes the exit more aggressive rather than less. A maker that wants a
one-sided skew flattens through its bracket (`take_profit` / `stop_loss`) and
lets the target say `hold()` rather than `0.0` while it is not quoting.

Bounding the inventory does not need `position()` at all when the target is the
bound: `.size(lit(0.05))` in `Units` mode holds at most 0.05, whatever the quote
does. The skew earns its place where the target does *not* bound anything — a
maker adding a clip per event (`pyramiding=True`) runs to whatever
`max_position_pct` allows without it, and settles at a book the skew chooses
with it. On three BTCUSDT days that is 0.42-0.46 BTC held at `k = 0` against
0.04-0.08 BTC at `k = 2000` price units per BTC, same days, same quote, one term
more.

**Later, not here.** Several resting orders per strategy, a ladder of quotes,
and a size that reads the inventory. The last one is not a grammar question: a
size is the position target itself, and a target that reads the position it
produces is a loop the batch pipeline is built around not having.

A strategy carrying `position()` runs on the general loop, like every other
resting order: the fast kernels and the CUDA sweep refuse a conditional entry by
name (`gpu-sweep-unsupported: the strategy has a conditional entry order`).

### Three things to watch

**A resting order keeps the level it was created with.** `signal=` is read once,
on the bar the order is placed, and held until the order fills, expires or is
cancelled. It does **not** follow the series afterwards. A time-limited order
does move, but in steps: it holds its level until it expires, and the order that
replaces it reads the series again (see [Time in force](#time-in-force)). That is the intended
behaviour of a resting order, and it is the trap for a band strategy: if the
band moves every bar, the order waits at a price the band has left, and a bar
that gaps past the stale level still fills there. Watch the out-of-range fill
warnings, which count exactly this.

If what you want is "fill wherever my level is on the bar that trades", that is
not a resting order at all: use
[`ExecutionPrice.custom`](#filling-at-a-computed-level), which re-reads the
level every bar. Keep a resting entry for what it models, a real order sitting
in the book at a price you chose.

**A resting entry can simply never fill.** A strategy whose entries never
trigger produces a flat equity curve with no drawdown, which reads as a clean
backtest. The engine counts unfilled entries and reports them:

```python
result = mbt.run(strategy, config, store)
for w in result.warnings:
    print(w)   # "N entry order(s) expired unfilled and M were still resting ..."
```

**Sizing uses the close, not the level.** In `FractionOfEquity` mode a target of
`1.0` is converted to units at the signal-bar close, so an entry resting 2% away
buys ~2% too much notional. `size_at_fill_price=True` sizes off the order's own
level instead. It is off by default because turning it on changes the results of
strategies written against the old behaviour.

### Cost

A conditional entry runs on the general simulation loop rather than the fast
kernel, so parameter sweeps over one are slower than sweeps over a market entry
and cannot use the GPU. `run_sweep` reports which setting took you off the fast
path.

This is specific to a **resting order**, which can stay unfilled across bars.
Filling at a computed level does not carry that cost: see
[Filling at a computed level](#filling-at-a-computed-level), which stays on the
fast kernel and on the GPU.

---

## Cross-Asset References

Use `mbt.symbol_ref()` to reference another symbol's data in multi-asset strategies:

```python
pair_close = mbt.symbol_ref("ETHUSDT", "close")
ratio = close / (pair_close + mbt.lit(1e-12))
```

> **Important:** Expressions using `symbol_ref()` must be registered as named signals (`.signal("name", expr)`), not passed directly to `.size()`. The multi-pass evaluator needs named signals to route cross-asset data correctly.

```python
# Required: symbol_names mapping
config = mbt.BacktestConfig(
    universe=[1, 2, 5],
    symbol_names={"BTCUSDT": 1, "ETHUSDT": 2, "BNBUSDT": 5},
    ...
)
```

---

## Dataset Auto-Resolution

The engine automatically selects the best dataset based on `bar_interval`:

| bar_interval     | Dataset loaded  | Bars (5 years) |
|------------------|-----------------|----------------|
| 1 min            | `bars_1m`       | ~2.6M          |
| 15 min           | `bars_15m`      | ~175k          |
| 1h - 23h         | `bars_1h`       | ~44k           |
| >= 24h           | `bars_1d`       | ~1.8k          |

When `bar_interval` doesn't exactly match a dataset (e.g. `4h`), the engine loads the closest smaller dataset (`bars_1h`) and pre-resamples to `4h` before simulation.

Override with `accuracy=True` to always load `bars_1m` (precise SL/TP fills).

Override manually with `dataset=`:
```python
store = mbt.DataStore(data_root="data", metadata_db="...", dataset="bars_1m")
```

---

## Diagnostics

```python
# Look-ahead bias detection: a static walk for every non-causal operator,
# then two re-runs over shorter windows. The two halves catch different
# things. The re-runs cannot see a fixed read-ahead (close.lead(1) trades
# the same in every window); the static walk names it. The static walk
# only knows the operators, so it cannot see a leak that comes from the
# data itself; the re-runs can.
#
# The static walk reads a registry in which every DSL operator is
# classified causal or not, so a new operator cannot be added without
# answering the question. Two are not causal: lead() and
# full_series_rank().
lookahead = mbt.diagnostics.detect_lookahead(strategy, config, store)
print(lookahead)  # PASS or FAIL with details

# Exposure stability (position consistency across time windows)
stability = mbt.diagnostics.check_exposure_stability(strategy, config, store)

# Post-run risk check
result = mbt.run(strategy, config, store)
risk = mbt.diagnostics.risk_check(result)
```

---

## Profiling

Every result includes microsecond-precision timing:

```python
result = mbt.run(strategy, config, store)
print(result.profile)
# {'data_load_us': 45000, 'align_us': 1000, 'signal_eval_us': 28000,
#  'runtime_prep_us': 500, 'simulation_us': 16000, 'output_build_us': 8000,
#  'total_us': 110000}

print(result.profile_summary())
# Profile (total: 110.0ms)
# ----------------------------------------
#   Data loading      45.0ms   40.9%  ################
#   Signal eval       28.0ms   25.5%  ##########
#   Simulation        16.0ms   14.5%  #####
#   ...
```

---

## Complete Examples

### Trend Following — EMA Crossover

```python
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Slippage, Interval

fast = ema(close, 12)
slow = ema(close, 50)

strategy = (
    mbt.Strategy.create("trend_following")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.5, 0.0))
    .stop_loss(pct=3.0)
)

start, end = time_range("2022-01-01", "2025-01-01")
config = mbt.BacktestConfig(
    universe=[1], time_range_start=start, time_range_end=end,
    bar_interval=Interval.hours(12), initial_capital=10_000,
    fees=mbt.FeeConfig.binance_perps(), slippage=Slippage.fixed_bps(2),
    warmup_bars=60,
)
store = mbt.DataStore(data_root="data", metadata_db="metadata/metadata.sqlite")
result = mbt.run(strategy, config, store)
print(result.summary())
```

### Parameter Sweep — 2D Heatmap

```python
fast = ema(close, mbt.param("fast", default=12))
slow = ema(close, mbt.param("slow", default=50))

strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.25, -0.25))
)

batch = mbt.run_sweep_lite(
    strategy,
    {"fast": list(range(5, 100)), "slow": list(range(10, 500))},
    config, store,
)

# Build metric grid and visualize
mbt.plot.heatmap_2d({...}, show=True)
mbt.plot.surface_3d({...}, show=True)
```

### Statistical Arbitrage — Cross-Asset

```python
pair_close = mbt.symbol_ref("ETHUSDT", "close")
ratio = close / (pair_close + mbt.lit(1e-12))
equilibrium = kalman(ratio, q=1e-4, r=1e-2)
spread_z = (ratio - equilibrium).zscore(28)

strategy = (
    mbt.Strategy.create("stat_arb")
    .signal("pair_close", pair_close)
    .signal("spread_z", spread_z)
    .signal("signal", -spread_z)
    .size(mbt.col("signal"))
)

config = mbt.BacktestConfig(
    universe=[1, 2, 5],
    symbol_names={"BTCUSDT": 1, "ETHUSDT": 2, "BNBUSDT": 5},
    ...
)
```

---

## Indicator Reference

### Moving Averages

| Function | Description |
|----------|-------------|
| `sma(source, period)` | Simple Moving Average |
| `ema(source, span)` | Exponential Moving Average |
| `dema(source, period)` | Double EMA |
| `tema(source, period)` | Triple EMA |
| `wma(source, period)` | Weighted MA |
| `hma(source, period)` | Hull MA |
| `kama(source, period)` | Kaufman Adaptive MA |

### Momentum

| Function | Description |
|----------|-------------|
| `rsi(source, period)` | Relative Strength Index [0-100] |
| `roc(source, period)` | Rate of Change |
| `momentum(source, period)` | Raw price difference |
| `macd(source, fast, slow)` | MACD line |
| `stoch_k(period)` | Stochastic %K |
| `stoch_d(period, d_period)` | Stochastic %D (SMA of %K) |
| `stoch_rsi(source, period, rsi_period)` | Stochastic RSI [0-1] |
| `williams_r(period)` | Williams %R |
| `cci(period)` | Commodity Channel Index |
| `adx(period)` | Average Directional Index |
| `plus_di(period)` | Wilder's +DI (the ADX's bullish half) |
| `minus_di(period)` | Wilder's -DI (the ADX's bearish half) |
| `aroon_up(period)` | Aroon Up [0-100] — how recent the window's high is |
| `aroon_down(period)` | Aroon Down [0-100] |
| `aroon_oscillator(period)` | `aroon_up - aroon_down` [-100, 100] |
| `ppo(source, fast, slow)` | Percentage Price Oscillator |
| `trix(source, period)` | Rate of change of a triple-smoothed EMA |

### Volatility

| Function | Description |
|----------|-------------|
| `atr(period)` | Average True Range |
| `natr(period)` | Normalized ATR |
| `bollinger_bands(source, period, num_std)` | Returns (upper, middle, lower) |
| `keltner_channels(period, multiplier)` | Returns (upper, middle, lower) |
| `donchian_channels(period)` | Returns (upper, middle, lower) |
| `vortex(period)` | Returns (vi_plus, vi_minus) |

### Volume

| Function | Description |
|----------|-------------|
| `obv(source, vol)` | On-Balance Volume |
| `vwap()` | Volume-Weighted Average Price |
| `mfi(period)` | Money Flow Index |
| `cmf(period)` | Chaikin Money Flow |

### Filters

| Function | Description |
|----------|-------------|
| `kalman(source, q, r)` | Kalman filter |
| `garch(source, omega, alpha, beta)` | GARCH volatility |

### Statistics

| Function | Description |
|----------|-------------|
| `source.zscore(window)` | Rolling z-score |
| `source.linreg_slope(window)` | Linear regression slope |
| `source.linreg_value(window)` | Linear regression fitted value |
| `source.linreg_r2(window)` | Linear regression R-squared |
| `source.rolling_median(window)` | Rolling median |
| `rolling_var(source, w)` | Rolling **population** variance (divides by `w`) |
| `rolling_skew(source, w)` | Rolling sample skewness (pandas `.skew()`) |
| `rolling_kurt(source, w)` | Rolling excess kurtosis (pandas `.kurt()`) |
| `rolling_rank(source, w)` | Percent-rank of the current value in the window [0-1] |
| `rolling_quantile(source, w, q)` | Rolling q-quantile, linear interpolation |
| `rolling_argmax(source, w)` | Bars since the window's max (0 = now) |
| `rolling_argmin(source, w)` | Bars since the window's min |
| `rolling_corr(a, b, w)` | Rolling Pearson correlation |
| `rolling_cov(a, b, w)` | Rolling sample covariance (ddof=1) |
| `rolling_beta(y, x, w)` | Rolling OLS beta of `y` on `x` |

`zscore`, `rolling_var`, `rolling_corr`, `rolling_cov` and `rolling_beta` also
take a duration (`Interval.seconds(30)`) instead of `w` -- see
[Windows in time](#windows-in-time).

### Signal state

Pine-style helpers. The first four take a **condition**, not a numeric series.

| Function | Description |
|----------|-------------|
| `bars_since(cond)` | Bars since `cond` was last true; NaN until it first is |
| `streak(cond)` | Length of the current consecutive run of true |
| `count_over(cond, w)` | Count of true rows in the trailing window |
| `time_since(cond)` | **Seconds** since `cond` was last true; NaN until it first is |
| `value_when(cond, source)` | `source` on the last bar where `cond` was true |
| `expr.ffill()` (method) | Last non-NaN value of `expr`, carried forward; NaN until the first one |
| `rising(source, n)` | 1.0 if strictly increasing on each of the last `n` steps |
| `falling(source, n)` | 1.0 if strictly decreasing |
| `pivot_high(source, left, right)` | Causal pivot high — **no lookahead** |
| `pivot_low(source, left, right)` | Causal pivot low |

A pivot only appears on its **confirmation bar**, `right` bars after the pivot
itself. That lag is what makes the signal tradable: a pivot detector that
reports on the pivot bar has read the future.

`ffill()` is a method on an expression, not a `condition -> series` helper, so it
reads its receiver as a series. It is the readable spelling of a persistent armed
state, and unlike `mbt.hold()` it composes with a regime filter -- see
[`mbt.hold()` holds the POSITION, not the value](#mbthold-holds-the-position-not-the-value).

### Cross-sectional (multi-asset)

These read the whole universe at one timestamp, so their argument must be a
column or a named signal — not a sub-expression. Define the sub-expression as
its own signal first.

| Function | Description |
|----------|-------------|
| `source.cs_mean()` | Cross-sectional mean, broadcast to every symbol |
| `source.cs_rank()` | Cross-sectional fractional rank [0-1] |
| `cs_zscore(source)` | `(v - mean) / std` across symbols, population std |
| `cs_demean(source)` | `v - mean` per timestamp |
| `cs_std(source)` | Cross-sectional population std, broadcast |
| `cs_scale(source)` | L1 (unit-gross) scaling: `v / sum(abs(v))` |
| `cs_winsorize(source, k)` | Clip to `[mean - k*std, mean + k*std]` |
| `cs_quantile(source, q)` | Cross-sectional q-quantile, broadcast |
| `cs_neutralize(source, factor)` | OLS residual of `source` on `factor` |

### Time

| Function | Description |
|----------|-------------|
| `source.lag(n)` | Value n bars ago |
| `source.lead(n)` | Value n bars ahead. **Future data**: in a strategy this is look-ahead by construction, and `detect_lookahead` fails it by name |
| `source.full_series_rank()` | Rank over the **whole series**, 1..N. **Future data**: the rank at bar `t` depends on bars after `t`, so it belongs in offline label building, never in a signal. `detect_lookahead` fails it. For a causal rank use `rolling_rank(w)` or `cs_rank()` |
| `source.diff(n)` | Difference over n bars |
| `source.pct_change(n)` | Percentage change over n bars |
| `source.rolling_mean(w)` | Rolling mean |
| `source.rolling_std(w)` | Rolling standard deviation |
| `source.cumsum()` | Cumulative sum |
| `hour()`, `minute()` | Clock components (UTC) |
| `day_of_week()` | 0 = Monday … 6 = Sunday |
| `month()`, `day_of_month()` | Calendar components |
| `year()` | Full year, e.g. 2024 |
| `week_of_year()` | ISO-8601 week [1-53] |
| `day_of_year()` | Ordinal day [1-366] |
| `is_month_start()`, `is_month_end()` | 1.0 / 0.0 |
| `is_quarter_end()` | 1.0 on the last day of Mar/Jun/Sep/Dec |
| `is_weekend()` | 1.0 on Saturday or Sunday |

ISO weeks belong to the year of their Thursday, so `week_of_year()` on 1 January
can read 52 or 53 — that is the definition, not a bug.

### TA-Lib compatibility

Bit-exact against TA-Lib 0.7.1, pinned by a stored fixture of 522 bars.

| Family | Functions |
|--------|-----------|
| Math transform | `sin` `cos` `tan` `asin` `acos` `atan` `sinh` `cosh` `log10` |
| Price transform | `median_price()` `typical_price()` `weighted_close()` `average_price()` |
| Pattern recognition | 38 `cdl_*` functions (see below) |

Every `cdl_*` returns one of `{-100, -80, 0, 80, 100}`: the sign is the
direction, the magnitude is TA-Lib's confidence, and **warmup bars are 0, not
NaN** — a detector's `0` already means "no pattern here".

```
cdl_doji  cdl_spinning_top  cdl_long_legged_doji  cdl_short_line  cdl_long_line
cdl_high_wave  cdl_rickshaw_man  cdl_marubozu  cdl_closing_marubozu
cdl_belt_hold  cdl_dragonfly_doji  cdl_gravestone_doji  cdl_engulfing
cdl_hammer  cdl_inverted_hammer  cdl_hanging_man  cdl_shooting_star  cdl_takuri
cdl_matching_low  cdl_homing_pigeon  cdl_harami  cdl_harami_cross
cdl_doji_star  cdl_piercing  cdl_thrusting  cdl_counterattack
cdl_three_inside  cdl_three_outside  cdl_morning_star  cdl_evening_star
cdl_dark_cloud_cover  cdl_three_white_soldiers  cdl_two_crows
cdl_identical_three_crows  cdl_tristar  cdl_separating_lines  cdl_on_neck
cdl_kicking
```

> All period/window arguments accept `mbt.param("name", default)` for sweep grids.

---

## Metrics Reference

Every result includes these performance metrics:

| Metric | Description |
|--------|-------------|
| `total_return` | Total return |
| `cagr` | Compound Annual Growth Rate |
| `volatility` | Annualized volatility |
| `sharpe` | Sharpe ratio |
| `sortino` | Sortino ratio |
| `calmar` | Calmar ratio |
| `max_drawdown` | Maximum drawdown |
| `tstat_sharpe` | t-statistic of Sharpe (sharpe * sqrt(years)) |
| `alpha` | Annualized CAPM alpha vs buy-and-hold benchmark |
| `beta` | Beta to benchmark |
| `tstat_alpha` | t-statistic of alpha (OLS regression) |
| `final_equity` | The last point of the equity curve, in the run's currency (`NaN` for a run without one) |
| `pnl` | `final_equity - initial_capital`: what the run made or lost, in money |

`final_equity` and `pnl` come after the other keys, in the metrics of every
result: `mbt.run` (bars or quotes), each result of `run_sweep` and
`run_batch`, and each of their lite counterparts, whose `final_equity` is also
the attribute `c.final_equity`. A `bt.sim` run's `res.metrics` has both,
against its `initial_cash`.

Everything but `total_return`, `max_drawdown`, `max_drawdown_duration_days`,
`final_equity`, `pnl` and the trade statistics is taken from the **daily**
returns: the equity at each UTC day's close, the first day measured from `initial_capital`, annualised
with `trading_days_per_year`. The bar size does not enter them.

**A run shorter than two days reports none of them.** One day is a single daily
return, and there is nothing to annualise: `sharpe`, `sortino`, `volatility`,
`cagr`, `calmar`, `tstat_sharpe`, `alpha`, `beta`, `tstat_alpha`, `skewness`,
`kurtosis`, `tail_ratio`, `omega_ratio`, `ulcer_index`, `best_day`,
`worst_day`, `avg_daily_return` and `pct_positive_days` are NaN, and
`result.warnings` says so. The run must reach into a second UTC day; a last
sample stamped exactly at the next midnight (a run from `2025-08-23` to
`2025-08-24`) still counts as one day. `total_return`, `max_drawdown` and the
trade statistics are reported whatever the length. This holds on every path:
`mbt.run`, `run_sweep`, `run_sweep_lite`, the batches, the GPU sweep and a
strategy that quotes.

The returns of each bar are not a substitute: a Sharpe annualised from them
multiplies by the square root of the bars in a year (17 800 at 100 ms), treats
steps that are not independent (a position marked to mid) as if they were, and
measures a different series than the same strategy run over a week. On a
one-day maker at 100 ms that gives a Sharpe of -1 000 and a CAGR of -99.7 % for
a day that lost 1.6 %.

A NaN is no value when ranking: `SweepResult.best()` and `worst()` skip it and
raise when every combination holds one, the walk-forward never selects it (and
refuses a fold where `optimize_metric` is NaN for every combination), and
pandas' `sort_values` puts it last. With arrays from `mbt.sweep_columns`, use
`np.nanargmax`: `np.argmax` returns the first NaN. In polars a NaN sorts above
every number: filter it out (`pl.col("sharpe").is_not_nan()`) before a
descending sort.

---

## Best Practices

1. **Set `signal_delay` deliberately.** It defaults to `0` (fill at the signal bar's close). Raise it to `1` when one bar is a realistic decision-to-fill latency, i.e. on fine-grained bars.
2. **Set `warmup_bars`** to at least the longest indicator period.
3. **Use `mbt.when()` for sizing.** Keep signal logic readable and composable.
4. **Run diagnostics** (`detect_lookahead`, `check_exposure_stability`) on new strategies.
5. **Start with `bar_interval=hours(12)` or `days(1)`** for fast iteration, then refine with smaller intervals.
6. **Use `accuracy=True`** only for final validation with SL/TP — it's 60x slower.
7. **Sweep with `run_sweep_lite`** for large grids. Use `run_sweep` only when you need full Result objects.
