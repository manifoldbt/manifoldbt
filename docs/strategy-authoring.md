# Strategy Authoring Guide

> **manifoldbt** — Python DSL for Declarative Strategy Definition

This guide describes how to define trading strategies using the manifoldbt Python DSL. Strategies are compiled into an optimized expression graph and executed by the Rust vectorized engine.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Indicators](#indicators)
3. [Signals & Sizing](#signals--sizing)
4. [Parameters & Sweeps](#parameters--sweeps)
5. [Backtest Configuration](#backtest-configuration)
6. [Execution Model](#execution-model)
7. [Fee & Slippage Models](#fee--slippage-models)
8. [Orders (SL/TP/Trailing)](#orders-sltptrailing)
9. [Entry Orders](#entry-orders)
10. [Cross-Asset References](#cross-asset-references)
11. [Dataset Auto-Resolution](#dataset-auto-resolution)
12. [Diagnostics](#diagnostics)
13. [Profiling](#profiling)
14. [Complete Examples](#complete-examples)
15. [Indicator Reference](#indicator-reference)

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
signal = mbt.when(rsi < 30, 1.0)  # buy oversold, hold otherwise
```

### Arithmetic on expressions

```python
trend = fast - slow
spread = close / (pair_close + mbt.lit(1e-12))   # mbt.lit() for constants in arithmetic
signal = -spread_z * mbt.lit(0.05)                # negation + scaling
```

> **Note:** `mbt.lit()` is needed for constants in arithmetic (`close + mbt.lit(1e-12)`). Numbers auto-coerce inside `mbt.when()`.

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
| `NaN`  | Hold current position unchanged         |

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
from manifoldbt.indicators import close, high, low, open
band_up, band_dn = sma * 1.012, sma * 0.992
exec_level = mbt.when(high >= band_up,
                      mbt.when(open >= band_up, open, band_up),   # gapped through
             mbt.when(low <= band_dn,
                      mbt.when(open <= band_dn, open, band_dn),
             close))
strat = strat.signal("exec_level", exec_level)
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

---

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

### Time in force

`"GTC"` (default, rests until filled or the signal changes), `{"GTB": n}`
(cancel after n bars), `"IOC"` (fill on the arrival bar or cancel).

### Three things to watch

**A resting order keeps the level it was created with.** `signal=` is read once,
on the bar the order is placed, and held until the order fills, expires or is
cancelled. It does **not** follow the series afterwards. That is the intended
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
result = mbt.run_backtest(strategy, config)
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
# Look-ahead bias detection: a static walk for lead(), then two re-runs
# over shorter windows. The re-runs cannot see a fixed read-ahead
# (close.lead(1) trades the same in every window); the static walk names it.
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

### Signal state

Pine-style helpers. The first four take a **condition**, not a numeric series.

| Function | Description |
|----------|-------------|
| `bars_since(cond)` | Bars since `cond` was last true; NaN until it first is |
| `streak(cond)` | Length of the current consecutive run of true |
| `count_over(cond, w)` | Count of true bars in the trailing window |
| `value_when(cond, source)` | `source` on the last bar where `cond` was true |
| `rising(source, n)` | 1.0 if strictly increasing on each of the last `n` steps |
| `falling(source, n)` | 1.0 if strictly decreasing |
| `pivot_high(source, left, right)` | Causal pivot high — **no lookahead** |
| `pivot_low(source, left, right)` | Causal pivot low |

A pivot only appears on its **confirmation bar**, `right` bars after the pivot
itself. That lag is what makes the signal tradable: a pivot detector that
reports on the pivot bar has read the future.

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

---

## Best Practices

1. **Set `signal_delay` deliberately.** It defaults to `0` (fill at the signal bar's close). Raise it to `1` when one bar is a realistic decision-to-fill latency, i.e. on fine-grained bars.
2. **Set `warmup_bars`** to at least the longest indicator period.
3. **Use `mbt.when()` for sizing.** Keep signal logic readable and composable.
4. **Run diagnostics** (`detect_lookahead`, `check_exposure_stability`) on new strategies.
5. **Start with `bar_interval=hours(12)` or `days(1)`** for fast iteration, then refine with smaller intervals.
6. **Use `accuracy=True`** only for final validation with SL/TP — it's 60x slower.
7. **Sweep with `run_sweep_lite`** for large grids. Use `run_sweep` only when you need full Result objects.
