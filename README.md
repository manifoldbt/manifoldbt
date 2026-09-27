<p align="center">
  <img src="https://raw.githubusercontent.com/manifoldbt/manifoldbt/master/assets/logo.png" width="110" alt="Manifold-BT logo">
</p>

<h1 align="center">Manifold-BT</h1>

<p align="center">
  <strong>Python backtesting with a Rust engine. Fast enough to sweep, strict about fills.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/manifoldbt/"><img src="https://img.shields.io/pypi/v/manifoldbt?logo=pypi&logoColor=white&color=2f6fed" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9%E2%80%933.13-3776AB?logo=python&logoColor=white" alt="Python 3.9 to 3.13">
  <a href="https://github.com/manifoldbt/manifoldbt/actions/workflows/ci.yml"><img src="https://github.com/manifoldbt/manifoldbt/actions/workflows/ci.yml/badge.svg?branch=master" alt="Tests against the PyPI wheel"></a>
  <a href="https://github.com/manifoldbt/manifoldbt/actions/workflows/bench-vs-vectorbt.yml"><img src="https://img.shields.io/badge/benchmarks-public%20CI-2ea44f?logo=githubactions&logoColor=white" alt="Benchmarks in public CI"></a>
  <a href="https://discord.gg/bvU6Wjc72d"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://www.manifoldbt.com/documentation">Documentation</a> &middot;
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#performance">Benchmarks</a> &middot;
  <a href="#correctness">Correctness</a> &middot;
  <a href="https://github.com/manifoldbt/manifoldbt/tree/master/examples">Examples</a> &middot;
  <a href="https://www.manifoldbt.com">Website</a>
</p>

---

Manifold-BT is a backtesting library for quantitative research. You describe a
strategy once, in a fluent Python DSL, and hand it to the engine: the whole
backtest then runs in Rust, signals, fills, fees, slippage, funding and equity
included. Python stays out of the hot loop.

- **Fast.** 10 million bars with a full performance summary in **337 ms**,
  where vectorbt takes 85.6 s. 10,000 parameter sets over 200,000 bars in
  **9.3 s inside 104 MB**. Measured on a standard 4-vCPU GitHub runner,
  [every run linked](#performance).
- **Checked.** Entry price, exit price and exit reason are asserted against
  vectorbt in CI, on the wheel published to PyPI. A benchmark workload the
  engines disagree on gets no timing at all.
- **Explicit.** Every fill rule is [written down](#execution-semantics),
  pessimistic wherever a choice existed, and so is
  [what the engine does not model](#what-is-not-covered).

```bash
pip install manifoldbt
```

Wheels for Linux, Windows and macOS. No Rust toolchain, no compiler, no account.

## Quick start

Two years of hourly Bitcoin, downloaded from Binance's public archive, and an
EMA crossover with perpetual-futures fees and slippage. It runs as-is on a
fresh install, in a couple of seconds including the download.

```python
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Interval, Slippage

# Two years of hourly BTCUSDT from Binance's public archive. No API key.
store = mbt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1, interval="1h",
                   start="2023-01-01T00:00:00Z", end="2025-01-01T00:00:00Z")

fast, slow = ema(close, 12), ema(close, 26)
strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.5, -0.5))   # half the equity, long or short
)

start, end = time_range("2023-01-01", "2025-01-01")
config = mbt.BacktestConfig(
    universe=[1], time_range_start=start, time_range_end=end,
    bar_interval=Interval.hours(1), initial_capital=10_000, warmup_bars=26,
    execution=mbt.ExecutionConfig(allow_short=True),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
)

result = mbt.run(strategy, config, store)
print(result.summary())
```

Output, trimmed:

```text
Strategy: ema_cross
----------------------------------------
  Total Return               -0.63%
  Sharpe                       0.11
  Max Drawdown              -36.38%
  ...
  Trades
  --------------------------------------
    Total                       605
    Total Fees              2807.07
```

A losing strategy, and that is the lesson. Remove the `fees=` and `slippage=`
lines and the same crossover returns **+51.8%**: 605 trades paid $2,807 on a
$10,000 account. Costs belong in the first run, not the last.

### Sweep a grid, read the surface

Replace a number with `mbt.param(...)` and the strategy becomes a grid. One
call runs every combination in parallel:

```python
# to_df() needs pandas: pip install "manifoldbt[pandas]"
fast = ema(close, mbt.param("fast"))
slow = ema(close, mbt.param("slow"))
strategy = (
    mbt.Strategy.create("ema_grid")
    .signal("fast", fast)
    .signal("slow", slow)
    .size(mbt.when(fast > slow, 0.5, -0.5))
)

sweep = mbt.run_sweep(strategy, {"fast": range(5, 55, 5), "slow": range(60, 260, 20)},
                      config, store)
surface = sweep.to_df(backend="pandas").pivot(
    index="param_fast", columns="param_slow", values="sharpe")
```

A hundred backtests come back as one table, a row per combination with its
parameters and its metrics. Read the shape rather than the maximum: a lone
peak in a flat field is usually a fitted accident, a broad plateau is a
candidate. Then take the candidate to [walk-forward](https://github.com/manifoldbt/manifoldbt/blob/master/examples/07_walk_forward.py)
and [Monte Carlo](https://github.com/manifoldbt/manifoldbt/blob/master/examples/10_monte_carlo.py).
With the `[plot]` extra, `sweep.plot_metric("sharpe")` draws the heatmap.

## Performance

Every number in this section comes from a benchmark that runs in public CI on a
standard GitHub runner. It installs each engine from PyPI the way a user would,
generates its own data, checks that the engines produced the **same result**,
and only then reports how long each took.

```text
SMA crossover with drawdown, Sharpe, Sortino and volatility, 10M bars
  Manifold-BT  ▎                                                  0.34 s
  raptorbt     ▍                                                  0.74 s
  vectorbt     ████████████████████████████████████████████████   85.6 s

Parameter sweep, 10,000 combinations over 200,000 bars
  Manifold-BT  ███                                                 9.3 s
  raptorbt     ████████████████████████████████████████████████  146.4 s
  vectorbt     not run: it materialises every combination, this grid needs tens of GB
```

**[Run 35590549822](https://github.com/manifoldbt/manifoldbt/actions/runs/35590549822)**:
Linux x86_64, 4 vCPU, Python 3.12, manifoldbt 0.26.0, vectorbt 0.28.4,
raptorbt 0.9.0. Seven interleaved repetitions (two for sweeps), medians
reported; the ratio is the median of the per-repetition ratios. All five rows
below agree with vectorbt exactly (relative tolerance 1e-9).

| Workload | Bars | Manifold-BT | vectorbt | raptorbt |
|---|---:|---:|---:|---:|
| SMA 30/150 crossover | 10M | **341 ms** | 22.0 s (×64.5) | 759 ms (×2.2) |
| ...plus drawdown, Sharpe, Sortino, volatility | 10M | **337 ms** | 85.6 s (**×250**) | 743 ms (×2.2) |
| ...with a 5 bps fee and 2 bps slippage | 10M | **335 ms** | 21.4 s (×63.7) | not supported |
| EMA 12/26 + RSI(14) filter, 5 bps fee | 1M | **52 ms** | 1.64 s (×31.5) | not supported |
| Five assets in one book, 5 bps fee | 1M | **161 ms** | 2.06 s (×12.4) | not supported |

The second row is the one worth reading twice. A performance summary costs
Manifold-BT nothing measurable, because it computes one during the run whether
you read it or not. vectorbt defers the equity curve until a risk metric asks
for it, then has to build one, and that costs it over a minute on this data.
"Not supported" means raptorbt has no way to express the workload (fixed
quantity sizing, a multi-asset book); the benchmark prints the reason instead
of an empty cell.

### Parameter sweeps

| Bars | Combinations | Manifold-BT | vectorbt | raptorbt |
|---:|---:|---:|---:|---:|
| 20,000 | 5,000 | **0.47 s**, 23 MB | 5.23 s, 2.5 GB | 6.65 s, 36 MB |
| 20,000 | 20,000 | **1.84 s**, 65 MB | not run | 26.8 s |
| 200,000 | 10,000 | **9.31 s**, 104 MB | not run | 146.4 s |

Past a certain grid the question stops being speed. vectorbt materialises the
simulation per combination, 1.57 MB of it at 20,000 bars, so the larger grids
would ask the runner for tens of gigabytes. Each sweep is checked for parity
before it is timed: every combination against raptorbt, and against vectorbt
on the full grid where it fits, or on a 250-combination anchor grid where it
does not. The sweep job runs with a Pro licence, since Community sweeps stop
at 256 combinations.

### Where it does not win

- **Against raptorbt the gap is about 2×, not orders of magnitude.** Both are
  compiled engines; the large ratios in the tables are against vectorbt.
- **Multi-asset is the smallest lead over vectorbt (×12.4).** Broadcasting a
  column per asset is close to free for vectorbt, while walking five books is
  not free for anything.
- **On a stop-loss/take-profit bracket, raptorbt is level** (×1.0 at 1M bars,
  ×1.15 at 10M). Deciding which side triggers first inside a bar is a
  sequential walk in both engines. That workload is also not in the table
  above: the engines disagree on whether to re-enter on the bar a bracket
  closes, so its timing sits in the benchmark's annex with the cause written
  down.

Reproduce any of it: fork the repository and press **Run workflow** on
[the benchmark](https://github.com/manifoldbt/manifoldbt/actions/workflows/bench-vs-vectorbt.yml),
or run [`benchmarks/vs_vectorbt/`](https://github.com/manifoldbt/manifoldbt/tree/master/benchmarks/vs_vectorbt)
locally. The method, the parity gate and the known divergences are in
[its README](https://github.com/manifoldbt/manifoldbt/blob/master/benchmarks/vs_vectorbt/README.md).
Each release re-runs it, and the latest results are on
[manifoldbt.com/benchmarks](https://www.manifoldbt.com/benchmarks).

Against an event-driven engine the comparison is only an order of magnitude:
backtrader runs an EMA(12/26) + RSI(14) strategy on 500K 1-minute bars in
46.9 s against 13 ms here
([`benchmarks/bench_vs_competitors.py`](https://github.com/manifoldbt/manifoldbt/blob/master/benchmarks/bench_vs_competitors.py),
median of 3, on a developer machine). It stays out of the CI suite because its
fills differ, and the parity gate publishes no timing for engines that did not
do the same work.

## Correctness

Speed is worth nothing if the fills are wrong. Everything in this section is a
test in this repository, and it runs in CI against the wheel **published on
PyPI**, not against the source, on Python 3.9 to 3.13. No licence is configured
in that job on purpose, so what it exercises is the experience of someone who
has just run `pip install manifoldbt`, and the run prints which tests skipped
rather than showing a green tick that hides them.

**Fill-level parity with vectorbt.**
[`test_parity_vectorbt.py`](https://github.com/manifoldbt/manifoldbt/blob/master/python/tests/test_parity_vectorbt.py)
checks **where a trade actually filled and why it exited**, not a summary
statistic. Each scenario is a short series built to produce one clean round
trip, and the entry price, exit price, exit reason and final return are all
asserted against vectorbt: market take-profit, market stop-loss,
stop-loss/take-profit brackets, shorts and trailing stops. A further scenario
pins the fee arithmetic across two round trips.

**Resting limit entries** have no vectorbt equivalent to compare against. They
are pinned instead by a NumPy model that computes the fill from raw OHLC with no
call into the engine. That catches an implementation drifting away from the
documented rule; it does not independently prove the rule is right, because the
rule was measured from the engine before being re-implemented.

**Look-ahead.** Regression tests in
[`python/tests/`](https://github.com/manifoldbt/manifoldbt/tree/master/python/tests),
the strictest of which corrupts every bar after a cut point, re-runs, and
requires the equity before that point to come back **bit-identical**: any
decision that read a future bar moves the prefix.

### Execution semantics

These rules are pinned by tests. They are written out so you can check them
against the wheel you installed rather than take them on faith. Where a choice
was available, the pessimistic one was taken.

| Situation | What the engine does |
|---|---|
| Stop-loss and take-profit both touched in the same bar | **The stop wins.** The path within a bar is unknown, so the unfavourable outcome is assumed. |
| Price gaps through a stop | Fills at the bar's **open**, not at the stop price. A stop at 100 on a bar opening at 95 fills at 95. |
| A limit entry is never reached | It expires at the end of its good-till window without filling. |
| A limit entry is immediate-or-cancel | It is cancelled, and will not fill on a later bar that would have triggered it. |
| A stop-limit's stop level is touched | The order arms and then rests at its limit, which may never fill. |
| An order is too large for the bar | It fills over several bars at the configured participation rate, and the partial position is bracketed **while** it fills. |
| Trailing stops | Never trigger on the bar they ratchet on, which would require reading the intra-bar path. |
| An execution-price expression evaluates to NaN | Falls back to the close and warns, rather than dropping the trade silently. |
| A short option's maintenance margin exceeds equity | The position is force-closed on that bar. |
| Fees, funding and borrow rates | Resolved per symbol, so venues inside one portfolio keep their own rates. |

### What is not covered

The edges, stated rather than left to be discovered:

- **No general liquidation model.** Margin force-close exists for short options
  only. A leveraged spot or perpetual position is not liquidated by the engine.
- **No corporate actions.** Yahoo prices arrive dividend-adjusted from the
  source (`dataset="raw"` opts out) and splits are whatever the provider
  returns. Nothing in the engine reconstructs either.
- The cross-engine scenarios run on short synthetic series, each built to
  isolate one behaviour. They are not a long backtest over market data compared
  trade for trade.
- Perpetual funding is exercised by the engine's own tests, but has no
  cross-engine parity test.
- The largest universe under test is a handful of instruments. Cross-sectional
  research across thousands of assets is exercised by the sweep benchmarks, not
  by the correctness suite.
- Look-ahead coverage is a set of regression tests and one worked example, not
  a systematic battery of leak archetypes.

The engine's own Rust suite is not published, which is why the rules above are
written out rather than linked. Independent verification is welcome, and a
reproduction showing a fill this engine gets wrong is the most useful report
this project can receive.

## Look-ahead

`mbt.detect_lookahead` re-runs a strategy over different windows and compares
the trades they have in common. That isolates bias coming from the engine or
from a strategy's own use of time.

One shape of leak is invisible to that comparison by construction: a signal
that reads a fixed number of bars ahead (`close.lead(1)`, or a column
precomputed with future bars and imported as data). Bar T sees the same T+1 in
every window, so the re-runs report PASS next to a Sharpe in the hundreds. The
detector therefore also walks the strategy's expressions and fails any explicit
`lead()`, naming the signal. For an implicit one, the check that works is to
perturb the bars after some K and require the equity up to K to be
bit-identical.

A parameter derived from the data *before* the backtest is a different
question: a threshold computed over the whole history in a notebook is the same
number in every run, so no re-run can weigh it. Re-derive it on the window under
test.
[`examples/25_lookahead_trap.py`](https://github.com/manifoldbt/manifoldbt/blob/master/examples/25_lookahead_trap.py)
runs every one of these checks on the same strategy and prints what each
concludes, so the difference is visible rather than asserted.

## What you can build

**Data.** Built-in connectors, all returning a store ready for `mbt.run`:
Binance, Bybit, Hyperliquid, dYdX, Bitstamp, Deribit, Yahoo Finance and
Dukascopy on every tier, Databento and Massive with Pro. Or bring your own CSV
(standard, MetaTrader 4 and MetaTrader 5 layouts are detected):

```python
store = mbt.import_csv("EURUSD_1m.csv", symbol="EURUSD", symbol_id=1,
                       interval="1m", asset_class="forex")

# Stocks, ETFs, indices, FX and futures from Yahoo Finance, no API key
store = mbt.ingest(provider="yahoo", symbol="AAPL", symbol_id=1, interval="1d",
                   asset_class="equity",
                   start="2015-01-01T00:00:00Z", end="2026-01-01T00:00:00Z")
```

Yahoo prices are dividend-adjusted, like `yfinance`'s `auto_adjust=True`; pass
`dataset="raw"` for unadjusted quotes. Deribit option contracts carry their
strike, expiry and settlement into the store and settle at intrinsic value on
the expiration bar.

**Signals.** Over 100 indicator functions and 38 candlestick patterns, from
moving averages and oscillators to rolling regression, Kalman filters, GARCH,
and cross-sectional z-scores and neutralization, all composable with
conditions, lags and references to other assets. The ones the library does not
ship
[you can write yourself](https://github.com/manifoldbt/manifoldbt/blob/master/examples/19_custom_indicators.py).

**Higher timeframes, without look-ahead.** Declare them next to the simulation
timeframe and read them with `mbt.tf(...)`. A higher-timeframe bar only becomes
readable once it has closed. `.apply(...)` computes an indicator on that
timeframe's own grid, so the period counts in *its* bars:

```python
from manifoldbt.indicators import close, sma

config = mbt.BacktestConfig(
    ...,
    bar_interval=Interval.minutes(1),                 # simulate on 1m
    extra_timeframes={"1h": Interval.hours(1)},       # also resample to 1h
)
band = mbt.tf("1h").apply(sma(close, 20))             # mean of 20 HOURLY closes
```

`sma(mbt.tf("1h").close, 20)` is not the same thing: it smooths the step-held
hourly series over 20 *simulation* bars.

**Sweep a choice, not just a number.** `mbt.choice(...)` makes an expression an
axis of the grid. Each combination resolves to its branch before simulation, so
the branches it did not pick cost nothing:

```python
band = mbt.choice("band", {
    "30m": mbt.tf("30m").apply(sma(close, mbt.param("len"))),
    "1h":  mbt.tf("1h").apply(sma(close, mbt.param("len"))),
})
strategy = (mbt.Strategy.create("band_cross")
            .signal("band", band)
            .size(mbt.when(close > mbt.col("band"), 1.0, 0.0)))
config.extra_timeframes = {"30m": Interval.minutes(30), "1h": Interval.hours(1)}

sweep = mbt.run_sweep(strategy, {"band": ["30m", "1h"], "len": range(10, 210, 10)},
                      config, store)
```

The same mechanism sweeps which exogenous column to use, which asset to
reference, or which indicator to apply.

**Execution.** Maker and taker fees per venue, perpetual funding, borrow costs,
slippage, partial fills at a participation rate, limit, stop and stop-limit
entries, brackets and trailing stops, and fills at a price the strategy
computes. Simulation on bars from one minute up, or from one second with Pro.

**Research.** Walk-forward optimization, Monte Carlo resampling, stochastic path
simulation (GBM, Heston, Merton jump-diffusion, GARCH with jumps), multi-asset
and multi-strategy portfolios, cross-exchange universes, exposure and risk
checks, and HTML tearsheets.

**Trades and quotes** (Researcher). Resolve stops, targets and resting entries
against the individual trades inside each bar instead of assuming the order of
high and low, attach a venue's quotes to each bar, and run order-flow or
market-making strategies on the tape itself, with a modelled queue.

## Examples

Clone the repository and run one:

```bash
python examples/13_stochastic_simulation.py
```

Most examples backtest real market data, which is not in the repository.
`python examples/setup_data.py` downloads it once, from free connectors that
need no API key, and
[examples/README.md](https://github.com/manifoldbt/manifoldbt/blob/master/examples/README.md)
says which files need it and which run on nothing at all. Charts come from the
optional `[plot]` extra: without it an example still runs to the end and skips
its chart.

Good places to start:
[01 Trend following](https://github.com/manifoldbt/manifoldbt/blob/master/examples/01_trend_following.py) &middot;
[05 Statistical arbitrage](https://github.com/manifoldbt/manifoldbt/blob/master/examples/05_stat_arb.py) &middot;
[07 Walk-forward](https://github.com/manifoldbt/manifoldbt/blob/master/examples/07_walk_forward.py) &middot;
[25 Look-ahead trap](https://github.com/manifoldbt/manifoldbt/blob/master/examples/25_lookahead_trap.py) &middot;
[26 Fill costs](https://github.com/manifoldbt/manifoldbt/blob/master/examples/26_fill_costs.py)

<details>
<summary><strong>All 28 examples</strong></summary>

| # | Example | What it shows |
|---|---------|---------------|
| 00 | [Template](https://github.com/manifoldbt/manifoldbt/blob/master/examples/00_template.py) | Minimal starting point |
| 01 | [Trend Following](https://github.com/manifoldbt/manifoldbt/blob/master/examples/01_trend_following.py) | EMA crossover, volume filter, stop-loss |
| 02 | [Mean Reversion](https://github.com/manifoldbt/manifoldbt/blob/master/examples/02_mean_reversion.py) | EMA crossover with parameter sweep |
| 03 | [Multi-Asset Momentum](https://github.com/manifoldbt/manifoldbt/blob/master/examples/03_multi_asset_momentum.py) | Cross-asset signals |
| 04 | [Linear Regression](https://github.com/manifoldbt/manifoldbt/blob/master/examples/04_linear_regression.py) | Regression-based signal |
| 05 | [Statistical Arbitrage](https://github.com/manifoldbt/manifoldbt/blob/master/examples/05_stat_arb.py) | Pairs trading, spread z-score |
| 06 | [Full Visualization](https://github.com/manifoldbt/manifoldbt/blob/master/examples/06_full_visualization.py) | Tearsheet and charts |
| 07 | [Walk-Forward](https://github.com/manifoldbt/manifoldbt/blob/master/examples/07_walk_forward.py) | Out-of-sample validation |
| 08 | [2D Sweep](https://github.com/manifoldbt/manifoldbt/blob/master/examples/08_sweep_2d_heatmap.py) | Parameter grid heatmap |
| 09 | [3D Surface](https://github.com/manifoldbt/manifoldbt/blob/master/examples/09_surface_3d.py) | Parameter surface plot |
| 10 | [Monte Carlo](https://github.com/manifoldbt/manifoldbt/blob/master/examples/10_monte_carlo.py) | Permutation-based robustness |
| 11 | [Portfolio](https://github.com/manifoldbt/manifoldbt/blob/master/examples/11_portfolio.py) | Multi-strategy portfolio |
| 12 | [Diagnostics](https://github.com/manifoldbt/manifoldbt/blob/master/examples/12_diagnostics.py) | Look-ahead and exposure checks |
| 13 | [Stochastic Simulation](https://github.com/manifoldbt/manifoldbt/blob/master/examples/13_stochastic_simulation.py) | SDE path simulation (GBM, Heston, ...) |
| 14 | [Multi-Timeframe](https://github.com/manifoldbt/manifoldbt/blob/master/examples/14_multi_timeframe.py) | Combining signals across timeframes |
| 15 | [Cross-Exchange](https://github.com/manifoldbt/manifoldbt/blob/master/examples/15_cross_exchange.py) | Signal on one venue, execute on another |
| 16 | [Exogenous Data](https://github.com/manifoldbt/manifoldbt/blob/master/examples/16_hashrate_exogene.py) | External series (e.g. hashrate) as a signal |
| 17 | [Per-Venue Fees](https://github.com/manifoldbt/manifoldbt/blob/master/examples/17_per_venue_fees.py) | Per-venue funding and borrow costs |
| 18 | [CSV Import](https://github.com/manifoldbt/manifoldbt/blob/master/examples/18_csv_import.py) | Load OHLCV from CSV (standard / MT4 / MT5) |
| 19 | [Custom Indicators](https://github.com/manifoldbt/manifoldbt/blob/master/examples/19_custom_indicators.py) | Write the ones the library does not ship |
| 20 | [Entry Orders](https://github.com/manifoldbt/manifoldbt/blob/master/examples/20_entry_orders.py) | Rest an entry at a price instead of taking the close |
| 21 | [Computed Fill Level](https://github.com/manifoldbt/manifoldbt/blob/master/examples/21_fill_at_computed_level.py) | Fill at a level the strategy computes |
| 22 | [Yahoo Equities](https://github.com/manifoldbt/manifoldbt/blob/master/examples/22_yahoo_equities.py) | Stocks, ETFs, indices, FX and futures |
| 23 | [Crypto Options](https://github.com/manifoldbt/manifoldbt/blob/master/examples/23_deribit_options.py) | Deribit contracts that actually expire |
| 24 | [Option Spread](https://github.com/manifoldbt/manifoldbt/blob/master/examples/24_option_spread.py) | A bull call spread, held to expiration |
| 25 | [Look-Ahead Trap](https://github.com/manifoldbt/manifoldbt/blob/master/examples/25_lookahead_trap.py) | Which audit answers which question |
| 26 | [Fill Costs](https://github.com/manifoldbt/manifoldbt/blob/master/examples/26_fill_costs.py) | The same signal filled four ways, and what each costs |
| 27 | [Bars vs Tape](https://github.com/manifoldbt/manifoldbt/blob/master/examples/27_bars_vs_tape.py) | A bracket resolved on the candle and on the trades inside it (Researcher) |

</details>

## Installation options

```bash
pip install manifoldbt              # engine only: backtests, sweeps, metrics
pip install manifoldbt[pandas]      # + DataFrame output
pip install manifoldbt[plot]        # + interactive charts and native windows (show=True)
pip install manifoldbt[all]         # everything: plots, windows, PNG export, pandas/polars
pip install manifoldbt[gpu]         # + NVIDIA runtime compiler, for device="cuda"
```

The base install stays light (no browser, no GUI) for scripts, servers and CI.
`[plot]` adds plotly and a native window backend; `[all]` also pulls kaleido
for static PNG/SVG export, which bundles a headless Chromium.

The Linux and Windows x86_64 wheels already carry the CUDA kernels, so `[gpu]`
only adds the NVIDIA runtime compiler (~180 MB) that compiles them on your
machine. Skip it if you already have a CUDA toolkit installed. An NVIDIA driver
is required, and GPU acceleration is a Researcher feature; everything else runs
at full speed on the CPU.

**Staying up to date.** manifoldbt asks PyPI once a day, in the background,
whether a newer release exists, and prints a one-line notice under the banner
when one does. It never delays an import (the notice is the previous run's
answer, read from a local cache) and the request is a plain GET of a public
JSON document. Set `MANIFOLDBT_NO_UPDATE_CHECK=1` to turn it off, or call
`mbt.check_for_update()` to ask on demand.

## Community, Pro and Researcher

Single backtests are free and run at full speed, with no run limit and no
account. The
paid tiers add scale and depth. See [pricing](https://www.manifoldbt.com/#pricing).

| | Community | Pro | Researcher |
|---|:---:|:---:|:---:|
| Single backtests (`mbt.run`) | Unlimited, full speed | Unlimited | Unlimited |
| Parameter sweeps and batches | Up to 256 backtests per sweep | Unlimited | Unlimited |
| Simulation resolution (`bar_interval`) | 1 minute and coarser | Down to 1 second | Down to 1 second |
| Output resolution | Daily | Down to 1 second | Down to 1 second |
| Monte Carlo | 1,000 simulations | Unlimited | Unlimited |
| Walk-forward optimization | - | Yes | Yes |
| Look-ahead detection (`detect_lookahead`) | - | Yes | Yes |
| Exposure and risk checks (`risk_check`) | Yes | Yes | Yes |
| Cross-exchange universes | - | Yes | Yes |
| Free connectors and CSV import | Yes | Yes | Yes |
| Databento and Massive connectors | - | Yes | Yes |
| Tearsheets and export | Yes | Yes | Yes |
| GPU acceleration (`device="cuda"`) | - | - | Yes |
| Trade tape and quotes | - | - | Yes |

## Documentation and community

- API reference, indicator list, configuration guide:
  **[manifoldbt.com/documentation](https://www.manifoldbt.com/documentation)**
- Writing strategies, from the first signal to the tape:
  [docs/strategy-authoring.md](https://github.com/manifoldbt/manifoldbt/blob/master/docs/strategy-authoring.md)
- Questions, ideas, or a backtest to share: [Discord](https://discord.gg/bvU6Wjc72d)
- Found a fill the engine gets wrong? [Open an issue](https://github.com/manifoldbt/manifoldbt/issues)
  with a reproduction. It is the most useful report this project can receive.

## License

Apache 2.0 with Commons Clause. The source is available, free to use, modify and
self-host. Reselling the software or offering it as a paid hosted service is not
permitted. See [LICENSE](https://github.com/manifoldbt/manifoldbt/blob/master/LICENSE)
for the full text.
