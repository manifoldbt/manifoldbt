<p align="center">
  <img src="https://raw.githubusercontent.com/manifoldbt/manifoldbt/master/assets/logo.png" width="110" alt="Manifold-BT logo">
</p>

<p align="center">
  <strong>Manifold-BT</strong><br>
  Rust-powered backtesting engine for quantitative research
</p>

<p align="center">
  <a href="https://discord.gg/bvU6Wjc72d"><img src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join the Manifold-BT Discord" height="34"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/manifoldbt/"><img src="https://img.shields.io/pypi/v/manifoldbt?logo=pypi&logoColor=white&color=2f6fed" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/core-Rust-dea584?logo=rust&logoColor=white" alt="Rust core">
  <a href="https://github.com/manifoldbt/manifoldbt/actions/workflows/bench-vs-vectorbt.yml"><img src="https://img.shields.io/badge/benchmarks-public%20CI-2ea44f?logo=githubactions&logoColor=white" alt="Benchmarks in public CI"></a>
</p>

<p align="center">
  <a href="https://www.manifoldbt.com">Website</a> &middot;
  <a href="https://www.manifoldbt.com/docs/documentation.html">Documentation</a> &middot;
  <a href="https://github.com/manifoldbt/manifoldbt/tree/master/examples">Examples</a>
</p>

---

Manifold-BT is a Python backtesting library with a Rust core. Strategies are written in a
fluent Python DSL, compiled to a vectorized Rust expression graph, then run through a
sequential fill simulation with realistic fees, slippage, funding and look-ahead protection.
**Vectorized speed with event-driven execution realism.**

## Why Manifold-BT

- **Fast**: 10M bars in 329 ms. 79x faster than vectorbt, and 311x once you also want drawdown and Sharpe. [Measured in public CI](#performance), every run linked.
- **Expressive**: fluent DSL with 104 indicators and 38 candlestick patterns, conditional logic, cross-asset references
- **Rigorous**: Monte Carlo, walk-forward, parameter sweeps, lookahead detection, exposure diagnostics
- **Portable**: `pip install`, no Rust toolchain needed. Works on Python 3.9+.

## Installation

```bash
pip install manifoldbt              # engine only: backtests, sweeps, metrics
pip install manifoldbt[plot]        # + interactive charts and native windows (show=True)
pip install manifoldbt[all]         # everything: plots, windows, PNG export, pandas/polars
pip install manifoldbt[gpu]         # + NVIDIA runtime compiler, for device="cuda" (Pro)
```

The base install stays light (no browser, no GUI) for scripts, servers and CI.
`[plot]` adds plotly and a native window backend; `[all]` also pulls kaleido for
static PNG/SVG export (which bundles a headless Chromium).

The Linux and Windows x86_64 wheels already carry the CUDA kernels, so `[gpu]`
only adds the NVIDIA runtime compiler (~180 MB) that compiles them on your
machine. Skip it if you already have a CUDA toolkit installed. An NVIDIA driver
is required, and GPU acceleration is a Pro feature; everything else runs at full
speed on the CPU.

### Staying up to date

manifoldbt asks PyPI once a day, in the background, whether a newer release
exists, and prints a one-line notice under the banner when one does. It never
delays an import (the notice is the previous run's answer, read from a local
cache) and it sends nothing: the request is a plain GET of a public JSON
document. Set `MANIFOLDBT_NO_UPDATE_CHECK=1` to turn it off, and
`mbt.check_for_update()` to ask on demand -- it returns the newer version, or
`None` when you are current.

## Quick Start

```python
import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Interval, Slippage

fast = ema(close, 12)
slow = ema(close, 26)

strategy = (
    mbt.Strategy.create("ema_crossover")
    .signal("fast", fast)
    .signal("slow", slow)
    .signal("signal", mbt.when(fast > slow, mbt.lit(1.0), mbt.lit(-1.0)))
    .size(mbt.col("signal") * mbt.lit(0.25))
)

start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe=[1],
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(12),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(allow_short=True, max_position_pct=0.5),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=30,
)

store = mbt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1,
                   start="2022-01-01T00:00:00Z", end="2025-01-01T00:00:00Z", interval="1h")
result = mbt.run(strategy, config, store)
print(result.summary())
```

## Loading data

Bring your own data, or pull it from a built-in connector. Both return a
`DataStore` ready for `mbt.run(...)`.

**CSV**, free on all tiers, auto-detects standard / MetaTrader 4 / MetaTrader 5:

```python
store = mbt.import_csv("EURUSD_1m.csv", symbol="EURUSD", symbol_id=1,
                       interval="1m", asset_class="forex")
```

**Market data connectors**: Binance, Bybit, Hyperliquid, dYdX, Bitstamp, Yahoo Finance (free);
Databento, Massive (Pro):

```python
store = mbt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1,
                   start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z")
```

Yahoo Finance covers stocks, ETFs, indices (`^GSPC`), FX (`EURUSD=X`), futures
(`ES=F`) and crypto (`BTC-USD`) without an API key. Prices are dividend-adjusted,
like `yfinance`'s `auto_adjust=True`; pass `dataset="raw"` for unadjusted quotes.
Yahoo's own history limits apply: 1m over the last 30 days, 1h over ~2 years,
daily back to the listing date.

```python
store = mbt.ingest(provider="yahoo", symbol="AAPL", symbol_id=1, interval="1d",
                   asset_class="equity",
                   start="2015-01-01T00:00:00Z", end="2026-01-01T00:00:00Z")
```

Or from the CLI:

```bash
manifoldbt import-csv data.csv --symbol EURUSD --symbol-id 1 --interval 1m
manifoldbt ingest --provider binance --symbol BTCUSDT --symbol-id 1 --start ... --end ...
```

## Higher timeframes

Declare the timeframes you want alongside the simulation one, then read them
with `mbt.tf(...)`. Columns are forward-filled onto the simulation grid, and a
bar's value only becomes readable once that bar has closed, so there is no
look-ahead.

```python
config = mbt.BacktestConfig(
    ...,
    bar_interval=Interval.minutes(1),                 # simulate on 1m
    extra_timeframes={"1h": Interval.hours(1)},       # also resample to 1h
)

h1 = mbt.tf("1h")
h1.close          # the last closed hourly close, held across the minute bars
```

For an **indicator** on a higher timeframe, use `.apply(...)`. It evaluates the
expression on that timeframe's own grid, so the period counts in *its* bars:

```python
from manifoldbt.indicators import close, sma

band = mbt.tf("1h").apply(sma(close, 20))     # mean of 20 HOURLY closes
```

> Careful: `sma(mbt.tf("1h").close, 20)` is **not** the same thing. That reads
> the step-held hourly series on the simulation grid, so the period counts in
> simulation bars: on a 1m simulation it is a 20-*minute* smoothing of an hourly
> staircase. Use `.apply(...)` whenever you want an indicator *of* the higher
> timeframe.

## Sweeping a choice, not just a number

`mbt.param(...)` sweeps numbers. `mbt.choice(...)` sweeps *expressions*: the
selector becomes a grid axis, and each combination resolves to its branch before
the simulation runs, so the branches it did not pick cost nothing.

```python
band = mbt.choice("band", {
    "30m": mbt.tf("30m").apply(sma(close, mbt.param("len"))),
    "1h":  mbt.tf("1h").apply(sma(close, mbt.param("len"))),
    "2h":  mbt.tf("2h").apply(sma(close, mbt.param("len"))),
})

sweep = mbt.run_sweep(strategy, {"band": ["30m", "1h", "2h"],
                                 "len": range(10, 210, 10)}, config, store)
```

The branches can hold any expression, so the same mechanism sweeps which
exogenous column to use, which asset to reference, or which indicator to apply.

## Examples

Clone the repository and run one:

```bash
python examples/13_stochastic_simulation.py
```

Most of them backtest real market data, which is not in the repository.
`python examples/setup_data.py` downloads it once, from free connectors that
need no API key; [examples/README.md](https://github.com/manifoldbt/manifoldbt/blob/master/examples/README.md)
says which files need it and which run on nothing at all. Charts come from the
optional `[plot]` extra: without it an example still runs to the end and skips
its chart.

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
| 12 | [Diagnostics](https://github.com/manifoldbt/manifoldbt/blob/master/examples/12_diagnostics.py) | Lookahead & exposure safety checks |
| 13 | [Stochastic Simulation](https://github.com/manifoldbt/manifoldbt/blob/master/examples/13_stochastic_simulation.py) | SDE path simulation (GBM, Heston, …) |
| 14 | [Multi-Timeframe](https://github.com/manifoldbt/manifoldbt/blob/master/examples/14_multi_timeframe.py) | Combining signals across timeframes |
| 15 | [Cross-Exchange](https://github.com/manifoldbt/manifoldbt/blob/master/examples/15_cross_exchange.py) | Signal on one venue, execute on another |
| 16 | [Exogenous Data](https://github.com/manifoldbt/manifoldbt/blob/master/examples/16_hashrate_exogene.py) | External series (e.g. hashrate) as a signal |
| 17 | [Per-Venue Fees](https://github.com/manifoldbt/manifoldbt/blob/master/examples/17_per_venue_fees.py) | Per-venue funding & borrow costs |
| 18 | [CSV Import](https://github.com/manifoldbt/manifoldbt/blob/master/examples/18_csv_import.py) | Load OHLCV from CSV (standard / MT4 / MT5) |
| 19 | [Custom Indicators](https://github.com/manifoldbt/manifoldbt/blob/master/examples/19_custom_indicators.py) | Write the ones the library does not ship |
| 20 | [Entry Orders](https://github.com/manifoldbt/manifoldbt/blob/master/examples/20_entry_orders.py) | Rest an entry at a price instead of taking the close |
| 21 | [Computed Fill Level](https://github.com/manifoldbt/manifoldbt/blob/master/examples/21_fill_at_computed_level.py) | Fill at a level the strategy computes |
| 22 | [Yahoo Equities](https://github.com/manifoldbt/manifoldbt/blob/master/examples/22_yahoo_equities.py) | Stocks, ETFs, indices, FX and futures |
| 23 | [Crypto Options](https://github.com/manifoldbt/manifoldbt/blob/master/examples/23_deribit_options.py) | Deribit contracts that actually expire |
| 24 | [Option Spread](https://github.com/manifoldbt/manifoldbt/blob/master/examples/24_option_spread.py) | A bull call spread, held to expiration |
| 25 | [Look-Ahead Trap](https://github.com/manifoldbt/manifoldbt/blob/master/examples/25_lookahead_trap.py) | Which audit answers which question |

## Look-ahead

`mbt.detect_lookahead` re-runs a strategy over different windows and compares
the trades they have in common. That is what isolates bias coming from the
engine or from a strategy's own use of time.

One shape of leak is invisible to that comparison by construction: a signal
that reads a fixed number of bars ahead (`close.lead(1)`, or a column
precomputed with future bars and imported as data). Bar T sees the same T+1
in every window, so the shorter run trades exactly like the full one and the
re-runs report PASS, next to a Sharpe in the hundreds. No choice of split
changes that. The detector therefore also walks the strategy's expressions
and fails any explicit `lead()`, naming the signal; for an implicit one, the
check that works is to perturb the bars after some K and require the equity
up to K to be bit-identical, which `examples/25_lookahead_trap.py` does.

A parameter derived from the data *before* the backtest is a different
question: a threshold computed over the whole history in a notebook and then
passed in as a number is the same number in every run, so no re-run-based
method can weigh it. Treat any parameter that came from data as part of the
pipeline, re-derive it on the window under test, and compare the results.

[`examples/25_lookahead_trap.py`](https://github.com/manifoldbt/manifoldbt/blob/master/examples/25_lookahead_trap.py)
runs both on the same strategy, including perturbing every future bar, and
prints what each method concludes, so the difference is visible rather than
asserted.

## Correctness

Speed is worth nothing if the fills are wrong. Everything in this section is a
test in this repository, and it runs in CI against the wheel **published on
PyPI**, not against the source, on five Python versions. No licence is
configured in that job on purpose, so what it exercises is the experience of
someone who has just run `pip install manifoldbt`, and the run prints which
tests skipped rather than showing a green tick that hides them.

**Fill-level parity with vectorbt.**
[`test_parity_vectorbt.py`](https://github.com/manifoldbt/manifoldbt/blob/master/python/tests/test_parity_vectorbt.py)
checks **where a trade actually filled and why it exited**, not a summary
statistic. Each scenario is a short series built to produce one clean round
trip, and the entry price, the exit price, the exit reason and the final return
are all asserted against vectorbt: market take-profit, market stop-loss,
stop-loss/take-profit brackets, shorts and trailing stops. A further scenario
pins the fee arithmetic across two round trips. The equity curve is deliberately
left out, because a Community build caps it to daily resolution and it would not
be an apples-to-apples comparison.

**Resting limit entries are checked without vectorbt**, which has no resting
order to compare against. They are pinned instead by a NumPy model that computes
the fill from raw OHLC with no call into the engine. Read it for what it is: it
catches an implementation drifting away from the documented rule, and it does
not independently prove the rule is the right one, because the rule was measured
from the engine before being re-implemented.

**Look-ahead.** Two regression tests in
[`python/tests/`](https://github.com/manifoldbt/manifoldbt/tree/master/python/tests), the stricter of which corrupts every bar after
a cut point, re-runs, and requires the equity before that point to come back
**bit-identical**: any decision that read a future bar moves the prefix. They sit
alongside the worked example above, which exists to demonstrate a leak the
detector **cannot** see and to say so plainly.

### Execution semantics

The rules below are pinned by tests. They are written out here so you can check
them against the wheel you installed rather than take them on faith. Where a
choice was available, the pessimistic one was taken.

| Situation | What the engine does |
|---|---|
| Stop-loss and take-profit are both touched in the same bar | **The stop wins.** The path within a bar is unknown, so the unfavourable outcome is assumed rather than the profitable one. |
| Price gaps through a stop | Fills at the bar's **open**, not at the stop price. A stop at 100 on a bar opening at 95 fills at 95. |
| A limit entry is never reached | It expires at the end of its good-till window without filling. |
| A limit entry is immediate-or-cancel | It is cancelled, and will not fill on a later bar that would have triggered it. |
| A stop-limit's stop level is touched | The order arms and then rests at its limit, which may never fill. |
| An order is too large for the bar | It fills over several bars at the configured participation rate, and the partially filled position is bracketed **while** it fills. |
| Trailing stops | Never trigger on the same bar they ratchet on, which would require reading the intra-bar path. |
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
- Look-ahead coverage is two regression tests and one worked example, not a
  systematic battery of leak archetypes.

The engine's own Rust suite is not published, which is why the rules above are
written out rather than linked. Independent verification is welcome, and a
reproduction showing a fill this engine gets wrong is the most useful report
this project can receive.

## Performance

Every number below comes from a benchmark that runs in public CI on a standard
GitHub runner, and links back to the run that produced it. It installs each
engine from PyPI the way a user would, generates its own data, checks that the
engines produced the **same result**, and only then reports how long each took:
a workload they disagree on gets no published timing at all.

**Latest run: [#13](https://github.com/manifoldbt/manifoldbt/actions/runs/32469701489)**
ran on Linux x86_64, 4 vCPU (AMD EPYC 7763), Python 3.12, manifoldbt 0.18.0 /
vectorbt 0.28.4 / raptorbt 0.9.0, 3 interleaved repetitions, medians reported.

| Workload | Bars | Manifold-BT | vectorbt | raptorbt |
|---|---:|---:|---:|---:|
| SMA crossover | 10M | **327 ms** | 26.12 s (x79) | 913 ms (x2.8) |
| ...with drawdown, Sharpe, Sortino, volatility | 10M | **329 ms** | 102.38 s (**x311**) | 909 ms (x2.8) |
| ...with a 5 bps fee and 2 bps slippage | 10M | **337 ms** | 26.08 s (x79) | not supported |
| EMA + RSI filter, 5 bps fee | 1M | **57 ms** | 2.35 s (x40) | not supported |
| Five assets in one book | 1M | **148 ms** | 2.54 s (x17) | not supported |
| Stop-loss and take-profit bracket | 10M | **934 ms** | 26.24 s (x28) | 916 ms (**x1.0**) |

The second row is the one worth reading twice. Asking for a performance summary
costs Manifold-BT nothing measurable, because it computes one during the run
whether you read it or not, and costs vectorbt 102 seconds, because it defers
the equity curve until a risk metric needs it and then has to build one.

The last two rows are the ones where Manifold-BT does worst, and they are
published for that reason. Broadcasting a column per asset is close to free for
vectorbt, while walking five books is not free for anything. And on a
stop-loss/take-profit bracket, raptorbt is level with us: the intra-bar check
that decides which of the two triggers first is a sequential walk in both
engines, so there is no vectorization left to win with.

### Parameter sweeps

| Bars | Combinations | Manifold-BT | vectorbt | raptorbt |
|---:|---:|---:|---:|---:|
| 20,000 | 5,000 | **446 ms**, 40 MB | 5.84 s, 2.5 GB | 7.08 s |
| 200,000 | 10,000 | **9.96 s**, 79 MB | out of memory | 164.50 s |

Past a certain grid the question stops being speed. vectorbt materialises the
simulation per combination, 1.57 MB of it at 20,000 bars, so the second row
would ask a machine for tens of gigabytes. Manifold-BT runs it in ten seconds
inside 79 MB.

Reproduce any of it yourself: fork the repository and press **Run workflow** on
[the benchmark](https://github.com/manifoldbt/manifoldbt/actions/workflows/bench-vs-vectorbt.yml),
or run it locally from
[`benchmarks/vs_vectorbt/`](https://github.com/manifoldbt/manifoldbt/tree/master/benchmarks/vs_vectorbt).
The method, the parity gate and the known divergences are written up in
[its README](https://github.com/manifoldbt/manifoldbt/blob/master/benchmarks/vs_vectorbt/README.md).

### Against an event-driven engine

backtrader runs the same EMA(12/26) + RSI(14) strategy on 500K 1-minute bars in
**46,944 ms**, against **13 ms** for Manifold-BT: a factor of **3,556**. Measured
with `benchmarks/bench_vs_competitors.py`, median of 3 runs, on a developer
machine and not the CI runner, so it is not comparable line-for-line with the
table above.

It sits outside the CI suite because its event-driven fills produce a different
PnL, and the parity gate publishes no timing for engines that did not do the
same work. Treat it as an order of magnitude, not a benchmark: the two engines
are not doing the same thing.

### How it compares

| | Manifold-BT | vectorbt | backtrader | Nautilus |
|---|---|---|---|---|
| Engine | Rust (vectorized + sequential fills) | Numba/NumPy (vectorized) | Python (event-driven) | Rust/Python (event-driven) |
| Execution realism¹ | High | Basic | High | High |
| Focus | Backtesting + research | Backtesting at scale | Backtest + live | Backtest + live (production) |

¹ fees, slippage, funding, partial fills, look-ahead detection.

> On GPU (Pro), the Monte Carlo engine runs **~36x faster** than the all-core CPU path (SDE path simulation, RTX 3090, f32).

## Documentation

Full API reference, indicator list, configuration guide, and best practices:

**[www.manifoldbt.com/docs/documentation.html](https://www.manifoldbt.com/docs/documentation.html)**

## Community vs Pro

| | Community | Pro |
|---|---|---|
| Single backtests (`mbt.run`) | Unlimited, full speed | Unlimited, full speed |
| Parameter sweeps & batches | Up to 256 backtests per sweep | Unlimited |
| Simulation resolution (`bar_interval`) | 1 minute and coarser | Down to 1 second |
| Output resolution | Daily | Down to 1 second |
| Monte Carlo | 1K sims | Unlimited |
| Walk-Forward | - | Anchored + Rolling |
| Parameter Stability | - | Yes |
| Free connectors (Binance, Bybit, Hyperliquid, dYdX, Bitstamp, Yahoo) | Yes | Yes |
| Databento & Massive connectors | - | Yes |
| GPU acceleration (`device="cuda"`) | - | Yes |
| Safety checks (lookahead, exposure) | - | Yes |
| Tearsheets & export | Yes | Yes |

## License

Apache 2.0 with Commons Clause. The source is available, free to use,
modify and self-host. Reselling the software or offering it as a paid
hosted service is not permitted. See [LICENSE](https://github.com/manifoldbt/manifoldbt/blob/master/LICENSE) for the full text.
