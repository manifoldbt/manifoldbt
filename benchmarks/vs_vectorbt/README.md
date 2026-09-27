# manifoldbt vs vectorbt vs raptorbt

An engine-to-engine benchmark you can re-run yourself. It installs every engine
from PyPI, generates its own data, checks that they produced the **same
result**, and only then reports how long each took.

```bash
pip install manifoldbt
pip install -r requirements-lock.txt
python bench.py --bars 10000 100000 1000000 --reps 7 --cold-start-reps 3 --memory-bars 2000000 --out results.json
python report.py results.json
```

No dataset to download, no API key, no configuration. The same command runs in
GitHub Actions on a public runner, so every published number has a run URL
behind it.

**Python 3.12, and not by preference.** raptorbt is built against pyo3 0.20.3,
whose maximum supported CPython is 3.12: no release up to 0.9.0 publishes a
cp313 wheel, and a source build refuses outright. Comparing engines means
running them in one environment, and the environment has to be one all of them
support. Add or drop a challenger with `--engines`; an engine that is not
installed is skipped with a printed line rather than crashing the run.

**manifoldbt is the reference.** Every parity check and every ratio is a
challenger against it, never two challengers against each other: three engines
make three pairs, and a table of pairs is a matrix, not a benchmark. The
reference gets no advantage from the position, it is simply the one every
timing is divided by.

## The rule this harness is built around

A speed comparison between backtesters is worthless unless they did the same
work. So parity comes first:

1. each workload runs once per engine;
2. total return, round-trip count and fees are compared against the reference;
3. **a workload an engine disagrees on gets no published timing for it.**

Three verdicts come out of that gate:

| Verdict | Meaning | What gets published |
|---|---|---|
| `exact` | agreement down to float-reordering noise (relative tolerance 1e-9) | the timing, in the headline table |
| `documented` | that engine disagrees, the workload declared it in advance *for that engine*, and the cause is written down | the timing, in an annex, with the cause and its measured size |
| `failed` | it disagrees and nobody predicted it | nothing. The run exits non-zero |

A fourth outcome sits beside the gate rather than inside it. `unsupported`
means an engine cannot express the workload at all: it is not run, and the
report prints the reason instead of an empty cell. A blank in a speed table
reads as a defeat, and "this engine has no fixed-quantity sizing" is not a
defeat, it is a different fact.

The `failed` path is not decoration. It is the reason the other numbers can be
trusted, and it makes the benchmark fail loudly if a future release of either
engine changes a fill rule.

## Method

**Interleaved repetitions.** The engines alternate inside each repetition
(A, B, A, B, ...) rather than running in two blocks. On a shared cloud runner
that slows down halfway through, two blocks would hand the penalty to whichever
engine ran second; alternating splits it evenly.

**The ratio is the headline, milliseconds are context.** Each ratio comes from
two measurements taken seconds apart on the same machine. Absolute timings from
a shared runner are worth much less than the ratio between them.

**Dispersion is published, and noise is flagged.** Every point carries min,
median, max and interquartile range. If the IQR exceeds 15% of the median the
point is marked `noisy` and is not headline material, however good it looks.

**Warmup is discarded, and that favours vectorbt on purpose.** The first call of
each engine is thrown away, which is where vectorbt pays its numba compilation.
Charging a one-off JIT cost to every repetition would inflate the result.

**Data loading is excluded on both sides.** manifoldbt is handed a prepared
store, vectorbt is handed prepared Series. What is timed is indicators plus
simulation plus reading the headline metrics, nothing else.

**Only public APIs.** manifoldbt is driven through `bt.run(strategy, config,
store)`, the documented entry point, not through an internal fast path.

## What is compared

| Workload | What it exercises | vectorbt | raptorbt |
|---|---|---|---|
| `sma_cross` | SMA 30/150 crossover, long-only, no cost | exact | exact |
| `sma_cross_metrics` | the same simulation, plus max drawdown, Sharpe, Sortino and volatility | exact | exact |
| `sma_cross_costs` | the same signal with a 5 bps fee and 2 bps of slippage | exact | unsupported |
| `multi_asset` | five assets in one shared book, capped at 1M bars | exact | unsupported |
| `ema_rsi_fees` | EMA 12/26 with an RSI(14) filter and a 5 bps taker fee, capped at 1M bars | exact | unsupported |
| `bracket_sl_tp` | the same entry with a 15 bps stop and a 30 bps target | documented | documented |

### Costs live in their own workload, and not by preference

A cost model cannot simply be switched on across the board. Measured on the
headline workload, a fee or a slippage applied to `FractionOfEquity` sizing puts
the engines 1.3e-4 and 2.1e-5 of capital apart against a 1e-9 tolerance, while
the round-trip counts stay identical: the trading agrees, the cost arithmetic
does not, because one charges the fee on top of the notional and the other
reserves it out of cash first. In fixed units both land exactly, to 1e-15.

So `sma_cross_costs` carries both costs on the same signal as the headline, and
the price of that is visible: x48.0 against x50.6 at 100k bars, x36.1 against
x36.7 at 1M. Costs change the result, not the ratio.

### The multi-asset workload is the one manifoldbt does worst on

It is here on purpose. A portfolio is what people actually run, and it is also
where the two designs differ in kind: manifoldbt walks a universe against one
shared book, while vectorbt broadcasts a column per asset and has to be told to
share cash at all.

Broadcasting wins.

| Workload | 100k bars | 1M bars |
|---|---:|---:|
| `sma_cross`, one asset | x50.6 | x36.7 |
| `multi_asset`, five assets | **x10.7** | **x8.8** |

The absolute timings say why: going from one asset to five costs manifoldbt 6.1x
(26.4 ms to 162.3 ms at 1M bars) and vectorbt 1.4x (989 ms to 1415 ms). Five
columns in one vectorised pass is close to free for it; five books are not free
for anything that walks them.

Publishing this lowers the average number on the page. It is the most useful
workload in the suite for anyone deciding whether to switch, which is the only
audience the benchmark has.

### Why the fee workload stops at 1M bars

A workload can stop being a comparison before it stops running. `ema_rsi_fees`
sizes in fixed units and pays 5 bps a side, and at 1-minute resolution it turns
over often enough that the fees compound into the account: measured, it ends at
-15% of capital on 1M bars, -74% on 5M, and exactly -100% on 10M, where fees
reach 99,611 of the 100,000 it started with.

Past that point the engines still agree on the equity, because both are sitting
at zero, and disagree by thousands of round-trips about how many more worthless
trades to book on a dead account. That is a fact about a bankrupt strategy, not
about either engine, so the workload carries a ceiling and the runner skips it
above that with the reason printed. The other workloads have no ceiling.

### What the windows are, and why they moved

`sma_cross` crosses on 30/150 rather than 10/50. The two were measured against
each other on the same 5M bars, and the slower pair is worth about 15% on the
ratio (x267 against x232 with a performance summary) because it books a third of
the trades and manifoldbt's cost, unlike vectorbt's, moves with the trade count.

That is a real effect and a small one, and it is worth knowing which way the
knobs turn before anyone quotes a number:

| Turn up | Effect on the ratio | Why |
|---|---|---|
| series length | **widens** | vectorbt materialises the simulation; its cost is linear in bars |
| asking for the summary | **widens sharply** | it has to build the equity curve it deferred |
| number of trades | narrows | near-free for vectorbt's per-bar loop, real for manifoldbt |
| number of indicators | narrows | same reason |

Measured at 5M bars on four levels of turnover, the ratio runs from x40 at
480,000 round-trips to x71 at 2,500. The floor matters more than the peak: even
in the busiest configuration tested, with half a million round-trips, the gap
holds at x40, and x151 with a performance summary.

Each of those runs across a range of series lengths. Two further axes, cold
start and memory, are measured in their own processes because they cannot be
measured honestly inside the main one.

**Scope, stated twice on purpose.** `sma_cross` and `sma_cross_metrics` run the
identical simulation; only the second one also produces a performance summary.
manifoldbt and raptorbt both compute that summary inside the run whether or not
you read it, while vectorbt defers the equity curve until a risk metric asks for
it and then pays to materialise it. Reporting both scopes is the only honest way
to present the result: a reader who only wants a total return should look at the
first number, and a reader who wants a Sharpe should look at the second. Parity
on the summary workload is gated on total return, round-trip count and max
drawdown, which match exactly across all three; the vectorbt ratios agree to
about 3e-4, because manifoldbt buckets its daily returns slightly differently,
and that residual is reported rather than smoothed over.

raptorbt annualises its ratios on its own basis and moves that basis between
releases (the same run reads Sharpe 0.21 on 0.4.1 and 3.43 on 0.9.0, against
manifoldbt's 8.14), so its Sharpe and Sortino are recorded but not compared:
subtracting them from the reference would publish a units mismatch as a
disagreement. It reports no volatility at all, and the harness leaves that cell
empty rather than recomputing it from the equity curve, which would credit
raptorbt with the harness's own arithmetic. None of this touches the gate, which
runs on money, round-trips and drawdown, all three basis-free. Its drawdown
matches the reference to 4e-15.

The vectorbt side of the summary is written out in pandas rather than through
`pf.sharpe_ratio()` for two reasons, both in vectorbt's favour or neutral.
manifoldbt computes its ratios on daily returns annualised by sqrt(365) and its
drawdown at full bar resolution, so the native accessors would return different
numbers and the comparison would be timing two different computations; and the
hand-written version is measurably faster than a single native accessor on the
same data, so vectorbt is credited with the quicker of its two paths.

**Cold start.** The steady-state table discards a warmup call, which is where
vectorbt compiles its numba kernels. A user pays that cost in every new
notebook, script or CI job, so it is measured rather than waved away: a fresh
process, an engine it has never imported, one backtest. The Python, numpy and
pandas baseline is measured the same way and reported alongside, so the engine's
own share can be read off instead of argued about.

**Memory added by the run.** Resident memory sampled while the backtest
executes, after a warmup. vectorbt materialises the simulation as arrays and its
footprint grows with the series; manifoldbt streams bars out of its store. What
is compared is what running a backtest costs *on top of* already holding the
data: building each engine's data representation is a different question and is
excluded on both sides, in manifoldbt's case a deliberately unflattering choice
since its one-off ingest is the memory-hungry part.

### Why raptorbt sits out the fee workload

Two blockers, either one sufficient, both measured rather than assumed.

*Sizing.* raptorbt has no fixed-quantity mode. `position_sizes` is a fraction of
equity (a constant 0.5 buys exactly half the equity of the bar before the entry),
`lot_size` rounds a computed size down to a multiple of itself, and
`alloted_capital` fixes the notional rather than the quantity. Reproducing
`units=5` would mean feeding a fraction derived from an equity curve that does
not exist until the run is over.

*Indicator.* `raptorbt.ema` seeds on a simple mean of the first `period` bars and
emits from bar `period-1`; manifoldbt seeds on the first observation and emits
from bar 0. Same recursion, different warmup, so the signal differs early and the
round-trip count with it. Its `sma` matches the reference to 3.3e-13 and its
`rsi` matches to the last bit, which is why the other three workloads run.

Running it anyway with a different size and a different indicator would produce
a number, and the number would not mean anything.

### Why the fee workload sizes in units

With `FractionOfEquity` sizing and a non-zero fee the engines size differently:
manifoldbt charges the fee on top of a full-equity notional, vectorbt reserves
it out of cash first. Both are legitimate product decisions, and comparing them
would compare policy rather than speed or correctness. Sizing in fixed units
isolates the fee arithmetic, which is the thing both engines must agree on.

### The documented divergence, in full

When a bracket fires intrabar and the entry condition still holds at that bar's
close, the three engines take three different roads:

- **manifoldbt** books two orders on that bar: the stop or target exit, then a
  fresh entry at the close.
- **vectorbt** processes one order per bar and re-enters on the next bar.
- **raptorbt** does not re-arm at all. The level still being true is not enough;
  it waits for the level to go false and true again.

None of them is wrong. On controlled bars the bracket fills themselves match
exactly, which the cross-engine parity suite shipped with the library pins test
by test; the divergence is purely about *when* a re-entry is allowed.

The harness counts the affected round-trips rather than hand-waving at them, and
it is one population, not three: at 10,000 bars manifoldbt books 214 round-trips
of which 82 re-enter on the exit bar, and raptorbt books exactly 214 - 82 = 132.
The report states the count and the share, so the size of the difference is
measured instead of asserted.

## Alignment choices, and why each one exists

Engines only produce identical numbers if they are told to do the same thing.
These are the conventions the harness sets, all of them visible in
[engine_mbt.py](engine_mbt.py), [engine_vbt.py](engine_vbt.py) and
[engine_rbt.py](engine_rbt.py):

- `signal_delay=0` with `execution_price="AtClose"`, matching what
  `Portfolio.from_signals` does by default: a signal fills at the close of the
  bar that produced it.
- `warmup_bars=0`, so the indicator's own NaN warmup is what suppresses early
  signals, identically on both sides.
- Signals are fed to vectorbt as a *level* (`entries` = the condition holds,
  `exits` = it no longer holds) rather than as transitions, which reproduces
  manifoldbt's target-position semantics.
- Indicators are mirrored on manifoldbt's exact definitions. The EMA seeds on
  the first observation with `alpha = 2/(span+1)`, which `ewm(span=n,
  adjust=False)` reproduces exactly. The RSI is Wilder's, seeded with the
  *simple* average of the first `period` deltas and emitted from bar `period`:
  a plain `ewm(alpha=1/period)` over the whole delta series is a different
  indicator, and using it would have made the engines disagree for a reason that
  has nothing to do with either engine.
- The generated bars have no opening gap (`open == previous close`). A bar that
  gaps through a stop is the one case where two engines can legitimately book
  different fill prices while both being correct, so that class of false
  failures is removed from the data rather than argued about in the report.

On the raptorbt side specifically:

- `upon_bar_close=True`, which fills at the close of the signal bar. Turning it
  off does not add a one-bar delay, it moves the fill to that same bar's *open*,
  so there is no setting on that side matching a delayed execution.
- Sizing is left at its default, which takes the whole equity of the bar before
  the entry. Since the account is flat at that point, that is the same number as
  the equity at the fill, and the three engines size identically: `sma_cross`
  comes back with manifoldbt's final equity to the last bit.
- The bracket is set on the config, in fractions rather than percent: 0.15 means
  a 15% stop, so the workload's 0.15% is `0.0015`. Passing the percent number is
  not an error, it is a stop so wide it never triggers.
- Indicators come from raptorbt's own Rust `sma` and `rsi`, not from numpy: that
  is what a user would write, and it is what deserves to be timed.

## Reading the numbers honestly

- On a shared runner with 4 vCPUs, an engine that parallelises is understated.
  These are floors, not peaks.
- The published wheels are CPU-only, and GitHub-hosted runners have no GPU, so
  nothing here says anything about GPU performance.
- vectorbt is the open-source package (`pip install vectorbt`), not vectorbtpro.
- raptorbt has no fan-out API for a parameter grid on one instrument, so its
  sweep column is a Python loop over `run_single_backtest`. That is not a
  handicap the harness imposed, it is the only spelling available, and the
  moving averages are hoisted out of the loop so it gets the same courtesy
  vectorbt gets on its own grid path.

## Files

- `data.py` - deterministic OHLCV generator and its content digest
- `probe_child.py` - the cold-start and memory probes, each in a fresh process
- `workloads.py` - the parameters every engine reads, and the per-engine notes
- `engines.py` - the registry: who is in the comparison, and who is the reference
- `engine_mbt.py` / `engine_vbt.py` / `engine_rbt.py` - one adapter per engine
- `parity.py` - the gate
- `bench.py` - the runner
- `sweep_child.py` - one parameter-grid point, in its own process
- `report.py` - JSON to Markdown, and to the GitHub job summary
- `ci_activate.py` - activates the licence the sweep job needs, and refuses to
  continue without one
- `ci/bench-vs-vectorbt.yml` - the workflow, deployed to the public repository

## Parameter sweeps

Sweeps run in their own CI job, because they need a licence: sweep timings are
published from a licensed run only, and the harness refuses to produce a number
without one.

Three points, sized from what the runner actually did rather than guessed.
Measured there: 87.5 us per combination for manifoldbt at 20,000 bars, 1.16 ms
for vectorbt, 1.34 ms for raptorbt, and 15.0 ms for raptorbt at 200,000 bars.

Only the first point is one vectorbt can hold. It materialises the simulation
per combination -- 1.57 MB of it at 20,000 bars -- so 5,000 combinations already
cost it 2.5 GB and 20,000 would need 31 GB. The other two are out of its scope,
and that is where a sweep stops being a speed comparison and becomes a
capability one: the question is no longer how long it takes but whether the
machine can hold it at all.

raptorbt sets the time budget rather than manifoldbt. With no fan-out API for a
parameter grid on one instrument, its sweep is a Python loop costing a full
backtest per cell, which is why the large point goes deep in combinations on a
short series instead of the reverse: 20,000 combinations on 20,000 bars costs it
27 seconds a call, where 5,000 combinations on a million bars would cost it 25
minutes.

Sweep ratios depend on the machine far more than single-backtest ratios do,
because manifoldbt is the only one of the three that fans out across cores. The
same point measured x13 on a four-vCPU cloud runner and x32 on two fast desktop
cores; the CI numbers are the conservative ones, and they are the ones published.
