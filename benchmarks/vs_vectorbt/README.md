# manifoldbt vs vectorbt vs raptorbt vs NautilusTrader

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
support. NautilusTrader asks for 3.12 or later, so it does not move the
interpreter. Add or drop a challenger with `--engines`; an engine that is not
installed is skipped with a printed line rather than crashing the run.

**manifoldbt is the reference.** Every parity check and every ratio is a
challenger against it, never two challengers against each other: four engines
make six pairs, and a table of pairs is a matrix, not a benchmark. The
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

A fifth, `not run above N bars`, is a time budget rather than a verdict, and it
applies to one engine only: an event-driven engine replays every bar through a
simulated exchange and costs seconds where the others cost milliseconds, so it
is not repeated on series it would need hours for. It is printed with its
reason, like `unsupported`, and the ceiling is a command-line option (see
[NautilusTrader is a different kind of engine](#nautilustrader-is-a-different-kind-of-engine)).

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
NautilusTrader is driven through its low-level `BacktestEngine`, a `Strategy`
subclass and its own indicators, which is what its documentation shows a user
writing.

## What is compared

| Workload | What it exercises | vectorbt | raptorbt | NautilusTrader |
|---|---|---|---|---|
| `sma_cross` | SMA 30/150 crossover, long-only, no cost | exact | exact | exact |
| `sma_cross_metrics` | the same simulation, plus max drawdown, Sharpe, Sortino and volatility | exact | exact | exact, on money and round-trips |
| `sma_cross_costs` | the same signal with a 5 bps fee and 2 bps of slippage | exact | unsupported | exact |
| `multi_asset` | five assets in one shared book, capped at 1M bars | exact | unsupported | exact |
| `ema_rsi_fees` | EMA 12/26 with an RSI(14) filter and a 5 bps taker fee, capped at 1M bars | exact | unsupported | exact |
| `bracket_sl_tp` | the same entry with a 15 bps stop and a 30 bps target | documented | documented | documented |

NautilusTrader is run up to 100,000 bars (see below), and on the summary
workload it is gated on total return and round-trips only, because it has no
drawdown statistic to compare.

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

## NautilusTrader is a different kind of engine

The other three engines are whole-series loops: vectorised, or compiled, or
both. NautilusTrader is event-driven. It replays the bars one at a time through
a simulated exchange with an order matching engine, and calls a Python strategy
on each of them, which is what lets the same strategy run unchanged against a
live venue. That design is the point of the engine, and it is also why it spends
seconds where the others spend milliseconds on a workload like this one: a
moving-average crossover is exactly the case a vectorised engine was built for.
The comparison below is what this workload costs on each design, and says
nothing about what each is for.

### What is timed

Everything that can be prepared once is prepared before timing, as for the other
engines: the instrument, the `Bar` objects (`BarDataWrangler`), the venue, the
account, the strategy, and the `add_data` call that hands the bars to the
engine. The timed closure is what a user repeats, `engine.reset()`,
`engine.run()` and reading the result. A replay is checked rather than assumed:
the first call is the one the parity gate reads, and every later call must
return exactly what it returned, or the run stops.

### Why it stops at 100,000 bars

Measured locally, pinned to four logical cores of a 13th-generation Core i5 (a
hosted runner is slower): one `sma_cross` call costs 3.4 s at 100,000 bars,
which is about 29,000 bars per second, and the cost is linear in the series. At
10,000,000 bars one call would take about six minutes, and the eight calls of
one workload (the gated call and seven repetitions) close to an hour, before the
other five workloads. The ceiling is therefore 100,000 bars, it is a time budget
and not a verdict, and each row beyond it says so instead of leaving a blank.
`bench.py --engine-max-bars nautilus_trader=1000000` lifts it for a local run.

The cost is not only the bars. Every fill is an order that goes through the risk
engine, the matching engine and the account, and costs on the order of a
millisecond (from the table: about 0.8 ms per extra fill between `sma_cross` and
`ema_rsi_fees`): `ema_rsi_fees`, which trades about 6,400 times per 100,000
bars, takes 7.6 s where `sma_cross` takes 3.4 s, and the five-asset
`multi_asset` takes 27 s because it replays five series.

### How each workload is expressed

- **Fill at the close of the signal bar.** The strategy submits a market order
  from `on_bar`. The simulated exchange has already moved the market to that
  bar's close, and the order is matched before the next bar arrives, so the fill
  lands on the same bar at the close. That is `signal_delay=0` with
  `execution_price="AtClose"`, and it needs no setting.
- **No volume-limited fills.** Nautilus builds its order book from a bar by
  splitting the bar's volume over four synthetic updates (open, high, low,
  close) and treats that quantity as the liquidity available. A market order
  larger than a quarter of the bar's volume is partially filled, and the
  remainder is booked one tick worse. The generated volumes are random between
  100 and 10,000 units, and full-equity sizing is about 1,000 units, so that
  rule fires on about four full-equity orders in ten. None of the other engines
  has a volume constraint, so the venue is given a fill model that answers
  market orders from a book of effectively unlimited depth at the close. It is
  the mechanism Nautilus ships for this (`BestPriceFillModel`), written out only
  because the bracket workload needs stop and limit orders left on the standard
  logic. `BestPriceFillModel` prices every order from the best bid and ask,
  stops and limits included, and with it `bracket_sl_tp` ends 6.2e-3 of capital
  away from the run that fills them at their own levels.
- **Slippage in basis points** is not a Nautilus setting: `prob_slippage` moves
  a fill by one *tick*, at random. The same fill model shifts the book instead,
  asks 2 bps above the close and bids 2 bps below, which is the documented
  extension point for custom slippage and gives what `Slippage.fixed_bps` gives.
- **Fees** are the instrument's taker fee, charged on notional by the venue's
  default fee model. A market order always takes.
- **Account and sizing.** A margin account with leverage 1, in a quote currency
  with eight decimals (USDT). Not USD, which has two: Nautilus rounds each
  realised PnL to the currency's precision, and measured with a USD quote,
  `sma_cross` ends 2.8e-7 of capital from the reference at 20,000 bars and
  2.1e-7 at 100,000, over two hundred times the gate. `FractionOfEquity` is the
  cash balance at the signal bar divided by the close, rounded *down* to the
  quantity precision; units sizing is the fixed quantity.
- **Indicators** are Nautilus's own `SimpleMovingAverage`,
  `ExponentialMovingAverage` and `RelativeStrengthIndex` (Wilder), registered on
  the bar type so the engine updates them before `on_bar` runs.
- **Brackets** are Nautilus's native contingent orders (`OrderFactory.bracket`):
  a stop-market and a limit, one-cancels-other, armed when the market entry
  fills, at levels computed from the signal bar's close. The venue's standard
  logic decides the fills: a stop the bar trades through fills at its trigger
  price, a limit the bar reaches fills at its limit price.

### Fixed-point rounding

Prices and quantities in Nautilus are fixed-point integers. The Windows wheel
carries 9 decimals and the Linux wheel 16, and the adapter uses 9 on both so
that the platform changes almost nothing: the parity results on the two agree to
at most 2e-16 of capital. Rounding the generated float64 prices to 9 decimals
moves the final equity by 3e-13 to 6e-11 of capital across the five workloads
that agree, at 100,000 bars, and up to 1.5e-10 on a 20,000-bar grid whose short windows trade
every few bars. The error grows with the square root of the number of fills. The
gate is 1e-9, and the number is printed in every run, not hidden.

### The RSI seed

manifoldbt seeds Wilder's RSI with the simple average of the first `period`
deltas and emits from bar `period`. Nautilus seeds it on the first observation
and reports it as initialised after `period` updates. Same recursion, different
seed, so the two series differ at the start and converge geometrically: on the
benchmark data the largest gap is 4.5e-2 (on a 0-1 scale) in the first 50 bars,
1.2e-4 between bars 100 and 200, and 2e-16 from bar 800 on. Whether that moves a
decision is what the gate answers, and on this data it does not: the entry
signals are identical, `ema_rsi_fees` comes back with the reference's
round-trips and fees. That is a property of this dataset, which is why the
difference is written down here rather than left for the gate to hide. The
adapter uses the engine's own indicator and its own `initialized` flag; the EMA
is identical to the reference to the last bit and the SMA to about 1e-13.

### The documented divergence on `bracket_sl_tp`

Entries, exits and fill prices are the same trades as the reference, round-trip
for round-trip (448 against 448 at 20,000 bars, 2,309 against 2,309 at 100,000).
Because the simulated exchange runs before the strategy sees the bar, a position
stopped out intrabar is entered again at that bar's close when the signal still
holds, exactly as the reference does, and not on the next bar as in vectorbt or
never as in raptorbt.

What differs is the quantity of that re-entry. The reference sizes it from the
equity at the close, counting the position it has just closed at the close
price, so it buys back the quantity it held. Nautilus sizes it from the cash
balance after the stop or target has been booked at its own price. Neither is
wrong, the two quantities differ by the gap between the exit price and the close
(a median of 2.7e-04 in relative terms, at most 1.1e-03), and the difference is
carried into every later trade: final equity ends 7.1e-06 of capital apart at
100,000 bars. The harness counts the affected re-entries and reports their
quantity gaps instead of asserting any of this.

Nautilus also books a stop in two pieces when a quarter of the bar's volume is
smaller than the position (a partial fill, then the remainder one tick worse, by
the engine's documented rule), so its fill count is higher than its round-trip
count implies. Fills are not gated, and the extra tick on the remainder is 1e-9.

### No drawdown, and ratios on their own basis

Nautilus's portfolio analyser reports volatility, Sharpe and Sortino and no
maximum drawdown. The harness does not rebuild one from an equity curve for it,
as it does not rebuild raptorbt's missing volatility: that column shows what the
engine hands a user. `sma_cross_metrics` is therefore gated on total return and
round-trips for Nautilus, which is said in the report. Its ratios are on their
own basis (labelled 252 days, built from the returns of closed positions), not
over 365 days from a daily equity curve, so they are recorded and not compared.

### Cold start and memory

Importing NautilusTrader and building an engine costs 1.2 s over the bare
interpreter (against 0.15 s for manifoldbt in the same run). Memory is measured
the same way as for the others, as what the run adds on top of data the process
already holds, and for Nautilus that is a few megabytes at any length: 0.4 MB at
100,000 bars and 2.6 MB at 2,000,000 (a one-off measurement, not part of the
run, since 2,000,000 bars take minutes). The probe measures it at its ceiling,
and the report prints the length beside the figure. What the exclusion hides is
the cost of holding the data: the `Bar` objects an engine keeps in memory take
230 to 280 MB per million bars (measured on the loaded engine), against 48 MB
for the same bars as a float64 frame. Building each engine's data representation
is a different question and is excluded on every side, but a reader sizing a job
for a long series should know the number.

## Alignment choices, and why each one exists

Engines only produce identical numbers if they are told to do the same thing.
These are the conventions the harness sets, all of them visible in
[engine_mbt.py](engine_mbt.py), [engine_vbt.py](engine_vbt.py),
[engine_rbt.py](engine_rbt.py) and [engine_nt.py](engine_nt.py):

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
- NautilusTrader has no fan-out API either, and its loop is the expensive kind:
  each cell is a full event-driven replay. It is looped the way the engine
  documents repeat runs (one engine, data loaded once, `reset()` and `run()` per
  cell). The moving averages cannot be hoisted out of the loop for it, because a
  Nautilus indicator lives inside the event loop; each cell computes its own,
  which is what a Nautilus strategy does.
- A simple signal strategy on bars is the cheapest thing an event-driven engine
  can be asked to do, and it is also the case a vectorised engine was designed
  around. What NautilusTrader buys, and these workloads do not use, is the
  matching engine, order types and contingency, order book and tick data, and the
  same strategy code in live trading. This benchmark measures bar-replay
  throughput of one simple strategy on each design; it does not measure what
  either is for.

## Files

- `data.py` - deterministic OHLCV generator and its content digest
- `probe_child.py` - the cold-start and memory probes, each in a fresh process
- `workloads.py` - the parameters every engine reads, and the per-engine notes
- `engines.py` - the registry: who is in the comparison, and who is the reference
- `engine_mbt.py` / `engine_vbt.py` / `engine_rbt.py` / `engine_nt.py` - one
  adapter per engine
- `parity.py` - the gate
- `bench.py` - the runner
- `sweep_child.py` - one parameter-grid point, in its own process
- `report.py` - JSON to Markdown, and to the GitHub job summary
- `ci_activate.py` - activates the licence the sweep job needs, and refuses to
  continue without one
- `test_report.py` / `test_harness.py` - guards for the report, the gate, the
  ceilings and the sweep plumbing; they need no engine installed
- `.github/workflows/bench-vs-vectorbt.yml` - the workflow

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
because manifoldbt is the only one of the four that fans out across cores. The
same point measured x13 on a four-vCPU cloud runner and x32 on two fast desktop
cores; the CI numbers are the conservative ones, and they are the ones published.

### NautilusTrader and the grid

A grid on NautilusTrader is a loop of full replays, so its cost is the number of
combinations times one backtest, and a backtest is seconds: 0.75 s on average
across the 20,000-bar grid above, 7.8 s at 200,000 bars, and more where the
windows are short enough to trade every few bars. Locally, the three points
above would cost it 63 minutes for 5,000 combinations at 20,000 bars, 4.7 hours
for 20,000, and 21.6 hours for 10,000 combinations at 200,000 bars, against
under three seconds for manifoldbt in each case. That is not a comparison a job
can repeat, so it is out of scope on all three, and the reason printed beside
each point carries the per-backtest time measured on 16 cells spread across that
point's own grid, and what that makes of the whole grid. Those 16 cells are also
run on manifoldbt and compared cell by cell, so the point still carries a
cross-engine check of the same code path.

The engine is held back by a sweep ceiling of 64 combinations, applied on its
own so that a point never needs to say so, and `--engine-max-combos
nautilus_trader=N` lifts it. Below the ceiling it runs, and the grid is checked
cell by cell like the others. A grid it can hold, run locally on the pinned
workstation:

```bash
python bench.py --workloads sma_cross --bars 100000 --reps 1 --cold-start-reps 0 \
    --sweep 20000:36 --sweep-reps 2
```

On 36 combinations at 20,000 bars that loop takes 41 s, against 11 ms for
manifoldbt's single fan-out call, 79 ms for vectorbt and 45 ms for raptorbt, and
the 36 cells agree with the reference to 1.2e-10 of capital. It adds 6 MB of
memory.
