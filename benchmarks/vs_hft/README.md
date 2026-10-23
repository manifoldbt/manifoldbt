# manifoldbt vs hftbacktest vs NautilusTrader: market making and HFT

An engine-to-engine benchmark you can re-run yourself. It installs every engine
from PyPI at a pinned version, downloads one pinned day of public market data
(checked against a SHA-256), runs the same trading rules in the three engines, and
reports, beside how long each took, how far they did **the same work**.

```bash
pip install manifoldbt
pip install -r requirements-lock.txt
python prep_cli.py 2025-08-13                      # download + fingerprint + prepare one day
python bench.py --day 2025-08-13 --reps 3 --out results-books.json
python sweep.py --day 2025-08-13 --window 36000 43200 --grids 4 16 64 --workers 4 --out results-sweeps-maker.json
python sweep_mbt.py --day 2025-08-13 --window 36000 43200 --grids 4 16 64 --workers 4 --identity 8 --out results-sweeps-mbt-maker.json
python report.py results-books.json
```

No account and no API key is needed for the data. manifoldbt itself needs a licence
key for this benchmark, as it does for the sweeps of the other one: the workflow
reads it from the `MANIFOLDBT_CI_LICENSE` secret and `ci_activate.py` runs before
anything else. The same commands run in GitHub Actions on public runners
(`.github/workflows/bench-vs-hft.yml`), so every published number has a run URL
behind it.

## What is compared

| Workload | Rule | Data the engines read |
|---|---|---|
| `mm_ladder_l2` | two post-only quotes every 100 ms, one level further back per clip of inventory held, inventory capped at 5 clips | 25 levels a side + trades |
| `mm_touch_bbo` | the same maker on a best bid/ask feed: it can only join the touch | best bid/ask + trades |
| `flow_taker_bbo` | follow the aggressor imbalance of the last 34 s (arm at +/-0.9), market orders, flip on the opposite crossing | best bid/ask + trades, trade clock |
| `mbo_join_mbo` | join the best bid and ask with a round lot, stand a side aside behind a crowded best or at the inventory bound | Nasdaq TotalView-ITCH, order by order (AAPL) |
| `probe_l2_zero` | open-loop probe: one post-only quote a side at the touch, cancelled and posted again at every wake-up, no inventory | 25 levels + trades, zero latency |
| `probe_l2_nominal` | the same probe with the venue latencies of the next section | 25 levels + trades |

The first four are strategies. The probes are not strategies but instruments:
their quotes depend on the market only, so each of them gets a verdict (served or
not, how much, how late) that does not depend on what the engine did before. That
is what isolates the queue and fill model of each engine.

All rules are written once, in [`spec.py`](spec.py), and each engine implements
them in its own idiom:

* **manifoldbt**: `Strategy.quote(...)` expressions (the market makers) and a
  trade-clock `Strategy` (the taker), run by `mbt.run`; sweeps by `mbt.run_sweep`
  and `mbt.run_sweep_lite`.
* **hftbacktest 2.4.4**: one `@njit` function per rule on
  `ROIVectorMarketDepthBacktest` (level 2) or `HashMapMarketDepthBacktest` (level 3),
  risk-adverse queue model, constant order latency.
* **NautilusTrader**: a `Strategy` subclass per rule on a `BacktestEngine`:
  `OrderBookDelta` (L2_MBP / L3_MBO) or `QuoteTick` (L1_MBP) plus `TradeTick`,
  `queue_position=True`, `trade_execution=True`, a `LatencyModel`.

## The rule this harness is built around

A speed comparison between backtesters is worthless unless they did the same work,
and with market making, counting fills is not enough to know whether they did.
An engine that serves an order in several smaller pieces shows more fills for the
same executed quantity; an engine that never serves a print that went *through* a
quote shows fewer for the same market. So parity is read in three layers:

1. **Actions**: which quotes were sent, at which wake-up, on which side, at which price.
2. **Verdicts**: for a quote both engines sent, was it served, for how much, how late.
3. **Totals**: fills, executed quantity, final position, P&L marked at the last mid.

A closed-loop strategy feeds its fills back into its next quote, so two engines
that disagree on one fill part ways from there on. The report therefore gives, for
each closed-loop workload, the wake-up where the actions first differ, and the
open-loop probe gives the verdict-level agreement without that feedback.

Parity is reported, not enforced. The vectorbt benchmark withholds a timing when two
engines disagree; here two engines are expected to differ, and the differences are
conventions of their models, named in "Why the engines differ". So the report prints
the counts of the three layers beside every timing and withholds nothing. What the
run does enforce is that every engine finished every workload: `--fail-on-blocked`
exits non-zero otherwise, and nothing partial is published.

## Data

One pinned day per source. The fingerprint of every archive is in
[`spec.py`](spec.py); a day that changed upstream stops the run instead of being
replaced silently.

* **Order books by price**: Bybit spot BTCUSDT, public archives
  (`quote-saver.bycsi.com/orderbook/spot/...ob200.data.zip` and
  `public.bybit.com/spot/...csv.gz`). The book archive is replayed once into
  states (updates that share a millisecond are merged, a snapshot replaces the
  book), and two tiers are cut from the same states: the best bid/ask and the top
  25 levels. Each tier is the *difference* between successive states, which is the
  one form every engine can replay without ambiguity. The trade tape is shared.
* **Order by order**: Nasdaq TotalView-ITCH 5.0, the day file Nasdaq publishes
  (`emi.nasdaq.com/ITCH/Nasdaq ITCH/S112825-v50.txt.gz`). It is read once as it
  inflates and only the messages of one symbol are kept (`itch.py`); manifoldbt
  ingests that extract, and `itch.py` also decodes it into the order events that
  hftbacktest and NautilusTrader are given.

## Venue model

The engines are set to the same venue wherever they can express it:

| | manifoldbt | hftbacktest | NautilusTrader |
|---|---|---|---|
| order / cancel / response / feed delay | `execution.latency` (5 / 5 / 5 / 2 ms) | `constant_order_latency(5 ms, 5 ms)`, feed 2 ms in the data | `LatencyModel` insert/cancel 5 ms + the feed delay; **no response delay**, emulated in the strategy |
| queue | `depth_source="book"`, risk-adverse | `risk_adverse_queue_model()` (level 2), `l3_fifo_queue_model()` (level 3) | `queue_position=True` |
| partial fills | yes | `no_partial_fill_exchange()` (the benchmark's choice: an order is served whole at its first piece) | yes |
| fees | maker 0, taker 4 bp | same | same |
| clock of the wake-ups | the first book state of the run + k x 100 ms | the same instants | the same instants |

Two rules come from the engines' own semantics and are applied to all three:

* the wake-up clock starts at the first *real* book state inside the run (the state
  in force when a sub-day window opens is handed to every engine, but it is not an
  event, so it does not start the clock);
* the flow taker's trailing window of 34 s gives no signal until a whole window of tape has gone by.

## Why the engines differ

The three engines read the same stream under the same venue settings, and they do
not serve exactly the same quotes. What separates them is the convention each model
uses for a resting order, not the data. Each convention is named below with what it
comes to on the pinned day. The report prints the same counts beside every timing,
so an engine that serves fewer quotes reads as doing less, not as doing the same
work faster. Figures are from manifoldbt 0.28.0, hftbacktest 2.4.4 and
NautilusTrader 1.231.0 unless a line says otherwise.

* **Tape or book (level 2).** manifoldbt reads a trade that prints through a resting
  quote off the trade tape and serves the quote there; hftbacktest waits for the book
  to show the level crossed. When the tape shows the crossing before the next book
  snapshot, only manifoldbt has served the quote. On the one-hour probe (71,994
  quotes, zero latency) manifoldbt serves 1,606 quotes and hftbacktest 1,590. The 16
  that only manifoldbt serves are all prints through the level, and none goes the
  other way.
* **Partial fills (closed loop).** manifoldbt serves an order in pieces, as the
  prints that reach it arrive. The benchmark runs hftbacktest with
  `no_partial_fill_exchange()`, which serves the order whole at its first piece. A
  strategy that reacts to its fills reacts to the first piece in one engine and to
  the whole order in the other, so the runs part at the first order where that
  matters: the quotes sent are identical until wake-up 310 for the level-2 maker and
  488 for the best bid/ask maker (13 August 2025). After that they are two different
  runs, which is why parity is read on actions and verdicts and not on fill counts.
* **Price priority (order by order).** When an order resting behind a quote, at a
  worse price on the same side, is executed, the quote has been reached. manifoldbt
  serves it then; hftbacktest serves a quote through executions at the quote's own
  price. Of the 6,318 quotes that at least one engine served and all three sent at one
  price, 166 are served by manifoldbt and not by hftbacktest, and the exact replay
  (`fifo_truth.py`, below) serves all 166.
* **Reading a `C` message.** ITCH reports an execution at a price other than the one
  an order is displayed at with a `C` message. hftbacktest reads it as an execution
  in the queue of the displayed level, and so did manifoldbt before 0.28.1; the exact
  replay reads it as an execution at its own price. A quote waiting at the displayed
  level is then served although the replay does not serve it: 90 quotes on the pinned
  day, the same 90 for both engines (of 4,866 served by manifoldbt 0.28.0 and 4,698
  by hftbacktest). manifoldbt 0.28.1 reads a `C` as the replay does: 0 of 4,769
  served, and each of the 5,170 quotes it serves over the whole run is served by an
  execution in the raw ITCH messages. hftbacktest keeps its reading: 81 of 4,684 in
  the same run.
* **NautilusTrader's queue.** With `queue_position=True` a resting order is served
  only by prints at its own price: a print through the level and a book that crosses
  the order do not serve it, and the queue ahead of the order is not capped by the
  size the book displays, which the other two engines bring down when that size
  shrinks (`micro_scenarios.py` reproduces each case on a three-line book, for all
  three engines). On the probe it serves 1,311 quotes: 295 fewer than manifoldbt and none
  that manifoldbt does not. Order by order, it takes an executed size off the queue
  ahead when the order is reduced and again when the trade arrives. The benchmark
  gives it both, as NautilusTrader's own Databento adapter does, so the same size
  counts twice (`micro_scenarios_l3.py` shows it on a five-order book). Against the
  exact replay, 1,468 of the 6,245 quotes it serves in the 0.28.0 run (1,454 of 6,226
  in the 0.28.1 run) are quotes the replay has not served.
* **Market orders.** On the pinned day the taker's decisions and orders are identical in
  the three engines. The fill price is the price of the first print at or after the order's
  arrival in manifoldbt's trade-clock engine, and the book at the arrival in the
  other two. The report gives the gap of the executed notional (-0.034 bp for
  hftbacktest and +0.006 bp for NautilusTrader on 13 August 2025). The execution time
  stamp in manifoldbt's trade log does not always equal the time of that print, so
  the report compares decisions, orders, quantities and prices, not those time
  stamps.

**The exact replay.** On an order-by-order market the right answer can be computed.
`fifo_truth.py` replays the day in explicit FIFO queues and judges every quote the
three engines sent at one price and at least one of them served: how many each engine
served although the replay had not, and how many it did not serve although the replay
had. Every engine is compared with the replay, not with another engine. The `orders`
job runs it on the journals of the run and publishes the table next to the timings. By
hand, on the journals that `bench.py` leaves in its `--tmp` directory:
`python fifo_truth.py AAPL.itch manifoldbt=<journal>.npz hftbacktest=<journal>.npz nautilus=<journal>.npz`.

**The order-by-order figures depend on the manifoldbt version installed.** The quotes
judged are the ones manifoldbt sent and served, so the same day gives different figures
with 0.28.0 and with 0.28.1 (both are quoted above). The benchmark records the version it
ran (`environment.versions.manifoldbt` in every result file), so a published figure
always names the version that produced it.

## Method

**Interleaved repetitions, one process per run.** Engines alternate inside each
repetition (A, B, C, A, B, C...). Each run is a fresh process, which is what makes
the cold start and the memory honest.

**Replay and strategy are timed apart.** `replay_only` runs the same data through
the same venue with no strategy; `strategy_run` runs the rule. Their difference is
what the strategy costs. Preparing each engine's input (hftbacktest's event array,
a manifoldbt store opened, NautilusTrader's data objects built) is reported
separately as `prepare` and `convert`.

**Cold start.** The first run in a process is kept and published: numba compiles
the hftbacktest kernels there. The steady state is the median of the runs after it.

**Dispersion is published, and noise is flagged.** Every point carries min, median,
max and interquartile range; above 15 % of the median the point is marked `~`.

**Memory added by the engine.** The process's peak resident memory minus what it
held after loading the shared neutral data.

## Parameter sweeps

manifoldbt sweeps a quoting strategy through `run_sweep` (every combination keeps
its order journal) and `run_sweep_lite` (metrics only): the book and tape are loaded
once and the combinations run in parallel inside the engine. The order delay is a
sweep axis of its own (`execution_grid`). The flow taker, on the trade clock, sweeps
through `run_sweep`. Each call runs in its own process so that its peak memory is
its own, and a sample of combinations is run alone with `mbt.run` and compared
bit for bit with the same combination inside the sweep (`sweep_mbt.py`).

hftbacktest and NautilusTrader have no sweep call; what is timed is the loop of
runs, and a pool of four processes (one per vCPU of the runner), each holding its
own copy of the data, with the pool created and warmed beforehand and reported
apart (`sweep.py`). The pool must return exactly what the loop returns, and the
result says whether it does.

The grids are nested (4, 16, 64, 256 combinations of order size, inventory bound
and the extra levels back of each side), plus a latency family (order delay x
inventory bound x levels back) and a taker family (flow threshold x order size).
NautilusTrader runs a bounded number of combinations per grid and the rest is
projected from them; the result says which. The first combinations of each engine are
recorded in the result (`sample`), so that they can be compared across engines, with the
same caveats as the single runs.

## Reading the numbers honestly

* A shared runner has 4 vCPUs. An engine that parallelises is understated; these
  are floors, not peaks.
* The runners have no GPU: nothing here says anything about GPU performance.
* The three engines model queues differently and the differences are listed
  in "Why the engines differ". A faster engine that serves fewer orders is doing less, not the same work
  faster; that is why parity is published beside every timing.
* One day per source is a sample. The statistics of a market maker depend on the
  day; the ratio between engines depends less on it than the absolute figures do.

## Files

* `spec.py` - every number every engine reads
* `data.py` - download, fingerprint, replay of the book archive into neutral streams
* `itch.py`, `itch_cli.py` - the ITCH day file, the extract, the order events
* `engine_mbt.py`, `engine_mbt_more.py`, `engine_hft.py`, `engine_nt.py`, `engine_nt_l3.py` - one adapter per engine
* `workloads.py`, `harness.py`, `child.py` - what runs, and one run in one process
* `parity.py` - the gate
* `bench.py`, `sweep.py`, `sweep_mbt.py`, `prep_cli.py` - the runners; `report.py`, `publish.py` - the output
* `micro_scenarios.py`, `micro_scenarios_l3.py` - the queue behaviours of the engines on tiny books
* `fifo_truth.py` - an exact FIFO replay of the order-by-order day, to judge the quotes the engines disagree on
* `ci_activate.py` - activates the benchmark licence and checks it
