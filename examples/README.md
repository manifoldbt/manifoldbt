# Examples

Each file demonstrates **one mechanism** of the engine and stops there. They are
documentation that happens to execute, not strategies: several would lose money
traded as written, and that is not a defect.

## The contract

Every example opens with the same four blocks, in this order:

```
"""One-line title -- the mechanism, named.

Demonstrates:
  - the two or three API surfaces the file exists to show

Data: <provenance>

Usage:
    python examples/NN_name.py
"""
```

`Data:` is the block that matters most, and it is not optional. An example is
read by someone deciding whether to trust a number, so where the number came
from is part of the example.

## The three data provenances

**`Data: shared store`** -- real market data from `data/`, ingested once and
reused. These examples need that store to exist (see below). Their results are
real in the sense that the prices are real; they are still not investment
advice, and none of the strategies is tuned.

**`Data: self-contained (network)`** -- the file ingests what it needs on each
run, from a free connector. Runs on a fresh clone, needs a network.

**`Data: synthetic (seed N)`** -- the file generates its own series. Always
reproducible, never a market claim.

## The rule on synthetic data

Synthetic data is legitimate and often the right choice: it isolates a
mechanism from market noise, and it lets an example run anywhere with no
dependency. It is used deliberately here, not as a fallback.

The one thing it must never do is let a reader mistake a fixture for a result.
So when the construction of the series **determines the outcome**, the file says
so before printing any figure, and marks the affected numbers. Two examples are
in that case:

| | What the fixture determines |
|---|---|
| `21_fill_at_computed_level.py` | the series is detrended, so mean reversion cannot lose on it -- only the *gap* between two execution prices is meaningful |
| `25_lookahead_trap.py` | the leak is deliberate; the file exists to show which audits fail to see it |

A synthetic example whose fixture does **not** predetermine the outcome carries
no such warning, because there is nothing to warn about.

## The rule on performance

No example is selected, tuned, or presented for its returns. Where a figure
appears it illustrates the mechanism under discussion. If an example ever
produced a reproducible edge on real data, it would belong in a private
repository rather than in documentation.

## Running them

```bash
pip install manifoldbt          # the engine
pip install manifoldbt[plot]    # and the charts, if you want them drawn
python examples/13_stochastic_simulation.py
```

Charts are an optional extra. Without it an example still runs to the end and
skips its chart, saying so -- no file here needs `[plot]` to be useful, except
`06_full_visualization.py`, which is nothing but charts.

## The shared store

Examples marked `Data: shared store` read `data/` and `metadata/`, which are
not in the repository: market data does not belong in a git repository.
Download what they need, once, from the repository root:

```bash
python examples/setup_data.py
```

That ingests the ten Binance perpetuals and the one dYdX perpetual these files
name, at 1h, over the range they cover. The engine aggregates upwards, so the
same 1h store serves the examples running on 12h or daily bars. Both connectors
are free on all tiers and need no API key.

Doing it by hand takes one call per symbol, and the `asset_class` is the part
that is easy to get wrong -- the examples ask for `BTC-USDT:perp`, and the
default (`crypto_spot`) stores a symbol they will not find:

```python
import manifoldbt as mbt
mbt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1, interval="1h",
           asset_class="crypto_perp",
           start="2021-01-01T00:00:00Z", end="2026-03-01T00:00:00Z")
```

The self-contained and synthetic examples need none of this.

## Index

| | Example | Mechanism | Data |
|---|---|---|---|
| 00 | `00_template.py` | the minimal shape of a backtest | shared store |
| 01 | `01_trend_following.py` | fluent builder, EMA, stop-loss, diagnostics | shared store |
| 02 | `02_mean_reversion.py` | bands, z-score, conditional sizing | shared store |
| 03 | `03_multi_asset_momentum.py` | cross-sectional ranking over a universe | shared store |
| 04 | `04_linear_regression.py` | rolling regression as a signal | shared store |
| 05 | `05_stat_arb.py` | pair spread, cross-asset references | shared store |
| 06 | `06_full_visualization.py` | the whole plotting surface | shared store |
| 07 | `07_walk_forward.py` | fold geometries, out-of-sample selection | shared store |
| 08 | `08_sweep_2d_heatmap.py` | a two-parameter sweep, read as a surface | shared store |
| 09 | `09_surface_3d.py` | the same surface in three dimensions | shared store |
| 10 | `10_monte_carlo.py` | bootstrap resampling, rare-event metrics | shared store |
| 11 | `11_portfolio.py` | several strategies in one book, rebalancing | shared store |
| 12 | `12_diagnostics.py` | look-ahead, exposure, risk checks | shared store |
| 13 | `13_stochastic_simulation.py` | SDE paths from the expression DSL | synthetic |
| 14 | `14_multi_timeframe.py` | higher timeframes, and how periods count | shared store |
| 15 | `15_cross_exchange.py` | signal on one venue, execution on another | shared store |
| 16 | `16_hashrate_exogene.py` | an exogenous series joined onto the bars | shared store |
| 17 | `17_per_venue_fees.py` | per-venue fee schedules in one book | shared store |
| 18 | `18_csv_import.py` | CSV / MT4 / MT5 import | synthetic |
| 19 | `19_custom_indicators.py` | composing an indicator from primitives | shared store |
| 20 | `20_entry_orders.py` | limit, stop and market-if-touched entries | shared store |
| 21 | `21_fill_at_computed_level.py` | filling at a level computed in advance | synthetic ⚠ |
| 22 | `22_yahoo_equities.py` | stocks, ETFs, indices, FX via Yahoo | network |
| 23 | `23_deribit_options.py` | option contracts that expire and settle | network |
| 24 | `24_option_spread.py` | a two-leg option structure | network |
| 25 | `25_lookahead_trap.py` | the look-ahead no re-run can detect | synthetic ⚠ |
| 26 | `26_fill_costs.py` | the same signal filled four ways, and what each costs | shared store |
| 27 | `27_bars_vs_tape.py` | the exit a candle has to guess, put back to the trades | synthetic, **not runnable yet** |

⚠ marks the two files whose fixture determines the outcome, as described above.

**27 does not run today, on any licence.** The tick layer it uses belongs to
the Pro+ tier, and that tier is not on sale: the code ships ahead of the plan
that will carry it. The file is there to be read, and it will run unchanged
when the tier opens. It is also the only example that wants pandas, for the
join between its trade log and its bars.

Two files here are not examples and carry no number:

| File | What it is |
|---|---|
| `setup_data.py` | downloads the shared store, once |
| `_bootstrap.py` | opens that store, and reports a missing plotting extra |
