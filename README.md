<p align="center">
  <img src="https://raw.githubusercontent.com/manifoldbt/manifoldbt/master/assets/logo.png" width="110" alt="ManifoldBT logo">
</p>

<p align="center">
  <strong>ManifoldBT</strong><br>
  Rust-powered backtesting engine for quantitative research
</p>

<p align="center">
  <a href="https://discord.gg/bvU6Wjc72d"><img src="https://img.shields.io/badge/Discord-join%20the%20community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://www.manifoldbt.com">Website</a> &middot;
  <a href="https://www.manifoldbt.com/docs/documentation.html">Documentation</a> &middot;
  <a href="https://github.com/manifoldbt/manifoldbt/tree/master/examples">Examples</a>
</p>

---

ManifoldBT compiles Python strategy definitions into an optimized Rust expression graph.
Write strategies in a fluent Python DSL — execute them on a vectorized Rust engine.

## Why ManifoldBT

- **Fast** — 500K bars in ~26ms. 161x faster than vectorbt, 1000x+ faster than backtrader.
- **Expressive** — fluent DSL with 30+ indicators, conditional logic, cross-asset references
- **Rigorous** — Monte Carlo, walk-forward, parameter sweeps, lookahead detection, exposure diagnostics
- **Portable** — `pip install`, no Rust toolchain needed. Works on Python 3.9+.

## Installation

```bash
pip install manifoldbt
```

With all extras (plotting, pandas, polars):

```bash
pip install manifoldbt[all]
```

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

Bring your own data, or pull it from a built-in connector — both return a
`DataStore` ready for `mbt.run(...)`.

**CSV** — free on all tiers, auto-detects standard / MetaTrader 4 / MetaTrader 5:

```python
store = mbt.import_csv("EURUSD_1m.csv", symbol="EURUSD", symbol_id=1,
                       interval="1m", asset_class="forex")
```

**Exchange connectors** — Binance, Bybit, Hyperliquid, dYdX, Bitstamp (free); Databento, Massive (Pro):

```python
store = mbt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1,
                   start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z")
```

Or from the CLI:

```bash
manifoldbt import-csv data.csv --symbol EURUSD --symbol-id 1 --interval 1m
manifoldbt ingest --provider binance --symbol BTCUSDT --symbol-id 1 --start ... --end ...
```

## Examples

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

## Performance

EMA(12/26) + RSI(14) on 500K synthetic 1-min bars (median of 5 runs):

| Engine | Time | vs ManifoldBT |
|--------|------|---------------|
| **ManifoldBT** (Rust) | **26 ms** | 1x |
| vectorbt (NumPy) | 4,094 ms | 161x slower |
| backtrader (Python) | — | ~1000x slower |

Reproduce: `python benchmarks/bench_vs_competitors.py --rows 500000 --runs 5`

## Documentation

Full API reference, indicator list, configuration guide, and best practices:

**[www.manifoldbt.com/docs/documentation.html](https://www.manifoldbt.com/docs/documentation.html)**

## Community vs Pro

| | Community | Pro |
|---|---|---|
| Output resolution | Daily | 1m, 5m, 15m, 1h |
| Monte Carlo | 1K sims | Unlimited |
| Walk-Forward | - | Anchored + Rolling |
| Parameter Stability | - | Yes |
| Crypto connectors (Binance, Bybit, Hyperliquid) | Yes | Yes |
| Databento & Massive connectors | - | Yes |
| Safety checks (lookahead, exposure) | - | Yes |
| Tearsheets & export | - | Yes |

## License

Apache 2.0 with Commons Clause. The source is available, free to use,
modify and self-host. Reselling the software or offering it as a paid
hosted service is not permitted. See [LICENSE](https://github.com/manifoldbt/manifoldbt/blob/master/LICENSE) for the full text.
