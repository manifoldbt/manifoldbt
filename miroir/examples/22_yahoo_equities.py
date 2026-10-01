"""Yahoo Finance -- stocks, ETFs, indices, FX and futures, free on all tiers.

Demonstrates:
  - mbt.ingest(provider="yahoo") -- no API key, no license required
  - Backtesting daily equity bars, exactly like a crypto connector
  - Dividend-adjusted prices (same convention as yfinance's auto_adjust=True)

Yahoo imposes its own history limits: 1m bars go back 30 days, 1h about two
years, daily bars back to the listing date. Tickers follow Yahoo's own
notation: AAPL, SPY, ^GSPC (index), EURUSD=X (FX), ES=F (future),
BTC-USD (crypto), AIR.PA (Euronext).

Pass `dataset="raw"` to keep unadjusted quotes.

Data: self-contained (network) — ingested on each run from a free connector

Usage:
    python examples/22_yahoo_equities.py
"""
import os
import tempfile

import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Interval

# -- 1. Pull daily bars from Yahoo (free, all tiers) --------------------------
tmp = tempfile.mkdtemp()
store = mbt.ingest(
    provider="yahoo",
    symbol="AAPL",
    symbol_id=1,
    start="2020-01-01T00:00:00Z",
    end="2024-01-01T00:00:00Z",
    interval="1d",
    asset_class="equity",
    data_root=os.path.join(tmp, "data"),
    metadata_db=os.path.join(tmp, "meta.sqlite"),
)
print("Ingested:", store.list_symbols())

# -- 2. Backtest on it like any other data ------------------------------------
strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", ema(close, 20))
    .signal("slow", ema(close, 50))
    .size(mbt.when(ema(close, 20) > ema(close, 50), 1.0, 0.0))
    .describe("EMA(20/50) crossover on daily AAPL bars from Yahoo Finance")
)

start, end = time_range("2020-01-01", "2024-01-01")
config = mbt.BacktestConfig(
    universe=[1],
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.days(1),
    initial_capital=10_000,
    warmup_bars=60,
)

if __name__ == "__main__":
    result = mbt.run(strategy, config, store)
    print(result.summary())
