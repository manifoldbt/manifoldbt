"""CSV Import -- load your own OHLCV data from a CSV file.

Demonstrates:
  - bt.import_csv() -- auto-detects standard / MetaTrader 4 / MetaTrader 5
  - Backtesting on the imported data, exactly like a built-in connector
  - Free on all tiers (no Pro license required)

The standard format is a header row + `timestamp,open,high,low,close,volume`
where timestamp is Unix milliseconds. MT4/MT5 exports are auto-detected.

Usage:
    python examples/18_csv_import.py
"""
import os
import tempfile

import manifoldbt as mbt
from manifoldbt.indicators import close, ema
from manifoldbt.helpers import time_range, Interval

# -- 1. A sample CSV ----------------------------------------------------------
# In practice you'd point `import_csv` straight at your own file. Here we
# synthesize a small one so the example runs out of the box.
tmp = tempfile.mkdtemp()
csv_path = os.path.join(tmp, "SAMPLE_1m.csv")

base_ms = 1_704_067_200_000  # 2024-01-01 00:00 UTC
px = 100.0
with open(csv_path, "w") as f:
    f.write("timestamp,open,high,low,close,volume\n")
    for i in range(3000):
        ts = base_ms + i * 60_000               # 1-minute bars
        nxt = px * (1.0 + (0.0009 if i % 3 else -0.0007))
        hi = max(px, nxt) + 0.05
        lo = min(px, nxt) - 0.05
        f.write(f"{ts},{px:.4f},{hi:.4f},{lo:.4f},{nxt:.4f},{1000 + i}\n")
        px = nxt

# -- 2. Import into the store (free, all tiers) -------------------------------
store = mbt.import_csv(
    csv_path,
    symbol="SAMPLE",
    symbol_id=1,
    interval="1m",
    data_root=os.path.join(tmp, "data"),
    metadata_db=os.path.join(tmp, "meta.sqlite"),
    asset_class="crypto_spot",
)
print("Imported:", store.list_symbols())

# -- 3. Backtest on it like any other data ------------------------------------
strategy = (
    mbt.Strategy.create("ema_cross")
    .signal("fast", ema(close, 10))
    .signal("slow", ema(close, 30))
    .size(mbt.when(ema(close, 10) > ema(close, 30), 0.5, 0.0))
    .describe("EMA(10/30) crossover on CSV-imported data")
)

start, end = time_range("2024-01-01", "2024-01-04")
config = mbt.BacktestConfig(
    universe=[1],
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.minutes(1),
    initial_capital=10_000,
    warmup_bars=30,
)

if __name__ == "__main__":
    result = mbt.run(strategy, config, store)
    print(result.summary())
