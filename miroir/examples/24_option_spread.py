"""Option strategy -- a bull call spread, held to expiration.

Demonstrates:
  - A two-leg option structure: long a low strike, short a higher one
  - Per-leg sizing with col("symbol_id")
  - A SHORT option paying margin under the venue's own formula
  - Both legs cash-settled at expiry, which is what caps the payoff

The structure: buy the 100k call, sell the 110k call, same expiration. The
short leg pays for part of the long one, and in exchange it caps the gain at
the distance between the strikes. Classic, and the cheapest way to see the
engine settle two contracts on the same day with different outcomes.

**A currency trap worth knowing.** Every leg of a strategy has to be quoted in
the same currency, because the engine carries one cash balance. On Deribit an
option is quoted in BTC, but `BTC-PERPETUAL` is quoted in USD. So a covered
call (long the perpetual, short a call) would add dollars to bitcoin in a single
number and produce a meaningless equity curve. A spread has both legs in BTC,
which is why this example is a spread. The perpetual appears below only as the
settlement reference, never as a position.

Data: self-contained (network) — ingested on each run from a free connector

Usage:
    python examples/24_option_spread.py
"""
import os
import tempfile

import manifoldbt as mbt
from manifoldbt.indicators import col
from manifoldbt.helpers import time_range, Interval

# Both legs expired on 2025-06-27, so the whole life of the trade is history.
UNDERLYING, UNDERLYING_ID = "BTC-PERPETUAL", 1
LONG_LEG, LONG_ID = "BTC-27JUN25-100000-C", 2
SHORT_LEG, SHORT_ID = "BTC-27JUN25-110000-C", 3
START, END = "2025-05-01T00:00:00Z", "2025-07-01T00:00:00Z"

tmp = tempfile.mkdtemp()
common = dict(
    start=START,
    end=END,
    interval="1d",
    data_root=os.path.join(tmp, "data"),
    metadata_db=os.path.join(tmp, "meta.sqlite"),
)

store = mbt.ingest(
    provider="deribit", symbol=UNDERLYING, symbol_id=UNDERLYING_ID,
    asset_class="crypto_perp", **common
)
for symbol, symbol_id in ((LONG_LEG, LONG_ID), (SHORT_LEG, SHORT_ID)):
    store = mbt.ingest(
        provider="deribit", symbol=symbol, symbol_id=symbol_id,
        asset_class="option", **common
    )

# -- The strategy --------------------------------------------------------------
# Legs are told apart by symbol id. Never discriminate on price level: a premium
# crossing the threshold would flip its own leg to zero and close the position.
size = (
    mbt.when(col("symbol_id") == float(LONG_ID), 1.0, 0.0)      # buy the 100k call
    + mbt.when(col("symbol_id") == float(SHORT_ID), -1.0, 0.0)  # sell the 110k call
)
strategy = (
    mbt.Strategy.create("bull_call_spread")
    .signal("leg", col("symbol_id"))
    .size(size)
    .describe("Long the 100k call, short the 110k call, held to expiration")
)

start, end = time_range("2025-05-01", "2025-07-01")
config = mbt.BacktestConfig(
    universe=[UNDERLYING_ID, LONG_ID, SHORT_ID],
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.days(1),
    initial_capital=10.0,                       # 10 BTC: everything here is in BTC
    currency="BTC",
    option_underlyings={LONG_ID: UNDERLYING_ID, SHORT_ID: UNDERLYING_ID},
    option_margin_model="deribit",              # the short leg posts margin
    execution=mbt.ExecutionConfig(position_sizing_mode="Units", allow_short=True),
)

if __name__ == "__main__":
    result = mbt.run(strategy, config, store)

    # `result.trades` is an Arrow table. Read it column-wise rather than through
    # `.to_pandas()`, so the file runs on the plain `pip install manifoldbt`.
    cols = ["symbol_id", "quantity", "fill_price", "exit_reason"]
    trades = [dict(zip(cols, row))
              for row in zip(*(result.trades.column(c).to_pylist() for c in cols))]

    names = {LONG_ID: "long 100k", SHORT_ID: "short 110k"}
    print("\nTrades, by leg:")
    for t in trades:
        if t["symbol_id"] == UNDERLYING_ID:
            continue
        what = "settled" if t["exit_reason"] == 5 else "traded"
        print(f"  {names[t['symbol_id']]:<12} {what:<8} {t['quantity']:>4.1f} "
              f"@ {t['fill_price']:.6f} BTC")

    equity = float(result.equity_curve[-1])
    print(f"\nFinal equity: {equity:.6f} BTC  ({equity - 10.0:+.6f})")
    # The short leg expiring worthless is what the spread pays for: it financed
    # part of the long call, and capped the gain at the strike distance.
    if result.warnings:
        print("Warnings:", result.warnings)
