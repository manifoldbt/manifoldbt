"""Crypto options -- Deribit contracts that actually expire.

Demonstrates:
  - mbt.ingest(provider="deribit", ...) -- no API key, expired contracts included
  - Contract terms (strike, expiry, side, settlement) recorded alongside the bars
  - Cash settlement at expiration, at intrinsic value
  - config.option_underlyings -- which price series settles the contract
  - Per-leg sizing with col("symbol_id"), for multi-leg structures

Why Deribit and not Binance: Deribit serves the history of contracts that have
already expired, which is the only data an option backtest can run on. Binance's
options API answers HTTP 400 for anything past its expiration date.

Three things worth knowing before reading the numbers:

  - **Everything is in BTC.** A Deribit BTC option is quoted, margined and
    settled in BTC, so `initial_capital` below is 10 BTC, not 10 dollars. The
    payoff of a call is `max(0, S - K) / S` BTC per contract.

  - **You choose the settlement reference.** Deribit settles against its own
    `BTC_USD` index, whose ticker matches no series you can ingest.
    `BTC-PERPETUAL` stands in for it here; the basis between the two is small
    but real, and it is not modelled.

  - **Positions are counted in units of the underlying.** On Deribit a contract
    is one unit, so the two are the same thing. On a 100-multiplier listed
    option, holding one contract means a position of 100.

Data: self-contained (network) — ingested on each run from a free connector

Usage:
    python examples/23_deribit_options.py
"""
import os
import tempfile

import manifoldbt as mbt
from manifoldbt.indicators import col
from manifoldbt.helpers import time_range, Interval

# Two contracts that expired on 2025-06-27, and the series that settles them.
UNDERLYING, UNDERLYING_ID = "BTC-PERPETUAL", 1
CALL_100K, CALL_ID = "BTC-27JUN25-100000-C", 2
PUT_90K, PUT_ID = "BTC-27JUN25-90000-P", 3
START, END = "2025-05-01T00:00:00Z", "2025-07-01T00:00:00Z"

tmp = tempfile.mkdtemp()
common = dict(
    start=START,
    end=END,
    interval="1d",
    data_root=os.path.join(tmp, "data"),
    metadata_db=os.path.join(tmp, "meta.sqlite"),
)

# -- 1. The settlement reference, then the contracts ---------------------------
store = mbt.ingest(
    provider="deribit", symbol=UNDERLYING, symbol_id=UNDERLYING_ID,
    asset_class="crypto_perp", **common
)
store = mbt.ingest(
    provider="deribit", symbol=CALL_100K, symbol_id=CALL_ID,
    asset_class="option", **common
)
store = mbt.ingest(
    provider="deribit", symbol=PUT_90K, symbol_id=PUT_ID,
    asset_class="option", **common
)

# The connector asked Deribit for the terms and the store kept them; nothing
# below is inferred from the instrument name.
for symbol_id, terms in sorted(store.option_contracts().items()):
    print(f"{symbol_id}: {terms['option_type']} {terms['strike']:.0f}, "
          f"settles {terms['settlement']}")

# -- 2. A risk reversal: long the 100k call, short the 90k put -----------------
# Legs are told apart by `col("symbol_id")`. Do NOT discriminate on price level
# (e.g. "close < 100"): a premium that crosses the threshold silently flips the
# leg to zero and the strategy closes its own position.
size = (
    mbt.when(col("symbol_id") == float(CALL_ID), 1.0, 0.0)
    + mbt.when(col("symbol_id") == float(PUT_ID), -1.0, 0.0)
)
strategy = (
    mbt.Strategy.create("risk_reversal")
    .signal("leg", col("symbol_id"))
    .size(size)
    .describe("Long the 100k call, short the 90k put, both held to expiration")
)

start, end = time_range("2025-05-01", "2025-07-01")
config = mbt.BacktestConfig(
    universe=[UNDERLYING_ID, CALL_ID, PUT_ID],
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.days(1),
    initial_capital=10.0,                                   # 10 BTC
    currency="BTC",
    option_underlyings={CALL_ID: UNDERLYING_ID, PUT_ID: UNDERLYING_ID},
    option_margin_model="deribit",                          # the short put pays margin
    execution=mbt.ExecutionConfig(position_sizing_mode="Units", allow_short=True),
)

if __name__ == "__main__":
    result = mbt.run(strategy, config, store)

    # `result.trades` is an Arrow table. Read it column-wise rather than through
    # `.to_pandas()`, so the file runs on the plain `pip install manifoldbt`.
    cols = ["symbol_id", "side", "quantity", "fill_price", "exit_reason"]
    trades = [dict(zip(cols, row))
              for row in zip(*(result.trades.column(c).to_pylist() for c in cols))]

    print("\nTrades on the option legs:")
    print(f"  {'symbol_id':>9} {'side':>6} {'quantity':>9} {'fill_price':>11} {'exit_reason':>11}")
    for t in trades:
        if t["symbol_id"] == UNDERLYING_ID:
            continue
        print(f"  {t['symbol_id']:>9} {str(t['side']):>6} {t['quantity']:>9.2f} "
              f"{t['fill_price']:>11.6f} {str(t['exit_reason']):>11}")
    # exit_reason 5 is OptionExpiry: the venue settled the contract, the
    # strategy did not sell it. The put settles at 0 because BTC finished far
    # above its 90k strike.
    print("\nFinal equity: %.6f BTC" % float(result.equity_curve[-1]))
    if result.warnings:
        print("Warnings:", result.warnings)
