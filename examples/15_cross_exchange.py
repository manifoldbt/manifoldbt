"""Example 15: Cross-Exchange — Signal Binance, Execution dYdX.

Simple RSI mean-reversion:
- RSI computed on Binance BTC perp data
- Trades executed at dYdX BTC-USD prices
- Both loaded via universe dict — no special config needed
- Per-venue fees: each symbol is charged its own exchange's fee schedule
  (see FeeConfig.multi_venue below)

Prerequisite:
    Binance perp data (bars_1m/201.arrow) + dYdX data (dydx/1h/BTC-USD.arrow)
"""

import time
import manifoldbt as mbt
from manifoldbt.indicators import rsi, ema
from manifoldbt.expr import col, symbol_ref, lit, when
from manifoldbt.helpers import time_range, Interval, Slippage

# =============================================================================
# Signal — RSI + EMA from Binance BTC, applied to dYdX BTC
# All SymbolRef expressions must be named signals (for pass 2b resolution)
# =============================================================================
bn_btc_close = symbol_ref("binance:BTC-USDT:perp", "close")
bn_btc_rsi = rsi(bn_btc_close, 14)
bn_ema_fast = ema(bn_btc_close, 15)
bn_ema_slow = ema(bn_btc_close, 30)
trend_up = bn_ema_fast > bn_ema_slow

# Size references named signals only (no inline SymbolRef)
signal = when(
    (col("trend") > lit(0.5)) & (col("bn_rsi") > lit(70.0)), 1.0,
    when((col("trend") < lit(0.5)) & (col("bn_rsi") < lit(30.0)), -1.0,
    0.0),
)

# =============================================================================
# Strategy
# =============================================================================
strategy = (
    mbt.Strategy.create("cross_exchange_rsi")
    .signal("bn_rsi", bn_btc_rsi)
    .signal("trend", when(trend_up, 1.0, 0.0))
    .size(signal)
    .describe("Signal: Binance RSI | Execution: dYdX")
)

# =============================================================================
# Config — everything in universe
# =============================================================================
START, END = time_range("2024-02-01", "2026-03-01")

config = mbt.BacktestConfig(
    universe={
        "dydx": ["BTC-USD:perp"],         # execution (fills here)
        "binance": ["BTC-USDT:perp"],      # signal source (via symbol_ref)
    },
    time_range_start=START,
    time_range_end=END,
    bar_interval=Interval.hours(6),
    initial_capital=10_000,
    warmup_bars=30,
    execution=mbt.ExecutionConfig(signal_delay=1),
    # Per-venue fees: each symbol pays the fee schedule of the exchange it
    # executes on. Fills happen on dYdX (the execution venue), so the dYdX
    # taker fee is what actually hits this strategy; the Binance entry is
    # signal-only. Symbols without a mapping fall back to `default`.
    fees=mbt.FeeConfig.multi_venue(
        default=mbt.VenueFees(maker_fee_bps=1.0, taker_fee_bps=2.5),
        venues={
            "dydx": mbt.VenueFees(maker_fee_bps=2.0, taker_fee_bps=5.0),
            "binance": mbt.VenueFees(maker_fee_bps=1.0, taker_fee_bps=2.5),
        },
        symbol_venue={
            "dydx:BTC-USD:perp": "dydx",        # execution venue (fills here)
            "binance:BTC-USDT:perp": "binance",  # signal source only
        },
    ),
    slippage=Slippage.fixed_bps(2),
)

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(root, "..", "data"))
    meta_db = os.path.join(root, "..", "metadata", "metadata.sqlite")

    store = mbt.DataStore(
        data_root=data_root,
        metadata_db=meta_db,
        arrow_dir=os.path.join(data_root, "mega"),
    )

    print("Running: cross_exchange_rsi")
    print("  Signal:    binance:BTC-USDT:perp (RSI + EMA)")
    print("  Execution: dydx:BTC-USD:perp")
    print()

    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\nElapsed: {elapsed:.3f}s")
    result.plot_equity(show=True)
