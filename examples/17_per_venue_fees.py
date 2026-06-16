"""Example 17: Per-Venue Fees — charge each symbol its own fee schedule.

Real desks route different assets to different exchanges (or liquidity tiers),
each with its own maker/taker fees, funding column and borrow rate. ``FeeConfig``
models this directly: a ``default`` venue plus named ``per_venue`` overrides and a
``symbol_venue`` map saying which symbol trades where.

Here a 4-asset momentum portfolio executes the majors (BTC, ETH) on a cheap
venue and the alts (XRP, DOT) on a more expensive one. Single-provider universe,
so it runs without Pro.

Usage:
    python examples/17_per_venue_fees.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, ema, roc, high, low
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Indicators ---------------------------------------------------------------
mom = ema(roc(close, 14), 6)
avg_range = (high - low).rolling_mean(14)
norm_vol = avg_range / (close + mbt.lit(1e-12))
safe_vol = mbt.when(norm_vol > 0.0005, norm_vol, 0.0005)

# -- Strategy -----------------------------------------------------------------
signal = mbt.when(mom > 0.0, mom / safe_vol, 0.0)

strategy = (
    mbt.Strategy.create("per_venue_momentum")
    .signal("momentum", mom)
    .signal("norm_vol", norm_vol)
    .size(signal * 0.01)
    .describe("Multi-asset momentum with per-venue fees")
)

# -- Per-venue fees -----------------------------------------------------------
# Majors fill on a cheap venue; alts on a pricier one. Symbols absent from
# `symbol_venue` would fall back to `default`. Keys are symbol names (qualified
# with the provider), resolved to SymbolIds automatically.
fees = mbt.FeeConfig.multi_venue(
    default=mbt.VenueFees(maker_fee_bps=2.0, taker_fee_bps=5.0),
    venues={
        "cheap":     mbt.VenueFees(maker_fee_bps=1.0, taker_fee_bps=3.0),
        "expensive": mbt.VenueFees(maker_fee_bps=5.0, taker_fee_bps=12.0),
    },
    symbol_venue={
        "binance:BTC-USDT:perp": "cheap",
        "binance:ETH-USDT:perp": "cheap",
        "binance:XRP-USDT:perp": "expensive",
        "binance:DOT-USDT:perp": "expensive",
    },
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe={
        "binance": ["BTC-USDT:perp", "ETH-USDT:perp",
                    "XRP-USDT:perp", "DOT-USDT:perp"],
    },
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(12),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        signal_delay=1,
        max_position_pct=0.3,
        allow_short=False,
    ),
    fees=fees,
    slippage=Slippage.fixed_bps(2),
    warmup_bars=25,
)

# -- Run ----------------------------------------------------------------------
if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..")
    data_root = os.path.abspath(os.path.join(root, "data"))
    store = mbt.DataStore(
        data_root=data_root,
        metadata_db=os.path.abspath(os.path.join(root, "metadata", "metadata.sqlite")),
        arrow_dir=os.path.join(data_root, "mega"),
    )

    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    elapsed = time.perf_counter() - t0

    print(result.summary())

    # Show that fees actually differ by venue: average fee in bps per symbol.
    trades = result.trades
    if trades.num_rows > 0:
        sids = trades.column("symbol_id").to_pylist()
        fee_vals = trades.column("fees").to_pylist()
        qty = trades.column("quantity").to_pylist()
        fill = trades.column("fill_price").to_pylist()
        agg: dict[int, list[float]] = {}
        for sid, f, q, p in zip(sids, fee_vals, qty, fill):
            notional = abs(q) * p
            if notional > 0:
                agg.setdefault(sid, []).append(f / notional * 10_000)
        print("\nRealized fee (bps) by symbol_id:")
        for sid in sorted(agg):
            bps = sum(agg[sid]) / len(agg[sid])
            print(f"  symbol {sid}: {bps:.2f} bps  ({len(agg[sid])} fills)")

    print(f"\nElapsed: {elapsed:.3f}s")
