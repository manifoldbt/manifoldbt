"""Example 16: BTC-Hashrate Spread — Exogenous Data Strategy.

Thesis: Bitcoin hashrate is a proxy for miner commitment and network
security.  When BTC price drops but hashrate holds (or rises), miners
are still profitable and the sell-off is likely transient — buy the dip.
When price rises but hashrate lags, the rally lacks fundamental backing.

The strategy normalizes both BTC price and hashrate via EMA ratios
(price/EMA and hashrate/EMA), then computes a spread between the two.
A rolling z-score of the spread generates the signal: negative z means
price is cheap relative to hashrate (long), positive means expensive.

Exogenous data flow:
    1. Fetch hashrate CSV (or use sample generator below)
    2. Register via  mbt.register_exo("hashrate", df)
    3. Declare in     BacktestConfig(exo_data=["hashrate"])
    4. Access with    exo("hashrate") in expressions

Prerequisite:
    Binance BTC perp data  +  hashrate exo registered in data/mega/exo/
"""

import time
import numpy as np
import manifoldbt as mbt
from manifoldbt.indicators import ema, close
from manifoldbt.expr import col, exo, lit, when, hold
from manifoldbt.helpers import time_range, Interval, Slippage


# =============================================================================
# Parameters
# =============================================================================
SMOOTH = 30          # EMA period for normalization
ZSCORE_WINDOW = 90   # Rolling z-score lookback (days)
ENTRY_Z = -1.5       # Long when spread z < -1.5 (price cheap vs hashrate)
EXIT_Z = 0.0         # Exit when spread reverts to mean
SHORT_Z = 1.5        # Short when spread z > 1.5 (price expensive vs hashrate)
SIZE = 0.5           # Position size (fraction of capital)


# =============================================================================
# Indicators
# =============================================================================

# Normalize price: ratio to its own EMA (>1 = above trend, <1 = below)
price_ratio = close / ema(close, SMOOTH)

# Normalize hashrate the same way
hr = exo("hashrate")
hr_ratio = hr / ema(hr, SMOOTH)

# Spread: price_ratio - hr_ratio
# Positive = price running ahead of hashrate, negative = price lagging
spread = price_ratio - hr_ratio

# Z-score of the spread (rolling mean & std)
spread_z = spread.zscore(ZSCORE_WINDOW)


# =============================================================================
# Sizing
# =============================================================================
z = col("spread_z")

size = when(
    z < lit(ENTRY_Z), lit(SIZE),           # price cheap vs hashrate -> long
    when(z > lit(SHORT_Z), -lit(SIZE),     # price expensive vs hashrate -> short
    when((z > lit(EXIT_Z)) & (z < lit(SHORT_Z)), 0.0,   # neutral zone -> flat
    hold())),
)


# =============================================================================
# Strategy
# =============================================================================
strategy = (
    mbt.Strategy.create("hashrate_spread")
    .signal("price_ratio", price_ratio)
    .signal("hr_ratio", hr_ratio)
    .signal("spread", spread)
    .signal("spread_z", spread_z)
    .size(size)
    .describe("BTC vs Hashrate spread z-score mean-reversion")
)


# =============================================================================
# Config
# =============================================================================
START, END = time_range("2021-06-01", "2026-03-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=START,
    time_range_end=END,
    bar_interval=Interval.days(1),
    initial_capital=10_000,
    warmup_bars=ZSCORE_WINDOW + SMOOTH,
    exo_data=["hashrate"],
    execution=mbt.ExecutionConfig(signal_delay=1, allow_short=True),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(3),
)


# =============================================================================
# Hashrate data helper
# =============================================================================
def fetch_hashrate_csv(path: str = "hashrate.csv"):
    """Load hashrate from a CSV with columns: timestamp, hashrate.

    Public sources (daily, free):
      - https://api.blockchain.info/charts/hash-rate?timespan=5years&format=csv
      - Glassnode, CoinMetrics (API key)

    The CSV should have:
      timestamp     — date or datetime (parsed automatically)
      hashrate      — daily avg hashrate in EH/s (float)
    """
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_sample_hashrate(start="2020-01-01", end="2026-03-01"):
    """Generate synthetic hashrate data for testing.

    Mimics the real BTC hashrate trajectory:
    - Exponential growth trend (~50% annual)
    - China ban crash (May-Jul 2021): -50%
    - Recovery + continued growth
    - Random noise (~5% daily vol)
    """
    import pandas as pd

    dates = pd.date_range(start, end, freq="D", tz="UTC")
    n = len(dates)

    # Base: exponential growth from ~120 EH/s to ~800 EH/s
    t = np.arange(n) / 365.25
    base = 120 * np.exp(0.40 * t)  # ~50% annual growth

    # China ban shock: May-Jul 2021
    ban_start = pd.Timestamp("2021-05-15", tz="UTC")
    ban_end = pd.Timestamp("2021-07-15", tz="UTC")
    recovery_end = pd.Timestamp("2022-01-01", tz="UTC")

    shock = np.ones(n)
    for i, d in enumerate(dates):
        if ban_start <= d <= ban_end:
            # Linear drop to 50%
            frac = (d - ban_start) / (ban_end - ban_start)
            shock[i] = 1.0 - 0.50 * frac
        elif ban_end < d < recovery_end:
            # Recovery from 50% back to 100%
            frac = (d - ban_end) / (recovery_end - ban_end)
            shock[i] = 0.50 + 0.50 * frac

    # Random noise (geometric brownian)
    rng = np.random.default_rng(42)
    noise = np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    noise /= noise[0]

    hashrate = base * shock * noise

    return pd.DataFrame({"timestamp": dates, "hashrate": hashrate})


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

    # -- Register hashrate exo data -------------------------------------------
    csv_path = os.path.join(root, "hashrate.csv")
    if os.path.exists(csv_path):
        print("Loading hashrate from CSV...")
        hr_df = fetch_hashrate_csv(csv_path)
    else:
        print("No hashrate.csv found — generating synthetic data for demo...")
        hr_df = generate_sample_hashrate()

    mbt.register_exo("hashrate", hr_df, store=store)
    print(f"  Registered {len(hr_df)} hashrate data points")
    print(f"  Range: {hr_df['timestamp'].iloc[0]} -> {hr_df['timestamp'].iloc[-1]}")
    print()

    # -- Run backtest ---------------------------------------------------------
    print("Running: hashrate_spread")
    print("  Long  when spread z < -1.5 (price cheap vs hashrate)")
    print("  Short when spread z > +1.5 (price expensive vs hashrate)")
    print()

    t0 = time.perf_counter()
    result = mbt.run(strategy, config, store)
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\nElapsed: {elapsed:.3f}s")
    result.plot_equity()
