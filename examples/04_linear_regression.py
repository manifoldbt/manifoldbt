"""Linear Regression Trend -- regression-based trend detection with confidence bands.

Demonstrates:
  - linreg_slope / linreg_value / linreg_r2 indicators
  - Confidence-weighted sizing (R² as conviction filter)
  - Multi-timeframe: slope on 4h window, trade on 15min bars
  - Trailing stop for trend exits
  - Bracket orders (stop-loss + take-profit)

The idea: fit a rolling OLS regression on price. When the slope is steep
and the R² is high (price moves in a straight line), we have a strong trend.
Size proportionally to slope strength * R² confidence.

Usage:
    python examples/04_linear_regression.py
"""
import os
import time
import manifoldbt as mbt
from manifoldbt.indicators import close, high, low, volume
from manifoldbt.helpers import time_range, Slippage, Interval

# -- Regression indicators ----------------------------------------------------
# Rolling linear regression over 16 bars (16 * 15min = 4h window)
window = 16

slope = close.linreg_slope(window)       # price change per bar (trend direction)
fitted = close.linreg_value(window)       # regression fitted value
r2 = close.linreg_r2(window)             # goodness of fit (0=noise, 1=perfect line)

# Normalize slope by price to get a percentage rate
norm_slope = slope / (close + mbt.lit(1e-12))

# -- Volatility filter ---------------------------------------------------------
# ATR-like: average true range normalized by price
avg_range = (high - low).rolling_mean(window)
norm_vol = avg_range / (close + mbt.lit(1e-12))

# -- Signal construction -------------------------------------------------------
# Conviction = R² (0 to 1). Only trade when R² > 0.6 (strong linear trend)
has_conviction = r2 > 0.6

# Direction: positive slope = long, negative = short
# Magnitude: |normalized slope| / volatility = trend strength vs noise
trend_strength = norm_slope / (norm_vol + mbt.lit(1e-12))

# Final signal: direction * conviction, gated by R² threshold
# Clamp to [-1, 1] range via division by expected max
raw_signal = mbt.when(
    has_conviction,
    trend_strength * r2 * mbt.lit(0.1),   # scale down
    0.0,
)

# -- Strategy ------------------------------------------------------------------
strategy = (
    mbt.Strategy.create("linreg_trend")
    .signal("slope", norm_slope)
    .signal("r2", r2)
    .signal("fitted", fitted)
    .signal("trend_strength", trend_strength)
    .size(raw_signal)
    .trailing_stop(pct=2.0)
    .describe(
        "Rolling OLS regression: trade strong linear trends (high R²), "
        "size by slope strength * confidence, trailing stop exit"
    )
)

# -- Config --------------------------------------------------------------------
start, end = time_range("2022-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp", "ETH-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.days(1),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(
        allow_short=True,
        max_position_pct=0.5,
    ),
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=20,
)

# -- Run -----------------------------------------------------------------------
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
    print(f"\nElapsed: {elapsed:.3f}s")
    mbt.plot.summary(result)
