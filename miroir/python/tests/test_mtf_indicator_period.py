"""Indicator periods over a higher timeframe count SIMULATION bars.

`bt.tf("1h").close` is the last closed hourly close, forward-filled onto the
simulation grid. An indicator over it counts rows of that grid, so on 1-minute
bars `sma(h1.close, 8)` averages 8 *minutes* of a step function — it tracks the
last closed hourly close instead of averaging 8 hours.

That reading is surprising enough that `tf()`'s own usage example used to show
`ema(h1.close, 20)` as if 20 meant hours. These tests pin the real semantics
and the conversion, measured rather than argued: on a ramp of +10 per hour, the
lag of an average over K hours is (K+1)/2 hours, plus the ~1 h the timeframe
itself costs (an hourly bar is only readable once closed).
"""
import os

import pytest

import manifoldbt as bt

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from manifoldbt.helpers import ExecutionPrice, Interval, Slippage  # noqa: E402

HOURS, PER_HOUR = 200, 60
N_BARS = HOURS * PER_HOUR
SLOPE = 10.0  # the price gains exactly this much per hour


def _ramp_bars():
    """Hourly closes of 100, 110, 120 … so a lag reads directly as hours."""
    ts = pd.date_range("2024-01-01", periods=N_BARS, freq="1min", tz="UTC")
    px = 100.0 + SLOPE * (np.arange(N_BARS) // PER_HOUR)
    return pd.DataFrame({
        "timestamp": ts, "open": px, "high": px * 1.5, "low": px * 0.5,
        "close": px, "volume": np.full(N_BARS, 1000.0),
    })


def _observed_series(period_or_expr, frame, tmp_path, tag):
    """The values a higher-timeframe band actually took, by timestamp.

    Takes either a period — read as `sma(tf("1h").close, period)`, the
    staircase form — or a ready-made expression, so the same probe serves both
    candidates.

    The indicator is read back through `ExecutionPrice.custom`, which fills at
    the value of a named signal: the trade log then carries the series itself.
    """
    from manifoldbt.indicators import close, sma
    from manifoldbt.expr import Expr

    root = os.path.join(str(tmp_path), tag)
    store = bt.import_dataframe(
        frame, symbol="S", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    band = (period_or_expr if isinstance(period_or_expr, Expr)
            else sma(bt.tf("1h").close, period_or_expr))
    strategy = (
        bt.Strategy.create("probe")
        .signal("band", band)
        .size(bt.when(close > sma(close, 3), 1.0, -1.0))
    )
    ts = pd.DatetimeIndex(frame["timestamp"])
    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=int(ts[0].value),
        time_range_end=int(ts[-1].value) + 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=1_000_000,
        execution=bt.ExecutionConfig(
            signal_delay=0,
            execution_price=ExecutionPrice.custom("band"),
            max_position_pct=0.1, allow_short=True,
            position_sizing_mode="FractionOfEquity",
        ),
        slippage=Slippage.fixed_bps(0),
        warmup_bars=700,  # clears the widest window under test
        extra_timeframes={"1h": Interval.hours(1)},
    )
    result = bt.run(strategy, config, store)
    trades = result.trades_df()
    assert len(trades) > 50, f"only {len(trades)} trades, too few to measure a lag"

    at = pd.DatetimeIndex(pd.to_datetime(trades["execution_timestamp"], utc=True))
    return pd.Series(trades["intended_price"].to_numpy(), index=at)


def _observed_lag_hours(period, frame, tmp_path, tag):
    """Average lag of `sma(tf("1h").close, period)`, in hours."""
    observed = _observed_series(period, frame, tmp_path, tag)
    ts = pd.DatetimeIndex(frame["timestamp"])
    price_now = pd.Series(frame["close"].to_numpy(), index=ts).reindex(observed.index)
    return float(np.median((price_now.to_numpy() - observed.to_numpy()) / SLOPE))


def test_a_bare_period_does_not_average_the_higher_timeframe(tmp_path):
    """`sma(h1.close, 8)` is NOT an 8-hour mean.

    An 8-hour mean would lag about (8+1)/2 + 1 = 5.5 hours. This lags under 2,
    which is the timeframe's own delay: it is tracking the last closed hourly
    close, not averaging eight of them.
    """
    lag = _observed_lag_hours(8, _ramp_bars(), tmp_path, "bare")
    assert lag < 2.5, f"lag {lag:.2f} h — this would be a real multi-hour average"
    assert lag > 0.5, f"lag {lag:.2f} h — the timeframe delay itself is missing"


def test_multiplying_the_period_by_the_interval_ratio_matches_the_lag(tmp_path):
    """`sma(h1.close, K * 60)` carries the LAG of a K-hour mean.

    Expected lag: (K+1)/2 from the average, plus ~1 h for the timeframe. This
    is the whole of what a ramp can establish, and it is not enough to call the
    result a K-hour mean: see `test_the_interval_ratio_is_not_the_hourly_mean`,
    which uses an impulse to show the two series apart.
    """
    frame = _ramp_bars()
    for hours, expected in ((4, 1 + (4 + 1) / 2), (8, 1 + (8 + 1) / 2)):
        lag = _observed_lag_hours(hours * PER_HOUR, frame, tmp_path, f"k{hours}")
        assert abs(lag - expected) < 0.5, (
            f"{hours}-hour mean lags {lag:.2f} h, expected about {expected:.2f} h"
        )


def _impulse_bars():
    """Hourly closes flat at 100 but for ONE hour at 200.

    A ramp cannot separate the two candidates: a box filter leaves a straight
    line straight, so a biased weighting still reports the right lag. An
    impulse makes each hour's weight readable in the value itself.

    The last minute of every hour keeps the base value, so the hourly closes --
    the only rows `tf("1h")` reads -- stay exactly 100 or 200. The intra-hour
    zigzag exists only to make the probe strategy trade on every bar.
    """
    n = 60 * PER_HOUR
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    base = np.where(np.arange(n) // PER_HOUR == 30, 200.0, 100.0)
    minute = np.arange(n) % PER_HOUR
    px = base + np.where(minute == PER_HOUR - 1, 0.0, np.where(minute % 2 == 0, 0.5, -0.5))
    return pd.DataFrame({
        "timestamp": ts, "open": px, "high": px * 1.5, "low": px * 0.5,
        "close": px, "volume": np.full(n, 1000.0),
    })


def test_the_interval_ratio_is_not_the_hourly_mean(tmp_path):
    """`sma(h1.close, 8 * 60)` is not an equal-weight mean of 8 hourly closes.

    It averages 480 rows of a step function, so a move ramps in over 60 minutes
    instead of stepping, and the window spans 9 hourly values with unequal
    weights rather than 8 with equal ones. On the impulse the true signal spans
    12.5 (100 to 112.5) and the gap reaches nearly all of it.
    """
    frame = _impulse_bars()
    observed = _observed_series(8 * PER_HOUR, frame, tmp_path, "impulse")

    ts = pd.DatetimeIndex(frame["timestamp"])
    hourly = pd.Series(frame["close"].to_numpy(), index=ts).resample("1h").last()
    assert set(np.round(hourly.dropna().unique(), 6)) <= {100.0, 200.0}
    truth = hourly.rolling(8).mean().shift(1).reindex(ts, method="ffill")

    gap = (observed - truth.reindex(observed.index)).dropna().abs()
    assert gap.max() > 10.0, (
        f"largest gap to a true 8-hour mean is {gap.max():.2f} on a signal "
        "spanning 12.5; the two would then be the same series"
    )


def test_apply_is_the_hourly_mean_and_reads_only_closed_bars(tmp_path):
    """`h1.apply(sma(close, 8))` IS the mean of 8 hourly closes — exactly.

    Same impulse that separates the two staircase forms. Two alignments are
    checked against, and they disagree on 120 bars, so matching one excludes
    the other: the shifted reference reads only CLOSED hours, the unshifted one
    would need the hour in progress. Landing on the shifted one is what rules
    out look-ahead.
    """
    from manifoldbt.indicators import close, sma

    frame = _impulse_bars()
    band = bt.tf("1h").apply(sma(close, 8))
    observed = _observed_series(band, frame, tmp_path, "apply")

    ts = pd.DatetimeIndex(frame["timestamp"])
    hourly = pd.Series(frame["close"].to_numpy(), index=ts).resample("1h").last()
    rolled = hourly.rolling(8).mean()
    safe = rolled.shift(1).reindex(ts, method="ffill")      # closed hours only
    leaking = rolled.reindex(ts, method="ffill")            # the hour in progress

    disagree = (safe - leaking).dropna().abs()
    assert (disagree > 1e-9).sum() > 50, "the two alignments must differ to discriminate"

    gap_safe = (observed - safe.reindex(observed.index)).dropna().abs()
    gap_leak = (observed - leaking.reindex(observed.index)).dropna().abs()
    assert gap_safe.max() < 1e-9, (
        f"apply() is off the true 8-hour mean by {gap_safe.max():.4f}"
    )
    assert gap_leak.max() > 1.0, (
        "apply() matches the alignment that reads the hour in progress"
    )


def test_a_longer_period_lags_more(tmp_path):
    """The ordering alone would catch a period silently ignored."""
    frame = _ramp_bars()
    short = _observed_lag_hours(4 * PER_HOUR, frame, tmp_path, "ord4")
    long = _observed_lag_hours(8 * PER_HOUR, frame, tmp_path, "ord8")
    assert long > short + 1.0, (
        f"8-hour mean lags {long:.2f} h vs {short:.2f} h for 4 hours; "
        "the period is not doing what it should"
    )
