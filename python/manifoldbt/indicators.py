"""Technical indicators built on top of the Expr DSL.

All functions return ``Expr`` objects that compose into the expression graph
evaluated by the Rust engine. No data is touched at definition time.

Usage::

    from manifoldbt.indicators import sma, rsi, macd, bollinger_bands

    fast = sma(close, 20)
    slow = sma(close, 60)
    my_rsi = rsi(close, 14)
"""
from __future__ import annotations

from typing import Tuple

from manifoldbt.expr import (
    Expr, MultiExpr, col, lit, when, _coerce, _resolve_period, _resolve_span, s, scan,
)

# ---------------------------------------------------------------------------
# Pre-built column references
# ---------------------------------------------------------------------------

open = col("open")
high = col("high")
low = col("low")
close = col("close")
volume = col("volume")

# What every band-shaped indicator returns, in order. Named here so the refusal
# a caller meets on misuse quotes the same words as the reference docs.
_BAND_NAMES = ("upper", "middle", "lower")
vwap = col("vwap")
timestamp = col("timestamp")

# ---------------------------------------------------------------------------
# Math helpers (wrapping Rust built-in functions)
# ---------------------------------------------------------------------------


def abs_val(x: Expr) -> Expr:
    """Absolute value (element-wise)."""
    return Expr("Function", "abs", [_coerce(x)])


def sqrt(x: Expr) -> Expr:
    """Square root (element-wise)."""
    return Expr("Function", "sqrt", [_coerce(x)])


def log(x: Expr) -> Expr:
    """Natural logarithm (element-wise)."""
    return Expr("Function", "log", [_coerce(x)])


def exp(x: Expr) -> Expr:
    """Exponential e^x (element-wise)."""
    return Expr("Function", "exp", [_coerce(x)])


def max_val(a: Expr, b: Expr) -> Expr:
    """Element-wise maximum of two expressions."""
    return Expr("Function", "max", [_coerce(a), _coerce(b)])


def min_val(a: Expr, b: Expr) -> Expr:
    """Element-wise minimum of two expressions."""
    return Expr("Function", "min", [_coerce(a), _coerce(b)])


# ---------------------------------------------------------------------------
# Trend / Moving averages
# ---------------------------------------------------------------------------


def sma(source: Expr, period) -> Expr:
    """Simple Moving Average. Period can be int or param()."""
    return source.rolling_mean(period)


def ema(source: Expr, span) -> Expr:
    """Exponential Moving Average (span-based). Span can be int or param()."""
    return source.ewm_mean(span)


def dema(source: Expr, period=14) -> Expr:
    """Double Exponential Moving Average. Period can be int or param()."""
    return source.dema(period)


def tema(source: Expr, period=14) -> Expr:
    """Triple Exponential Moving Average. Period can be int or param()."""
    return source.tema(period)


def wma(source: Expr, period=14) -> Expr:
    """Weighted Moving Average. Period can be int or param()."""
    return source.wma(period)


def hma(source: Expr, period=14) -> Expr:
    """Hull Moving Average. Period can be int or param()."""
    return source.hma(period)


def kama(source: Expr, period=10) -> Expr:
    """Kaufman Adaptive Moving Average. Period can be int or param()."""
    return source.kama(period)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def roc(source: Expr, period=1) -> Expr:
    """Rate of Change. Period can be int or param()."""
    return source.roc(period)


def momentum(source: Expr, period=1) -> Expr:
    """Momentum (raw price difference). Period can be int or param()."""
    return source.diff(period)


def rsi(source: Expr, period=14) -> Expr:
    """Relative Strength Index (native Rust, Wilder's smoothing, single-pass O(n)).

    Returns an expression in [0, 100]. Values below 30 are typically
    considered oversold, above 70 overbought.
    """
    return source.rsi(period)


def stoch_k(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Stochastic %K oscillator (native Rust).

    Args:
        h, l, c: Custom high/low/close columns (e.g. exo columns).
                 Defaults to native bar columns.
    """
    return Expr("StochK", h or high, l or low, c or close, _resolve_period(period))


def stochastic_k(period: int = 14, source: Expr = None) -> Expr:
    """Stochastic %K oscillator (DSL-based fallback).

    ``(close - lowest_low) / (highest_high - lowest_low) * 100``
    """
    c = source if source is not None else close
    lowest = c.rolling_min(period)
    highest = c.rolling_max(period)
    return (c - lowest) / (highest - lowest + lit(1e-12)) * lit(100.0)


def williams_r(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Williams %R oscillator (native Rust).

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    return Expr("WilliamsR", h or high, l or low, c or close, _resolve_period(period))


def cci(period: int = 20, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Commodity Channel Index (native Rust).

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    return Expr("Cci", h or high, l or low, c or close, _resolve_period(period))


def adx(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Average Directional Index (native Rust).

    Args:
        h, l, c: Custom high/low/close columns (e.g. exo columns).
                 Defaults to native bar columns.
    """
    return Expr("Adx", h or high, l or low, c or close, _resolve_period(period))


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def bollinger_bands(
    source: Expr, period: int = 20, num_std: float = 2.0
) -> Tuple[Expr, Expr, Expr]:
    """Bollinger Bands (native Rust).

    Returns:
        ``(upper, middle, lower)`` — three ``Expr`` objects.
    """
    upper = source.bollinger_upper(period, num_std)
    middle = source.rolling_mean(period)
    lower = source.bollinger_lower(period, num_std)
    return MultiExpr((upper, middle, lower), "bollinger_bands", _BAND_NAMES)


def bollinger_width(source: Expr, period: int = 20, num_std: float = 2.0) -> Expr:
    """Bollinger Bandwidth (native Rust)."""
    return source.bollinger_width(period, num_std)


def atr(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Average True Range (native Rust, Wilder's smoothing, single-pass O(n)).

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    return Expr("Atr", h or high, l or low, c or close, _resolve_period(period))


def true_range(*, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """True Range (native Rust).

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    return Expr("TrueRange", h or high, l or low, c or close)


def natr(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Normalized ATR (native Rust).

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    return Expr("Natr", h or high, l or low, c or close, _resolve_period(period))


def keltner_channels(
    period: int = 20, multiplier: float = 1.5,
    *, h: Expr = None, l: Expr = None, c: Expr = None,
) -> Tuple[Expr, Expr, Expr]:
    """Keltner Channels (native Rust).

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    _h, _l, _c = h or high, l or low, c or close
    upper = Expr("KeltnerUpper", _h, _l, _c, _resolve_period(period), _resolve_span(multiplier))
    # La bande centrale est une EMA de meme longueur. `ewm_mean` prend un span
    # (DynFloat), donc un `param()` doit passer par la resolution DynFloat et non
    # par un `float()` qui le casserait.
    middle = _c.ewm_mean(period if isinstance(period, Expr) else float(period))
    lower = Expr("KeltnerLower", _h, _l, _c, _resolve_period(period), _resolve_span(multiplier))
    return MultiExpr((upper, middle, lower), "keltner_channels", _BAND_NAMES)


def supertrend(
    period: int = 10, multiplier: float = 3.0,
    *, h: Expr = None, l: Expr = None, c: Expr = None,
) -> Expr:
    """SuperTrend indicator (native Rust).

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    return Expr("SuperTrend", h or high, l or low, c or close, _resolve_period(period),
                _resolve_span(multiplier))


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    source: Expr,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[Expr, Expr, Expr]:
    """Moving Average Convergence Divergence (native Rust).

    Returns:
        ``(macd_line, signal_line, histogram)`` — three ``Expr`` objects.
    """
    macd_line = source.macd_line(fast_period, slow_period)
    signal_line = source.macd_signal(fast_period, slow_period, signal_period)
    histogram = source.macd_hist(fast_period, slow_period, signal_period)
    return MultiExpr(
        (macd_line, signal_line, histogram),
        "macd",
        ("macd_line", "signal_line", "histogram"),
    )


# ---------------------------------------------------------------------------
# Crossover signals
# ---------------------------------------------------------------------------


def crossover(a: Expr, b: Expr) -> Expr:
    """True on bars where ``a`` crosses above ``b`` (native Rust)."""
    return a.cross_above(b)


def crossunder(a: Expr, b: Expr) -> Expr:
    """True on bars where ``a`` crosses below ``b`` (native Rust)."""
    return a.cross_below(b)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def obv(source: Expr = None, vol: Expr = None) -> Expr:
    """On-Balance Volume (native Rust).

    Args:
        source: Price series. Defaults to ``close``.
        vol: Volume series. Defaults to ``volume``.
    """
    return Expr("Obv", source if source is not None else close,
                vol if vol is not None else volume)


def vwap(*, h: Expr = None, l: Expr = None, c: Expr = None, v: Expr = None) -> Expr:
    """Volume Weighted Average Price (native Rust).

    Args:
        h, l, c, v: Custom high/low/close/volume columns. Defaults to native bar columns.
    """
    return Expr("Vwap", h or high, l or low, c or close, v or volume)


def ad_line(*, h: Expr = None, l: Expr = None, c: Expr = None, v: Expr = None) -> Expr:
    """Accumulation/Distribution Line (native Rust).

    Args:
        h, l, c, v: Custom high/low/close/volume columns. Defaults to native bar columns.
    """
    return Expr("AdLine", h or high, l or low, c or close, v or volume)


def mfi(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None, v: Expr = None) -> Expr:
    """Money Flow Index (native Rust).

    Args:
        h, l, c, v: Custom high/low/close/volume columns. Defaults to native bar columns.
    """
    return Expr("Mfi", h or high, l or low, c or close, v or volume, _resolve_period(period))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def rolling_median(source: Expr, window: int) -> Expr:
    """Rolling median (native Rust)."""
    return source.rolling_median(window)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def parabolic_sar(af_start: float = 0.02, af_max: float = 0.2) -> Expr:
    """Parabolic SAR (native Rust, uses high/low)."""
    return Expr("ParabolicSar", high, low, _resolve_span(af_start), _resolve_span(af_max))


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


def linreg_slope(source: Expr, window: int) -> Expr:
    """Rolling linear regression slope (native Rust, single-pass O(n)).

    Fits y = a + b*x over a rolling window and returns the slope b.
    """
    return source.linreg_slope(window)


def linreg_value(source: Expr, window: int) -> Expr:
    """Rolling linear regression predicted value (native Rust, single-pass O(n)).

    Returns the predicted y at the last point of the rolling window.
    Equivalent to ``mean + slope * (window - 1) / 2``.
    """
    return source.linreg_value(window)


def linreg_r2(source: Expr, window: int) -> Expr:
    """Rolling linear regression R-squared (native Rust, single-pass O(n)).

    Returns the coefficient of determination in [0, 1].
    NaN when the series is constant within the window.
    """
    return source.linreg_r2(window)


# ---------------------------------------------------------------------------
# Datetime extraction
# ---------------------------------------------------------------------------


def hour(source: Expr = None) -> Expr:
    """Extract hour (0-23 UTC) from a timestamp column.

    Defaults to the ``timestamp`` bar column if no source given.

    Usage::

        # Trade only during US equity hours (14:30-21:00 UTC)
        us_hours = (hour() >= 14) & (hour() < 21)
    """
    return (source if source is not None else timestamp).hour()


def minute(source: Expr = None) -> Expr:
    """Extract minute (0-59) from a timestamp column.

    Defaults to the ``timestamp`` bar column if no source given.
    """
    return (source if source is not None else timestamp).minute()


def day_of_week(source: Expr = None) -> Expr:
    """Extract day of week from a timestamp column (0=Monday, 6=Sunday).

    Defaults to the ``timestamp`` bar column if no source given.

    Usage::

        # Only trade on weekdays
        is_weekday = day_of_week() < 5
    """
    return (source if source is not None else timestamp).day_of_week()


def month(source: Expr = None) -> Expr:
    """Extract month (1-12) from a timestamp column.

    Defaults to the ``timestamp`` bar column if no source given.

    Usage::

        # Seasonal filter: trade only Q4 (Oct-Dec)
        is_q4 = month() >= 10
    """
    return (source if source is not None else timestamp).month()


def day_of_month(source: Expr = None) -> Expr:
    """Extract day of month (1-31) from a timestamp column.

    Defaults to the ``timestamp`` bar column if no source given.
    """
    return (source if source is not None else timestamp).day_of_month()


# ---------------------------------------------------------------------------
# Scan-based indicators (arbitrary stateful computations)
# ---------------------------------------------------------------------------


def kalman(source: Expr = None, q: float = 1e-5, r: float = 1e-2) -> Expr:
    """Kalman filter (1-D constant-velocity model).

    Uses the ``scan`` primitive — runs entirely in Rust as a flat scalar VM.

    Args:
        source: Input price series. Defaults to ``close``.
        q: Process noise covariance (how much the true value can change per step).
        r: Measurement noise covariance (how noisy the observations are).

    Returns:
        Smoothed estimate ``Expr`` (Float64 array).
    """
    src = source if source is not None else close
    return scan(
        state={"x": src, "p": lit(1.0)},
        update={
            "p_pred": s.prev("p") + _coerce(q),
            "k": s.var("p_pred") / (s.var("p_pred") + _coerce(r)),
            "x": s.prev("x") + s.var("k") * (src - s.prev("x")),
            "p": (lit(1.0) - s.var("k")) * s.var("p_pred"),
        },
        output="x",
    )


def garch(source: Expr = None, omega: float = 1e-6, alpha: float = 0.1, beta: float = 0.85) -> Expr:
    """GARCH(1,1) conditional volatility estimator.

    Uses the ``scan`` primitive — runs entirely in Rust.

    Args:
        source: Return series. Defaults to ``close.pct_change(1)``.
        omega: Long-run variance weight.
        alpha: Weight on lagged squared return (ARCH term).
        beta: Weight on lagged conditional variance (GARCH term).

    Returns:
        Conditional standard deviation ``Expr`` (Float64 array).
    """
    src = source if source is not None else close.pct_change(1)
    return scan(
        state={"sigma2": lit(omega / (1.0 - alpha - beta)), "ret": src},
        update={
            "ret": src,
            "sigma2": _coerce(omega)
                + _coerce(alpha) * s.prev("ret") * s.prev("ret")
                + _coerce(beta) * s.prev("sigma2"),
            "sigma": Expr("Function", "sqrt", [s.var("sigma2")]),
        },
        output="sigma",
    )


# ---------------------------------------------------------------------------
# Directional movement / Aroon
# ---------------------------------------------------------------------------


def plus_di(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Wilder's +DI (native Rust).

    The bullish half of the ADX, on the same window, so ``plus_di(n)`` and
    ``minus_di(n)`` are exactly what ``adx(n)`` combines internally.

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    return Expr("PlusDi", h or high, l or low, c or close, _resolve_period(period))


def minus_di(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Wilder's -DI (native Rust). The bearish half of the ADX.

    Args:
        h, l, c: Custom high/low/close columns. Defaults to native bar columns.
    """
    return Expr("MinusDi", h or high, l or low, c or close, _resolve_period(period))


def aroon_up(period: int = 25, *, h: Expr = None) -> Expr:
    """Aroon Up (native Rust): how recent the window's high is, as a percent.

    ``100`` when the current bar is the highest of the trailing ``period + 1``
    bars, ``0`` when that high is ``period`` bars old.
    """
    return Expr("AroonUp", h or high, _resolve_period(period))


def aroon_down(period: int = 25, *, l: Expr = None) -> Expr:
    """Aroon Down (native Rust): how recent the window's low is, as a percent."""
    return Expr("AroonDown", l or low, _resolve_period(period))


def aroon_oscillator(period: int = 25, *, h: Expr = None, l: Expr = None) -> Expr:
    """Aroon Oscillator: ``aroon_up - aroon_down``, in [-100, 100]."""
    return aroon_up(period, h=h) - aroon_down(period, l=l)


def ppo(source: Expr = None, fast: int = 12, slow: int = 26) -> Expr:
    """Percentage Price Oscillator: the MACD line as a percent of the slow EMA."""
    src = source if source is not None else close
    fast_ema = src.ewm_mean(fast)
    slow_ema = src.ewm_mean(slow)
    return (fast_ema - slow_ema) / (slow_ema + lit(1e-12)) * lit(100.0)


def trix(source: Expr = None, period: int = 15) -> Expr:
    """TRIX: rate of change of a triple-smoothed EMA."""
    src = source if source is not None else close
    return src.ewm_mean(period).ewm_mean(period).ewm_mean(period).roc(1)


def stoch_rsi(source: Expr = None, period: int = 14, rsi_period: int = 14) -> Expr:
    """Stochastic RSI: the stochastic oscillator applied to the RSI, in [0, 1]."""
    r = rsi(source if source is not None else close, rsi_period)
    lowest = r.rolling_min(period)
    highest = r.rolling_max(period)
    return (r - lowest) / (highest - lowest + lit(1e-12))


def stoch_d(period: int = 14, d_period: int = 3,
            *, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """Stochastic %D: a ``d_period``-bar SMA of the native stochastic %K."""
    return stoch_k(period, h=h, l=l, c=c).rolling_mean(d_period)


# ---------------------------------------------------------------------------
# Channels and extra volatility
# ---------------------------------------------------------------------------


def donchian_channels(period: int = 20, *, h: Expr = None, l: Expr = None) -> Tuple[Expr, Expr, Expr]:
    """Donchian Channels.

    Returns:
        ``(upper, middle, lower)`` — the rolling high, their midpoint, the
        rolling low.
    """
    upper = (h or high).rolling_max(period)
    lower = (l or low).rolling_min(period)
    return MultiExpr(
        (upper, (upper + lower) / lit(2.0), lower), "donchian_channels", _BAND_NAMES
    )


def vortex(period: int = 14, *, h: Expr = None, l: Expr = None, c: Expr = None) -> Tuple[Expr, Expr]:
    """Vortex Indicator.

    Returns:
        ``(vi_plus, vi_minus)`` — the two directional movement sums normalized
        by the true-range sum over the same window.
    """
    eps = lit(1e-12)
    tr_sum = true_range(h=h, l=l, c=c).rolling_sum(period)
    vm_plus = abs_val((h or high) - (l or low).lag(1))
    vm_minus = abs_val((l or low) - (h or high).lag(1))
    return MultiExpr(
        (
            vm_plus.rolling_sum(period) / (tr_sum + eps),
            vm_minus.rolling_sum(period) / (tr_sum + eps),
        ),
        "vortex",
        ("vi_plus", "vi_minus"),
    )


def cmf(period: int = 20, *, h: Expr = None, l: Expr = None, c: Expr = None, v: Expr = None) -> Expr:
    """Chaikin Money Flow: volume-weighted close position over the window."""
    _h, _l, _c, _v = h or high, l or low, c or close, v or volume
    eps = lit(1e-12)
    mfm = ((_c - _l) - (_h - _c)) / (_h - _l + eps)
    return (mfm * _v).rolling_sum(period) / (_v.rolling_sum(period) + eps)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def rolling_var(source: Expr, window: int) -> Expr:
    """Rolling population variance (native Rust): divides by the window, not n-1."""
    return source.rolling_var(window)


def rolling_skew(source: Expr, window: int) -> Expr:
    """Rolling sample skewness (native Rust), matching pandas ``.skew()``.

    Needs ``window >= 3``; a constant window is NaN.
    """
    return source.rolling_skew(window)


def rolling_kurt(source: Expr, window: int) -> Expr:
    """Rolling excess kurtosis (native Rust), matching pandas ``.kurt()``.

    Needs ``window >= 4``; a constant window is NaN.
    """
    return source.rolling_kurt(window)


def rolling_rank(source: Expr, window: int) -> Expr:
    """Rolling percent-rank of the current value in the window, in [0, 1]."""
    return source.rolling_rank(window)


def rolling_quantile(source: Expr, window: int, q: float = 0.5) -> Expr:
    """Rolling q-quantile over the window (native Rust, linear interpolation)."""
    return source.rolling_quantile(window, q)


def rolling_argmax(source: Expr, window: int) -> Expr:
    """Bars since the window's maximum (native Rust). 0 = the current bar."""
    return source.rolling_argmax(window)


def rolling_argmin(source: Expr, window: int) -> Expr:
    """Bars since the window's minimum (native Rust). 0 = the current bar."""
    return source.rolling_argmin(window)


def rolling_corr(a: Expr, b: Expr, window: int) -> Expr:
    """Rolling Pearson correlation of ``a`` and ``b`` (native Rust)."""
    return a.rolling_corr(b, window)


def rolling_cov(a: Expr, b: Expr, window: int) -> Expr:
    """Rolling sample covariance, ddof=1 (native Rust)."""
    return a.rolling_cov(b, window)


def rolling_beta(y: Expr, x: Expr, window: int) -> Expr:
    """Rolling OLS beta of ``y`` on ``x`` (native Rust): ``cov(y, x) / var(x)``."""
    return y.rolling_beta(x, window)


# ---------------------------------------------------------------------------
# Signal state (Pine-style)
# ---------------------------------------------------------------------------


def bars_since(condition: Expr) -> Expr:
    """Bars since ``condition`` was last true (native Rust).

    ``0`` on a bar where it is true, ``1`` one bar later; NaN until it first is.
    """
    return condition.bars_since()


def streak(condition: Expr) -> Expr:
    """Length of the current consecutive run of true ending at this bar."""
    return condition.streak()


def count_over(condition: Expr, window: int) -> Expr:
    """Count of bars where ``condition`` is true in the trailing window."""
    return condition.count_over(window)


def value_when(condition: Expr, source: Expr) -> Expr:
    """Value of ``source`` on the last bar (<= now) where ``condition`` was true."""
    return condition.value_when(source)


def rising(source: Expr, n: int) -> Expr:
    """1.0 if ``source`` strictly increased on each of the last ``n`` steps."""
    return source.rising(n)


def falling(source: Expr, n: int) -> Expr:
    """1.0 if ``source`` strictly decreased on each of the last ``n`` steps."""
    return source.falling(n)


def pivot_high(source: Expr, left: int, right: int) -> Expr:
    """Causal pivot high (native Rust) — NO lookahead.

    The value appears on the confirmation bar, ``right`` bars after the pivot
    itself; that lag is the price of the signal being tradable.
    """
    return source.pivot_high(left, right)


def pivot_low(source: Expr, left: int, right: int) -> Expr:
    """Causal pivot low (native Rust) — NO lookahead. See :func:`pivot_high`."""
    return source.pivot_low(left, right)


# ---------------------------------------------------------------------------
# Cross-sectional (multi-asset)
# ---------------------------------------------------------------------------
#
# These read the whole universe at a timestamp, so their argument must be a
# column or a named signal, not a sub-expression — define the sub-expression as
# its own signal first.


def cs_zscore(source: Expr) -> Expr:
    """Cross-sectional z-score across symbols, on the population std."""
    return source.cs_zscore()


def cs_demean(source: Expr) -> Expr:
    """Cross-sectional demeaning: ``v - cross_mean`` per timestamp."""
    return source.cs_demean()


def cs_std(source: Expr) -> Expr:
    """Cross-sectional population std, broadcast to every symbol."""
    return source.cs_std()


def cs_scale(source: Expr) -> Expr:
    """Cross-sectional L1 (unit-gross) scaling: ``v / sum(|v_j|)``."""
    return source.cs_scale()


def cs_winsorize(source: Expr, k: float = 3.0) -> Expr:
    """Clip each value to ``[mean - k*std, mean + k*std]`` across symbols."""
    return source.cs_winsorize(k)


def cs_quantile(source: Expr, q: float) -> Expr:
    """Cross-sectional q-quantile, broadcast to every symbol."""
    return source.cs_quantile(q)


def cs_neutralize(source: Expr, factor: Expr) -> Expr:
    """Cross-sectional OLS residual of ``source`` regressed on ``factor``.

    Residuals are orthogonal to the factor and sum to zero across the symbols
    that participate (both series finite at that timestamp). A factor with no
    cross-sectional variance degenerates to plain demeaning.
    """
    return source.cs_neutralize(factor)


# ---------------------------------------------------------------------------
# Calendar components
# ---------------------------------------------------------------------------


def year(source: Expr = None) -> Expr:
    """Extract the full year (e.g. 2024) from a timestamp column (UTC)."""
    return (source if source is not None else timestamp).year()


def week_of_year(source: Expr = None) -> Expr:
    """Extract the ISO-8601 week number (1-53) from a timestamp column (UTC).

    ISO weeks belong to the year of their Thursday, so the 1st of January can
    land in week 52 or 53 of the previous year. That is the definition, not a bug.
    """
    return (source if source is not None else timestamp).week_of_year()


def day_of_year(source: Expr = None) -> Expr:
    """Extract the ordinal day of year (1-366) from a timestamp column (UTC)."""
    return (source if source is not None else timestamp).day_of_year()


def is_month_start(source: Expr = None) -> Expr:
    """1.0 on the first day of the month, else 0.0 (UTC)."""
    return (source if source is not None else timestamp).is_month_start()


def is_month_end(source: Expr = None) -> Expr:
    """1.0 on the last day of the month, else 0.0 (UTC)."""
    return (source if source is not None else timestamp).is_month_end()


def is_quarter_end(source: Expr = None) -> Expr:
    """1.0 on the last day of a calendar quarter (Mar/Jun/Sep/Dec), else 0.0."""
    return (source if source is not None else timestamp).is_quarter_end()


def is_weekend(source: Expr = None) -> Expr:
    """1.0 on Saturday or Sunday, else 0.0 (UTC).

    Usage::

        weekday_only = is_weekend() == 0
    """
    return (source if source is not None else timestamp).is_weekend()


# ---------------------------------------------------------------------------
# TA-Lib Math Transform
# ---------------------------------------------------------------------------
#
# Element-wise, zero lookback. Out-of-domain inputs are left as TA-Lib leaves
# them: asin/acos outside [-1, 1] give NaN, log10(0) gives -inf.


def sin(x: Expr) -> Expr:
    """Sine (element-wise)."""
    return Expr("Function", "sin", [_coerce(x)])


def cos(x: Expr) -> Expr:
    """Cosine (element-wise)."""
    return Expr("Function", "cos", [_coerce(x)])


def tan(x: Expr) -> Expr:
    """Tangent (element-wise)."""
    return Expr("Function", "tan", [_coerce(x)])


def asin(x: Expr) -> Expr:
    """Arcsine (element-wise). NaN outside [-1, 1]."""
    return Expr("Function", "asin", [_coerce(x)])


def acos(x: Expr) -> Expr:
    """Arccosine (element-wise). NaN outside [-1, 1]."""
    return Expr("Function", "acos", [_coerce(x)])


def atan(x: Expr) -> Expr:
    """Arctangent (element-wise)."""
    return Expr("Function", "atan", [_coerce(x)])


def sinh(x: Expr) -> Expr:
    """Hyperbolic sine (element-wise)."""
    return Expr("Function", "sinh", [_coerce(x)])


def cosh(x: Expr) -> Expr:
    """Hyperbolic cosine (element-wise)."""
    return Expr("Function", "cosh", [_coerce(x)])


def log10(x: Expr) -> Expr:
    """Base-10 logarithm (element-wise)."""
    return Expr("Function", "log10", [_coerce(x)])


# ---------------------------------------------------------------------------
# TA-Lib Price Transform
# ---------------------------------------------------------------------------


def median_price(*, h: Expr = None, l: Expr = None) -> Expr:
    """TA-Lib ``MEDPRICE``: ``(high + low) / 2``."""
    return Expr("Function", "median_price", [h or high, l or low])


def typical_price(*, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """TA-Lib ``TYPPRICE``: ``(high + low + close) / 3``."""
    return Expr("Function", "typical_price", [h or high, l or low, c or close])


def weighted_close(*, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """TA-Lib ``WCLPRICE``: ``(high + low + 2*close) / 4``."""
    return Expr("Function", "weighted_close", [h or high, l or low, c or close])


def average_price(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """TA-Lib ``AVGPRICE``: ``(open + high + low + close) / 4``."""
    return Expr("Function", "average_price", [o or open, h or high, l or low, c or close])


# ---------------------------------------------------------------------------
# TA-Lib Pattern Recognition
# ---------------------------------------------------------------------------
#
# 38 of TA-Lib's 61 ``CDL*`` functions, bit-exact against TA-Lib 0.7.1 (pinned by
# ``crates/bt-expr/tests/talib_candles.rs``). Each returns one of
# ``{-100, -80, 0, 80, 100}``: the sign is the direction, the magnitude is
# TA-Lib's confidence, and warmup bars are **0** rather than NaN because a
# detector's 0 already means "no pattern here".


def _cdl(name: str, o: Expr, h: Expr, l: Expr, c: Expr) -> Expr:
    """Build a candlestick pattern call over the four OHLC columns."""
    return Expr("Function", name, [o or open, h or high, l or low, c or close])


def cdl_doji(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLDOJI``: body no larger than the doji share of the high-low range."""
    return _cdl("cdl_doji", o, h, l, c)


def cdl_spinning_top(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLSPINNINGTOP``: small body with both shadows longer than it."""
    return _cdl("cdl_spinning_top", o, h, l, c)


def cdl_long_legged_doji(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLLONGLEGGEDDOJI``: doji body with at least one long shadow."""
    return _cdl("cdl_long_legged_doji", o, h, l, c)


def cdl_short_line(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLSHORTLINE``: short body, both shadows short."""
    return _cdl("cdl_short_line", o, h, l, c)


def cdl_long_line(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLLONGLINE``: long body, both shadows short."""
    return _cdl("cdl_long_line", o, h, l, c)


def cdl_high_wave(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLHIGHWAVE``: short body with both shadows very long."""
    return _cdl("cdl_high_wave", o, h, l, c)


def cdl_rickshaw_man(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLRICKSHAWMAN``: long-legged doji straddling the range midpoint."""
    return _cdl("cdl_rickshaw_man", o, h, l, c)


def cdl_marubozu(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLMARUBOZU``: long body, both shadows very short."""
    return _cdl("cdl_marubozu", o, h, l, c)


def cdl_closing_marubozu(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLCLOSINGMARUBOZU``: long body, very short shadow on the close side."""
    return _cdl("cdl_closing_marubozu", o, h, l, c)


def cdl_belt_hold(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLBELTHOLD``: long body, very short shadow on the *open* side."""
    return _cdl("cdl_belt_hold", o, h, l, c)


def cdl_dragonfly_doji(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLDRAGONFLYDOJI``: doji with a very short upper and long lower shadow."""
    return _cdl("cdl_dragonfly_doji", o, h, l, c)


def cdl_gravestone_doji(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLGRAVESTONEDOJI``: doji with a very short lower and long upper shadow."""
    return _cdl("cdl_gravestone_doji", o, h, l, c)


def cdl_engulfing(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLENGULFING``: this body swallows the previous one."""
    return _cdl("cdl_engulfing", o, h, l, c)


def cdl_hammer(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLHAMMER``: small body, long lower shadow, at or below the prior low."""
    return _cdl("cdl_hammer", o, h, l, c)


def cdl_inverted_hammer(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLINVERTEDHAMMER``: gapped-down small body with a long upper shadow."""
    return _cdl("cdl_inverted_hammer", o, h, l, c)


def cdl_hanging_man(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLHANGINGMAN``: a hammer's shape at or above the previous high."""
    return _cdl("cdl_hanging_man", o, h, l, c)


def cdl_shooting_star(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLSHOOTINGSTAR``: gapped-up small body with a long upper shadow."""
    return _cdl("cdl_shooting_star", o, h, l, c)


def cdl_takuri(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLTAKURI``: a dragonfly doji whose lower shadow is *very* long."""
    return _cdl("cdl_takuri", o, h, l, c)


def cdl_matching_low(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLMATCHINGLOW``: two black candles closing at the same level."""
    return _cdl("cdl_matching_low", o, h, l, c)


def cdl_homing_pigeon(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLHOMINGPIGEON``: a small black body inside the previous long black."""
    return _cdl("cdl_homing_pigeon", o, h, l, c)


def cdl_harami(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLHARAMI``: a small body contained within the previous long body."""
    return _cdl("cdl_harami", o, h, l, c)


def cdl_harami_cross(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLHARAMICROSS``: a harami whose inside bar is a doji."""
    return _cdl("cdl_harami_cross", o, h, l, c)


def cdl_doji_star(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLDOJISTAR``: a doji gapping away from the previous long body."""
    return _cdl("cdl_doji_star", o, h, l, c)


def cdl_piercing(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLPIERCING``: a long white opening below the prior low, closing past its midpoint."""
    return _cdl("cdl_piercing", o, h, l, c)


def cdl_thrusting(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLTHRUSTING``: a piercing that fails, closing at or below the midpoint."""
    return _cdl("cdl_thrusting", o, h, l, c)


def cdl_counterattack(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLCOUNTERATTACK``: two long opposite-coloured bodies closing level."""
    return _cdl("cdl_counterattack", o, h, l, c)


def cdl_three_inside(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDL3INSIDE``: a harami then a close beyond the first body."""
    return _cdl("cdl_three_inside", o, h, l, c)


def cdl_three_outside(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDL3OUTSIDE``: an engulfing then a confirming close."""
    return _cdl("cdl_three_outside", o, h, l, c)


def cdl_morning_star(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLMORNINGSTAR``: long black, a gapped-down small body, then a long white."""
    return _cdl("cdl_morning_star", o, h, l, c)


def cdl_evening_star(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLEVENINGSTAR``: the mirror of the morning star."""
    return _cdl("cdl_evening_star", o, h, l, c)


def cdl_dark_cloud_cover(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLDARKCLOUDCOVER``: a black opening above the prior high, closing well inside."""
    return _cdl("cdl_dark_cloud_cover", o, h, l, c)


def cdl_three_white_soldiers(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDL3WHITESOLDIERS``: three rising white bodies with short upper shadows."""
    return _cdl("cdl_three_white_soldiers", o, h, l, c)


def cdl_two_crows(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDL2CROWS``: a long white, a gapped-up black, then a black closing back inside."""
    return _cdl("cdl_two_crows", o, h, l, c)


def cdl_identical_three_crows(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLIDENTICAL3CROWS``: three declining blacks each opening at the previous close."""
    return _cdl("cdl_identical_three_crows", o, h, l, c)


def cdl_tristar(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLTRISTAR``: three dojis, the middle one gapped away from both."""
    return _cdl("cdl_tristar", o, h, l, c)


def cdl_separating_lines(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLSEPARATINGLINES``: opposite colours sharing an open, the second long."""
    return _cdl("cdl_separating_lines", o, h, l, c)


def cdl_on_neck(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLONNECK``: a white opening below a long black's low, closing back at it."""
    return _cdl("cdl_on_neck", o, h, l, c)


def cdl_kicking(*, o: Expr = None, h: Expr = None, l: Expr = None, c: Expr = None) -> Expr:
    """``CDLKICKING``: two opposite-coloured marubozu separated by a gap."""
    return _cdl("cdl_kicking", o, h, l, c)
