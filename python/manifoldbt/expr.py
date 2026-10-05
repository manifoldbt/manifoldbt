"""Expression AST builder — the core of the Python DSL.

Builds an expression tree that serializes to the externally-tagged JSON
format the engine reads.
"""
from __future__ import annotations

import builtins
import warnings

from typing import Any, Dict, Optional, Union

from manifoldbt._serde import scalar_value_to_json

Numeric = Union[int, float, "Expr"]
Period = Union[int, "Expr", Dict[str, int]]
Span = Union[float, int, "Expr"]


class _ParamRef(str):
    """The name of a ``param()`` passed as an indicator period or span.

    Such an argument serialises as the bare name (``DynPeriod::Param`` /
    ``DynFloat::Param`` on the Rust side), so the ``Parameter`` node that held
    the default is gone once the indicator is built. The name stays a ``str``,
    and the JSON is byte for byte what it was, but it carries the metadata of
    the ``param()`` it came from: the strategy that holds the expression reads
    its own default here.

    It replaces a module-level registry, keyed by name, where the last
    ``param()`` built anywhere in the process won. A strategy built with
    ``param("slow", default=20)`` then serialised after a variant built with
    ``default=41`` ran with 41, in silence.
    """

    __slots__ = ("_param_meta",)

    def __new__(cls, name: str, meta: Dict[str, Any]) -> "_ParamRef":
        self = super().__new__(cls, name)
        self._param_meta = meta
        return self

    def __reduce__(self) -> Any:
        # copy, deepcopy and pickle rebuild it with its metadata, whatever the
        # protocol, instead of falling back to a plain str.
        return (_ParamRef, (str(self), self._param_meta))


def _param_name(value: "Expr") -> Any:
    """What a ``param()`` period or span serialises as: its name, carrying its
    metadata when it has some."""
    name = value._args[0]
    if value._param_meta is None:
        return name
    return _ParamRef(name, value._param_meta)


# Nanoseconds per unit of an ``Interval`` (see manifoldbt.helpers.Interval).
_NANOS_PER_UNIT = {
    "Millis": 1_000_000,
    "Seconds": 1_000_000_000,
    "Minutes": 60_000_000_000,
    "Hours": 3_600_000_000_000,
    "Days": 86_400_000_000_000,
}


def _interval_to_nanos(value: Any) -> Optional[int]:
    """Nanoseconds behind an ``Interval`` dict, or None if it is not one."""
    if not isinstance(value, dict) or len(value) != 1:
        return None
    unit, count = next(iter(value.items()))
    per = _NANOS_PER_UNIT.get(unit)
    if per is None:
        return None
    return int(count) * per


def _resolve_period(value: Period, allow_duration: bool = False) -> Any:
    """Convert a period argument for DynPeriod serialization.

    - int → int (serializes as JSON number → DynPeriod::Fixed)
    - param("name") Expr → "name" (serializes as JSON string → DynPeriod::Param)
    - Interval.seconds(30) → {"duration_ns": ...} → DynPeriod::Duration,
      accepted only by the operators that implement a window in time.
    """
    # `isinstance` first, and not merely for tidiness: `Expr.__eq__` builds a
    # comparison expression rather than answering a bool, so `value == "Trades"`
    # on a `param()` is a truthy object and would raise on every swept period.
    if isinstance(value, str) and value == "Trades":
        raise TypeError(
            "Interval.trades() is a simulation clock, not a window: it says how "
            "often the engine steps, not how far back an indicator looks. Pass "
            "an integer to count EVENTS (rolling_sum(34) = thirty-four prints) "
            "or Interval.seconds(n) to count time (rolling_sum(Interval."
            "seconds(34)) = thirty-four seconds of prints)."
        )
    nanos = _interval_to_nanos(value)
    if nanos is not None:
        if not allow_duration:
            raise TypeError(
                "this operator counts BARS, not time: pass an integer. "
                "Windows in time are available on rolling_mean, rolling_sum, "
                "rolling_std, rolling_min, rolling_max, zscore, rolling_var, "
                "rolling_corr, rolling_cov, rolling_beta, count_over, and on "
                "ewm_mean(halflife=...)"
            )
        if nanos <= 0:
            raise ValueError("a window in time must be a positive duration")
        return {"duration_ns": nanos}
    if isinstance(value, dict):
        raise TypeError(
            f"unknown window {value!r}: use an integer, param(...), or "
            "Interval.seconds/minutes/hours/days(n)"
        )
    if isinstance(value, Expr) and value._variant == "Parameter":
        return _param_name(value)
    if isinstance(value, Expr):
        raise TypeError("Only param() expressions can be used as indicator periods, not arbitrary expressions")
    return int(value)


def _resolve_span(value: Span) -> Any:
    """Convert a span/float argument for DynFloat serialization."""
    if isinstance(value, Expr) and value._variant == "Parameter":
        return _param_name(value)
    if isinstance(value, Expr):
        raise TypeError("Only param() expressions can be used as indicator spans, not arbitrary expressions")
    return float(value)


def _round_step(step: Any, what: str) -> float:
    """Check a price-grid step here, where the message can name the caller.

    The engine refuses the same thing at compile time; catching it in Python
    turns a backtest that fails on its first bar into a TypeError/ValueError on
    the line that wrote it. A ``param()`` is refused outright: the step is read
    once as an exact decimal, so it cannot be an axis of a sweep.
    """
    if isinstance(step, Expr):
        raise TypeError(
            f"{what}(step=...) takes a literal number, not an expression: the step is "
            "read once as the exact decimal it is written as, so it cannot be a "
            "param() axis"
        )
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise TypeError(f"{what}(step=...) takes a number, got {type(step).__name__}")
    step = float(step)
    if step != step or step in (float("inf"), float("-inf")):
        raise ValueError(f"{what}(step=...) must be a finite number, got {step}")
    if step <= 0.0:
        raise ValueError(f"{what}(step=...) must be strictly positive, got {step}")
    return step


# Variants that wrap a single Box<Expr>
_UNARY_BOX = frozenset(
    [
        "Not", "CumSum", "CumProd", "Rank", "CrossSectionalMean", "CrossSectionalRank",
        "Hour", "Minute", "DayOfWeek", "Month", "DayOfMonth",
        # Coupe transversale (multi-actif)
        "CsZScore", "CsDemean", "CsStd", "CsScale",
        # Etat de signal (l'argument est une CONDITION)
        "BarsSince", "TimeSince", "Streak",
        # Etat de signal dont l'argument est une SERIE
        "Ffill",
        # Composants calendaires
        "Year", "WeekOfYear", "DayOfYear",
        "IsMonthStart", "IsMonthEnd", "IsQuarterEnd", "IsWeekend",
    ]
)

# Variants with two Box<Expr>
_BINARY_BOX = frozenset(["Add", "Sub", "Mul", "Div", "Gt", "Lt", "Eq", "And", "Or"])

# Variants with Box<Expr> + usize (or f64 for EwmMean)
_EXPR_SCALAR = frozenset(
    [
        "Lag",
        "Lead",
        "RollingMean",
        "RollingStd",
        "RollingSum",
        "RollingMin",
        "RollingMax",
        "EwmMean",
        # Box<Expr> + DurationNs, serialise en {"duration_ns": n}
        "EwmMeanHalflife",
        "Diff",
        "PctChange",
        "ZScore",
        "Rsi",
        "LinRegSlope",
        "LinRegValue",
        "LinRegR2",
        # New indicators (Box<Expr>, usize)
        "Dema",
        "Tema",
        "Wma",
        "Hma",
        "Kama",
        "Roc",
        "RollingMedian",
        # Statistiques glissantes (Box<Expr>, usize)
        "RollingVar",
        "RollingSkew",
        "RollingKurt",
        "RollingRank",
        "RollingArgMax",
        "RollingArgMin",
        # Aroon et etat de signal (Box<Expr>, usize)
        "AroonUp",
        "AroonDown",
        "CountOver",
        "Rising",
        "Falling",
        # Coupe transversale (Box<Expr>, f64)
        "CsWinsorize",
        "CsQuantile",
        # Grille de prix (Box<Expr>, f64) -- le pas est un LITTERAL, pas un
        # param() : il est lu une fois a la compilation comme le decimal qu'il
        # ecrit.
        "RoundTo",
        "FloorTo",
        "CeilTo",
    ]
)

# Box<Expr> + usize + usize (ou usize + f64 pour RollingQuantile)
_EXPR_2SCALAR = frozenset(["Macd", "PivotHigh", "PivotLow", "RollingQuantile"])

# Box<Expr> + usize + usize + usize
_EXPR_3SCALAR = frozenset(["MacdSignal", "MacdHist"])

# Box<Expr> + usize + f64
_EXPR_SCALAR_F64 = frozenset(["BollingerUpper", "BollingerLower", "BollingerWidth"])

# 3×Box<Expr> (no extra scalar)
_HLC_NO_SCALAR = frozenset(["TrueRange"])

# 3×Box<Expr> + usize — same layout as Atr
_HLC_USIZE = frozenset(["StochK", "WilliamsR", "Cci", "Adx", "Natr", "PlusDi", "MinusDi"])

# 3×Box<Expr> + usize + f64
_HLC_USIZE_F64 = frozenset(["KeltnerUpper", "KeltnerLower", "SuperTrend"])

# 2×Box<Expr>
_BINARY_EXPR = frozenset(["Obv", "CrossAbove", "CrossBelow", "ValueWhen", "CsNeutralize"])

# 2×Box<Expr> + usize — statistiques glissantes de paire
_BINARY_EXPR_USIZE = frozenset(["RollingCorr", "RollingCov", "RollingBeta"])

# 4×Box<Expr>
_HLCV_NO_SCALAR = frozenset(["Vwap", "AdLine"])

# 4×Box<Expr> + usize
_HLCV_USIZE = frozenset(["Mfi"])


class Expr:
    """AST node representing a backtester expression."""

    __slots__ = ("_variant", "_args", "_param_meta")
    __hash__ = None  # not hashable (we override __eq__)

    def __init__(self, variant: str, *args: Any) -> None:
        self._variant = variant
        self._args = args
        self._param_meta = None

    # -- Serialization -------------------------------------------------------

    def to_json(self) -> Any:
        """Serialize to a dict/value matching Rust ``Expr`` serde format."""
        v = self._variant
        args = self._args

        if v in _UNARY_BOX:
            return {v: args[0].to_json()}

        if v in _BINARY_BOX:
            return {v: [args[0].to_json(), args[1].to_json()]}

        if v in _EXPR_SCALAR:
            return {v: [args[0].to_json(), args[1]]}

        if v in _EXPR_2SCALAR:
            # e.g. Macd(Box<Expr>, usize, usize)
            return {v: [args[0].to_json(), args[1], args[2]]}

        if v in _EXPR_3SCALAR:
            # e.g. MacdSignal(Box<Expr>, usize, usize, usize)
            return {v: [args[0].to_json(), args[1], args[2], args[3]]}

        if v in _EXPR_SCALAR_F64:
            # e.g. BollingerUpper(Box<Expr>, usize, f64)
            return {v: [args[0].to_json(), args[1], args[2]]}

        if v in _HLC_NO_SCALAR:
            # e.g. TrueRange(Box<Expr>, Box<Expr>, Box<Expr>)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json()]}

        if v == "Atr" or v in _HLC_USIZE:
            # Atr/StochK/WilliamsR/Cci/Adx/Natr(Box<Expr>, Box<Expr>, Box<Expr>, usize)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json(), args[3]]}

        if v in _HLC_USIZE_F64:
            # KeltnerUpper/KeltnerLower/SuperTrend(Box<Expr>, Box<Expr>, Box<Expr>, usize, f64)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json(), args[3], args[4]]}

        if v in _BINARY_EXPR:
            # Obv/CrossAbove/CrossBelow(Box<Expr>, Box<Expr>)
            return {v: [args[0].to_json(), args[1].to_json()]}

        if v in _BINARY_EXPR_USIZE:
            # RollingCorr/RollingCov/RollingBeta(Box<Expr>, Box<Expr>, usize)
            return {v: [args[0].to_json(), args[1].to_json(), args[2]]}

        if v in _HLCV_NO_SCALAR:
            # Vwap/AdLine(Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json(), args[3].to_json()]}

        if v in _HLCV_USIZE:
            # Mfi(Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>, usize)
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json(), args[3].to_json(), args[4]]}

        if v == "ParabolicSar":
            # ParabolicSar(Box<Expr>, Box<Expr>, f64, f64)
            return {v: [args[0].to_json(), args[1].to_json(), args[2], args[3]]}

        if v == "IfElse":
            return {v: [args[0].to_json(), args[1].to_json(), args[2].to_json()]}

        if v == "Choice":
            # Choice(String, Vec<(String, Expr)>) -- serde attend une liste de
            # paires, pas un dict : l'ORDRE des branches est porteur (la
            # premiere sert de defaut a la compilation initiale).
            return {"Choice": [args[0], [[k, e.to_json()] for k, e in args[1]]]}

        if v == "OnTimeframe":
            # OnTimeframe(String, Box<Expr>) -- l'expression est evaluee sur la
            # grille de la timeframe nommee puis etalee en escalier.
            return {"OnTimeframe": [args[0], args[1].to_json()]}

        if v == "Position":
            # Variante unitaire cote Rust : une chaine nue, pas un objet.
            return "Position"

        if v == "SimState":
            # SimState(SimStateRef) : ("Cash",) -> "Cash" ; (kind, side) ->
            # {kind: side} ; ("BookPriceAt", side, level) -> un objet.
            kind = args[0]
            if kind in ("Cash", "PositionAge"):
                return {"SimState": kind}
            if kind == "BookPriceAt":
                return {"SimState": {"BookPriceAt": {"side": args[1], "level": args[2].to_json()}}}
            return {"SimState": {kind: args[1]}}

        if v == "Column":
            return {"Column": args[0]}
        if v == "Literal":
            return {"Literal": scalar_value_to_json(args[0])}
        if v == "Parameter":
            return {"Parameter": args[0]}

        if v == "Function":
            return {"Function": [args[0], [a.to_json() for a in args[1]]]}

        if v == "SymbolRef":
            return {"SymbolRef": [args[0], args[1].to_json()]}

        if v == "Scan":
            state_names, init_exprs, update_names, update_exprs, output = args
            return {
                "Scan": {
                    "state_names": list(state_names),
                    "init_exprs": [e.to_json() for e in init_exprs],
                    "update_names": list(update_names),
                    "update_exprs": [e.to_json() for e in update_exprs],
                    "output": output,
                }
            }
        if v == "ScanPrev":
            return {"ScanPrev": args[0]}
        if v == "ScanVar":
            return {"ScanVar": args[0]}

        raise ValueError(f"Unknown Expr variant: {v}")

    # -- Arithmetic operators ------------------------------------------------

    def __add__(self, other: Numeric) -> Expr:
        return Expr("Add", self, _coerce(other))

    def __radd__(self, other: Numeric) -> Expr:
        return Expr("Add", _coerce(other), self)

    def __sub__(self, other: Numeric) -> Expr:
        return Expr("Sub", self, _coerce(other))

    def __rsub__(self, other: Numeric) -> Expr:
        return Expr("Sub", _coerce(other), self)

    def __mul__(self, other: Numeric) -> Expr:
        return Expr("Mul", self, _coerce(other))

    def __rmul__(self, other: Numeric) -> Expr:
        return Expr("Mul", _coerce(other), self)

    def __truediv__(self, other: Numeric) -> Expr:
        return Expr("Div", self, _coerce(other))

    def __rtruediv__(self, other: Numeric) -> Expr:
        return Expr("Div", _coerce(other), self)

    def __neg__(self) -> Expr:
        return Expr("Mul", Expr("Literal", -1.0), self)

    def __pow__(self, exponent: Any, modulo: Any = None) -> Expr:
        return _power(self, exponent, modulo)

    def __rpow__(self, base: Any, modulo: Any = None) -> Expr:
        raise TypeError(
            f"{base!r} ** expr is not supported: the engine has no power "
            "function, so only an expression raised to a literal integer works "
            "(x ** 2 is x * x). For a positive base write "
            "exp(expr * math.log(base)) with manifoldbt.indicators.exp."
        )

    # -- Truth value ---------------------------------------------------------

    def __bool__(self) -> bool:
        # An Expr is a series the engine evaluates later, bar by bar. It has
        # no truth value NOW, and answering one (every object is truthy by
        # default) let `max(expr, lit(0.1))` return an operand picked by its
        # position, with no error. Refusing is what numpy and polars do.
        raise TypeError(
            "an expression has no truth value in Python: it is a series the "
            "engine evaluates bar by bar, so `if`, `and`, `or`, `not`, a "
            "chained comparison (a < b < c), `in` and the builtins max()/min() "
            "cannot read it (they would silently pick one operand). Write "
            "instead: max_val(a, b) / min_val(a, b) from manifoldbt.indicators "
            "for an elementwise max/min; (a > 0) & (b < 1), | and ~ for "
            "and/or/not, each comparison in parentheses; (a < b) & (b < c) for "
            "a chained comparison; when(cond, x, y) for if/else; "
            "`x is None` to test whether an argument was given."
        )

    # -- Comparison operators ------------------------------------------------

    def __gt__(self, other: Numeric) -> Expr:
        return Expr("Gt", self, _coerce(other))

    def __lt__(self, other: Numeric) -> Expr:
        return Expr("Lt", self, _coerce(other))

    def __eq__(self, other: Numeric) -> Expr:  # type: ignore[override]
        return Expr("Eq", self, _coerce(other))

    def __ne__(self, other: Numeric) -> Expr:  # type: ignore[override]
        # Without it Python answered `not (self == other)`, a plain False
        # whatever the series held; now that an Expr has no truth value that
        # would raise instead. `~(a == b)` is what `!=` means on a series.
        return ~(self == other)

    def __ge__(self, other: Numeric) -> Expr:
        return (self > other) | (self == other)

    def __le__(self, other: Numeric) -> Expr:
        return (self < other) | (self == other)

    # -- Boolean operators ---------------------------------------------------

    def __and__(self, other: Expr) -> Expr:
        return Expr("And", self, other)

    def __or__(self, other: Expr) -> Expr:
        return Expr("Or", self, other)

    def __invert__(self) -> Expr:
        return Expr("Not", self)

    # -- Time-series methods -------------------------------------------------

    def lag(self, n: Period) -> Expr:
        return Expr("Lag", self, _resolve_period(n))

    def lead(self, n: Period) -> Expr:
        return Expr("Lead", self, _resolve_period(n))

    def diff(self, n: Period = 1) -> Expr:
        return Expr("Diff", self, _resolve_period(n))

    def pct_change(self, n: Period = 1) -> Expr:
        return Expr("PctChange", self, _resolve_period(n))

    def rolling_mean(self, window: Period) -> Expr:
        """Rolling mean over ``window``.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingMean", self, _resolve_period(window, allow_duration=True))

    def rolling_std(self, window: Period) -> Expr:
        """Rolling population standard deviation over ``window``.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingStd", self, _resolve_period(window, allow_duration=True))

    def rolling_sum(self, window: Period) -> Expr:
        """Rolling sum over ``window``.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingSum", self, _resolve_period(window, allow_duration=True))

    def rolling_min(self, window: Period) -> Expr:
        """Rolling minimum over ``window``.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingMin", self, _resolve_period(window, allow_duration=True))

    def rolling_max(self, window: Period) -> Expr:
        """Rolling maximum over ``window``.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingMax", self, _resolve_period(window, allow_duration=True))

    def ewm_mean(self, span: Optional[Span] = None, halflife: Any = None) -> Expr:
        """Exponential moving mean.

        Two forms, one per axis:

        - ``ewm_mean(span=20)`` counts BARS, and is the plain recurrence
          (``pandas.ewm(span=20, adjust=False)``);
        - ``ewm_mean(halflife=Interval.seconds(2))`` decays with the TIME
          elapsed between two rows: a row ``dt`` old weighs
          ``0.5 ** (dt / halflife)``. It equals
          ``pandas.ewm(halflife="2s", times=...).mean()``, the weighted-average
          form, which is the only one pandas offers on an irregular axis.

        The halflife form has no window, hence no warmup: the value exists from
        the first row, like the span form. A positional ``Interval`` is read as
        a halflife.
        """
        if halflife is None and _interval_to_nanos(span) is not None:
            span, halflife = None, span
        if (span is None) == (halflife is None):
            raise TypeError(
                "ewm_mean takes exactly one of span= (bars) or halflife= (a duration)"
            )
        if halflife is not None:
            nanos = _interval_to_nanos(halflife)
            if nanos is None:
                raise TypeError(
                    "ewm_mean(halflife=...) takes an Interval, e.g. Interval.seconds(2)"
                )
            if nanos <= 0:
                raise ValueError("an ewm halflife must be a positive duration")
            return Expr("EwmMeanHalflife", self, {"duration_ns": nanos})
        return Expr("EwmMean", self, _resolve_span(span))

    def zscore(self, window: Period) -> Expr:
        """Rolling z-score over ``window`` (population standard deviation).

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("ZScore", self, _resolve_period(window, allow_duration=True))

    def rsi(self, period: Period = 14) -> Expr:
        """Native Rust RSI (Wilder's smoothing, single-pass O(n))."""
        return Expr("Rsi", self, _resolve_period(period))

    def linreg_slope(self, window: Period) -> Expr:
        """Rolling linear regression slope (single-pass O(n))."""
        return Expr("LinRegSlope", self, _resolve_period(window))

    def linreg_value(self, window: Period) -> Expr:
        """Rolling linear regression predicted value at end of window."""
        return Expr("LinRegValue", self, _resolve_period(window))

    def linreg_r2(self, window: Period) -> Expr:
        """Rolling linear regression R-squared (single-pass O(n))."""
        return Expr("LinRegR2", self, _resolve_period(window))

    # -- New indicators (native Rust) ----------------------------------------

    def dema(self, period: Period) -> Expr:
        """Double Exponential Moving Average."""
        return Expr("Dema", self, _resolve_period(period))

    def tema(self, period: Period) -> Expr:
        """Triple Exponential Moving Average."""
        return Expr("Tema", self, _resolve_period(period))

    def wma(self, period: Period) -> Expr:
        """Weighted Moving Average."""
        return Expr("Wma", self, _resolve_period(period))

    def hma(self, period: Period) -> Expr:
        """Hull Moving Average."""
        return Expr("Hma", self, _resolve_period(period))

    def kama(self, period: Period) -> Expr:
        """Kaufman Adaptive Moving Average."""
        return Expr("Kama", self, _resolve_period(period))

    def roc(self, period: Period) -> Expr:
        """Rate of Change."""
        return Expr("Roc", self, _resolve_period(period))

    def rolling_median(self, window: Period) -> Expr:
        """Rolling median."""
        return Expr("RollingMedian", self, _resolve_period(window))

    def rolling_var(self, window: Period) -> Expr:
        """Rolling population variance (divides by the row count, not n-1).

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingVar", self, _resolve_period(window, allow_duration=True))

    def rolling_skew(self, window: Period) -> Expr:
        """Rolling sample skewness (adjusted Fisher-Pearson, like pandas)."""
        return Expr("RollingSkew", self, _resolve_period(window))

    def rolling_kurt(self, window: Period) -> Expr:
        """Rolling excess kurtosis (bias-corrected Fisher, like pandas)."""
        return Expr("RollingKurt", self, _resolve_period(window))

    def rolling_rank(self, window: Period) -> Expr:
        """Rolling percent-rank of the current value in the window, in [0, 1]."""
        return Expr("RollingRank", self, _resolve_period(window))

    def rolling_argmax(self, window: Period) -> Expr:
        """Bars since the window's maximum (0 = the current bar is the max)."""
        return Expr("RollingArgMax", self, _resolve_period(window))

    def rolling_argmin(self, window: Period) -> Expr:
        """Bars since the window's minimum (0 = the current bar is the min)."""
        return Expr("RollingArgMin", self, _resolve_period(window))

    def rolling_quantile(self, window: Period, q: Span = 0.5) -> Expr:
        """Rolling q-quantile over the window (linear interpolation)."""
        return Expr("RollingQuantile", self, _resolve_period(window), _resolve_span(q))

    def rolling_corr(self, other: "Expr", window: Period) -> Expr:
        """Rolling Pearson correlation with another series.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingCorr", self, _coerce(other),
                    _resolve_period(window, allow_duration=True))

    def rolling_cov(self, other: "Expr", window: Period) -> Expr:
        """Rolling sample covariance (ddof=1) with another series.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingCov", self, _coerce(other),
                    _resolve_period(window, allow_duration=True))

    def rolling_beta(self, other: "Expr", window: Period) -> Expr:
        """Rolling OLS beta of self on ``other``: ``cov(self, other) / var(other)``.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.
        """
        return Expr("RollingBeta", self, _coerce(other),
                    _resolve_period(window, allow_duration=True))

    def macd_line(self, fast: int = 12, slow: int = 26) -> Expr:
        """MACD line (fast EMA - slow EMA)."""
        return Expr("Macd", self, _resolve_period(fast), _resolve_period(slow))

    def macd_signal(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Expr:
        """MACD signal line."""
        return Expr("MacdSignal", self, _resolve_period(fast), _resolve_period(slow),
                    _resolve_period(signal))

    def macd_hist(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Expr:
        """MACD histogram."""
        return Expr("MacdHist", self, _resolve_period(fast), _resolve_period(slow),
                    _resolve_period(signal))

    def bollinger_upper(self, period: int = 20, num_std: float = 2.0) -> Expr:
        """Bollinger upper band."""
        return Expr("BollingerUpper", self, _resolve_period(period), _resolve_span(num_std))

    def bollinger_lower(self, period: int = 20, num_std: float = 2.0) -> Expr:
        """Bollinger lower band."""
        return Expr("BollingerLower", self, _resolve_period(period), _resolve_span(num_std))

    def bollinger_width(self, period: int = 20, num_std: float = 2.0) -> Expr:
        """Bollinger bandwidth."""
        return Expr("BollingerWidth", self, _resolve_period(period), _resolve_span(num_std))

    def cross_above(self, other: "Expr") -> Expr:
        """True when self crosses above other."""
        return Expr("CrossAbove", self, _coerce(other))

    def cross_below(self, other: "Expr") -> Expr:
        """True when self crosses below other."""
        return Expr("CrossBelow", self, _coerce(other))

    # -- Signal state (Pine-style) -------------------------------------------
    #
    # ``bars_since``/``streak``/``count_over`` read *self* as a condition, not
    # as a numeric series, so they belong on the boolean side of a strategy.

    def bars_since(self) -> Expr:
        """Bars since this condition was last true. NaN until it first is."""
        return Expr("BarsSince", self)

    def time_since(self) -> Expr:
        """Seconds since this condition was last true (0.0 on a true row).

        NaN until it first is. The twin of ``bars_since`` for a grid whose rows
        are not evenly spaced: on an irregular grid, ten bars
        ago can be ten seconds ago or two minutes ago.
        """
        return Expr("TimeSince", self)

    def streak(self) -> Expr:
        """Length of the current consecutive run of true ending at this bar."""
        return Expr("Streak", self)

    def count_over(self, window: Period) -> Expr:
        """Count of rows where this condition is true in the trailing window.

        A window is a BAR COUNT (``20``) or a DURATION
        (``Interval.seconds(30)``). A duration reads the window
        ``(t - d, t]`` on the bar timestamps, so a gappy grid -- one-second
        bars on an irregular grid, where an empty period prints no bar
        -- gets a real thirty seconds instead of thirty bars. NaN until the
        series holds a full ``d`` of history, then equal to
        ``pandas.rolling("30s")``. Durations are literal: ``param()`` does not
        sweep one yet. See "Windows in time" in the strategy authoring
        guide.

        With an always-true condition and a duration it counts the ROWS
        themselves, which is how a strategy measures how gappy its own grid
        is: the number of bars printed over the last thirty seconds.
        """
        return Expr("CountOver", self, _resolve_period(window, allow_duration=True))

    def value_when(self, source: "Expr") -> Expr:
        """Value of ``source`` on the last bar where this condition was true."""
        return Expr("ValueWhen", self, _coerce(source))

    def ffill(self) -> Expr:
        """Last non-NaN value of *this* series, carried forward. NaN until the
        first one.

        Unlike the four helpers above, ``ffill`` reads self as a SERIES, not as
        a condition. Paired with ``when(cond, value)`` -- whose false branch is
        NaN by default -- it is the readable spelling of a persistent armed
        state::

            state = mbt.when(imb >= thr, 1.0,
                    mbt.when(imb <= -thr, -1.0)).ffill()
            pos   = mbt.when(regime_open, state, 0.0)   # composes with a filter

        The state is a real series, so masking it does not destroy it: when the
        filter reopens, ``pos`` picks the state back up on the same bar. That is
        what ``hold()`` cannot do -- see :func:`hold`.
        """
        return Expr("Ffill", self)

    # -- Price grid ----------------------------------------------------------

    def round_to(self, step: float) -> Expr:
        """Nearest multiple of ``step``; an exact half goes UP (towards +inf).

        A venue quotes on a grid: an order at a price the grid does not carry
        finds no depth in the book, and under the queue model there is nothing
        for it to stand behind. Snap a computed level onto the venue's tick
        size before resting on it::

            quote = (col("fair") - lit(0.5) * col("spread")).round_to(0.1)

        ``step`` is a strictly positive **literal**, not a ``param()``: it is
        read once at compile time as the decimal it is written as (``0.1`` is
        one tenth, not the binary double just above it), which is what a tick
        size is. A step that is not a positive number is refused by name at
        compile time.

        The result is the double the corresponding literal denotes:
        ``100.34`` on a ``0.1`` grid gives exactly ``100.3``, where
        ``round(x / step) * step`` gives ``100.30000000000001`` -- one ulp
        away, which is a different price level to a book lookup.

        Half UP, not half to even: ``0.25`` on a ``0.1`` grid is ``0.3`` here
        and ``0.2`` under ``numpy.round``.

        Snapping twice is snapping once: a value already on the grid comes back
        unchanged, by any of the three methods and in any order.
        """
        return Expr("RoundTo", self, _round_step(step, "round_to"))

    def floor_to(self, step: float) -> Expr:
        """Largest multiple of ``step`` at or below the value.

        The passive direction for a **bid**: rounding a buy level down never
        makes the order more aggressive than the author asked for. Same rules
        for ``step`` as :meth:`round_to`.
        """
        return Expr("FloorTo", self, _round_step(step, "floor_to"))

    def ceil_to(self, step: float) -> Expr:
        """Smallest multiple of ``step`` at or above the value.

        The passive direction for an **ask**. Same rules for ``step`` as
        :meth:`round_to`.
        """
        return Expr("CeilTo", self, _round_step(step, "ceil_to"))

    def rising(self, n: Period) -> Expr:
        """1.0 if self strictly increased on each of the last ``n`` steps."""
        return Expr("Rising", self, _resolve_period(n))

    def falling(self, n: Period) -> Expr:
        """1.0 if self strictly decreased on each of the last ``n`` steps."""
        return Expr("Falling", self, _resolve_period(n))

    def pivot_high(self, left: Period, right: Period) -> Expr:
        """Causal pivot high, confirmed ``right`` bars later (no lookahead)."""
        return Expr("PivotHigh", self, _resolve_period(left), _resolve_period(right))

    def pivot_low(self, left: Period, right: Period) -> Expr:
        """Causal pivot low, confirmed ``right`` bars later (no lookahead)."""
        return Expr("PivotLow", self, _resolve_period(left), _resolve_period(right))

    # -- Cumulative ----------------------------------------------------------

    def cumsum(self) -> Expr:
        """Cumulative sum from the first bar to the current one."""
        return Expr("CumSum", self)

    def cumprod(self) -> Expr:
        """Cumulative product from the first bar to the current one."""
        return Expr("CumProd", self)

    # -- Not causal ----------------------------------------------------------
    #
    # Its own section. This used to sit inside `-- Cumulative --`, right under
    # `cumprod`, with no docstring: a reader filed it with its neighbours and
    # read it as expanding, which is exactly the mistake it invites. A section
    # heading is not documentation, but it is what a reader skims.

    def full_series_rank(self) -> Expr:
        """Rank of each value over the **whole series**, 1 to N.

        **This operator is not causal.** It sorts the entire column, so the
        rank at bar ``t`` depends on bars that come after ``t``. Feed it to an
        entry signal or to position sizing and the backtest is worth nothing:
        the strategy is choosing with information it could not have had. Two
        operators in the DSL behave this way, this one and ``lead(n)``, and
        ``detect_lookahead`` fails a strategy that lets either one decide.

        What it is for: building labels and targets offline, where seeing the
        whole series is the point.

        For a rank you can trade on, use ``rolling_rank(window)``, which ranks
        within a trailing window, or ``cs_rank()``, which ranks across symbols
        at the same bar.
        """
        return Expr("Rank", self)

    def rank(self) -> Expr:
        """Deprecated spelling of :meth:`full_series_rank`.

        The bare name said nothing about the whole-series sort, and sat among
        the cumulative operators, so it read as causal. It is kept working so
        existing code still runs, and warns so that code gets fixed.
        """
        warnings.warn(
            "Expr.rank() is not causal: it ranks over the whole series, so the "
            "rank at bar t depends on bars after t. Renamed to "
            "full_series_rank() to say so. For a causal rank use "
            "rolling_rank(window), or cs_rank() across symbols.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.full_series_rank()

    # -- Cross-sectional -----------------------------------------------------

    def cs_mean(self) -> Expr:
        return Expr("CrossSectionalMean", self)

    def cs_rank(self) -> Expr:
        return Expr("CrossSectionalRank", self)

    def cs_zscore(self) -> Expr:
        """Cross-sectional z-score across symbols, on the population std."""
        return Expr("CsZScore", self)

    def cs_demean(self) -> Expr:
        """Cross-sectional demeaning: ``v - cross_mean`` per timestamp."""
        return Expr("CsDemean", self)

    def cs_std(self) -> Expr:
        """Cross-sectional population std, broadcast to every symbol."""
        return Expr("CsStd", self)

    def cs_scale(self) -> Expr:
        """Cross-sectional L1 (unit-gross) scaling: ``v / sum(|v_j|)``."""
        return Expr("CsScale", self)

    def cs_winsorize(self, k: Span = 3.0) -> Expr:
        """Clip to ``[mean - k*std, mean + k*std]`` across symbols."""
        return Expr("CsWinsorize", self, _resolve_span(k))

    def cs_quantile(self, q: Span) -> Expr:
        """Cross-sectional q-quantile, broadcast to every symbol."""
        return Expr("CsQuantile", self, _resolve_span(q))

    def cs_neutralize(self, factor: "Expr") -> Expr:
        """Cross-sectional OLS residual of self regressed on ``factor``.

        The residuals are orthogonal to the factor and sum to zero across the
        symbols that participate (both series finite at that timestamp).
        """
        return Expr("CsNeutralize", self, _coerce(factor))

    # -- Cross-asset reference -----------------------------------------------

    def of_symbol(self, symbol: str) -> Expr:
        """Reference this column from a specific symbol's data.

        Example::

            btc_close = col("close").of_symbol("BTCUSDT")
            signal = col("close") - btc_close  # ETH close minus BTC close
        """
        return Expr("SymbolRef", symbol, self)

    # -- Datetime extraction -------------------------------------------------

    def hour(self) -> Expr:
        """Extract hour (0-23) from a timestamp column (UTC)."""
        return Expr("Hour", self)

    def minute(self) -> Expr:
        """Extract minute (0-59) from a timestamp column (UTC)."""
        return Expr("Minute", self)

    def day_of_week(self) -> Expr:
        """Extract day of week from a timestamp column (0=Monday, 6=Sunday)."""
        return Expr("DayOfWeek", self)

    def month(self) -> Expr:
        """Extract month (1-12) from a timestamp column (UTC)."""
        return Expr("Month", self)

    def day_of_month(self) -> Expr:
        """Extract day of month (1-31) from a timestamp column (UTC)."""
        return Expr("DayOfMonth", self)

    def year(self) -> Expr:
        """Extract the full year (e.g. 2024) from a timestamp column (UTC)."""
        return Expr("Year", self)

    def week_of_year(self) -> Expr:
        """Extract the ISO-8601 week number (1-53) from a timestamp (UTC)."""
        return Expr("WeekOfYear", self)

    def day_of_year(self) -> Expr:
        """Extract the ordinal day of year (1-366) from a timestamp (UTC)."""
        return Expr("DayOfYear", self)

    def is_month_start(self) -> Expr:
        """1.0 on the first day of the month, else 0.0 (UTC)."""
        return Expr("IsMonthStart", self)

    def is_month_end(self) -> Expr:
        """1.0 on the last day of the month, else 0.0 (UTC)."""
        return Expr("IsMonthEnd", self)

    def is_quarter_end(self) -> Expr:
        """1.0 on the last day of a calendar quarter (Mar/Jun/Sep/Dec) (UTC)."""
        return Expr("IsQuarterEnd", self)

    def is_weekend(self) -> Expr:
        """1.0 on Saturday or Sunday, else 0.0 (UTC)."""
        return Expr("IsWeekend", self)

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        if self._variant in ("Column", "Parameter", "Literal"):
            return f"Expr.{self._variant}({self._args[0]!r})"
        return f"Expr.{self._variant}(...)"


class MultiExpr(tuple):
    """The several series a multi-output indicator returns.

    It unpacks and indexes exactly like the plain tuple it replaces::

        upper, middle, lower = bollinger_bands(close, 20)
        upper = bollinger_bands(close, 20)[0]

    What it adds is a refusal that says something. Handing the whole tuple to
    something that wants ONE series used to fail deep in the plumbing: an
    ``AttributeError`` on ``_param_meta`` (an engine-private attribute the
    caller never wrote) when it reached a signal, and a bare
    ``'>' not supported between instances of 'tuple' and 'int'`` when it was
    compared. Neither names the call, so neither is actionable -- least of all
    for a caller that only sees the traceback. Every such use now names the
    call, lists what it returns, and shows the unpacking.

    Equality is kept: ``==`` against another tuple still compares element by
    element (two expressions are equal when they build the same tree), so
    ordinary container checks keep working. Only a
    comparison against a number or an expression -- always a mistake -- is
    refused.
    """

    # Class-level defaults: attribute lookup must find these without falling
    # through to __getattr__, which would recurse while building its message.
    _call = "this indicator"
    _names: tuple = ()

    def __new__(cls, values: Any, call: str = "this indicator", names: Any = ()) -> "MultiExpr":
        self = super().__new__(cls, values)
        self._call = call
        self._names = tuple(names)
        return self

    # -- Refusals ------------------------------------------------------------
    #
    # Python resolves operators on the tuple before any engine code runs, so
    # this class is the only place left where the mistake can still be named.

    def _reject(self, misuse: str) -> TypeError:
        """The error to raise for `misuse`. Returned, not raised, so callers
        outside this class can `raise value._reject(...)` at their own site."""
        members = ", ".join(self._names)
        return TypeError(
            f"{self._call}() returns {len(self)} series, not one: ({members}). "
            f"{misuse}\n"
            f"Unpack it and use the series you meant:\n"
            f"    {members} = {self._call}(...)"
        )

    def _reject_op(self, op: str) -> None:
        raise self._reject(f"`{op}` needs a single series on each side.")

    def __gt__(self, other: Any) -> Any:
        self._reject_op(">")

    def __lt__(self, other: Any) -> Any:
        self._reject_op("<")

    def __ge__(self, other: Any) -> Any:
        self._reject_op(">=")

    def __le__(self, other: Any) -> Any:
        self._reject_op("<=")

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, (Expr, int, float)):
            self._reject_op("==")
        return self._same_as(other)

    def __ne__(self, other: Any) -> Any:
        if isinstance(other, (Expr, int, float)):
            self._reject_op("!=")
        same = self._same_as(other)
        return same if same is NotImplemented else not same

    def _same_as(self, other: Any) -> Any:
        # Not tuple.__eq__: it compares the members with `==`, which on two
        # expressions builds an Eq node instead of answering, and an
        # expression has no truth value. Two members are the same when they
        # are the same object or serialise to the same tree.
        if not isinstance(other, tuple):
            return NotImplemented
        if len(self) != len(other):
            return False
        for a, b in zip(self, other):
            if a is b:
                continue
            if isinstance(a, Expr) or isinstance(b, Expr):
                if not (isinstance(a, Expr) and isinstance(b, Expr)):
                    return False
                if a.to_json() != b.to_json():
                    return False
            elif a != b:
                return False
        return True

    # Defining __eq__ would otherwise drop the inherited hash.
    __hash__ = tuple.__hash__

    def __add__(self, other: Any) -> Any:
        self._reject_op("+")

    def __radd__(self, other: Any) -> Any:
        self._reject_op("+")

    def __sub__(self, other: Any) -> Any:
        self._reject_op("-")

    def __rsub__(self, other: Any) -> Any:
        self._reject_op("-")

    def __mul__(self, other: Any) -> Any:
        self._reject_op("*")

    def __rmul__(self, other: Any) -> Any:
        self._reject_op("*")

    def __truediv__(self, other: Any) -> Any:
        self._reject_op("/")

    def __rtruediv__(self, other: Any) -> Any:
        self._reject_op("/")

    def __neg__(self) -> Any:
        self._reject_op("-")

    def __and__(self, other: Any) -> Any:
        self._reject_op("&")

    def __rand__(self, other: Any) -> Any:
        self._reject_op("&")

    def __or__(self, other: Any) -> Any:
        self._reject_op("|")

    def __ror__(self, other: Any) -> Any:
        self._reject_op("|")

    def __invert__(self) -> Any:
        self._reject_op("~")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            # copy, pickle and hasattr() probe dunders: let them fail normally.
            raise AttributeError(name)
        if name.startswith("_"):
            # An engine-internal probe (a signal on its way to JSON, say). The
            # caller never wrote this name, so do not quote it back at them.
            raise self._reject("It was used where a single expression belongs.")
        raise self._reject(f"`.{name}` belongs to one series, not to the group.")

    def __repr__(self) -> str:
        return f"{self._call}(...) -> ({', '.join(self._names)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Largest |n| that ``expr ** n`` expands. The expansion is a chain of n - 1
# multiplications, and each rounds once: past a handful of factors the result
# drifts from a real power by more than an ulp, and a larger exponent on a
# series is more likely a mistake than a need.
_MAX_POWER = 8


def _power(base: Expr, exponent: Any, modulo: Any = None) -> Expr:
    """``base ** exponent`` for a literal integer ``exponent``, as products.

    The engine has no power function, so ``x ** 3`` is written for it as
    ``x * x * x`` and ``x ** -2`` as ``1 / (x * x)``: the same nodes as the
    hand-written form, hence the same values everywhere an expression is
    evaluated (in batch and in the program a quote runs at each wake-up), a
    NaN or a null propagating as it does through ``*`` and ``/``. ``x ** 2``
    is bit for bit ``x * x``; from ``x ** 3`` on, each product rounds once,
    so the result can differ from ``numpy.power`` in the last bit.
    """
    if modulo is not None:
        raise TypeError("pow(expr, n, mod) is not supported on an expression")
    if isinstance(exponent, (Expr, MultiExpr)):
        raise TypeError(
            "expr ** expr is not supported: the engine has no power function, "
            "so the exponent must be a literal integer (x ** 2 is x * x). For "
            "a positive base write exp(y * log(x)) with manifoldbt.indicators."
        )
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        raise TypeError(
            f"expr ** n takes a literal integer n, got {type(exponent).__name__}"
        )
    if isinstance(exponent, float):
        if not exponent.is_integer():
            hint = (
                "sqrt(x) from manifoldbt.indicators"
                if exponent == 0.5
                else "exp(p * log(x)) from manifoldbt.indicators, for a positive x"
            )
            raise TypeError(
                f"expr ** {exponent!r}: only a literal integer exponent is "
                f"supported (the engine has no power function). Write {hint}."
            )
        exponent = int(exponent)
    if exponent == 0:
        raise ValueError(
            "expr ** 0 is the constant 1 whatever the series holds (a NaN "
            "included): write lit(1.0) if that is what you mean"
        )
    n = abs(exponent)
    if n > _MAX_POWER:
        raise ValueError(
            f"expr ** {exponent}: exponents are expanded into products and "
            f"stop at +/-{_MAX_POWER}. For a larger one write the products "
            "yourself (y = x * x; y * y * ...) or exp(p * log(x)) for a "
            "positive x"
        )
    out = base
    for _ in range(n - 1):
        out = Expr("Mul", out, base)
    if exponent < 0:
        out = Expr("Div", Expr("Literal", 1.0), out)
    return out


def _coerce(value: Any) -> Expr:
    """Coerce a raw Python value into an Expr.Literal.

    int values are promoted to float so the Rust type-checker never sees
    Int64 vs Float64 mismatches in arithmetic/comparison expressions.
    """
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        return Expr("Literal", value)
    if isinstance(value, int):
        return Expr("Literal", float(value))
    if isinstance(value, (float, str)) or value is None:
        return Expr("Literal", value)
    if isinstance(value, MultiExpr):
        raise value._reject("It was used where a single expression belongs.")
    if isinstance(value, (list, tuple)) and any(isinstance(v, Expr) for v in value):
        raise TypeError(
            f"Cannot use a {type(value).__name__} of {len(value)} expressions where one "
            f"is expected; unpack it and pass the one you meant."
        )
    raise TypeError(f"Cannot coerce {type(value).__name__} to Expr")


# ---------------------------------------------------------------------------
# Module-level factory functions (public API)
# ---------------------------------------------------------------------------


def col(name: str) -> Expr:
    """Reference a data column (e.g. ``'close'``, ``'volume'``)."""
    return Expr("Column", name)


def lit(value: Any) -> Expr:
    """Create a literal constant expression."""
    return Expr("Literal", value)


def position() -> Expr:
    """The position held, in signed units, at the instant an order is posted.

    Not a column. A column is computed for every bar before the simulation
    starts; the position at bar ``i`` is the result of what the strategy did
    with bars ``0..i``, and only the simulation knows it. So ``position()`` is
    readable in exactly ONE place: the level of an entry order declared on the
    strategy. That is what a market maker needs, and it is what it is for::

        quote = col("fair") - lit(0.5) * col("spread") - lit(k) * position()

        strategy = (
            mbt.Strategy.create("maker")
            .signal("quote", quote)
            .size(col("wants_to_quote") * lit(qty))
            .limit_entry(signal="quote", time_in_force=Interval.seconds(1))
        )

    The market data in that level is delayed like every other signal
    (``signal_delay``); the inventory is not, because it is the strategy's own
    book, known without a round trip.

    Inside that level it combines with ``+``, ``-``, ``*``, ``/``, ``abs``,
    ``min`` and ``max``, and with any sub-expression that does not itself read
    it. A size, a plain signal, a comparison, a ``when()``, or any operator
    with memory (a rolling window, a lag, an indicator) refuses it BY NAME at
    compile time rather than reading it as something else.

    See "Reading the position from a level" in the strategy authoring guide.

    In the expressions of a quote (``Strategy.quote``) it is readable
    everywhere, with every operator a wake-up takes: there it is the position
    as the fills that have reached the strategy say, at each wake-up
    (``execution.latency["response"]`` after the venue booked them). See
    "Quoting in the DSL" in the guide.
    """
    return Expr("Position")


# ---------------------------------------------------------------------------
# Reading the simulation state at a wake-up
# ---------------------------------------------------------------------------
#
# Like ``position()``, none of these is a column: each is what the venue knows
# at the instant the engine wakes up, and exists only there. They are readable
# in the expressions of a quote (its price, its size, its enabled condition),
# which the engine evaluates at each wake-up without calling Python. A signal,
# a size, or any operator with memory (a rolling window, a lag, an indicator)
# refuses them BY NAME at compile time, as it refuses ``position()``.
#
# What they combine with there: ``+ - * /``, comparisons, ``& | ~``,
# ``when()``, ``abs``, ``min``, ``max``, :func:`round`, :func:`clip`,
# ``sqrt``, ``log``, ``exp``, ``round_to``, ``floor_to``, ``ceil_to``, and any
# sub-expression that does not read the state (that part is computed in batch,
# with every operator).


def _side(side: Any) -> str:
    """``"bid"`` / ``"ask"`` as the engine spells them."""
    if not isinstance(side, str) or side.lower() not in ("bid", "ask"):
        raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")
    return "Bid" if side.lower() == "bid" else "Ask"


def live_qty(side: str) -> Expr:
    """What is left to fill of the strategy's live orders on ``side``.

    ``side`` is ``"bid"`` (the buy orders) or ``"ask"`` (the sell orders). The
    sum, in units, of the unfilled size of every order of that side that may
    still fill, as the strategy knows them: an order sent and not yet
    acknowledged counts, an order whose cancellation is on its way counts
    until the answer arrives. ``0`` when there is none.

    Like every reading of the simulation state, only the expressions of a
    quote (``Strategy.quote``) may use it; see "Quoting in the DSL" in the
    guide.
    """
    return Expr("SimState", "LiveQty", _side(side))


def order_age(side: str) -> Expr:
    """Seconds since the strategy sent its newest live order on ``side``
    (``"bid"`` or ``"ask"``), on the strategy's clock; NaN when there is none.

    Readable only in the expressions of a quote (``Strategy.quote``).
    """
    return Expr("SimState", "OrderAge", _side(side))


def last_fill_px(side: str) -> Expr:
    """Price of the strategy's last fill on ``side`` (``"bid"``: a buy,
    ``"ask"``: a sell) that has reached it; NaN before the first.

    A fill reaches the strategy ``execution.latency["response"]`` after the
    venue booked it. Readable only in the expressions of a quote
    (``Strategy.quote``).
    """
    return Expr("SimState", "LastFillPx", _side(side))


def queue_ahead(side: str) -> Expr:
    """Size queued ahead of the strategy's newest live order on ``side``
    (``"bid"`` or ``"ask"``), at its price.

    The venue's estimate as of its last answer about that order (its
    acknowledgement, a partial fill), delayed like every answer; NaN when
    there is no live order on that side, or before the first answer about it
    has arrived. On a market stored order by order, under its exact queue,
    it is the volume ahead of the order exactly, as the feed shows it at the
    strategy's clock (``feed`` behind): it moves at every execution or
    cancellation ahead, once the acknowledgement has told the strategy which
    order is its own. Readable only in the expressions of a quote
    (``Strategy.quote``).
    """
    return Expr("SimState", "QueueAhead", _side(side))


def order_price(side: str) -> Expr:
    """Price of the strategy's newest live order on ``side`` (``"bid"`` or
    ``"ask"``), as the strategy knows it; NaN when there is none.

    The order ``order_age(side)`` and ``queue_ahead(side)`` read: with
    several quotes on one side, the one sent last. An order sent and not yet
    acknowledged has its price already (the price it was sent at, on the
    symbol's tick grid); an order whose cancellation is on its way keeps it
    until the answer arrives.

    What it is for: withdrawing a quote whose price the market left, instead
    of moving it (a moved price cancels AND posts again, rule 4 of
    ``Strategy.quote``)::

        from manifoldbt.indicators import abs_val

        best = bt.book.bid_price_at(1)
        left_behind = abs_val(bt.order_price("bid") - best) > 1e-6   # False without an order (NaN)

    Readable only in the expressions of a quote (``Strategy.quote``).
    """
    return Expr("SimState", "OrderPrice", _side(side))


def position_age() -> Expr:
    """Seconds since the position left zero, or changed sign, as the
    strategy knows it; NaN while it is flat.

    The position is the one ``position()`` reads: the fills that have
    reached the strategy, ``execution.latency["response"]`` after the venue
    booked them. The age counts from the instant the fill that opened the
    position (or flipped its sign) reached the strategy, on its clock, so at
    the first wake-up that sees the position it is already the time elapsed
    since that fill arrived. A fill that adds to or trims the position
    without crossing zero leaves the age as it is; a position within 1e-12
    of zero is flat. A maximum holding time is one comparison::

        stale = bt.position_age() >= 2.0     # False while flat (NaN)

    Readable only in the expressions of a quote (``Strategy.quote``).
    """
    return Expr("SimState", "PositionAge")


def last_fill_age(side: Optional[str] = None) -> Expr:
    """Seconds since the strategy's last fill on ``side`` reached it
    (``"bid"``: a buy, ``"ask"``: a sell); without ``side``, since its last
    fill on either. NaN before the first.

    Counted, on the strategy's clock, from the instant the fill reached it:
    ``execution.latency["response"]`` after the venue booked it, the instant
    ``position()``, ``cash()`` and ``last_fill_px(side)`` change. Without
    ``side`` it is ``min_val`` of the two sides, which ignores a side that
    never filled.

    Readable only in the expressions of a quote (``Strategy.quote``).
    """
    if side is None:
        return Expr(
            "Function",
            "min",
            [Expr("SimState", "LastFillAge", "Bid"), Expr("SimState", "LastFillAge", "Ask")],
        )
    return Expr("SimState", "LastFillAge", _side(side))


def last_cancel_age(side: str) -> Expr:
    """Seconds since the strategy SENT its last cancellation of an order on
    ``side`` (``"bid"`` or ``"ask"``), on its clock; NaN before the first.

    Counted from the sending, not from the venue's answer: the strategy
    knows when it cancelled without waiting, whatever the latencies, and a
    cancellation that arrives too late (the order filled first) was still
    sent. Every cancellation of the side counts, whichever quote sent it: a
    quote withdrawn (``enabled`` false, a size of zero) and a quote moved to
    another price (cancelled, then posted again) alike. An order that fills,
    is refused, or expires (the rest of an IOC) is not a cancellation.

    A pause before a quote posts again after it cancelled is
    ``.quote(..., cooldown=Interval.millis(500))``. Written with this age
    instead, it is the pause of the side::

        paused = bt.last_cancel_age("bid") < 0.5
        .quote("buy", price, size, enabled=... & ~paused)

    Write it as ``~(age < d)``, not ``age >= d``: before the first
    cancellation the age is NaN, a comparison with NaN is false, and
    ``age >= d`` would keep the quote off until something is cancelled.

    Readable only in the expressions of a quote (``Strategy.quote``).
    """
    return Expr("SimState", "LastCancelAge", _side(side))


def cash() -> Expr:
    """Cash of the account as the fills that have reached the strategy say:
    ``initial_capital``, minus what the buys cost, plus what the sells
    brought, fees included.

    Readable only in the expressions of a quote (``Strategy.quote``).
    """
    return Expr("SimState", "Cash")


def _book_price_at(side: str, level: Any) -> Expr:
    if isinstance(level, bool):
        raise TypeError("level must be an int or an expression, got bool")
    if isinstance(level, int) and level < 1:
        raise ValueError(f"level must be 1 or more (1 = best), got {level}")
    return Expr("SimState", "BookPriceAt", _side(side), _coerce(level))


# ---------------------------------------------------------------------------
# round / clip
# ---------------------------------------------------------------------------
#
# Named after the Python builtins on purpose, and they answer as the builtins
# do when nothing in the call is an expression: ``mbt.round(2.5) == 2``, so a
# ``from manifoldbt import *`` that shadows the builtin changes nothing for
# plain numbers.


def round(x: Any, ndigits: Optional[int] = None) -> Any:  # noqa: A001 - mirrors the builtin
    """Round to the nearest integer, an exact half going to the EVEN one.

    The rule of Python's ``round`` (``round(0.5) == 0``, ``round(1.5) == 2``,
    ``round(-2.5) == -2``), applied to a series. A null stays null, a NaN stays
    NaN. For a price on a tick grid use ``expr.round_to(step)``, whose half
    goes up.

    On plain numbers, this IS the builtin ``round``.
    """
    if isinstance(x, Expr):
        if ndigits is not None:
            raise TypeError(
                "round(expr) rounds to an integer and takes no ndigits; to round "
                "onto a price grid use expr.round_to(step)"
            )
        return Expr("Function", "round", [x])
    if isinstance(x, MultiExpr):
        raise x._reject("It was passed to round().")
    return builtins.round(x) if ndigits is None else builtins.round(x, ndigits)


def clip(x: Any, lo: Any, hi: Any) -> Any:
    """``x`` brought into ``[lo, hi]``.

    A NaN in any of the three gives NaN (as ``numpy.clip``: an unknown value
    does not become a bound), a null gives null, and ``lo > hi`` gives ``hi``.
    Each argument may be a number or an expression.
    """
    if any(isinstance(v, (Expr, MultiExpr)) for v in (x, lo, hi)):
        return Expr("Function", "clip", [_coerce(x), _coerce(lo), _coerce(hi)])
    x, lo, hi = float(x), float(lo), float(hi)
    if x != x or lo != lo or hi != hi:
        return float("nan")
    return min(max(x, lo), hi)


def hold() -> Expr:
    """NaN. The engine reads it as: leave the POSITION where it is.

    What it holds is the position the simulator currently carries, not the
    value of the expression that contains it. The distinction only shows up
    once something else can move the position:

    * after a stop-loss or take-profit fired, ``hold()`` keeps the position
      FLAT. It does not replay the last signal value and buy back in.
    * under a regime filter, ``hold()`` holds whatever the filter last wrote::

          state = mbt.when(imb >= thr, 1.0,
                  mbt.when(imb <= -thr, -1.0, mbt.hold()))
          pos   = mbt.when(regime_open, state, 0.0)   # <- reads as a trap

      While the filter is closed the target is 0.0, so the position goes flat.
      When it reopens on a bar where ``state`` falls through to ``hold()``, the
      engine holds that flat position. Exposure only returns on the NEXT
      threshold crossing, which can be far away: a filter open 55% of the time
      has been measured to leave under 1% exposure. Nothing warns. Routing the
      state through a named ``.signal()`` does not change it either -- the NaN
      still means "hold the position" wherever it is read.

    Both readings are legitimate; pick the one you meant.

    Exit and re-enter only on a NEW signal -- keep ``hold()`` inside the when::

        pos = mbt.when(regime_open,
              mbt.when(imb >= thr, 1.0,
              mbt.when(imb <= -thr, -1.0, mbt.hold())), 0.0)

    Persistent state that you mask -- build the state as a real series with
    :meth:`Expr.ffill`, then mask it::

        state = mbt.when(imb >= thr, 1.0,
                mbt.when(imb <= -thr, -1.0)).ffill()   # false branch is NaN
        pos   = mbt.when(regime_open, state, 0.0)

    Masking a ``ffill`` state does not destroy it, so the position comes back
    on the bar the filter reopens. ``cond.value_when(value)`` writes the same
    series with an explicit trigger.

    With no filter and nothing else moving the position (no stop, no take
    profit), the three spellings give the same trades.
    """
    return Expr("Literal", float("nan"))


def param(
    name: str,
    *,
    default: Any = None,
    range: Any = None,
    description: str = "",
) -> Expr:
    """Create a parameter reference.

    The returned ``Expr`` serializes as ``Expr::Parameter(name)``.
    Metadata (default, range, description) is stored as ``_param_meta``
    and picked up by :class:`Strategy` when building the ``ParamSpec``.

    The default belongs to the expression, so to the strategy that holds it:
    two strategies built in one process, each with its own
    ``param("slow", default=...)``, each run with their own default. Within
    one strategy, a name may appear several times; a use without a default
    reads the default another use gives, and two different defaults are
    refused when the strategy is serialised, unless the strategy declares the
    parameter itself with ``Strategy.param(name, default=...)``, which then
    wins over every default written in its expressions.
    """
    expr = Expr("Parameter", name)
    expr._param_meta = {
        "name": name,
        "default": default,
        "range": range,
        "description": description,
    }
    return expr


def when(condition: Expr, true_value: Any = 1.0, false_value: Any = float("nan")) -> Expr:
    """Conditional expression (if/else).

    Omit true_value to default to 1.0 (full position, clamped by max_position_pct).
    Omit false_value and the false branch is NaN. At the TOP of the position
    expression the engine reads that NaN as "hold the position"; anywhere else
    it is just NaN, and an enclosing ``when`` can overwrite it. See :func:`hold`
    and :meth:`Expr.ffill`.
    """
    # The condition goes through _coerce like the two branches: unchecked, a
    # multi-output indicator handed in whole reached the JSON as a tuple and
    # only failed there, on an attribute the caller never wrote.
    return Expr("IfElse", _coerce(condition), _coerce(true_value), _coerce(false_value))


def choice(name: str, branches: "dict[str, Expr]", *, description: str = "") -> Expr:
    """Balayer un CHOIX d'expression, pas seulement un nombre.

    Un ``param()`` ordinaire porte une valeur numerique. ``choice()`` porte un
    NOM, et chaque nom designe une sous-expression differente. Le moteur
    remplace le noeud entier par la branche choisie AVANT de simuler, donc une
    combinaison n'evalue que sa propre branche : les autres n'existent plus.

    C'est ce qui le distingue d'un ``when()`` imbrique, qui construit toutes
    les variantes et tranche barre par barre.

    Usage (balayer la timeframe d'une bande, sim en 1m)::

        bande = mbt.choice("band", {
            "30m": mbt.tf("30m").apply(sma(close, mbt.param("len"))),
            "1h":  mbt.tf("1h").apply(sma(close, mbt.param("len"))),
            "2h":  mbt.tf("2h").apply(sma(close, mbt.param("len"))),
        })
        # grille : {"len": [10, 20, 30], "band": ["30m", "1h", "2h"]}

    Noter le ``apply()``. Ecrit ``sma(mbt.tf("30m").close, param("len"))``, le
    balayage n'aurait pas le sens attendu : la periode compterait des barres de
    SIMULATION sur une colonne etalee en escalier, donc les trois branches
    lisseraient le meme nombre de MINUTES au lieu de 10, 20 ou 30 bougies de
    leur timeframe. Voir :func:`tf`.

    Les branches acceptent n'importe quelle expression, donc le meme mecanisme
    balaie une colonne exogene, un actif ou un type d'indicateur.

    Args:
        name: nom du parametre selecteur, a mettre dans la grille.
        branches: nom de branche -> expression. L'ordre compte : la premiere
            sert de defaut quand le parametre est absent.
        description: metadonnee libre.

    Raises:
        ValueError: si ``branches`` est vide.
    """
    if not branches:
        raise ValueError(
            f"choice({name!r}) needs at least one branch; an empty choice has "
            f"nothing to resolve to."
        )
    items = [(str(k), _coerce(v)) for k, v in branches.items()]
    expr = Expr("Choice", name, items)
    # Declare le selecteur comme un parametre a part entiere, sans quoi le
    # balayer serait refuse par la validation ("parameter not declared").
    expr._param_meta = {
        "name": name,
        "default": items[0][0],
        "range": None,
        "description": description,
    }
    return expr


def exo(name: str, column: Optional[str] = None) -> Expr:
    """Reference an exogenous data column.

    Exogenous data is registered via ``bt.register_exo()`` and declared
    in ``BacktestConfig(exo_data=[...])``.

    Args:
        name: Exo series name (e.g. ``"hashrate"``).
        column: Column name within the exo series. If ``None``, defaults
                to ``name`` (convenient when the series has a single value column
                with the same name as the series).

    Returns:
        An ``Expr`` referencing ``col("exo.{name}.{column}")``.

    Example::

        # Single-column shorthand
        signal = rsi(exo("hashrate"), 14) > 70

        # Multi-column explicit
        signal = exo("onchain", "active_addresses") > 1_000_000
    """
    col_name = column if column is not None else name
    return col(f"exo.{name}.{col_name}")


def symbol_ref(symbol: str, column: str) -> Expr:
    """Reference a column from a specific symbol's data.

    Args:
        symbol: Symbol name (e.g., "BTCUSDT").
        column: Column or signal name to reference.

    Example::

        btc_momentum = symbol_ref("BTCUSDT", "momentum")
    """
    return Expr("SymbolRef", symbol, col(column))


class AssetRef:
    """Reference to a specific symbol for cross-asset column access."""

    __slots__ = ("_symbol",)

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol

    def col(self, name: str) -> Expr:
        """Reference a column from this symbol's data."""
        return Expr("SymbolRef", self._symbol, Expr("Column", name))

    def __repr__(self) -> str:
        return f"AssetRef({self._symbol!r})"


def asset(symbol: str) -> AssetRef:
    """Reference a specific symbol for cross-asset data access.

    Usage::

        btc_close = bt.asset("BTCUSDT").col("close")
        # Then define as a signal and use in downstream expressions:
        relative = bt.col("close") / bt.col("btc_close")
    """
    return AssetRef(symbol)


class TimeframeRef:
    """Reference columns from a higher timeframe.

    The columns are forward-filled: a completed 1h bar's value becomes
    available at the start of the *next* 1h bar and persists until that
    bar completes.  This avoids lookahead bias.

    Requires ``extra_timeframes`` in ``BacktestConfig``.
    """

    __slots__ = ("_tf",)

    def __init__(self, tf: str) -> None:
        self._tf = tf

    @property
    def open(self) -> Expr:
        return col(f"{self._tf}.open")

    @property
    def high(self) -> Expr:
        return col(f"{self._tf}.high")

    @property
    def low(self) -> Expr:
        return col(f"{self._tf}.low")

    @property
    def close(self) -> Expr:
        return col(f"{self._tf}.close")

    @property
    def volume(self) -> Expr:
        return col(f"{self._tf}.volume")

    def col(self, name: str) -> Expr:
        """Reference any column from this timeframe."""
        return col(f"{self._tf}.{name}")

    def apply(self, expr: "Expr") -> Expr:
        """Evaluate *expr* ON this timeframe's own grid, then step-hold the
        result back onto the simulation grid (forward-filled, no lookahead:
        a completed bar's value becomes readable from the next bar on).

        This is what makes higher-timeframe INDICATORS correct. Periods
        inside *expr* count in THIS timeframe's bars::

            h1 = bt.tf("1h")
            band = h1.apply(sma(close, mbt.param("len")))   # len = HOURS

        is a true SMA of ``len`` hourly closes, sweepable like any param.
        By contrast ``sma(h1.close, 20)`` counts 20 SIMULATION bars over a
        step-held hourly series -- on a 1m simulation that is a 20-MINUTE
        smoothing of a staircase, not a 20-hour average.

        Inside *expr*, ``close``/``open``/... refer to this timeframe's own
        resampled columns. Requires ``extra_timeframes`` to declare the
        timeframe. Nesting ``apply`` inside another ``apply`` is rejected.
        """
        return Expr("OnTimeframe", self._tf, _coerce(expr))

    def __repr__(self) -> str:
        return f"TimeframeRef({self._tf!r})"


def tf(timeframe: str) -> TimeframeRef:
    """Reference a higher timeframe for multi-TF strategies.

    Two different things, and the distinction matters::

        h1 = bt.tf("1h")

        h1.close                        # a COLUMN: the last closed hourly
                                        # close, held across the minute bars
        h1.apply(ema(close, 20))        # an INDICATOR on the hourly grid:
                                        # 20 counts hourly candles

    Requires ``extra_timeframes={"1h": Interval.hours(1)}`` in config.

    .. warning::
       **An indicator applied to** ``h1.close`` **counts SIMULATION bars, not
       candles of the higher timeframe.** The column is forward-filled onto the
       simulation grid, so an indicator over it counts rows of that grid.

       On 1-minute bars, ``sma(h1.close, 8)`` averages the last 8 *minutes* of
       a step function — which is the last closed hourly close, not an 8-hour
       average. Measured on a ramp of +10/hour, it lags 1.63 h where a true
       8-hour mean lags 5.45 h.

       Multiplying by the ratio of the two intervals does not fix it either.
       ``sma(h1.close, 8 * 60)`` averages 480 rows of the step function: at
       every move it ramps in over 60 minutes instead of stepping, and its
       window spans 9 hourly values with unequal weights rather than 8 with
       equal ones. Measured against a true 8-hour mean on an impulse (one hour
       at 200, base 100, so the true signal spans 12.5): the error reaches
       12.29, or 98 % of that span. A ramp cannot reveal this — a box filter
       leaves a straight line straight — which is why a lag measurement alone
       reads correct.

       Use :meth:`TimeframeRef.apply`, which evaluates on the hourly grid and
       then step-holds the result. On that same impulse it matches the true
       8-hour mean exactly, on every bar::

           sma(h1.close, 8)          # 8 minutes of a step (gap 12.29)
           sma(h1.close, 8 * 60)     # 480 minutes of a step (gap 12.29)
           h1.apply(sma(close, 8))   # the 8-hour mean (gap 0.00)

       The ~1 h of lag common to all three is the timeframe itself: an hourly
       bar is only readable once closed, which is what makes it free of
       look-ahead.
    """
    return TimeframeRef(timeframe)


# ---------------------------------------------------------------------------
# Scan (stateful fold) support
# ---------------------------------------------------------------------------


class _ScanState:
    """Helper to build ``ScanPrev`` / ``ScanVar`` references inside a scan.

    Usage::

        from manifoldbt.expr import s, scan

        kalman = scan(
            state={"x": col("close"), "p": lit(1.0)},
            update={
                "p_pred": s.prev("p") + param("q"),
                "k":      s.var("p_pred") / (s.var("p_pred") + param("r")),
                "x":      s.prev("x") + s.var("k") * (col("close") - s.prev("x")),
                "p":      (lit(1.0) - s.var("k")) * s.var("p_pred"),
            },
            output="x",
        )
    """

    __slots__ = ()

    def prev(self, name: str) -> Expr:
        """Reference a state variable's value at t-1."""
        return Expr("ScanPrev", name)

    def var(self, name: str) -> Expr:
        """Reference a variable computed earlier in the current scan step."""
        return Expr("ScanVar", name)


s = _ScanState()
"""Singleton for building scan state references: ``s.prev("x")``, ``s.var("k")``."""


def scan(
    state: "dict[str, Expr]",
    update: "dict[str, Expr]",
    output: str,
) -> Expr:
    """Create a stateful scan (fold) expression.

    The scan executes entirely in Rust as a flat register-based scalar VM —
    no Python callbacks, no Arrow overhead per row.

    Args:
        state: Initial state variables. Keys are names, values are ``Expr``
            objects whose first-row value seeds the state.
        update: Ordered dict of update expressions. Each expression can
            reference ``s.prev("name")`` for previous state and
            ``s.var("name")`` for variables computed earlier in the same step.
            If an update name matches a state name, it writes back to that state.
        output: Name of the update variable to emit as the scan output.

    Returns:
        An ``Expr`` that evaluates to a Float64 array.

    Example::

        # Exponential moving average via scan
        ema_scan = scan(
            state={"ema": col("close")},
            update={"ema": s.prev("ema") * lit(0.9) + col("close") * lit(0.1)},
            output="ema",
        )
    """
    state_names = list(state.keys())
    init_exprs = [_coerce(v) for v in state.values()]
    update_names = list(update.keys())
    update_exprs = [_coerce(v) for v in update.values()]
    return Expr("Scan", state_names, init_exprs, update_names, update_exprs, output)
