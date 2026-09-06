"""Indicators travel Python -> JSON -> the engine's expression compiler.

Two separate things are checked here, and the second is the one that catches
real regressions:

1. The SHAPE of the JSON, variant by variant. The serialiser in `expr.py` is
   driven by sets of names; filing a variant under the wrong set produces
   plausible JSON that the engine rejects or, worse, accepts and misreads.
2. That the compiler actually ACCEPTS each expression. A shape test on its own
   would pass with a variant name that does not exist in the engine at all.

Numeric values are pinned on the engine side, against reference
implementations, rather than here.
"""
import json

import manifoldbt as bt
from manifoldbt import indicators as ind

CLOSE = {"Column": "close"}
HIGH = {"Column": "high"}
LOW = {"Column": "low"}
OPEN = {"Column": "open"}
TS = {"Column": "timestamp"}


def _compile(expr):
    """Compile an expression as the only signal and return the engine summary."""
    strategy = bt.Strategy(
        name="probe",
        signals={"probe": expr},
        position_sizing=bt.col("probe"),
    )
    return json.loads(bt.compile_strategy_json(strategy.to_json()))


# ---------------------------------------------------------------------------
# JSON shape
# ---------------------------------------------------------------------------


def test_directionnels_ont_la_forme_hlc_periode():
    assert ind.plus_di(14).to_json() == {"PlusDi": [HIGH, LOW, CLOSE, 14]}
    assert ind.minus_di(20).to_json() == {"MinusDi": [HIGH, LOW, CLOSE, 20]}


def test_aroon_prend_une_seule_serie():
    assert ind.aroon_up(25).to_json() == {"AroonUp": [HIGH, 25]}
    assert ind.aroon_down(25).to_json() == {"AroonDown": [LOW, 25]}


def test_statistiques_glissantes_a_une_serie():
    c = ind.close
    assert ind.rolling_var(c, 20).to_json() == {"RollingVar": [CLOSE, 20]}
    assert ind.rolling_skew(c, 20).to_json() == {"RollingSkew": [CLOSE, 20]}
    assert ind.rolling_kurt(c, 20).to_json() == {"RollingKurt": [CLOSE, 20]}
    assert ind.rolling_rank(c, 20).to_json() == {"RollingRank": [CLOSE, 20]}
    assert ind.rolling_argmax(c, 20).to_json() == {"RollingArgMax": [CLOSE, 20]}
    assert ind.rolling_argmin(c, 20).to_json() == {"RollingArgMin": [CLOSE, 20]}


def test_quantile_glissant_porte_une_fenetre_et_un_flottant():
    assert ind.rolling_quantile(ind.close, 20, 0.25).to_json() == {
        "RollingQuantile": [CLOSE, 20, 0.25]
    }


def test_statistiques_de_paire_portent_deux_series_et_une_fenetre():
    a, b = ind.close, ind.high
    assert ind.rolling_corr(a, b, 30).to_json() == {"RollingCorr": [CLOSE, HIGH, 30]}
    assert ind.rolling_cov(a, b, 30).to_json() == {"RollingCov": [CLOSE, HIGH, 30]}
    assert ind.rolling_beta(a, b, 30).to_json() == {"RollingBeta": [CLOSE, HIGH, 30]}


def test_etat_de_signal():
    rising = ind.close > ind.close.lag(1)
    cond = rising.to_json()
    assert ind.bars_since(rising).to_json() == {"BarsSince": cond}
    assert ind.streak(rising).to_json() == {"Streak": cond}
    assert ind.count_over(rising, 10).to_json() == {"CountOver": [cond, 10]}
    assert ind.value_when(rising, ind.close).to_json() == {"ValueWhen": [cond, CLOSE]}
    # ffill est une METHODE : son receveur est une serie, pas une condition.
    assert ind.close.ffill().to_json() == {"Ffill": CLOSE}
    assert ind.rising(ind.close, 3).to_json() == {"Rising": [CLOSE, 3]}
    assert ind.falling(ind.close, 3).to_json() == {"Falling": [CLOSE, 3]}


def test_les_pivots_portent_leurs_deux_bras():
    assert ind.pivot_high(ind.high, 2, 5).to_json() == {"PivotHigh": [HIGH, 2, 5]}
    assert ind.pivot_low(ind.low, 2, 5).to_json() == {"PivotLow": [LOW, 2, 5]}


def test_coupe_transversale():
    c = ind.close
    assert ind.cs_zscore(c).to_json() == {"CsZScore": CLOSE}
    assert ind.cs_demean(c).to_json() == {"CsDemean": CLOSE}
    assert ind.cs_std(c).to_json() == {"CsStd": CLOSE}
    assert ind.cs_scale(c).to_json() == {"CsScale": CLOSE}
    assert ind.cs_winsorize(c, 2.5).to_json() == {"CsWinsorize": [CLOSE, 2.5]}
    assert ind.cs_quantile(c, 0.9).to_json() == {"CsQuantile": [CLOSE, 0.9]}
    assert ind.cs_neutralize(c, ind.high).to_json() == {"CsNeutralize": [CLOSE, HIGH]}


def test_composants_calendaires():
    assert ind.year().to_json() == {"Year": TS}
    assert ind.week_of_year().to_json() == {"WeekOfYear": TS}
    assert ind.day_of_year().to_json() == {"DayOfYear": TS}
    assert ind.is_month_start().to_json() == {"IsMonthStart": TS}
    assert ind.is_month_end().to_json() == {"IsMonthEnd": TS}
    assert ind.is_quarter_end().to_json() == {"IsQuarterEnd": TS}
    assert ind.is_weekend().to_json() == {"IsWeekend": TS}


def test_transformations_de_prix_passent_par_le_registre_de_fonctions():
    assert ind.median_price().to_json() == {"Function": ["median_price", [HIGH, LOW]]}
    assert ind.typical_price().to_json() == {
        "Function": ["typical_price", [HIGH, LOW, CLOSE]]
    }
    assert ind.weighted_close().to_json() == {
        "Function": ["weighted_close", [HIGH, LOW, CLOSE]]
    }
    assert ind.average_price().to_json() == {
        "Function": ["average_price", [OPEN, HIGH, LOW, CLOSE]]
    }


def test_une_figure_en_chandeliers_recoit_les_quatre_colonnes_ohlc():
    assert ind.cdl_doji().to_json() == {
        "Function": ["cdl_doji", [OPEN, HIGH, LOW, CLOSE]]
    }


# ---------------------------------------------------------------------------
# The compiler accepts it
# ---------------------------------------------------------------------------

MONTE = ind.close > ind.close.lag(1)

CASES = {
    "plus_di": ind.plus_di(14),
    "minus_di": ind.minus_di(14),
    "aroon_up": ind.aroon_up(25),
    "aroon_down": ind.aroon_down(25),
    "aroon_oscillator": ind.aroon_oscillator(25),
    "ppo": ind.ppo(ind.close),
    "trix": ind.trix(ind.close),
    "stoch_rsi": ind.stoch_rsi(ind.close),
    "stoch_d": ind.stoch_d(14, 3),
    "donchian_upper": ind.donchian_channels(20)[0],
    "vortex_plus": ind.vortex(14)[0],
    "cmf": ind.cmf(20),
    "rolling_var": ind.rolling_var(ind.close, 20),
    "rolling_skew": ind.rolling_skew(ind.close, 20),
    "rolling_kurt": ind.rolling_kurt(ind.close, 20),
    "rolling_rank": ind.rolling_rank(ind.close, 20),
    "rolling_quantile": ind.rolling_quantile(ind.close, 20, 0.25),
    "rolling_argmax": ind.rolling_argmax(ind.close, 20),
    "rolling_argmin": ind.rolling_argmin(ind.close, 20),
    "rolling_corr": ind.rolling_corr(ind.close, ind.high, 20),
    "rolling_cov": ind.rolling_cov(ind.close, ind.high, 20),
    "rolling_beta": ind.rolling_beta(ind.close, ind.high, 20),
    "bars_since": ind.bars_since(MONTE),
    "streak": ind.streak(MONTE),
    "count_over": ind.count_over(MONTE, 10),
    "value_when": ind.value_when(MONTE, ind.close),
    "ffill": ind.close.ffill(),
    "rising": ind.rising(ind.close, 3),
    "falling": ind.falling(ind.close, 3),
    "pivot_high": ind.pivot_high(ind.high, 3, 3),
    "pivot_low": ind.pivot_low(ind.low, 3, 3),
    "year": ind.year(),
    "week_of_year": ind.week_of_year(),
    "day_of_year": ind.day_of_year(),
    "is_month_start": ind.is_month_start(),
    "is_month_end": ind.is_month_end(),
    "is_quarter_end": ind.is_quarter_end(),
    "is_weekend": ind.is_weekend(),
    "sin": ind.sin(ind.close),
    "cos": ind.cos(ind.close),
    "tan": ind.tan(ind.close),
    "asin": ind.asin(ind.close),
    "acos": ind.acos(ind.close),
    "atan": ind.atan(ind.close),
    "sinh": ind.sinh(ind.close),
    "cosh": ind.cosh(ind.close),
    "log10": ind.log10(ind.close),
    "median_price": ind.median_price(),
    "typical_price": ind.typical_price(),
    "weighted_close": ind.weighted_close(),
    "average_price": ind.average_price(),
}


def test_chaque_indicateur_compile_cote_rust():
    for name, expr in CASES.items():
        summary = _compile(expr)
        assert "probe" in summary["signal_names"], name


def test_cross_sectional_ops_compile_too():
    # Separate: these ops require a COLUMN as their argument, not a
    # sub-expression, and the orchestrator intercepts them instead of evaluating
    # them symbol by symbol.
    for name, expr in {
        "cs_zscore": ind.cs_zscore(ind.close),
        "cs_demean": ind.cs_demean(ind.close),
        "cs_std": ind.cs_std(ind.close),
        "cs_scale": ind.cs_scale(ind.close),
        "cs_winsorize": ind.cs_winsorize(ind.close, 2.5),
        "cs_quantile": ind.cs_quantile(ind.close, 0.9),
        "cs_neutralize": ind.cs_neutralize(ind.close, ind.high),
    }.items():
        summary = _compile(expr)
        assert "probe" in summary["signal_names"], name


# The 38 ported candlestick patterns, in the order the engine declares them.
FIGURES = [
    "cdl_doji", "cdl_spinning_top", "cdl_long_legged_doji", "cdl_short_line",
    "cdl_long_line", "cdl_high_wave", "cdl_rickshaw_man", "cdl_marubozu",
    "cdl_closing_marubozu", "cdl_belt_hold", "cdl_dragonfly_doji",
    "cdl_gravestone_doji", "cdl_engulfing", "cdl_hammer", "cdl_inverted_hammer",
    "cdl_hanging_man", "cdl_shooting_star", "cdl_takuri", "cdl_matching_low",
    "cdl_homing_pigeon", "cdl_harami", "cdl_harami_cross", "cdl_doji_star",
    "cdl_piercing", "cdl_thrusting", "cdl_counterattack", "cdl_three_inside",
    "cdl_three_outside", "cdl_morning_star", "cdl_evening_star",
    "cdl_dark_cloud_cover", "cdl_three_white_soldiers", "cdl_two_crows",
    "cdl_identical_three_crows", "cdl_tristar", "cdl_separating_lines",
    "cdl_on_neck", "cdl_kicking",
]


def test_every_period_argument_accepts_a_swept_param():
    """Every period argument accepts `mbt.param()`, so any of them can be swept.

    The failure mode this guards against is quiet and lands far from its cause:
    an indicator that forwards a raw scalar to the serialiser instead of an
    expression raises `TypeError: Object of type Expr is not JSON serializable`
    when the strategy is serialised, nowhere near the call that built it. The
    whole HLC family and the MACD / Bollinger helpers are covered here so the
    promise holds across all of them, new and old.
    """
    p = bt.param("p", default=14, range=(8, 30))
    f = bt.param("m", default=2.0, range=(1.0, 4.0))
    rising = ind.close > ind.close.lag(1)

    cases = {
        # Pre-existing.
        "adx": ind.adx(p),
        "atr": ind.atr(p),
        "cci": ind.cci(p),
        "natr": ind.natr(p),
        "stoch_k": ind.stoch_k(p),
        "williams_r": ind.williams_r(p),
        "mfi": ind.mfi(p),
        "supertrend": ind.supertrend(p, f),
        "keltner_upper": ind.keltner_channels(p, f)[0],
        "keltner_middle": ind.keltner_channels(p, f)[1],
        "parabolic_sar": ind.parabolic_sar(f, f),
        "macd_line": ind.macd(ind.close, p, p, p)[0],
        "bollinger_upper": ind.bollinger_bands(ind.close, p, f)[0],
        # Newer additions.
        "plus_di": ind.plus_di(p),
        "minus_di": ind.minus_di(p),
        "aroon_up": ind.aroon_up(p),
        "aroon_down": ind.aroon_down(p),
        "aroon_oscillator": ind.aroon_oscillator(p),
        "rolling_var": ind.rolling_var(ind.close, p),
        "rolling_rank": ind.rolling_rank(ind.close, p),
        "rolling_quantile": ind.rolling_quantile(ind.close, p, f),
        "rolling_corr": ind.rolling_corr(ind.close, ind.high, p),
        "count_over": ind.count_over(rising, p),
        "rising": ind.rising(ind.close, p),
        "pivot_high": ind.pivot_high(ind.high, p, p),
        "cs_winsorize": ind.cs_winsorize(ind.close, f),
        "cs_quantile": ind.cs_quantile(ind.close, f),
        "stoch_d": ind.stoch_d(p, p),
        "trix": ind.trix(ind.close, p),
        "ppo": ind.ppo(ind.close, p, p),
        "donchian_upper": ind.donchian_channels(p)[0],
        "vortex_plus": ind.vortex(p)[0],
        "cmf": ind.cmf(p),
    }
    for name, expr in cases.items():
        # The parameter name must come out as a STRING in the JSON: that is
        # what the engine reads as a swept period / factor rather than a
        # literal.
        rendered = json.dumps(expr.to_json())
        assert '"p"' in rendered or '"m"' in rendered, f"{name}: the param did not survive serialisation"
        summary = _compile(expr)
        assert "probe" in summary["signal_names"], name


def test_les_38_figures_sont_exposees_et_compilent():
    assert len(FIGURES) == 38
    for name in FIGURES:
        factory = getattr(ind, name, None)
        assert factory is not None, f"{name} is not exposed by manifoldbt.indicators"
        summary = _compile(factory())
        assert "probe" in summary["signal_names"], name
        # A pattern reads all four columns, including the ones a partial OHLC
        # would let through: this is what guarantees the engine loads them.
        for column in ("open", "high", "low", "close"):
            assert column in summary["required_columns"], f"{name}: {column}"
