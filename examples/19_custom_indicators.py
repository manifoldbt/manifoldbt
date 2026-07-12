"""Créer ses propres indicateurs (indicateurs absents de la base).

    python examples/19_custom_indicators.py

────────────────────────────────────────────────────────────────────────────
LE MODÈLE MENTAL
────────────────────────────────────────────────────────────────────────────
Un indicateur, ici, n'est RIEN d'autre qu'une fonction Python qui renvoie un
`Expr`. Un `Expr` est un *nœud dans un graphe de calcul* : quand vous écrivez
`(high + low) / 2`, aucune donnée n'est touchée — vous décrivez une opération.
Le graphe complet est ensuite compilé et évalué **en Rust**, en une passe,
vectorisé. C'est pour ça que vos indicateurs maison tournent à la vitesse des
indicateurs natifs : ils finissent dans le même moteur.

Toute la lib `manifoldbt.indicators` est écrite comme ça (`sma` ==
`source.rolling_mean(period)`). Donc « ajouter un indicateur » = « écrire une
fonction qui compose des `Expr` ». Trois niveaux, du plus simple au plus rare.
"""

import os
from time import perf_counter

import manifoldbt as mbt
# Colonnes de base (ce sont déjà des Expr) + quelques helpers.
from manifoldbt.indicators import open, high, low, close, volume, sma, rsi, ema
# Briques bas niveau : lit (constante), col (colonne par nom), when (if/else),
# scan/s (état récursif), param (paramètre balayable).
from manifoldbt.expr import lit, col, when, scan, s, param
from manifoldbt.helpers import time_range, Slippage, Interval


# ═══════════════════════════════════════════════════════════════════════════
# NIVEAU 1 — COMPOSER LES PRIMITIVES  (99 % des cas)
# ═══════════════════════════════════════════════════════════════════════════
# On combine colonnes + opérateurs (+ - * /, > < >= & | ~) + méthodes d'Expr
# (rolling_mean/std/min/max/median, ewm_mean, zscore, pct_change, diff, lag,
#  rsi, linreg_*, cross_above/below, cumsum, rank, ...). Chaque appel renvoie
# un Expr, donc tout se chaîne.

def awesome_oscillator(fast=5, slow=34):
    """Awesome Oscillator (Bill Williams) — ABSENT de la base.

        AO = SMA(prix médian, 5) − SMA(prix médian, 34),  prix médian = (H+L)/2

    Momentum : positif = pression acheteuse, négatif = vendeuse.
    """
    median_price = (high + low) / 2          # Expr : opération sur 2 colonnes
    return sma(median_price, fast) - sma(median_price, slow)   # Expr résultat


def dist_to_ma_pct(period=20):
    """Écart en % du prix à sa moyenne mobile — ABSENT de la base.

    Négatif = le prix est SOUS sa moyenne (survendu) → brique idéale pour du
    retour à la moyenne. Une seule ligne de composition.
    """
    ma = sma(close, period)
    return (close - ma) / ma * 100.0


def intraday_range_pct():
    """Amplitude de la bougie en % du close — ABSENT de la base.

    Un proxy de volatilité instantané. Montre qu'on mélange librement les
    colonnes OHLC.
    """
    return (high - low) / close * 100.0


def rsi_zscore(period=14, lookback=365):
    """RSI standardisé : à quel point le RSI est extrême vs SA PROPRE histoire.

    Compose un indicateur natif (rsi) avec des stats roulantes. C'est
    exactement le motif utilisé dans strategies/rsi_dynamic_alloc.py.
    """
    r = rsi(close, period)
    return (r - r.rolling_mean(lookback)) / r.rolling_std(lookback)


# ═══════════════════════════════════════════════════════════════════════════
# NIVEAU 2 — `scan` : INDICATEURS À ÉTAT / RÉCURSIFS
# ═══════════════════════════════════════════════════════════════════════════
# Quand la valeur d'aujourd'hui dépend de celle d'HIER (récursion) et qu'aucun
# rolling ne suffit, on utilise `scan`. Il tourne comme une petite VM scalaire,
# entièrement en Rust (pas de callback Python par barre).
#
#   scan(state=..., update=..., output=...)
#     • state  : variables d'état + leur valeur initiale (1re ligne)
#     • update : expressions évaluées à chaque barre, DANS L'ORDRE
#                - s.prev("x") = valeur de "x" à la barre précédente
#                - s.var("k")  = valeur calculée plus tôt DANS LE MÊME pas
#                - si un nom d'update == un nom d'état, on réécrit cet état
#     • output : quelle variable émettre comme résultat
#
# Preuve que c'est puissant : le Kalman et le GARCH livrés sont écrits
# UNIQUEMENT avec scan (voir manifoldbt/indicators.py).

def up_streak():
    """Nombre de bougies HAUSSIÈRES consécutives — ABSENT de la base, et
    impossible avec un simple rolling (il faut un compteur qui se réinitialise).

        streak = streak_précédent + 1  si close > close(-1),  sinon 0
    """
    is_up = close > close.lag(1)             # Expr booléen (1.0 / 0.0) par barre
    return scan(
        state={"n": lit(0.0)},               # compteur initialisé à 0
        update={
            # if is_up: prev(n) + 1  else: 0
            "n": when(is_up, s.prev("n") + lit(1.0), lit(0.0)),
        },
        output="n",
    )


def ema_from_scratch(alpha=0.1):
    """EMA « à la main » via scan — juste pour illustrer le mécanisme.
    (L'EMA existe en natif : `ema(close, span)`. Ici c'est pédagogique.)

        ema = alpha * close + (1 - alpha) * ema_précédent
    """
    return scan(
        state={"ema": close},                # graine = 1er close
        update={"ema": lit(alpha) * close + lit(1.0 - alpha) * s.prev("ema")},
        output="ema",
    )


# ═══════════════════════════════════════════════════════════════════════════
# NIVEAU 3 — LES LIMITES (À CONNAÎTRE)
# ═══════════════════════════════════════════════════════════════════════════
# • PAS de callback Python par barre : `scan` s'exécute en Rust, on ne peut pas
#   y injecter une fonction Python appelée sur chaque bougie (ce serait lent).
#   Tant que la logique s'exprime avec Expr + when + scan, ça passe.
# • Un indicateur VRAIMENT nouveau, non exprimable ainsi, demande d'ajouter un
#   variant `Expr` + son kernel côté Rust — chemin contributeur, pas utilisateur.
# • Données externes (hashrate, funding, sentiment…) : `mbt.register_exo(...)`
#   puis `exo("nom")` renvoie un Expr utilisable comme n'importe quelle colonne.


# ═══════════════════════════════════════════════════════════════════════════
# BONUS — RENDRE SON INDICATEUR BALAYABLE (sweep)
# ═══════════════════════════════════════════════════════════════════════════
# Les périodes acceptent `param(...)` à la place d'un entier. Le moteur
# recompile alors une fois par combinaison et balaie la grille en parallèle,
# sans changer une ligne de l'indicateur :
#
#   ao = awesome_oscillator(fast=param("fast"), slow=param("slow"))
#   # puis, avec la grille passée séparément (l'indicateur ne change pas) :
#   #   batch = mbt.run_sweep_lite(
#   #       strategy,
#   #       {"fast": [3, 5, 8], "slow": [21, 34, 55]},
#   #       config, store,
#   #   )
#
# (voir examples/08_sweep_2d_heatmap.py pour le sweep complet.)


# ═══════════════════════════════════════════════════════════════════════════
# METTRE UN INDICATEUR MAISON DANS UNE STRATÉGIE + BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
# On utilise `dist_to_ma_pct` (retour à la moyenne) : long quand le prix est
# nettement sous sa moyenne, on sort quand il l'a rejointe.

dist = dist_to_ma_pct(period=48)             # notre indicateur maison
streak = up_streak()                         # et un second, pour l'exposer aussi

signal = when(dist < -5.0, 1.0,              # >5 % sous la MM → achat du creux
         when(dist > 0.0, 0.0))              # revenu à la MM → sortie, sinon hold

strategy = (
    mbt.Strategy.create("custom_indicator_demo")
    .signal("dist_to_ma_%", dist)            # .signal() = exposer pour le rapport
    .signal("up_streak", streak)
    .size(signal)
    .describe("Retour à la moyenne piloté par un indicateur maison (écart à la MM)")
)

# -- Config -------------------------------------------------------------------
start, end = time_range("2021-01-01", "2026-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(1),
    initial_capital=10_000,
    execution=mbt.ExecutionConfig(allow_short=False, max_position_pct=1.0),
    fees=mbt.FeeConfig.zero(),               # sans frais, pour l'exemple
    slippage=Slippage.fixed_bps(2),
    warmup_bars=60,                          # >= la plus longue fenêtre utilisée
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

    t0 = perf_counter()
    result = mbt.run(strategy, config, store)
    print(result.summary())
    print(f"\nElapsed: {perf_counter() - t0:.2f}s")
