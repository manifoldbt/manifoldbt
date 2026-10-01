"""`round_to` / `floor_to` / `ceil_to`, de Python jusqu'au moteur.

Trois choses separees, et la troisieme est celle qui compte.

1. La FORME du JSON. Le pas voyage en flottant a cote de l'expression, comme la
   portee d'une EMA : range sous le mauvais jeu de noms, il produirait un JSON
   plausible que le moteur rejette, ou pire, accepte et relit de travers.
2. Les REFUS. Un pas nul, negatif, non fini ou balaye est refuse a l'ecriture,
   sur la ligne qui l'a ecrit, et non a la premiere barre du backtest.
3. Les VALEURS, contre une reference exacte en `fractions` (un `Fraction`
   construit sur un flottant vaut EXACTEMENT ce flottant, et `float()` d'un
   `Fraction` est correctement arrondi), et contre `numpy` -- dont la
   convention d'egalite est l'AUTRE, moitie vers le pair, ce que le dernier
   test epingle valeur par valeur. Un quatrieme test dit que l'arrondi est
   idempotent : arrondir un niveau deja sur la grille ne le bouge pas.

Comment on lit une serie depuis Python : par les positions, comme
`test_fenetres_en_temps.py`. Les barres sont journalieres et valent 1.0, le
capital vaut 1.0, la taille est l'expression divisee par une puissance de deux,
donc la quantite en position EST la valeur de l'expression, au bit pres. La
serie qui porte l'information voyage dans `volume`, que rien ne contraint. La
sortie journaliere du moteur rend une ligne par barre : le test dit la meme
chose avec ou sans licence.

UNE barre sur deux porte un zero, et c'est ce qui rend la lecture exacte au
bit pres. Le moteur atteint une position en ACCUMULANT des ecarts
(`position += cible - position`), donc une position lue apres trois cents
barres a derive de quelques ulps de la cible. En repassant a plat entre deux
valeurs, chaque valeur est atteinte depuis zero : `0 + (v - 0)` vaut `v`
exactement, et `v + (0 - v)` vaut zero exactement.
"""
import json
from decimal import Decimal
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, param  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

VOLUME = {"Column": "volume"}
CLOSE = {"Column": "close"}
# Division exacte : la taille passe par /K et revient par *K sans arrondi.
K = 1024.0

# Les grilles des venues, avec le decimal EXACT qu'elles s'ecrivent.
GRILLES = ["0.1", "0.01", "0.0001", "0.00001", "0.5", "1", "5", "25"]
MODES = ["round_to", "floor_to", "ceil_to"]


# ---------------------------------------------------------------------------
# Forme du JSON
# ---------------------------------------------------------------------------


def test_les_trois_operateurs_portent_une_expression_et_un_flottant():
    assert col("close").round_to(0.1).to_json() == {"RoundTo": [CLOSE, 0.1]}
    assert col("close").floor_to(0.01).to_json() == {"FloorTo": [CLOSE, 0.01]}
    assert col("close").ceil_to(25).to_json() == {"CeilTo": [CLOSE, 25.0]}


def test_le_moteur_accepte_les_trois():
    for methode in MODES:
        expr = getattr(col("close"), methode)(0.1)
        strategy = bt.Strategy(
            name="probe", signals={"probe": expr}, position_sizing=col("probe")
        )
        json.loads(bt.compile_strategy_json(strategy.to_json()))


# ---------------------------------------------------------------------------
# Refus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("methode", MODES)
def test_un_pas_qui_nest_pas_positif_est_refuse_a_lecriture(methode):
    arrondir = getattr(col("close"), methode)
    for mauvais in (0, 0.0, -0.1, -25, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError) as leve:
            arrondir(mauvais)
        assert methode in str(leve.value)


@pytest.mark.parametrize("methode", MODES)
def test_le_pas_est_un_litteral_pas_un_axe_de_balayage(methode):
    arrondir = getattr(col("close"), methode)
    with pytest.raises(TypeError, match="param"):
        arrondir(param("tick", default=0.1))
    with pytest.raises(TypeError):
        arrondir(col("close"))
    with pytest.raises(TypeError):
        arrondir("0.1")


# ---------------------------------------------------------------------------
# Les valeurs
# ---------------------------------------------------------------------------


def _valeurs() -> np.ndarray:
    """Des prix hors grille, plus les egalites exactes que la convention decide.

    `x + 0.25` et `x + 0.05` tombent pile entre deux crans d'une grille a 0.1 et
    a 0.01 quand ils sont representables, ce qui est le seul endroit ou "moitie
    vers le haut" et "moitie vers le pair" different.
    """
    rng = np.random.default_rng(20260908)
    aleatoires = rng.uniform(90.0, 110.0, 180)
    egalites = np.array(
        [100.0 + k * 0.25 for k in range(-8, 9)]
        + [100.0 + k * 0.5 for k in range(-4, 5)]
        + [0.125, -0.125, 2.5, -2.5, 12.5, -12.5]
    )
    return np.concatenate([aleatoires, -aleatoires, egalites])


_VALEURS = _valeurs()


def _indice(x: float, pas: str, mode: str) -> int:
    """Le cran de grille exact, en arithmetique rationnelle.

    `floor` et `ceil` bornent par le DOUBLE qu'un cran s'ecrit, pas par le
    decimal exact : c'est ce qui rend l'arrondi idempotent (le double ecrit
    `1.23` est un cheveu SOUS cent vingt-trois centiemes, donc un plancher
    strict repondrait `1.22`). Au plus un cran d'ecart, et seulement quand `x`
    est exactement ce double.
    """
    quotient = Fraction(x) / Fraction(pas)
    plancher = quotient.numerator // quotient.denominator
    reste = quotient - plancher
    if mode == "floor_to":
        indice = plancher
        if float(Fraction(indice + 1) * Fraction(pas)) <= x:
            indice += 1
    elif mode == "ceil_to":
        indice = plancher + (1 if reste > 0 else 0)
        if float(Fraction(indice - 1) * Fraction(pas)) >= x:
            indice -= 1
    else:  # moitie vers le HAUT
        indice = plancher + (1 if reste >= Fraction(1, 2) else 0)
    return indice


def _cran(valeur: float, pas: str) -> int:
    """Le cran qu'une valeur DEJA sur la grille occupe."""
    quotient = Fraction(valeur) / Fraction(pas)
    return round(quotient.numerator / quotient.denominator)


def _reference(x: float, pas: str, mode: str) -> float:
    """La reponse exacte, en arithmetique rationnelle.

    `Fraction(x)` vaut exactement le flottant `x`, `Fraction(pas)` vaut
    exactement le decimal ecrit, et `float()` d'un `Fraction` est une division
    entiere correctement arrondie : donc le resultat EST le double le plus
    proche du multiple exact.
    """
    return float(Fraction(_indice(x, pas, mode)) * Fraction(pas))


# Une barre sur deux porte un zero : la position repasse a plat entre deux
# valeurs, donc chaque valeur est atteinte depuis zero et lue au bit pres.
_ENTRELACE = np.zeros(2 * len(_VALEURS))
_ENTRELACE[0::2] = _VALEURS


@pytest.fixture(scope="module")
def banc(tmp_path_factory):
    """Un store journalier, prix constant a 1.0, les valeurs dans `volume`."""
    n = len(_ENTRELACE)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    un = np.ones(n)
    df = pd.DataFrame({
        "timestamp": ts, "open": un, "high": un, "low": un, "close": un,
        "volume": _ENTRELACE,
    })
    root = tmp_path_factory.mktemp("arrondi_au_pas")
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1d",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )
    return store, ts


def _config(ts) -> bt.BacktestConfig:
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=int(ts[-1].value) + 86_400_000_000_000,
        bar_interval=Interval.days(1),
        initial_capital=1.0,
        warmup_bars=0,
        execution=bt.ExecutionConfig(
            signal_delay=0, execution_price="AtClose", max_position_pct=100.0,
            allow_short=True, position_sizing_mode="FractionOfInitialCapital",
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
    )


def _lit(banc, expr) -> np.ndarray:
    """La serie que l'expression calcule, lue dans les positions (barres paires)."""
    store, ts = banc
    strat = (bt.Strategy.create("probe")
             .signal("probe", expr / lit(K))
             .size(col("probe")))
    res = bt.run(strat, _config(ts), store)
    positions = res.positions_df()
    assert len(positions) == len(ts), "la sortie doit rendre une ligne par barre"
    return positions["position"].to_numpy()[0::2] * K


def test_le_banc_rend_la_colonne_au_bit_pres(banc):
    """Sans quoi les comparaisons qui suivent mesureraient le harnais."""
    lu = _lit(banc, col("volume"))
    assert np.array_equal(lu, _VALEURS)


@pytest.mark.parametrize("pas", GRILLES)
@pytest.mark.parametrize("mode", MODES)
def test_les_valeurs_sont_le_double_que_le_litteral_designe(banc, pas, mode):
    obtenu = _lit(banc, getattr(col("volume"), mode)(float(pas)))
    attendu = np.array([_reference(x, pas, mode) for x in _VALEURS])
    ecarts = np.nonzero(obtenu != attendu)[0]
    assert len(ecarts) == 0, (
        f"pas={pas} {mode}: {len(ecarts)} ecarts, premier sur "
        f"x={_VALEURS[ecarts[0]]!r} -> {obtenu[ecarts[0]]!r} au lieu de "
        f"{attendu[ecarts[0]]!r}"
    )


def test_le_pas_est_lu_comme_le_decimal_ecrit_pas_comme_son_double(banc):
    """`100.34` sur une grille a 0.1 rend `100.3`, pas `100.30000000000001`.

    C'est toute la difference entre l'operateur et le `round(x / pas) * pas` que
    tout le monde ecrit : un ulp, soit un niveau de prix different pour un
    carnet. Chaque valeur rendue est comparee au double qu'un litteral decimal a
    une decimale designe -- construit ici en `Decimal`, dont la conversion en
    flottant est correctement arrondie.
    """
    obtenu = _lit(banc, col("volume").round_to(0.1))
    naif = np.round(_VALEURS / 0.1) * 0.1
    assert np.any(obtenu != naif), "le test ne prouverait rien si les deux coincidaient"
    for x, got in zip(_VALEURS, obtenu):
        indice = _indice(x, "0.1", "round_to")
        assert got == float(Decimal(indice) * Decimal("0.1")), (
            f"x={x!r}: {got!r} au lieu de {indice} crans de 0.1"
        )


@pytest.mark.parametrize("pas", GRILLES)
@pytest.mark.parametrize("mode", MODES)
def test_arrondir_un_niveau_deja_sur_la_grille_ne_le_bouge_pas(banc, pas, mode):
    """La propriete sur laquelle repose le calage au tick cote moteur.

    Sans elle, un moteur qui recale un ordre au repos descendrait son niveau
    d'un tick a chaque requote.
    """
    une_fois = _lit(banc, getattr(col("volume"), mode)(float(pas)))
    for x in une_fois:
        for encore in MODES:
            assert _reference(x, pas, encore) == x, (
                f"pas={pas} {mode} puis {encore}: {x!r} a bouge"
            )


@pytest.mark.parametrize("pas", ["0.1", "0.01", "0.5", "1"])
def test_numpy_arrondit_a_moitie_vers_le_pair_et_le_moteur_vers_le_haut(banc, pas):
    """La convention est fixe et ne depend PAS de la parite de la valeur.

    `numpy.round` casse l'egalite vers le nombre pair, ce qui fait de
    `round(0.5)` un `0` et de `round(1.5)` un `2`. Le moteur casse toujours vers
    le haut. Hors egalite les deux disent le meme cran ; sur les egalites ils
    different, et ce test le montre plutot que de le contourner.
    """
    obtenu = _lit(banc, col("volume").round_to(float(pas)))
    indices_moteur = [_cran(v, pas) for v in obtenu]
    indices_numpy = np.round(_VALEURS / float(pas)).astype(np.int64)

    egalite = np.array(
        [(Fraction(x) / Fraction(pas)) % 1 == Fraction(1, 2) for x in _VALEURS]
    )
    assert egalite.any(), f"pas={pas}: aucune egalite dans l'echantillon"

    attendus = np.array([_indice(x, pas, "round_to") for x in _VALEURS], dtype=np.int64)
    moteur = np.array(indices_moteur, dtype=np.int64)
    assert np.array_equal(moteur, attendus)
    # Hors egalite, numpy dit la meme chose.
    assert np.array_equal(moteur[~egalite], indices_numpy[~egalite])
    # Sur une egalite, il descend au moins une fois -- la difference epinglee.
    assert np.any(moteur[egalite] != indices_numpy[egalite]), (
        f"pas={pas}: numpy devrait aller vers le pair sur au moins une egalite"
    )
