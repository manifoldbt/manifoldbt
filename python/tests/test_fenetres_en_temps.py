"""Les fenetres en duree, contre pandas, sur une grille a trous.

Deux choses separees ici, et la seconde est celle qui attrape les regressions.

1. La FORME du JSON. Une duree se serialise `{"duration_ns": n}`, un objet, la
   ou un compte de barres est un nombre et un parametre balaye une chaine. Les
   trois formes sont disjointes : c'est ce qui permet a l'ancien format de ne
   pas bouger. Un operateur sans version en temps refuse la duree.

2. Les VALEURS, contre `pandas.rolling("3D")` et
   `pandas.ewm(halflife=..., times=...)`, sur un calendrier a trous.

Comment on lit une serie d'indicateur depuis Python : par les positions. Les
barres sont journalieres et valent 1.0, le capital vaut 1.0, la taille est
l'indicateur divise par une puissance de deux, donc la quantite en position EST
la valeur de l'expression, au bit pres. Le calendrier journalier a trous joue
exactement le role d'une grille a la seconde a trous, et la sortie journaliere
du moteur y rend une ligne par barre : le test dit la meme chose avec ou sans
licence.
"""
import json

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import manifoldbt as bt  # noqa: E402
from manifoldbt import indicators as ind  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

CLOSE = {"Column": "close"}
# Division exacte : la taille passe par /K et revient par *K sans arrondi.
K = 1024.0


# ---------------------------------------------------------------------------
# Forme du JSON
# ---------------------------------------------------------------------------


def test_une_duree_se_serialise_en_objet_et_un_compte_de_barres_en_nombre():
    assert col("close").rolling_mean(30).to_json() == {"RollingMean": [CLOSE, 30]}
    assert col("close").rolling_mean(Interval.seconds(30)).to_json() == {
        "RollingMean": [CLOSE, {"duration_ns": 30_000_000_000}]
    }
    assert col("close").rolling_sum(Interval.minutes(5)).to_json() == {
        "RollingSum": [CLOSE, {"duration_ns": 300_000_000_000}]
    }
    assert col("close").rolling_std(Interval.hours(1)).to_json() == {
        "RollingStd": [CLOSE, {"duration_ns": 3_600_000_000_000}]
    }
    assert col("close").zscore(Interval.days(2)).to_json() == {
        "ZScore": [CLOSE, {"duration_ns": 172_800_000_000_000}]
    }


def test_les_paires_et_letat_de_signal_portent_la_meme_duree():
    a, b = col("close"), col("volume")
    vol = {"Column": "volume"}
    d30 = {"duration_ns": 30_000_000_000}
    assert a.rolling_corr(b, Interval.seconds(30)).to_json() == {"RollingCorr": [CLOSE, vol, d30]}
    assert a.rolling_cov(b, Interval.seconds(30)).to_json() == {"RollingCov": [CLOSE, vol, d30]}
    assert a.rolling_beta(b, Interval.seconds(30)).to_json() == {"RollingBeta": [CLOSE, vol, d30]}
    cond = (a > lit(1.0))
    assert cond.count_over(Interval.seconds(30)).to_json() == {
        "CountOver": [cond.to_json(), d30]
    }
    assert cond.time_since().to_json() == {"TimeSince": cond.to_json()}


def test_ewm_prend_une_portee_en_barres_ou_une_demi_vie_en_temps():
    assert col("close").ewm_mean(12).to_json() == {"EwmMean": [CLOSE, 12.0]}
    attendu = {"EwmMeanHalflife": [CLOSE, {"duration_ns": 2_000_000_000}]}
    assert col("close").ewm_mean(halflife=Interval.seconds(2)).to_json() == attendu
    # Une duree en positionnel se lit comme une demi-vie : une portee en temps
    # n'a pas de sens, et l'ecriture courte est celle qu'on veut lire.
    assert col("close").ewm_mean(Interval.seconds(2)).to_json() == attendu
    with pytest.raises(TypeError):
        col("close").ewm_mean(12, halflife=Interval.seconds(2))
    with pytest.raises(TypeError):
        col("close").ewm_mean()


def test_count_over_seul_compte_les_lignes_de_la_fenetre():
    # `count_over(Interval)` sans condition : le nombre de barres imprimees.
    j = ind.count_over(Interval.seconds(30)).to_json()
    assert j == {"CountOver": [{"Literal": {"Bool": True}}, {"duration_ns": 30_000_000_000}]}


def test_un_operateur_sans_horloge_refuse_la_duree_des_la_construction():
    for construire in (
        lambda: col("close").rsi(Interval.seconds(30)),
        lambda: col("close").lag(Interval.seconds(30)),
        lambda: col("close").rolling_median(Interval.seconds(30)),
        lambda: col("close").rolling_quantile(Interval.seconds(30), 0.5),
    ):
        with pytest.raises(TypeError, match="counts BARS"):
            construire()


def test_le_moteur_accepte_une_fenetre_en_duree():
    # Une forme plausible mais fausse serait acceptee par le serialiseur et
    # rejetee ici : ce test dit que le moteur compile bien ce qu'on ecrit.
    strategy = bt.Strategy(
        name="probe",
        signals={"probe": col("close").rolling_mean(Interval.seconds(30))},
        position_sizing=col("probe"),
    )
    json.loads(bt.compile_strategy_json(strategy.to_json()))


# ---------------------------------------------------------------------------
# Les valeurs, contre pandas
# ---------------------------------------------------------------------------


def _calendrier(n_jours: int = 70) -> pd.DatetimeIndex:
    """Un calendrier a trous : des jours isoles absents, et un bloc de six."""
    tous = pd.date_range("2023-01-01", periods=n_jours, freq="1D", tz="UTC")
    garde = [
        j for i, j in enumerate(tous)
        if i % 5 != 2 and i % 9 != 7 and not (30 <= i < 36)
    ]
    return pd.DatetimeIndex(garde)


def _serie_source(n: int) -> np.ndarray:
    rng = np.random.default_rng(20260907)
    return np.cumsum(rng.normal(0.0, 1.0, n)) + 10.0


@pytest.fixture(scope="module")
def banc(tmp_path_factory):
    """Un store journalier a trous, prix constant a 1.0, la source en volume.

    Le prix est constant pour que la quantite en position soit exactement la
    valeur de la taille ; la serie qui porte l'information voyage donc dans
    `volume`, que rien ne contraint.
    """
    ts = _calendrier()
    n = len(ts)
    un = np.ones(n)
    src = _serie_source(n)
    df = pd.DataFrame({
        "timestamp": ts, "open": un, "high": un, "low": un, "close": un,
        "volume": src,
    })
    root = tmp_path_factory.mktemp("fenetres_en_temps")
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1d",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )
    return store, ts, src


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
    """La serie que l'expression calcule, lue dans les positions.

    Prix 1.0, capital 1.0 : quantite = taille. La taille passe par /K pour
    rester sous le plafond de position, et revient par *K, une puissance de
    deux, donc sans arrondi. Une ligne NaN sort a 0.0 (pas de position).
    """
    store, ts, _ = banc
    strat = (bt.Strategy.create("probe")
             .signal("probe", expr / lit(K))
             .size(col("probe")))
    res = bt.run(strat, _config(ts), store)
    pos = res.positions_df()
    assert len(pos) == len(ts), "la sortie doit rendre une ligne par barre"
    return pos["position"].to_numpy() * K


def _compare(obtenu, attendu_pd, ts, duree_jours, nom, rtol=1e-9):
    """Compare hors chauffe, et exige la chauffe a zero (donc NaN cote moteur)."""
    chaud = (ts - ts[0]) >= pd.Timedelta(days=duree_jours)
    attendu = np.asarray(attendu_pd, dtype=float)
    froid = ~chaud
    assert np.all(obtenu[froid] == 0.0), (
        f"{nom}: la chauffe doit etre NaN (donc position nulle), "
        f"obtenu {obtenu[froid][:5]}"
    )
    a, b = obtenu[chaud], attendu[chaud]
    fini = np.isfinite(b)
    assert fini.sum() > 20, f"{nom}: seulement {fini.sum()} lignes comparees"
    np.testing.assert_allclose(a[fini], b[fini], rtol=rtol, atol=1e-12,
                               err_msg=f"{nom} contre pandas")


@pytest.mark.parametrize("nom,fabrique,reference", [
    ("rolling_mean", lambda s: s.rolling_mean(Interval.days(7)), lambda r: r.mean()),
    ("rolling_sum", lambda s: s.rolling_sum(Interval.days(7)), lambda r: r.sum()),
    ("rolling_std", lambda s: s.rolling_std(Interval.days(7)), lambda r: r.std(ddof=0)),
    ("rolling_min", lambda s: s.rolling_min(Interval.days(7)), lambda r: r.min()),
    ("rolling_max", lambda s: s.rolling_max(Interval.days(7)), lambda r: r.max()),
    ("rolling_var", lambda s: s.rolling_var(Interval.days(7)), lambda r: r.var(ddof=0)),
])
def test_la_famille_glissante_colle_a_pandas_rolling_7d(banc, nom, fabrique, reference):
    _, ts, src = banc
    # rolling_sum sur 7 jours depasse le plafond de position : on normalise la
    # source, ce que la reference pandas fait aussi.
    src_n = src / 100.0
    obtenu = _lit(banc, fabrique(col("volume") / lit(100.0)))
    serie = pd.Series(src_n, index=ts)
    attendu = reference(serie.rolling("7D"))
    _compare(obtenu, attendu, ts, 7, nom)


def test_le_zscore_colle_a_pandas(banc):
    _, ts, src = banc
    obtenu = _lit(banc, col("volume").zscore(Interval.days(7)))
    serie = pd.Series(src, index=ts)
    f = serie.rolling("7D")
    attendu = (serie - f.mean()) / f.std(ddof=0)
    _compare(obtenu, attendu, ts, 7, "zscore", rtol=1e-8)


@pytest.mark.parametrize("nom", ["corr", "cov", "beta"])
def test_les_statistiques_de_paire_collent_a_pandas(banc, nom):
    _, ts, src = banc
    a = col("volume") / lit(100.0)
    b = (col("volume") / lit(100.0)).lag(1)
    fabrique = {
        "corr": lambda: a.rolling_corr(b, Interval.days(10)),
        "cov": lambda: a.rolling_cov(b, Interval.days(10)),
        "beta": lambda: a.rolling_beta(b, Interval.days(10)),
    }[nom]
    obtenu = _lit(banc, fabrique())
    sa = pd.Series(src / 100.0, index=ts)
    sb = sa.shift(1)
    if nom == "corr":
        attendu = sa.rolling("10D").corr(sb)
    elif nom == "cov":
        attendu = sa.rolling("10D").cov(sb)
    else:
        attendu = sa.rolling("10D").cov(sb) / sb.rolling("10D").var(ddof=1)
    # Une seule ligne NaN dans la fenetre suffit a la supprimer des deux cotes :
    # `lag(1)` en pose une en tete de serie, donc la premiere fenetre pleine
    # arrive plus tard que la chauffe, et pandas la donne NaN aussi.
    _compare(obtenu, attendu, ts, 10, f"rolling_{nom}", rtol=1e-7)


def test_ewm_en_demi_vie_colle_a_pandas_avec_times(banc):
    _, ts, src = banc
    obtenu = _lit(banc, col("volume").ewm_mean(halflife=Interval.days(3)))
    serie = pd.Series(src, index=ts)
    attendu = serie.ewm(halflife="3D", times=ts).mean()
    # Pas de fenetre, donc pas de chauffe : toutes les lignes se comparent.
    np.testing.assert_allclose(obtenu, np.asarray(attendu, dtype=float), rtol=1e-9)


def test_count_over_compte_les_barres_reellement_imprimees(banc):
    _, ts, src = banc
    obtenu = _lit(banc, ind.count_over(Interval.days(7)))
    serie = pd.Series(1.0, index=ts)
    attendu = serie.rolling("7D").sum()
    _compare(obtenu, attendu, ts, 7, "count_over")
    chaud = (ts - ts[0]) >= pd.Timedelta(days=7)
    # Le calendrier a des trous : le compte doit tomber sous 7 quelque part,
    # sinon le test ne prouverait rien de plus qu'une grille pleine.
    assert obtenu[chaud].min() < 7.0


def test_time_since_compte_les_jours_pas_les_barres(banc):
    _, ts, src = banc
    seuil = float(np.median(src))
    obtenu = _lit(banc, ind.time_since(col("volume") > lit(seuil)) / lit(86400.0))
    vrai = src > seuil
    attendu = np.full(len(ts), np.nan)
    derniere = None
    for i in range(len(ts)):
        if vrai[i]:
            derniere = ts[i]
        if derniere is not None:
            attendu[i] = (ts[i] - derniere).total_seconds() / 86400.0
    defini = np.isfinite(attendu)
    np.testing.assert_allclose(obtenu[defini], attendu[defini], rtol=1e-9, atol=1e-12)
    # Et l'ecart avec le compte de barres est reel sur ce calendrier.
    barres = _lit(banc, ind.bars_since(col("volume") > lit(seuil)))
    ecarts = int(np.sum(np.abs(barres[defini] - attendu[defini]) > 0.5))
    assert ecarts > 5, f"seulement {ecarts} lignes ou barres et jours different"


def test_la_fenetre_en_duree_et_le_compte_de_barres_different_sur_grille_a_trous(banc):
    _, ts, src = banc
    en_temps = _lit(banc, col("volume").rolling_mean(Interval.days(7)))
    en_barres = _lit(banc, col("volume").rolling_mean(7))
    ecarts = int(np.sum(np.abs(en_temps - en_barres) > 1e-9))
    assert ecarts > 20, (
        "sur une grille a trous, sept jours et sept barres ne sont pas la meme "
        f"fenetre, seulement {ecarts} lignes different"
    )
