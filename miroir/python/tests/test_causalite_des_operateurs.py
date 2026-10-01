"""Le detecteur de biais voit TOUS les operateurs non causaux, pas seulement un.

Un operateur est causal si sa valeur a la barre `t` ne depend que des barres
`0..=t`. `Expr.rank()` trie la serie entiere, donc le rang d'une barre bouge
quand on ajoute des barres APRES elle. C'est un look-ahead, et il alimentait
des signaux d'entree sans que rien ne le signale : le sous-test statique de
`detect_lookahead` ne cherchait qu'un seul variant, `Lead`, ecrit en dur dans
son parcours.

Ces tests epinglent les trois moities du correctif :

- la mesure, par troncature a plusieurs points de coupe, qui montre que
  `rank()` change des decisions deja prises et que `rolling_rank()` non ;
- le detecteur, qui doit maintenant nommer l'operateur et echouer ;
- la surface Python, renommee en `full_series_rank()` avec l'ancien nom garde
  en alias qui previent.

La methode de mesure importe. Un seul point de coupe ne prouve rien : un
look-ahead de portee finie ne perturbe que la fin du prefixe et peut disparaitre
au point choisi. On prend donc plusieurs coupes, et `lead(5)` sert de temoin
non causal connu : si la methode ne le rattrape pas, elle ne conclut rien.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
from manifoldbt.expr import col, lit, when
from manifoldbt.indicators import sma

# Le detecteur est Pro. La CI publique tourne la roue PyPI sans licence, donc
# en Community : ces tests-la y sauteraient au lieu d'y rougir.
pro_seulement = pytest.mark.skipif(
    not bt._is_pro(), reason="detect_lookahead est Pro")

N_BARRES = 600
COUPES = (250, 350, 450)


def _barres(n: int) -> pd.DataFrame:
    """Marche aleatoire sans derive : aucun operateur causal ne doit y gagner."""
    rng = np.random.default_rng(20260915)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close * 1.001,
        "low": close * 0.999, "close": close, "volume": np.full(n, 1000.0),
    })


_COMPLET = _barres(N_BARRES)


def _magasin(df: pd.DataFrame):
    racine = tempfile.mkdtemp(prefix="causalite_")
    return bt.import_dataframe(
        df, symbol="CAUSAL", symbol_id=1, interval="1m",
        data_root=os.path.join(racine, "data"),
        metadata_db=os.path.join(racine, "meta.sqlite"))


def _config() -> bt.BacktestConfig:
    debut, fin = bt.time_range("2023-12-31", "2024-01-03")
    return bt.BacktestConfig(
        universe=[1], time_range_start=debut, time_range_end=fin,
        initial_capital=10_000.0,
        execution=bt.ExecutionConfig(execution_price="AtClose", signal_delay=1))


def _strategie(condition) -> bt.Strategy:
    taille = when(condition, lit(1.0), lit(0.0))
    return bt.Strategy.create("causalite").signal("pos", taille).size(col("pos"))


def _fills(df: pd.DataFrame, condition) -> list:
    res = bt.run(_strategie(condition), _config(), _magasin(df))
    trades = res.trades_df()
    if trades is None or len(trades) == 0:
        return []
    colonne = "timestamp" if "timestamp" in trades.columns else trades.columns[0]
    return [str(v) for v in trades[colonne].tolist()]


def _coupes_divergentes(condition) -> list:
    """Les coupes ou tronquer la serie change les fills deja decides."""
    entiers = _fills(_COMPLET, condition)
    divergentes = []
    for k in COUPES:
        tronquee = _COMPLET.iloc[:k].reset_index(drop=True)
        horizon = str(tronquee["timestamp"].iloc[-1])
        attendu = [ts for ts in entiers if ts <= horizon]
        if attendu != _fills(tronquee, condition):
            divergentes.append(k)
    return divergentes


# ------------------------------------------------------------- la mesure ----

def test_le_temoin_non_causal_est_bien_rattrape():
    """Sans ce test, les suivants ne prouvent rien.

    `lead(5)` lit cinq barres devant par construction. Si la troncature ne le
    voit pas, la methode est aveugle et son verdict sur le rang ne vaut rien.

    Une divergence SUFFIT, et exiger les trois serait faux : une lecture de
    portee finie ne perturbe que les toutes dernieres barres du prefixe, donc
    elle peut ne rien changer a une coupe donnee. C'est la raison pour laquelle
    il faut plusieurs coupes. Mesure ici : `lead(5)` diverge a une coupe sur
    trois, la lecture de la serie entiere aux trois.
    """
    assert _coupes_divergentes(col("close").lead(5) > col("close")) != []


@pytest.mark.parametrize("nom,condition", [
    ("sma", col("close") > sma(col("close"), 20)),
    ("rolling_rank", col("close").rolling_rank(60) > lit(0.9)),
    ("cumsum", col("close").cumsum() > lit(0.0)),
])
def test_les_operateurs_causaux_ne_changent_pas_le_passe(nom, condition):
    assert _coupes_divergentes(condition) == [], nom


def test_le_rang_sur_serie_entiere_change_des_decisions_deja_prises():
    """Le defaut, mesure de bout en bout.

    Le seuil est un rang absolu (la moitie de la serie complete), donc le
    nombre de barres qui le passent depend de la longueur de la serie. C'est
    exactement la forme du defaut : la meme barre est classee differemment
    selon ce qui vient apres elle.
    """
    condition = col("close").full_series_rank() > lit(float(N_BARRES) * 0.5)
    # Une divergence suffirait pour conclure ; on epingle les trois parce que
    # ce look-ahead-la est GLOBAL, pas de portee finie, donc il doit se voir
    # partout. Le jour ou l'une des trois passe, c'est que la semantique de
    # l'operateur a change, et le test doit le dire.
    assert _coupes_divergentes(condition) == list(COUPES)


# ----------------------------------------------------------- le detecteur ---

@pro_seulement
def test_le_sous_test_statique_nomme_le_rang_sur_serie_entiere():
    """C'est le point du correctif : avant, ce sous-test rendait PASS."""
    strategie = _strategie(
        col("close").full_series_rank() > lit(float(N_BARRES) * 0.5))
    rapport = bt.diagnostics.detect_lookahead(
        strategie, _config(), _magasin(_COMPLET), mode="static")
    texte = str(rapport)
    assert "PASS" not in texte or "FAIL" in texte, texte
    assert "whole series" in texte, texte
    assert "rolling_rank" in texte, "un refus doit nommer la sortie: " + texte


@pro_seulement
def test_le_sous_test_statique_voit_encore_lead():
    """Le seul operateur qu'il connaissait avant doit rester couvert."""
    strategie = _strategie(col("close").lead(3) > col("close"))
    texte = str(bt.diagnostics.detect_lookahead(
        strategie, _config(), _magasin(_COMPLET), mode="static"))
    assert "3 bars ahead" in texte, texte


@pro_seulement
@pytest.mark.parametrize("nom,condition", [
    ("rolling_rank", col("close").rolling_rank(60) > lit(0.9)),
    ("sma", col("close") > sma(col("close"), 20)),
])
def test_le_sous_test_statique_laisse_passer_les_operateurs_causaux(nom, condition):
    """Un detecteur qui refuse tout ne sert a rien non plus."""
    texte = str(bt.diagnostics.detect_lookahead(
        _strategie(condition), _config(), _magasin(_COMPLET), mode="static"))
    assert "FAIL" not in texte, nom + ": " + texte


# ------------------------------------------------------- la surface Python --

def test_lancien_nom_marche_encore_et_previent():
    """Une strategie deja ecrite continue de tourner, en disant quoi corriger."""
    with pytest.warns(DeprecationWarning) as prises:
        expr = col("close").rank()
    message = str(prises[0].message)
    assert "whole series" in message, message
    assert "rolling_rank" in message, message
    # Le JSON ne bouge pas : c'est le nom Python qui change, pas le format,
    # donc une strategie serialisee par une version anterieure se relit.
    assert expr.to_json() == col("close").full_series_rank().to_json()


def test_le_nouveau_nom_ne_previent_pas():
    """Sinon l'avertissement devient du bruit que personne ne lit."""
    import warnings
    with warnings.catch_warnings(record=True) as prises:
        warnings.simplefilter("always")
        col("close").full_series_rank()
    assert [p for p in prises if issubclass(p.category, DeprecationWarning)] == []


def test_la_docstring_dit_que_loperateur_nest_pas_causal():
    """Le defaut d'origine tenait autant a l'absence de docstring qu'au code."""
    doc = bt.expr.Expr.full_series_rank.__doc__ or ""
    assert "not causal" in doc, doc
    assert "rolling_rank" in doc, doc
