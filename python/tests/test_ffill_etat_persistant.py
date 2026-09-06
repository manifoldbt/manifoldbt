"""`ffill()` compose avec un filtre de regime, `hold()` non.

`hold()` retient la POSITION que porte le simulateur, pas la valeur de
l'expression qui le contient. Sous un filtre qui ecrit 0.0 pendant qu'il est
ferme, c'est ce 0.0 que `hold()` retient quand le filtre rouvre : l'exposition
ne revient qu'a la prochaine traversee de seuil.

Ce fichier epingle les deux comportements. Celui de `hold()` n'est PAS un bug a
corriger -- "sortir quand le regime est hostile, ne rentrer que sur un NOUVEAU
signal" est une strategie legitime -- mais il doit rester un choix, pas une
surprise, donc il est mesure ici pour que personne ne le change par accident.
"""
import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
from manifoldbt import indicators as ind
from manifoldbt.expr import col, hold, lit, when
from manifoldbt.helpers import Interval, Slippage

N_BARS = 3000
FENETRE, SEUIL = 200, 2.5


def _bars(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, N_BARS)))
    ts = pd.date_range("2023-01-01", periods=N_BARS, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close * 1.002,
        "low": close * 0.998, "close": close, "volume": 1000.0,
    })


_DF = _bars()
_END_NS = int(_DF["timestamp"].iloc[-1].value) + 86_400_000_000_000
# Le filtre est l'heure de la barre : ouvert exactement une barre sur deux, sans
# dependre des donnees. C'est le taux auquel l'exposition de `ffill` est comparee.
TAUX_OUVERTURE = 0.5


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("ffill_etat")
    return bt.import_dataframe(
        _DF, symbol="TEST", symbol_id=1, interval="1h",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )


def _config() -> bt.BacktestConfig:
    return bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=_END_NS,
        bar_interval=Interval.hours(1), initial_capital=10_000.0,
        execution=bt.ExecutionConfig(
            signal_delay=1, execution_price="AtOpen", max_position_pct=1.0,
            allow_short=True, position_sizing_mode="FractionOfInitialCapital",
        ),
        fees=bt.FeeConfig.zero(), slippage=Slippage.none(),
        warmup_bars=FENETRE + 10,
    )


_Z = col("close").zscore(FENETRE)
_HAUT, _BAS = _Z >= lit(SEUIL), _Z <= lit(-SEUIL)
_FILTRE = ind.hour() < lit(12.0)

# La branche fausse omise vaut NaN, donc ffill retient le dernier etat arme.
_ETAT_FFILL = when(_HAUT, lit(1.0), when(_BAS, lit(-1.0))).ffill()
_ETAT_HOLD = when(_HAUT, lit(1.0), when(_BAS, lit(-1.0), hold()))


def _positions(store, taille, nom: str) -> np.ndarray:
    strat = bt.Strategy.create(nom).signal("d", col("close")).size(taille)
    res = bt.run(strat, _config(), store)
    return res.positions.column("position").to_numpy()


def _exposition(positions: np.ndarray) -> float:
    return float(np.mean(positions != 0.0))


def test_sans_filtre_ffill_reproduit_hold_barre_par_barre(store):
    """Sans filtre, les deux ecritures sont la meme strategie.

    C'est ce qui autorise a presenter `ffill()` comme la version qui compose :
    si les deux divergeaient deja sans filtre, ce serait un autre operateur.
    """
    par_ffill = _positions(store, _ETAT_FFILL, "ffill_nu")
    par_hold = _positions(store, _ETAT_HOLD, "hold_nu")
    np.testing.assert_array_equal(par_ffill, par_hold)
    # Et la strategie trade vraiment, sinon l'egalite ne prouverait rien.
    assert _exposition(par_ffill) > 0.5


def test_sous_filtre_ffill_tient_l_exposition(store):
    """L'exposition suit le taux d'ouverture du filtre, aux barres non armees pres."""
    nu = _exposition(_positions(store, _ETAT_FFILL, "ffill_nu2"))
    sous_filtre = _exposition(
        _positions(store, when(_FILTRE, _ETAT_FFILL, lit(0.0)), "ffill_filtre")
    )
    # Le filtre est ouvert une barre sur deux et l'etat est arme `nu` du temps ;
    # les deux sont independants ici (l'heure ne dit rien du z-score).
    attendu = TAUX_OUVERTURE * nu
    assert abs(sous_filtre - attendu) < 0.03, (sous_filtre, attendu)


def test_sous_filtre_hold_s_effondre(store):
    """Le comportement de `hold()` sous filtre, epingle tel qu'il est.

    A ne PAS "corriger" : c'est la lecture "ne rentrer que sur un nouveau
    signal". Ce test existe pour que le jour ou quelqu'un change la semantique
    de `hold()`, il le fasse en connaissance de cause.
    """
    par_ffill = _exposition(
        _positions(store, when(_FILTRE, _ETAT_FFILL, lit(0.0)), "ffill_filtre2")
    )
    par_hold = _exposition(
        _positions(store, when(_FILTRE, _ETAT_HOLD, lit(0.0)), "hold_filtre")
    )
    assert par_ffill > 0.35, par_ffill
    # Mesure sur ces donnees : ~42.6% contre ~4.6%. Le seuil est large exprès :
    # ce qui est epingle est l'ordre de grandeur de l'ecart, pas un chiffre.
    assert par_hold < par_ffill / 4.0, (par_hold, par_ffill)


def test_passer_par_un_signal_nomme_ne_change_rien_a_hold(store):
    """`.signal("etat", ...)` puis `col("etat")` donne exactement le meme resultat.

    Le detour par un signal nomme est le premier reflexe quand l'exposition
    s'effondre. Il ne change rien : le NaN veut dire "retiens la position"
    partout ou il est lu, pas seulement dans l'expression ou il est ecrit.
    """
    direct = _positions(store, when(_FILTRE, _ETAT_HOLD, lit(0.0)), "hold_direct")
    strat = (bt.Strategy.create("hold_nomme")
             .signal("d", col("close"))
             .signal("etat", _ETAT_HOLD)
             .size(when(_FILTRE, col("etat"), lit(0.0))))
    via_signal = bt.run(strat, _config(), store).positions.column("position").to_numpy()
    np.testing.assert_array_equal(direct, via_signal)
