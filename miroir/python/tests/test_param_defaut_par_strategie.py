"""Le defaut d'un `param()` appartient a la strategie qui le declare.

Un `param()` passe comme periode ou comme portee d'indicateur
(`sma(close, param("slow", default=20))`) se serialise en son seul nom. Son
defaut etait range dans un registre du module, par nom, ou la DERNIERE
declaration du processus gagnait : une strategie construite avec `slow=20`,
puis une variante construite avec `slow=41`, tournait avec 41 des qu'elle
etait serialisee apres la variante. En silence, dans `run`, `run_batch`, les
sweeps (l'axe non balaye prenait le defaut de l'autre), le walk-forward.

Ce fichier epingle la regle : chaque strategie tourne avec les defauts ecrits
dans SES expressions, quel que soit l'ordre de construction, le fil qui l'a
construite ou la copie qu'on en fait, et un nom declare avec deux defauts
differents dans une meme strategie est refuse plutot que tranche au hasard du
parcours.
"""
import copy
import json
import pickle
import threading

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt import _validate_swept_params  # noqa: E402
from manifoldbt.exceptions import StrategyError  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from manifoldbt.indicators import atr, close, ema, sma  # noqa: E402

N_BARS = 3_000


def croisement(slow=20, fast=5):
    """La strategie du defaut constate : deux moyennes, la lente balayable."""
    f = sma(close, bt.param("fast", default=fast))
    s = sma(close, bt.param("slow", default=slow))
    return (
        bt.Strategy.create("croisement")
        .signal("fast", f)
        .signal("sl", s)
        .size(bt.when(f > s, 1.0, 0.0))
    )


def defauts(strategy):
    """``{"slow": 20, ...}`` : les defauts que la strategie envoie au moteur."""
    out = {}
    for name, spec in json.loads(strategy.to_json())["parameters"].items():
        d = spec["default"]
        out[name] = next(iter(d.values())) if isinstance(d, dict) else d
    return out


# ---------------------------------------------------------------------------
# Serialisation : aucune strategie ne lit le defaut d'une autre
# ---------------------------------------------------------------------------


def test_une_variante_construite_apres_ne_change_pas_la_premiere():
    a20 = croisement(slow=20)
    a41 = croisement(slow=41)
    assert defauts(a20) == {"fast": 5, "slow": 20}
    assert defauts(a41) == {"fast": 5, "slow": 41}


def test_l_ordre_de_serialisation_ne_compte_pas():
    a20, a41 = croisement(slow=20), croisement(slow=41)
    json41, json20 = a41.to_json(), a20.to_json()
    assert json20 == croisement(slow=20).to_json()
    assert json41 == croisement(slow=41).to_json()


def test_une_portee_et_une_periode_suivent_la_meme_regle():
    def s(span, k):
        e = ema(close, bt.param("span", default=span))
        band = close.rolling_std(20) * bt.param("k", default=k)
        return bt.Strategy.create("e").signal("e", e).size(bt.when(close > e + band, 1.0, 0.0))

    a, b = s(10, 1.5), s(30, 2.5)
    assert defauts(a) == {"span": 10, "k": 1.5}
    assert defauts(b) == {"span": 30, "k": 2.5}


def test_une_colonne_homonyme_d_un_parametre_d_une_autre_strategie_n_est_pas_un_parametre():
    # "signal" est une periode ailleurs dans le processus ; ici c'est le nom
    # d'une colonne. L'ancien registre en faisait un parametre fantome.
    autre = bt.Strategy.create("macd").signal(
        "m", sma(close, bt.param("signal", default=9))
    )
    assert "signal" in defauts(autre)
    s = bt.Strategy.create("simple").signal("signal", bt.lit(1.0)).size(bt.col("signal"))
    assert json.loads(s.to_json())["parameters"] == {}


# ---------------------------------------------------------------------------
# Un meme nom plusieurs fois dans une strategie
# ---------------------------------------------------------------------------


def test_deux_defauts_differents_dans_une_strategie_sont_refuses():
    s = (
        bt.Strategy.create("ambigue")
        .signal("a", sma(close, bt.param("n", default=10)))
        .signal("b", sma(close, bt.param("n", default=20)))
    )
    with pytest.raises(ValueError) as err:
        s.to_json()
    msg = str(err.value)
    assert "'n'" in msg and "10" in msg and "20" in msg and ".param(" in msg


def test_la_strategie_tranche_avec_param():
    s = (
        bt.Strategy.create("tranchee")
        .signal("a", sma(close, bt.param("n", default=10)))
        .signal("b", sma(close, bt.param("n", default=20)))
        .param("n", default=15)
    )
    assert defauts(s) == {"n": 15}


def test_une_declaration_sans_defaut_lit_celle_qui_en_donne_un():
    # Dans les deux ordres de parcours : valeur d'un signal d'abord, puis periode,
    # et l'inverse.
    for ordre in (("v", "p"), ("p", "v")):
        s = bt.Strategy.create("fusion")
        for key in ordre:
            if key == "v":
                s.signal("v", close * bt.param("n"))
            else:
                s.signal("p", sma(close, bt.param("n", default=20)))
        assert defauts(s) == {"n": 20}, ordre


def test_param_sans_defaut_sur_la_strategie_n_efface_pas_celui_des_expressions():
    s = (
        bt.Strategy.create("s")
        .signal("p", sma(close, bt.param("n", default=20)))
        .param("n", range=(5, 50), description="periode")
    )
    spec = json.loads(s.to_json())["parameters"]["n"]
    assert spec["default"] == {"Int64": 20}
    assert spec["range"] == [{"Int64": 5}, {"Int64": 50}]
    assert spec["description"] == "periode"


def test_la_meme_valeur_ecrite_deux_fois_n_est_pas_un_conflit():
    s = (
        bt.Strategy.create("s")
        .signal("a", sma(close, bt.param("n", default=20)))
        .signal("b", close.rolling_max(bt.param("n", default=20.0)))
    )
    assert defauts(s) == {"n": 20}


def test_un_parametre_dans_un_signal_et_dans_une_periode():
    n = bt.param("n", default=20)
    meme_objet = bt.Strategy.create("s").signal("p", sma(close, n)).size(close > n)
    deux_appels = (
        bt.Strategy.create("s")
        .signal("p", sma(close, bt.param("n", default=20)))
        .size(close > bt.param("n", default=20))
    )
    croisement(slow=41)
    assert defauts(meme_objet) == defauts(deux_appels) == {"n": 20}


def test_un_parametre_declare_mais_absent_de_la_strategie_n_est_pas_declare():
    bt.param("orphelin", default=5)
    close.rolling_mean(bt.param("orphelin", default=5))  # construit, jamais utilise
    s = croisement()
    assert "orphelin" not in defauts(s)
    with pytest.raises(StrategyError):
        _validate_swept_params(s, ["orphelin"], "run_sweep")
    # Declare sur la strategie, il l'est, comme avant.
    assert defauts(croisement().param("orphelin", default=5))["orphelin"] == 5


# ---------------------------------------------------------------------------
# Cotes, niveaux d'entree, brackets
# ---------------------------------------------------------------------------


def test_un_parametre_dans_une_cote():
    def maker(edge, n):
        mid = sma(close, bt.param("n", default=n))
        return bt.Strategy.create("maker").quote(
            "buy", mid - bt.param("edge", default=edge), 1.0
        )

    a, b = maker(0.1, 10), maker(0.3, 30)
    assert defauts(a) == {"n": 10, "edge": 0.1}
    assert defauts(b) == {"n": 30, "edge": 0.3}


def test_un_parametre_dans_un_niveau_d_entree_et_un_bracket():
    def s(n, k):
        return (
            bt.Strategy.create("entree")
            .signal("niveau", sma(close, bt.param("n", default=n)))
            .signal("dist", atr(bt.param("a", default=n)) / close * bt.param("k", default=k))
            .size(bt.lit(1.0))
            .limit_entry(signal="niveau")
            .stop_loss(signal="dist")
        )

    a, b = s(10, 2.0), s(30, 3.0)
    assert defauts(a) == {"n": 10, "a": 10, "k": 2.0}
    assert defauts(b) == {"n": 30, "a": 30, "k": 3.0}


# ---------------------------------------------------------------------------
# Copie, pickle, fils
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dupliquer",
    [copy.copy, copy.deepcopy, lambda s: pickle.loads(pickle.dumps(s))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_une_copie_garde_ses_defauts(dupliquer):
    a20 = croisement(slow=20)
    attendu = a20.to_json()
    a20._json_cache = None  # que la copie reserialise, pas qu'elle relise
    double = dupliquer(a20)
    croisement(slow=41)
    assert double.to_json() == attendu
    (periode,) = [a for a in double.signals["sl"]._args if isinstance(a, str)]
    assert periode._param_meta["default"] == 20  # le nom porte sa declaration


def test_des_strategies_construites_dans_des_fils():
    construites = {}

    def construire(slow):
        construites[slow] = croisement(slow=slow)

    fils = [threading.Thread(target=construire, args=(v,)) for v in range(10, 50, 4)]
    for t in fils:
        t.start()
    for t in fils:
        t.join()
    for slow, s in construites.items():
        assert defauts(s)["slow"] == slow


# ---------------------------------------------------------------------------
# Moteur : run, run_batch, sweeps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def marche(tmp_path_factory):
    rng = np.random.default_rng(11)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, N_BARS)))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=N_BARS, freq="1h", tz="UTC"),
        "open": px, "high": px * 1.002, "low": px * 0.998, "close": px,
        "volume": 1000.0,
    })
    root = tmp_path_factory.mktemp("param_defaut")
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1h",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )
    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=int(df["timestamp"].iloc[-1].value) + 86_400_000_000_000,
        bar_interval=Interval.hours(1),
        initial_capital=100_000.0,
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )
    return store, config


def _rendement(result):
    return result.metrics["total_return"]


def test_run_tourne_chaque_variante_avec_son_defaut(marche):
    store, config = marche
    a20 = croisement(slow=20)
    a41 = croisement(slow=41)
    r20, r41 = bt.run(a20, config, store), bt.run(a41, config, store)
    seule = bt.run(croisement(slow=20), config, store)
    assert _rendement(r20) != _rendement(r41), "le parametre doit compter sur ces donnees"
    assert r20.manifest["parameters"]["slow"] == {"Int64": 20}
    assert _rendement(r20) == _rendement(seule)


def test_run_batch_de_variantes(marche):
    store, config = marche
    variantes = [croisement(slow=v) for v in (20, 41, 30)]
    batch = bt.run_batch(variantes, config, store)
    for v, r in zip((20, 41, 30), batch):
        assert _rendement(r) == _rendement(bt.run(croisement(slow=v), config, store)), v


def test_un_sweep_garde_le_defaut_de_sa_strategie(marche):
    # L'axe non balaye ("slow") prend le defaut de LA strategie balayee.
    store, config = marche
    a20 = croisement(slow=20)
    croisement(slow=41)
    lite = bt.run_sweep_lite(a20, {"fast": [5, 8]}, config, store)
    full = bt.run_sweep(a20, {"fast": [5, 8]}, config, store)
    for fast, rl, rf in zip((5, 8), lite, full):
        seule = _rendement(bt.run(croisement(slow=20, fast=fast), config, store))
        assert rf.manifest["parameters"]["slow"] == {"Int64": 20}
        assert _rendement(rf) == seule, fast
        assert abs(_rendement(rl) - seule) <= 1e-9 * max(1.0, abs(seule)), fast


def test_un_sweep_balaye_les_valeurs_demandees(marche):
    store, config = marche
    a20 = croisement(slow=20)
    croisement(slow=99)
    sweep = bt.run_sweep(a20, {"slow": [20, 41]}, config, store)
    for slow, r in zip((20, 41), sweep):
        assert r.manifest["parameters"]["slow"] == {"Int64": slow}
        assert _rendement(r) == _rendement(bt.run(croisement(slow=slow), config, store))
