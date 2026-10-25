"""run() refuse un param() lu sans valeur, au lieu de ne rien trader.

Un param() sans defaut n'a de valeur que dans un balayage. Un run seul le
lisait comme une serie vide : EMA sans periode, aucun signal, 0 trade, aucune
erreur (la strategie de l'exemple 08 lancee seule). Les balayages, eux,
fixent chaque nom balaye et ne passent pas par cette garde.
"""
import os

import pytest

import manifoldbt as bt
from manifoldbt.exceptions import StrategyError
from manifoldbt.indicators import close, ema

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def monte(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("param")
    n = 2400
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    pas = np.sin(np.arange(n) / 37.0) * 2.0 + np.cos(np.arange(n) / 11.0)
    px = 100.0 + np.cumsum(pas) * 0.05
    df = pd.DataFrame({"open": px, "high": px * 1.004, "low": px * 0.996,
                       "close": px, "volume": np.full(n, 1000.0)}, index=idx)
    racine = str(tmp / "data")
    os.makedirs(racine, exist_ok=True)
    store = bt.import_dataframe(df.reset_index(), symbol="ZPSV", symbol_id=1,
                                interval="1h", asset_class="equity",
                                exchange="TEST", data_root=racine,
                                metadata_db=str(tmp / "meta.sqlite"))
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=bt.Interval.hours(1), symbol_names={"ZPSV": 1})
    return store, cfg


def _croisement(fast, slow):
    f, s = ema(close, fast), ema(close, slow)
    return (bt.Strategy.create("croisement").signal("f", f).signal("s", s)
            .size(bt.when(f > s, 0.25, bt.when(f < s, -0.25, 0.0))))


def test_un_param_lu_sans_valeur_est_refuse_et_nomme(monte):
    store, cfg = monte
    with pytest.raises(StrategyError, match="has no value in a plain run") as exc:
        bt.run(_croisement(bt.param("fast"), bt.param("slow")), cfg, store)
    assert "'fast'" in str(exc.value) and "'slow'" in str(exc.value)


def test_un_param_avec_defaut_tourne(monte):
    store, cfg = monte
    r = bt.run(_croisement(bt.param("fast", default=12), bt.param("slow", default=26)),
               cfg, store)
    assert r.trade_count > 0


def test_une_declaration_strategy_param_donne_la_valeur(monte):
    store, cfg = monte
    s = (_croisement(bt.param("fast"), bt.param("slow"))
         .param("fast", default=12).param("slow", default=26))
    assert bt.run(s, cfg, store).trade_count > 0


def test_un_param_declare_mais_lu_par_rien_ne_bloque_pas(monte):
    store, cfg = monte
    s = _croisement(12, 26).param("seuil")
    assert bt.run(s, cfg, store).trade_count > 0


def test_un_balayage_fixe_les_valeurs_et_passe(monte):
    store, cfg = monte
    if bt.license_info()[0] not in ("Pro", "Researcher"):
        pytest.skip("requires Pro beyond the Community combination limit")
    b = bt.run_sweep_lite(_croisement(bt.param("fast"), bt.param("slow")),
                          {"fast": [8, 12], "slow": [26]}, cfg, store)
    assert len(b) == 2
