"""Erreur d'un balayage situee, parametres d'un balayage lite, mediane des WFE.

1. Une combinaison dont l'equite passe sous zero fait lever `DataError` a
   tout le balayage. Le message nomme la premiere combinaison en cause, dit
   combien ont echoue, et indique `liquidation` pour modeliser la ruine.
   Un lot de strategies nomme la strategie en cause.

2. `run_sweep_lite` rend une liste qui connait sa grille : `params(i)` et
   `to_df()` etiquettent chaque resultat dans l'ordre du moteur.

3. Le walk-forward rend la mediane des WFE a cote de leur moyenne.
"""
import os
import statistics

import pytest

import manifoldbt as bt
from manifoldbt.exceptions import DataError
from manifoldbt.expr import lit, param
from manifoldbt.indicators import close

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _hausse(tmp_path):
    """100 jours de barres horaires, prix de 100 a 250. La taille est plafonnee a
    l'equite (`max_position_pct` = 1) : un short a 1x y perd 1,5 fois le compte,
    un short a 0,5x 75 %."""
    n = 2400
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    px = np.linspace(100.0, 250.0, n)
    df = pd.DataFrame({"open": px, "high": px * 1.001, "low": px * 0.999,
                       "close": px, "volume": np.full(n, 1000.0)}, index=idx)
    racine = str(tmp_path / "data")
    meta = str(tmp_path / "meta.sqlite")
    os.makedirs(racine, exist_ok=True)
    store = bt.import_dataframe(df.reset_index(), symbol="ZRUI", symbol_id=1,
                                interval="1h", asset_class="equity",
                                exchange="TEST", data_root=racine, metadata_db=meta)
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=bt.Interval.hours(1), symbol_names={"ZRUI": 1})
    cfg.warmup_bars = 0
    cfg.execution = bt.ExecutionConfig(allow_short=True)
    return store, cfg


def _short(name="short"):
    return bt.Strategy.create(name).signal("c", close).size(lit(0.0) - param("lev", default=0.5))


def test_sweep_lite_nomme_la_combinaison_ruinee(tmp_path):
    store, cfg = _hausse(tmp_path)
    with pytest.raises(DataError) as err:
        bt.run_sweep_lite(_short(), {"lev": [0.5, 1.0, 3.0]}, cfg, store)
    msg = str(err.value)
    assert msg.startswith("data error: equity values must be > 0")
    # La premiere combinaison en echec, et le compte des autres.
    assert "combo 2 of 3: lev=1" in msg
    assert "2 of 3 combos failed" in msg
    # Comment modeliser la ruine au lieu de la subir.
    assert "liquidation" in msg


def test_sweep_complet_nomme_la_combinaison_ruinee(tmp_path):
    store, cfg = _hausse(tmp_path)
    with pytest.raises(DataError) as err:
        bt.run_sweep(_short(), {"lev": [0.5, 3.0]}, cfg, store)
    assert "combo 2 of 2: lev=3" in str(err.value)
    assert "1 of 2 combos failed" in str(err.value)


def test_batch_lite_nomme_la_strategie_ruinee(tmp_path):
    store, cfg = _hausse(tmp_path)
    sain = bt.Strategy.create("prudent").signal("c", close).size(lit(-0.5))
    ruine = bt.Strategy.create("levier").signal("c", close).size(lit(-3.0))
    with pytest.raises(DataError) as err:
        bt.run_batch_lite([sain, ruine], cfg, store)
    msg = str(err.value)
    assert "strategy \"levier\", index 1 of 2" in msg
    assert "1 of 2 strategies failed" in msg


def test_run_seul_dit_comment_modeliser_la_ruine(tmp_path):
    store, cfg = _hausse(tmp_path)
    with pytest.raises(DataError) as err:
        bt.run(bt.Strategy.create("levier").signal("c", close).size(lit(-3.0)), cfg, store)
    msg = str(err.value)
    assert msg.startswith("data error: equity values must be > 0")
    assert "liquidation" in msg
    # Un run seul n'a pas de combinaison a nommer.
    assert "combo" not in msg


def test_sweep_lite_connait_ses_parametres(tmp_path):
    store, cfg = _hausse(tmp_path)
    strat = bt.Strategy.create("s").signal("c", close).size(
        lit(0.0) - param("lev", default=0.1) * param("a_poids", default=1.0))
    # Cles volontairement hors ordre alphabetique : le moteur trie les axes.
    grille = {"lev": [0.1, 0.2, 0.3], "a_poids": [1.0, 0.5]}
    res = bt.run_sweep_lite(strat, grille, cfg, store)
    assert len(res) == 6
    # a_poids est l'axe lent, lev le rapide.
    assert res.params(0) == {"lev": 0.1, "a_poids": 1.0}
    assert res.params(1) == {"lev": 0.2, "a_poids": 1.0}
    assert res.params(3) == {"lev": 0.1, "a_poids": 0.5}
    assert res.params(-1) == {"lev": 0.3, "a_poids": 0.5}
    with pytest.raises(IndexError):
        res.params(6)
    # L'etiquette est la bonne : chaque ligne vaut le run seul de ses parametres.
    for i in (0, 4):
        p = res.params(i)
        seul = bt.run(bt.Strategy.create("s").signal("c", close).size(lit(-p["lev"] * p["a_poids"])), cfg, store)
        assert res[i].metrics["total_return"] == pytest.approx(
            seul.metrics["total_return"], rel=1e-12)
    df = res.to_df(backend="pandas")
    assert list(df["param_lev"]) == [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    assert list(df["param_a_poids"]) == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    assert "<SweepLiteResults" in repr(res) and "to_df" in repr(res)


@pytest.mark.skipif(
    bt.license_info()[0] not in ("Pro", "Researcher"),
    reason="walk-forward requires Pro: skipped on a Community wheel",
)
def test_walk_forward_rend_la_mediane_des_wfe(tmp_path):
    n = 2400
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    pas = np.sin(np.arange(n) / 37.0) * 2.0 + np.cos(np.arange(n) / 11.0)
    px = 100.0 + np.cumsum(pas) * 0.05
    df = pd.DataFrame({"open": px, "high": px * 1.004, "low": px * 0.996,
                       "close": px, "volume": np.full(n, 1000.0)}, index=idx)
    racine = str(tmp_path / "data")
    meta = str(tmp_path / "meta.sqlite")
    os.makedirs(racine, exist_ok=True)
    store = bt.import_dataframe(df.reset_index(), symbol="ZMED", symbol_id=1,
                                interval="1h", asset_class="equity",
                                exchange="TEST", data_root=racine, metadata_db=meta)
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=bt.Interval.hours(1), symbol_names={"ZMED": 1})
    cfg.warmup_bars = 0
    from manifoldbt.indicators import close, sma
    from manifoldbt.expr import when
    rapide = sma(close, param("n", default=10))
    strat = bt.Strategy.create("x").signal("f", rapide).size(when(close > rapide, 1.0, 0.0))
    wf = {"geometry": "anchored", "n_splits": 4, "train_ratio": 0.6,
          "optimize_metric": "total_return", "param_grid": {"n": [5, 10, 20]}}
    out = bt.run_walk_forward(strat, wf, cfg, store)
    assert "walk_forward_efficiency_median" in out
    finis = [f["wfe"] for f in out["folds"] if f.get("wfe") is not None]
    if finis:
        assert out["walk_forward_efficiency_median"] == pytest.approx(statistics.median(finis))
        assert "median" in repr(out)
    else:
        assert out["walk_forward_efficiency_median"] is None
