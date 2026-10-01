"""Geometrie du walk-forward et chauffe hors echantillon.

Trois contrats poses par la refonte :

1. Le run OOS est CHAUFFE : il simule depuis le debut de l'apprentissage du
   pli et ne trade qu'a partir du test. Le test le prouve avec un SMA plus
   long que la fenetre de test : a froid l'indicateur resterait nul sur toute
   la fenetre et l'equity serait PLATE ; chauffe, il est disponible des la
   premiere barre tradable.

2. Les geometries `pardo` et `custom` derivent le nombre de plis des
   longueurs de fenetres, et `custom` sait exprimer des tests recouvrants --
   signales par `folds_overlap` et repondus par `effective_folds`.

3. `method="Rolling"` est refuse avec un message qui nomme le remplacant :
   ce mode faisait des blocs disjoints, pas le rolling de Pardo.
"""
import os

import pytest

import manifoldbt as bt

# Walk-forward optimization is a Pro feature: on a Community wheel,
# run_walk_forward raises LicenseError before any geometry is read, so
# every assertion below about folds and warmup would fail on the gate,
# not on what it tests. The suite skips, and names why.
pytestmark = pytest.mark.skipif(
    bt.license_info()[0] != "Pro",
    reason="requires Pro: skipped on a Community wheel",
)


pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

JOUR_NS = 86_400 * 1_000_000_000


def _monte(tmp_path):
    """100 jours de barres horaires, prix cyclique pour garantir des trades."""
    n = 2400
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    pas = np.sin(np.arange(n) / 37.0) * 2.0 + np.cos(np.arange(n) / 11.0)
    px = 100.0 + np.cumsum(pas) * 0.05
    df = pd.DataFrame({"open": px, "high": px * 1.004, "low": px * 0.996,
                       "close": px, "volume": np.full(n, 1000.0)}, index=idx)
    racine = str(tmp_path / "data")
    meta = str(tmp_path / "meta.sqlite")
    os.makedirs(racine, exist_ok=True)
    store = bt.import_dataframe(df.reset_index(), symbol="ZWFG", symbol_id=1,
                                interval="1h", asset_class="equity",
                                exchange="TEST", data_root=racine, metadata_db=meta)
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=bt.Interval.hours(1), symbol_names={"ZWFG": 1})
    cfg.warmup_bars = 0
    return store, cfg


def _strategie_sma_long():
    """SMA plus long (400 barres) que toute fenetre de test des tests ci-dessous."""
    from manifoldbt.indicators import close, sma
    m = sma(close, 400) * (bt.lit(1.0) + bt.param("dev") * 0.0)
    return (bt.Strategy.create("s")
            .signal("m", m)
            .size(bt.when(close > m, 1.0, 0.0)))


def test_oos_est_chauffe_l_indicateur_est_disponible(tmp_path):
    store, cfg = _monte(tmp_path)
    wf = {"geometry": "anchored", "n_splits": 2, "train_ratio": 0.8,
          "optimize_metric": "sharpe", "param_grid": {"dev": [0.0, 1.0]}}
    r = bt.run_walk_forward(_strategie_sma_long(), wf, cfg, store)
    assert r["n_folds"] == 2
    for f in r["folds"]:
        eq = f["oos_equity"]
        ts = f["oos_timestamps"]
        # la courbe rendue couvre les seules barres du test, chauffe exclue
        assert len(eq) == len(ts) > 0
        assert ts[0] >= f["test_range"]["start"]
        assert ts[-1] < f["test_range"]["end"]
        # fenetre de test = 10 jours = 240 barres < SMA(400) : a froid,
        # l'indicateur serait nul sur TOUTE la fenetre et l'equity plate.
        assert len(eq) <= 400, "le test doit etre plus court que le SMA"
        assert max(eq) != min(eq), (
            "equity OOS plate : l'indicateur n'a pas ete chauffe")


def test_pardo_derive_le_nombre_de_plis(tmp_path):
    store, cfg = _monte(tmp_path)
    wf = {"geometry": "pardo",
          "train": {"length": {"Days": 50}},
          "test": {"length": {"Days": 10}},
          "optimize_metric": "sharpe", "param_grid": {"dev": [0.0]}}
    r = bt.run_walk_forward(_strategie_sma_long(), wf, cfg, store)
    # 100 jours : premier test a j50, puis 5 fenetres de 10 jours
    assert r["n_folds"] == 5
    assert r["folds_overlap"] is False
    assert r["effective_folds"] == 5.0
    for f in r["folds"]:
        tr, te = f["train_range"], f["test_range"]
        assert te["start"] - tr["start"] == 50 * JOUR_NS
        assert te["end"] - te["start"] == 10 * JOUR_NS


def test_custom_recouvrant_expose_les_plis_effectifs(tmp_path):
    store, cfg = _monte(tmp_path)
    wf = {"geometry": "custom",
          "train": {"mode": "anchored", "min_length": {"Days": 60}},
          "test": {"length": {"Days": 10}, "step": {"Days": 5}},
          "optimize_metric": "sharpe", "param_grid": {"dev": [0.0]}}
    r = bt.run_walk_forward(_strategie_sma_long(), wf, cfg, store)
    # tests possibles de j60 a j90 par pas de 5 -> 7 plis, union 40 jours
    assert r["n_folds"] == 7
    assert r["folds_overlap"] is True
    assert r["effective_folds"] == pytest.approx(4.0)


def test_rolling_est_refuse_avec_le_remplacant_nomme(tmp_path):
    store, cfg = _monte(tmp_path)
    wf = {"method": "Rolling", "n_splits": 2, "train_ratio": 0.7,
          "optimize_metric": "sharpe", "param_grid": {"dev": [0.0]}}
    with pytest.raises(Exception) as exc:
        bt.run_walk_forward(_strategie_sma_long(), wf, cfg, store)
    msg = str(exc.value)
    assert "blocked" in msg and "pardo" in msg


def test_wfe_est_rendu(tmp_path):
    store, cfg = _monte(tmp_path)
    wf = {"geometry": "anchored", "n_splits": 2, "train_ratio": 0.8,
          "optimize_metric": "sharpe", "param_grid": {"dev": [0.0]}}
    r = bt.run_walk_forward(_strategie_sma_long(), wf, cfg, store)
    assert "walk_forward_efficiency" in r
    for f in r["folds"]:
        assert "wfe" in f
