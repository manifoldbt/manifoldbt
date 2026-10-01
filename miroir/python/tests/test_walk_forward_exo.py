"""Le walk-forward doit accepter les series EXOGENES, comme tous les autres
chemins du moteur.

Il chargeait ses colonnes exo pour les deux runs qui tracent les courbes
d'equite, mais appelait le moteur avec des tables VIDES pour la selection du
reglage. Une branche sur trois etait oubliee, et c'etait celle qui decide :
toute strategie lisant `exo.<nom>.<colonne>` echouait sur "unknown input
column" alors que la meme strategie tourne dans `run`, `run_sweep_lite` et
`run_batch_lite`.

Le second test verifie le point qui rend la correction sure : la selection lit
desormais des colonnes DECOUPEES une fois pour toutes, la ou les courbes les
rechargent par fenetre. Les deux chemins doivent rendre la meme metrique
d'in-sample pour le reglage retenu, sinon le decoupage est faux.
"""
import os

import pytest

import manifoldbt as bt

# Walk-forward optimization is a Pro feature: on a Community wheel the
# gate raises before the exogenous columns are ever loaded.
pytestmark = pytest.mark.skipif(
    bt.license_info()[0] != "Pro",
    reason="requires Pro: skipped on a Community wheel",
)


pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _monte(tmp_path):
    """Une serie horaire regulière, plus une moyenne posee en serie exogene."""
    n = 2400
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    pas = np.sin(np.arange(n) / 37.0) * 2.0 + np.cos(np.arange(n) / 11.0)
    px = 100.0 + np.cumsum(pas) * 0.05
    df = pd.DataFrame({"open": px, "high": px * 1.004, "low": px * 0.996,
                       "close": px, "volume": np.full(n, 1000.0)}, index=idx)
    racine = str(tmp_path / "data")
    meta = str(tmp_path / "meta.sqlite")
    os.makedirs(racine, exist_ok=True)
    store = bt.import_dataframe(df.reset_index(), symbol="ZEXO", symbol_id=1,
                                interval="1h", asset_class="equity",
                                exchange="TEST", data_root=racine, metadata_db=meta)
    moy = pd.Series(px).rolling(24).mean().to_numpy()
    bt.register_exo("moyenne", pd.DataFrame({"timestamp": idx, "sma": moy}),
                    store=store, data_root=racine, timeframe="1h")
    # la plage doit coller aux donnees : 2400 heures = 100 jours
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=bt.Interval.hours(1), symbol_names={"ZEXO": 1})
    cfg.warmup_bars = 0
    cfg.exo_data = ["moyenne"]
    return store, cfg


def _strategie():
    from manifoldbt.indicators import close, col
    m = col("exo.moyenne.sma")
    bande = m * (bt.lit(1.0) - bt.param("dev"))
    return (bt.Strategy.create("s")
            .signal("aux", bande)
            .size(bt.when(close < bande, 1.0, bt.when(close > m, 0.0, bt.hold()))))


WF = {"method": "Anchored", "n_splits": 3, "train_ratio": 0.5,
      "optimize_metric": "sharpe",
      "param_grid": {"dev": [0.002, 0.005, 0.01]}}


def test_walk_forward_accepte_une_serie_exogene(tmp_path):
    store, cfg = _monte(tmp_path)
    r = bt.run_walk_forward(_strategie(), WF, cfg, store)
    assert len(r["folds"]) == 3
    # chaque pli doit avoir EVALUE la grille, pas l'avoir sautee
    for f in r["folds"]:
        assert len(f["all_is_results"]) == 3, "la grille n'a pas ete evaluee"


def test_selection_et_courbes_voient_les_memes_colonnes(tmp_path):
    """La selection tranche les colonnes une fois, les courbes les rechargent
    par fenetre : le meme reglage doit donner le meme in-sample des deux cotes."""
    store, cfg = _monte(tmp_path)
    r = bt.run_walk_forward(_strategie(), WF, cfg, store)
    for f in r["folds"]:
        meilleur = max(f["all_is_results"],
                       key=lambda x: x["metrics"].get("sharpe", float("-inf")))
        a = meilleur["metrics"]["sharpe"]
        b = f["is_metrics"]["sharpe"]
        assert a == b, "selection {} contre courbe {} au pli {}".format(
            a, b, f["fold_index"])
