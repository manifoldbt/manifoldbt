"""Le walk-forward doit accepter les sources exogenes PAR SYMBOLE (`exo_sources`).

Une strategie lisant `exo.<colonne>` d'une source par symbole tournait dans
`run`, mais le walk-forward passait des sources vides a ses trois runs
(selection, run d'apprentissage, run hors echantillon) et levait
"unknown input column".
"""
import os

import pytest

import manifoldbt as bt

pytestmark = pytest.mark.skipif(
    bt.license_info()[0] not in ("Pro", "Researcher"),
    reason="requires Pro: skipped on a Community wheel",
)

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _monte(tmp_path):
    """Le titre trade, une seconde source porte une moyenne causale du titre."""
    n = 2400
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    pas = np.sin(np.arange(n) / 37.0) * 2.0 + np.cos(np.arange(n) / 11.0)
    px = 100.0 + np.cumsum(pas) * 0.05
    aux = pd.Series(px).rolling(24, min_periods=1).mean().to_numpy()

    def barres(p):
        return pd.DataFrame({"open": p, "high": p * 1.004, "low": p * 0.996,
                             "close": p, "volume": np.full(n, 1000.0)},
                            index=idx).reset_index()

    racine = str(tmp_path / "data")
    meta = str(tmp_path / "meta.sqlite")
    os.makedirs(racine, exist_ok=True)
    store = bt.import_dataframe(barres(px), symbol="ZESM", symbol_id=1, interval="1h",
                                asset_class="equity", exchange="TEST",
                                data_root=racine, metadata_db=meta)
    bt.import_dataframe(barres(aux), symbol="ZESA", symbol_id=2, interval="1h",
                        asset_class="equity", exchange="AUX",
                        data_root=racine, metadata_db=meta)
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=bt.Interval.hours(1), symbol_names={"ZESM": 1})
    cfg.warmup_bars = 0
    cfg.exo_sources = {1: ("AUX", "ZESA")}
    return store, cfg


def _strategie():
    from manifoldbt.indicators import close, col
    m = col("exo.close")
    bande = m * (bt.lit(1.0) - bt.param("dev", default=0.002))
    return (bt.Strategy.create("s")
            .signal("b", bande)
            .size(bt.when(close < bande, 1.0, bt.when(close > m, 0.0, bt.hold()))))


WF = {"method": "Anchored", "n_splits": 3, "train_ratio": 0.5,
      "optimize_metric": "sharpe", "param_grid": {"dev": [0.002, 0.005, 0.01]}}


def test_la_strategie_trade_dans_run(tmp_path):
    store, cfg = _monte(tmp_path)
    assert bt.run(_strategie(), cfg, store).trade_count > 0


def test_walk_forward_accepte_une_source_par_symbole(tmp_path):
    store, cfg = _monte(tmp_path)
    r = bt.run_walk_forward(_strategie(), WF, cfg, store)
    assert r["n_folds"] == 3
    # La source est lue dans les trois runs : la selection (metriques IS) et
    # les deux courbes. Sans elle, aucun pli ne trade.
    assert any(f["oos_metrics"]["total_return"] != 0.0 for f in r["folds"])
    assert any(len(set(f["is_equity"])) > 1 for f in r["folds"])
