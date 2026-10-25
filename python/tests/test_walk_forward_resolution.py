"""Walk-forward sur un signal plus grossier que le store.

Un signal 12h sur des barres 1h (mode hybride) rendait, en sortie
automatique, une courbe OOS sous-echantillonnee au pas du signal. Sa longueur
ne valait plus celle des barres natives, et le walk-forward retombait sur les
metriques du run ENTIER, chauffe plate comprise : Sharpe hors echantillon
dilue, WFE faux, sans un avertissement. Avec `output_resolution` a 1h, le meme
walk-forward donnait le bon chiffre.

Le contrat : les metriques hors echantillon ne dependent pas de la resolution
de sortie, et la courbe rendue suit celle-ci, avec autant d'horodatages que de
points.
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

HEURE_NS = 3_600 * 1_000_000_000


def _magasin_horaire(tmp_path):
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
    store = bt.import_dataframe(df.reset_index(), symbol="ZWFR", symbol_id=1,
                                interval="1h", asset_class="equity",
                                exchange="TEST", data_root=racine, metadata_db=meta)
    return store


def _config(signal):
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=signal, symbol_names={"ZWFR": 1})
    cfg.warmup_bars = 0
    return cfg


def _strategie():
    from manifoldbt.indicators import close, sma
    m = sma(close, bt.param("n"))
    return (bt.Strategy.create("s")
            .signal("m", m)
            .size(bt.when(close > m, 1.0, 0.0)))


WF = {"geometry": "anchored", "n_splits": 2, "train_ratio": 0.6,
      "optimize_metric": "total_return", "param_grid": {"n": [4, 8]}}


def test_les_metriques_oos_ne_dependent_pas_de_la_sortie(tmp_path):
    store = _magasin_horaire(tmp_path)
    auto = bt.run_walk_forward(_strategie(), WF, _config(bt.Interval.hours(12)), store)
    cfg_native = _config(bt.Interval.hours(12))
    cfg_native.output_resolution = bt.Interval.hours(1)
    native = bt.run_walk_forward(_strategie(), WF, cfg_native, store)

    assert auto["n_folds"] == native["n_folds"] == 2
    for fa, fn in zip(auto["folds"], native["folds"]):
        assert fa["best_params"] == fn["best_params"]
        for k in ("sharpe", "cagr", "total_return", "volatility", "max_drawdown"):
            assert fa["oos_metrics"][k] == pytest.approx(fn["oos_metrics"][k], rel=1e-12), k
        assert fa["oos_metrics"]["sharpe"] != 0.0, "le pli doit trader"
        if fn["wfe"] is None:
            assert fa["wfe"] is None
        else:
            assert fa["wfe"] == pytest.approx(fn["wfe"], rel=1e-12)


def test_la_courbe_oos_suit_la_resolution_de_sortie(tmp_path):
    store = _magasin_horaire(tmp_path)
    auto = bt.run_walk_forward(_strategie(), WF, _config(bt.Interval.hours(12)), store)
    cfg_native = _config(bt.Interval.hours(12))
    cfg_native.output_resolution = bt.Interval.hours(1)
    native = bt.run_walk_forward(_strategie(), WF, cfg_native, store)
    for fa, fn in zip(auto["folds"], native["folds"]):
        eq, ts = fa["oos_equity"], fa["oos_timestamps"]
        assert len(eq) == len(ts) > 0
        assert ts[0] >= fa["test_range"]["start"]
        assert ts[-1] < fa["test_range"]["end"]
        # Un point par seau de sortie, au plus 12h d'ecart...
        pas = np.diff(np.asarray(ts))
        assert pas.max() == 12 * HEURE_NS and pas.min() > 0
        # ... et chaque point est celui de la courbe native a cet instant.
        natif = dict(zip(fn["oos_timestamps"], fn["oos_equity"]))
        assert len(natif) == len(fn["oos_timestamps"]) > len(ts)
        for t, e in zip(ts, eq):
            assert natif[t] == e
