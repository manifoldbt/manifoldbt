"""Les diagnostics doivent accepter les series EXOGENES, comme `run`.

`detect_lookahead` relancait la strategie avec des tables exo vides, et
`AlignedData` (sur lequel `check_exposure_stability` relance ses runs) n'en
portait aucune : une strategie lisant `exo.<nom>.<colonne>` levait
"unknown input column" dans les deux, alors que `mbt.run` la faisait tourner.
"""
import os

import pytest

import manifoldbt as bt

# Safety diagnostics are a Pro feature.
pytestmark = pytest.mark.skipif(
    bt.license_info()[0] not in ("Pro", "Researcher"),
    reason="requires Pro: skipped on a Community wheel",
)

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _monte(tmp_path):
    """Serie horaire reguliere, plus une moyenne posee en serie exogene."""
    n = 2400
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    pas = np.sin(np.arange(n) / 37.0) * 2.0 + np.cos(np.arange(n) / 11.0)
    px = 100.0 + np.cumsum(pas) * 0.05
    df = pd.DataFrame({"open": px, "high": px * 1.004, "low": px * 0.996,
                       "close": px, "volume": np.full(n, 1000.0)}, index=idx)
    racine = str(tmp_path / "data")
    meta = str(tmp_path / "meta.sqlite")
    os.makedirs(racine, exist_ok=True)
    store = bt.import_dataframe(df.reset_index(), symbol="ZDEX", symbol_id=1,
                                interval="1h", asset_class="equity",
                                exchange="TEST", data_root=racine, metadata_db=meta)
    # Moyenne CAUSALE (fenetre glissante sur le passe) : aucune fuite.
    moy = pd.Series(px).rolling(24).mean().to_numpy()
    bt.register_exo("moyenne", pd.DataFrame({"timestamp": idx, "sma": moy}),
                    store=store, data_root=racine, timeframe="1h")
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=bt.Interval.hours(1), symbol_names={"ZDEX": 1})
    cfg.warmup_bars = 0
    cfg.exo_data = ["moyenne"]
    return store, cfg


def _strategie():
    from manifoldbt.indicators import close, col
    m = col("exo.moyenne.sma")
    return (bt.Strategy.create("s")
            .signal("m", m)
            .size(bt.when(close > m, 1.0, 0.0)))


def test_la_strategie_a_exo_trade_dans_run(tmp_path):
    store, cfg = _monte(tmp_path)
    r = bt.run(_strategie(), cfg, store)
    assert r.trade_count > 0


def test_detect_lookahead_accepte_une_serie_exogene(tmp_path):
    store, cfg = _monte(tmp_path)
    rapport = bt.diagnostics.detect_lookahead(_strategie(), cfg, store, mode="all")
    # Une moyenne glissante sur le passe ne fuit pas : les re-runs tronques
    # doivent retrouver les trades du run complet.
    assert rapport.passed, rapport


def test_check_exposure_stability_accepte_une_serie_exogene(tmp_path):
    store, cfg = _monte(tmp_path)
    rapport = bt.diagnostics.check_exposure_stability(_strategie(), cfg, store)
    assert rapport.passed, rapport
