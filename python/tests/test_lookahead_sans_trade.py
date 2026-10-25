"""Un sous-test de re-run qui n'a rien compare ne rend pas PASS.

detect_lookahead rendait PASS sur une strategie qui ne trade jamais (un
param() sans valeur suffit), ou quand le run tronque echouait : un verdict
vide lu comme un feu vert. Il rend maintenant INCONCLUSIVE, `.passed` vaut
False et `assert_clean()` leve, en disant pourquoi.
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
    n = 2400
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    pas = np.sin(np.arange(n) / 37.0) * 2.0 + np.cos(np.arange(n) / 11.0)
    px = 100.0 + np.cumsum(pas) * 0.05
    df = pd.DataFrame({"open": px, "high": px * 1.004, "low": px * 0.996,
                       "close": px, "volume": np.full(n, 1000.0)}, index=idx)
    racine = str(tmp_path / "data")
    os.makedirs(racine, exist_ok=True)
    store = bt.import_dataframe(df.reset_index(), symbol="ZLSV", symbol_id=1,
                                interval="1h", asset_class="equity",
                                exchange="TEST", data_root=racine,
                                metadata_db=str(tmp_path / "meta.sqlite"))
    tr0, tr1 = bt.time_range("2021-01-01", "2021-04-11")
    cfg = bt.BacktestConfig(universe=[1], time_range_start=tr0, time_range_end=tr1,
                            initial_capital=1000.0, provider="TEST",
                            bar_interval=bt.Interval.hours(1), symbol_names={"ZLSV": 1})
    cfg.warmup_bars = 0
    return store, cfg


def _jamais():
    from manifoldbt.indicators import close
    return (bt.Strategy.create("jamais")
            .signal("c", close)
            .size(bt.lit(0.0)))


def _croise():
    from manifoldbt.indicators import close, sma
    m = sma(close, 24)
    return bt.Strategy.create("croise").signal("m", m).size(bt.when(close > m, 1.0, 0.0))


def test_une_strategie_qui_ne_trade_pas_est_inconclusive(tmp_path):
    store, cfg = _monte(tmp_path)
    rapport = bt.diagnostics.detect_lookahead(_jamais(), cfg, store, mode="all")
    assert not rapport.passed
    re_runs = [r for r in rapport.reports if r.method != "static"]
    assert {r.method for r in re_runs} == {"extension", "truncation"}
    for r in re_runs:
        assert r.inconclusive and "made no trade" in r.inconclusive
        assert "INCONCLUSIVE" in str(r)
    statique = [r for r in rapport.reports if r.method == "static"]
    assert statique and statique[0].passed
    assert str(rapport).startswith("Lookahead diagnostics: INCONCLUSIVE")
    with pytest.raises(AssertionError, match="inconclusive"):
        rapport.assert_clean()


def test_une_strategie_causale_qui_trade_reste_pass(tmp_path):
    store, cfg = _monte(tmp_path)
    rapport = bt.diagnostics.detect_lookahead(_croise(), cfg, store, mode="all")
    assert rapport.passed, rapport
    assert all(r.inconclusive is None for r in rapport.reports)
    assert str(rapport).startswith("Lookahead diagnostics: PASS")
