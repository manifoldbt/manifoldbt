"""Cas limites d'un run : parametres, periodes, membres, exo et funding.

Un param() sans valeur en lot, une periode fractionnaire, un membre qui demarre
tard sous une cible constante, une valeur exo datee d'avant le run, un NaN dans
le funding. Plus deux avertissements : des barres spot rangees comme perpetuel,
et un symbole que seul symbol_ref() nomme et qui trade.
"""
import glob
import os
import warnings

import pytest

import manifoldbt as bt
from manifoldbt.exceptions import StrategyError
from manifoldbt.expr import col, lit, symbol_ref, when
from manifoldbt.indicators import close, ema, sma

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")

pro = pytest.mark.skipif(
    bt.license_info()[0] not in ("Pro", "Researcher"),
    reason="requires Pro: skipped on a Community wheel",
)

H1 = bt.Interval.hours(1)


def _barres(idx, px=None):
    n = len(idx)
    if px is None:
        bruit = np.random.default_rng(7).normal(0.0, 1.0, n)
        px = 200.0 + np.cumsum(np.sin(np.arange(n) / 7.0) + bruit)
    return pd.DataFrame({"timestamp": idx, "open": px, "high": px * 1.002, "low": px * 0.998,
                         "close": px, "volume": np.full(n, 1000.0)})


def _magasin(tmp_path, cadres):
    racine = str(tmp_path / "data")
    os.makedirs(racine, exist_ok=True)
    ids, store = {}, None
    for i, (nom, df) in enumerate(cadres.items(), 1):
        store = bt.import_dataframe(df, symbol=nom, symbol_id=i, interval="1h",
                                    asset_class="equity", exchange="TEST", data_root=racine,
                                    metadata_db=str(tmp_path / "meta.sqlite"))
        ids[nom] = i
    return store, racine, ids


def _config(ids, debut, fin, **attrs):
    t0, t1 = bt.time_range(debut, fin)
    cfg = bt.BacktestConfig(universe=list(ids.values()), time_range_start=t0, time_range_end=t1,
                            initial_capital=100000.0, provider="TEST", bar_interval=H1,
                            symbol_names=dict(ids))
    for nom, valeur in attrs.items():
        setattr(cfg, nom, valeur)
    return cfg


def _strategie(nom, taille, **signaux):
    strat = bt.Strategy.create(nom).signal("c", close)
    for cle, expr in signaux.items():
        strat = strat.signal(cle, expr)
    return strat.size(taille)


def _ajoute_funding(racine, nom, taux):
    """Une colonne de funding sur le fichier d'un symbole. `taux` garde ses NaN."""
    chemin = glob.glob(os.path.join(racine, "**", f"{nom}.arrow"), recursive=True)[0]
    with open(chemin, "rb") as f:
        table = pa.ipc.open_file(pa.BufferReader(f.read())).read_all()
    table = table.append_column("funding_rate", pa.array(taux, type=pa.float64()))
    with pa.ipc.new_file(chemin, table.schema) as w:
        w.write_table(table)


IDX = pd.date_range("2021-01-01", periods=600, freq="1h", tz="UTC", name="timestamp")


@pytest.fixture
def seul(tmp_path):
    store, _, ids = _magasin(tmp_path, {"AAA": _barres(IDX)})
    return store, _config(ids, "2021-01-01", "2021-01-20")


def _sma_sans_valeur():
    return _strategie("u", when(close > col("m"), 1.0, 0.0), m=sma(close, bt.param("p")))


def test_run_batch_refuse_un_param_sans_valeur(seul):
    store, cfg = seul
    with pytest.raises(StrategyError, match="'p'"):
        bt.run_batch([_sma_sans_valeur()], cfg, store)


def test_run_batch_lite_refuse_un_param_sans_valeur(seul):
    store, cfg = seul
    with pytest.raises(StrategyError, match="'p'"):
        bt.run_batch_lite([_sma_sans_valeur()], cfg, store)


@pro
def test_detect_lookahead_refuse_un_param_sans_valeur(seul):
    store, cfg = seul
    with pytest.raises(StrategyError, match="'p'"):
        bt.diagnostics.detect_lookahead(_sma_sans_valeur(), cfg, store, mode="all")


def test_un_balayage_fixe_toujours_le_param(seul):
    store, cfg = seul
    res = bt.run_sweep_lite(_sma_sans_valeur(), {"p": [3, 5]}, cfg, store)
    assert len(res) == 2


@pytest.mark.parametrize("periode", [5.4, 5.6, 2.5])
def test_une_periode_fractionnaire_est_refusee(periode):
    with pytest.raises(ValueError, match="whole number"):
        sma(close, periode)


def test_une_periode_fractionnaire_par_param_est_refusee(seul):
    store, cfg = seul
    strat = _strategie("f", when(close > col("m"), 1.0, 0.0),
                       m=sma(close, bt.param("p", default=5.6)))
    with pytest.raises(Exception, match="whole number"):
        bt.run(strat, cfg, store)


def test_une_grille_fractionnaire_est_refusee(seul):
    store, cfg = seul
    with pytest.raises(Exception, match="whole number"):
        bt.run_sweep_lite(_sma_sans_valeur(), {"p": [5.0, 5.4]}, cfg, store)


def test_une_periode_ronde_ecrite_en_flottant_reste_acceptee(seul):
    store, cfg = seul
    def taille(p):
        return _strategie(f"s{p}", when(close > col("m"), 1.0, 0.0), m=sma(close, p))
    assert (bt.run(taille(5.0), cfg, store).metrics["total_return"]
            == bt.run(taille(5), cfg, store).metrics["total_return"])


def test_la_span_d_une_ema_reste_un_reel(seul):
    store, cfg = seul
    def taille(s):
        return _strategie(f"e{s}", when(close > col("m"), 1.0, 0.0), m=ema(close, s))
    assert (bt.run(taille(5.4), cfg, store).metrics["total_return"]
            != bt.run(taille(5), cfg, store).metrics["total_return"])


@pytest.fixture
def tardif(tmp_path):
    """BBB ne commence que dix jours apres AAA, a prix constant."""
    tot = pd.date_range("2024-02-12", periods=480, freq="1h", tz="UTC", name="timestamp")
    store, _, ids = _magasin(tmp_path, {"AAA": _barres(tot, np.full(480, 100.0)),
                                        "BBB": _barres(tot[240:], 100.0 + np.arange(240.0) * 0.1)})
    return store, _config(ids, "2024-02-12", "2024-03-03")


def _par_symbole(res):
    return res.trades_df(backend="pandas").groupby("symbol_id").size().to_dict()


def test_un_membre_tardif_entre_a_sa_premiere_barre_sous_une_cible_constante(tardif):
    store, cfg = tardif
    res = bt.run(_strategie("k", lit(0.2)), cfg, store)
    assert set(_par_symbole(res)) == {1, 2}
    t = res.trades_df(backend="pandas")
    premier = pd.Timestamp(t[t["symbol_id"] == 2]["execution_timestamp"].iloc[0])
    assert premier == pd.Timestamp("2024-02-22", tz="UTC")


def test_cible_constante_et_cible_qui_lit_les_barres_font_le_meme_run(tardif):
    store, cfg = tardif
    constante = bt.run(_strategie("k", lit(0.2)), cfg, store)
    lue = bt.run(_strategie("l", when(col("close") > 0, 0.2, 0.0)), cfg, store)
    assert constante.metrics["total_return"] == lue.metrics["total_return"]


def test_le_membre_tardif_entre_aussi_sur_les_chemins_rapides(tardif):
    store, cfg = tardif
    strat = _strategie("k", lit(0.2))
    attendu = bt.run(strat, cfg, store).metrics["total_return"]
    assert bt.run_batch_lite([strat], cfg, store)[0].metrics["total_return"] == pytest.approx(attendu, abs=1e-12)
    grille = _strategie("g", bt.param("w", default=0.2))
    balaye = bt.run_sweep_lite(grille, {"w": [0.2]}, cfg, store)
    assert balaye[0].metrics["total_return"] == pytest.approx(attendu, abs=1e-12)


def _exo(racine, store, nom, stamps, valeurs):
    bt.register_exo(nom, pd.DataFrame({"timestamp": pd.to_datetime(stamps, utc=True), "v": valeurs}),
                    store=store, data_root=racine, timeframe="1h")


@pytest.fixture
def exo_mensuel(tmp_path):
    idx = pd.date_range("2021-03-01", periods=1200, freq="1h", tz="UTC", name="timestamp")
    store, racine, ids = _magasin(tmp_path, {"AAA": _barres(idx, np.full(1200, 100.0))})
    return store, racine, ids


def _premier_fill(store, ids, debut="2021-03-20", nom="m"):
    cfg = _config(ids, debut, "2021-04-10", exo_data=[nom])
    v = col(f"exo.{nom}.v")
    res = bt.run(_strategie("e", when(v > 0.5, 1.0, 0.0), v=v), cfg, store)
    t = res.trades_df(backend="pandas")
    return pd.Timestamp(t["execution_timestamp"].iloc[0]) if len(t) else None


def test_la_valeur_exo_d_avant_le_run_est_en_vigueur_a_son_debut(exo_mensuel):
    store, racine, ids = exo_mensuel
    _exo(racine, store, "m", ["2021-03-01", "2021-04-01"], [1.0, 2.0])
    assert _premier_fill(store, ids) == pd.Timestamp("2021-03-20", tz="UTC")


def test_seule_la_derniere_valeur_d_avant_le_run_compte(exo_mensuel):
    store, racine, ids = exo_mensuel
    # 1 puis 0 avant le run : la valeur en vigueur est 0, le run ne trade pas avant avril.
    _exo(racine, store, "m", ["2021-03-01", "2021-03-10", "2021-04-01"], [1.0, 0.0, 1.0])
    assert _premier_fill(store, ids) == pd.Timestamp("2021-04-01", tz="UTC")


def test_une_serie_qui_commence_dans_le_run_ne_remonte_pas_avant_elle(exo_mensuel):
    store, racine, ids = exo_mensuel
    _exo(racine, store, "m", ["2021-03-25"], [1.0])
    assert _premier_fill(store, ids) == pd.Timestamp("2021-03-25", tz="UTC")


def test_aucune_valeur_future_n_est_lue(exo_mensuel):
    store, racine, ids = exo_mensuel
    _exo(racine, store, "m", ["2021-04-20"], [1.0])
    assert _premier_fill(store, ids) is None


@pytest.fixture
def funding(tmp_path):
    idx = IDX[:192]
    store, racine, ids = _magasin(tmp_path, {"AAA": _barres(idx, np.full(192, 100.0))})
    return store, racine, ids


def _cfg_funding(ids):
    cfg = _config(ids, "2021-01-01", "2021-01-09")
    cfg.fees = bt.FeeConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, funding_rate_column="funding_rate")
    return cfg


def test_un_nan_de_funding_vaut_un_taux_absent_et_se_dit(funding):
    store, racine, ids = funding
    taux = np.full(192, -1e-4)
    taux[50] = np.nan
    _ajoute_funding(racine, "AAA", taux)
    res = bt.run(_strategie("c", when(col("close") > 0, 10.0, 0.0)), _cfg_funding(ids), store)
    assert not np.isnan(np.asarray(res.equity_curve, dtype=float)).any()
    assert any("1 NaN" in w for w in res.warnings), res.warnings


def test_nan_et_taux_absent_font_le_meme_run_partout(funding, tmp_path):
    store, racine, ids = funding
    taux = np.full(192, -1e-4)
    taux[::3] = np.nan
    _ajoute_funding(racine, "AAA", taux)
    strat = _strategie("c", when(col("close") > 0, 1.0, 0.0))
    cfg = _cfg_funding(ids)
    assert (bt.run(strat, cfg, store).metrics["total_return"]
            == pytest.approx(bt.run_batch_lite([strat], cfg, store)[0].metrics["total_return"], abs=1e-12))


def test_un_taux_sans_nan_ne_dit_rien(funding):
    store, racine, ids = funding
    _ajoute_funding(racine, "AAA", np.full(192, -1e-4))
    res = bt.run(_strategie("c", when(col("close") > 0, 10.0, 0.0)), _cfg_funding(ids), store)
    assert not [w for w in res.warnings if "NaN" in w]


@pytest.mark.parametrize("fournisseur,attendu", [("binance", True), ("bybit", True), ("hyperliquid", False)])
def test_des_barres_spot_rangees_comme_perpetuel_se_disent(monkeypatch, fournisseur, attendu):
    monkeypatch.setattr(bt, "_ingest_native", lambda **kw: object())
    with warnings.catch_warnings(record=True) as vus:
        warnings.simplefilter("always")
        bt.ingest(provider=fournisseur, symbol="BTCUSDT", symbol_id=1, start="2021-01-01T00:00:00Z",
                  end="2021-01-02T00:00:00Z", asset_class="crypto_perp", progress=False)
    assert any("spot bars" in str(w.message) for w in vus) is attendu


def test_le_spot_ne_dit_rien(monkeypatch):
    monkeypatch.setattr(bt, "_ingest_native", lambda **kw: object())
    with warnings.catch_warnings(record=True) as vus:
        warnings.simplefilter("always")
        bt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1, start="2021-01-01T00:00:00Z",
                  end="2021-01-02T00:00:00Z", progress=False)
    assert not [w for w in vus if "spot bars" in str(w.message)]


@pytest.fixture
def paire(tmp_path):
    idx = IDX[:400]
    store, _, _ = _magasin(tmp_path, {"SOLUSDT": _barres(idx, np.full(400, 100.0)),
                                      "ETHUSDT": _barres(idx, 50.0 + np.arange(400.0) * 0.1)})
    t0, t1 = bt.time_range("2021-01-01", "2021-01-16")
    cfg = bt.BacktestConfig(universe=["SOLUSDT"], time_range_start=t0, time_range_end=t1,
                            initial_capital=100000.0, provider="TEST", bar_interval=H1)
    return store, cfg


def _avertis(strat, cfg, store):
    with warnings.catch_warnings(record=True) as vus:
        warnings.simplefilter("always")
        res = bt.run(strat, cfg, store)
    return res, [str(w.message) for w in vus if "symbol_ref()" in str(w.message)]


def test_un_symbole_reference_qui_trade_se_dit(paire):
    store, cfg = paire
    ratio = col("close") / symbol_ref("ETHUSDT", "close")
    pos = when(ratio > lit(0.0), lit(0.5), lit(0.0))
    res, messages = _avertis(bt.Strategy.create("r").signal("pos", pos).size(col("pos")), cfg, store)
    assert 2 in _par_symbole(res)
    assert messages and "ETHUSDT" in messages[0]


def test_un_symbole_reference_qui_reste_plat_ne_dit_rien(paire):
    store, cfg = paire
    ratio = col("close") / symbol_ref("ETHUSDT", "close")
    pos = when(ratio > lit(0.0), lit(0.5), lit(0.0))
    filtre = when(col("symbol_id") == lit(1.0), col("pos"), lit(0.0))
    res, messages = _avertis(bt.Strategy.create("f").signal("pos", pos).size(filtre), cfg, store)
    assert set(_par_symbole(res)) == {1}
    assert not messages
