"""A bar whose price is NaN no longer aborts a run.

A NaN in ``open``/``close`` (a value, not a null: Arrow keeps it) used to be
booked as a fill price. The cash went NaN, then the equity, and the next sized
signal reached ``f64::clamp`` with NaN bounds and panicked. A NaN ``high`` or
``low`` reached the stop-loss fill clamp the same way, and so did a bar whose
low sits above its high. These tests pin the outcome on every path a run can
take: the general loop (``bt.run`` with a stop-loss), the lite sweep on its
fast kernel and on the general loop, and the two must agree.
"""
import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
from manifoldbt.expr import col, when
from manifoldbt.helpers import Interval, Slippage
from manifoldbt.indicators import sma

pa = pytest.importorskip("pyarrow")

N = 1500
CAPITAL = 10_000.0
TROUS = np.arange(100, N, 37)


def _table(colonnes=(), inverse=False):
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.006, N)))
    data = {
        "open": close.copy(), "high": close * 1.004, "low": close * 0.996,
        "close": close.copy(), "volume": rng.uniform(500.0, 1500.0, N),
    }
    for c in colonnes:
        data[c][TROUS] = np.nan
    if inverse:
        data["high"][TROUS], data["low"][TROUS] = (
            data["low"][TROUS].copy(), data["high"][TROUS].copy())
    ts = pd.date_range("2023-01-01", periods=N, freq="1h", tz="UTC")
    # `from_pandas=False` keeps NaN as a value: pandas would turn it into a
    # null, which the import refuses for a price column.
    cols = {"timestamp": pa.array(ts)}
    cols.update({k: pa.array(v, from_pandas=False) for k, v in data.items()})
    return pa.table(cols)


def _store(tmp_path, nom, **kw):
    root = tmp_path / nom
    return bt.import_dataframe(
        _table(**kw), symbol="T", symbol_id=1, interval="1h",
        data_root=str(root / "data"), metadata_db=str(root / "m.sqlite"))


def _config(prix):
    fin = int(pd.Timestamp("2023-01-01", tz="UTC").value) + (N + 24) * 3_600 * 10**9
    return bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=fin,
        bar_interval=Interval.hours(1), initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=1, execution_price=prix, max_position_pct=1.0,
            allow_short=True, position_sizing_mode="FractionOfEquity"),
        fees=bt.FeeConfig(taker_fee_bps=5.0, maker_fee_bps=5.0),
        slippage=Slippage.none(), warmup_bars=15)


def _strategie():
    # Driven by the volume, which has no hole: the signal keeps moving across
    # the NaN bars, so fills do land on them.
    rapide, lente = sma(col("volume"), 3), sma(col("volume"), 10)
    return (bt.Strategy.create("trous").signal("v", col("volume"))
            .size(when(rapide > lente, 0.5, -0.5)).stop_loss(pct=0.3))


def _lite(store, prix, monkeypatch, general):
    if general:
        monkeypatch.setenv("MBT_NO_FAST_PATH", "1")
    else:
        monkeypatch.delenv("MBT_NO_FAST_PATH", raising=False)
    [r] = bt.run_sweep_lite(
        _strategie(), {"stop_loss": [0.3]}, _config(prix), store, device="cpu")
    monkeypatch.delenv("MBT_NO_FAST_PATH", raising=False)
    return r


CAS = [
    ("cloture", dict(colonnes=("close",)), "AtClose"),
    ("ouverture", dict(colonnes=("open",)), "AtOpen"),
    ("haut", dict(colonnes=("high",)), "AtClose"),
    ("bas", dict(colonnes=("low",)), "AtClose"),
    ("haut_ouverture", dict(colonnes=("high",)), "AtOpen"),
    ("bas_ouverture", dict(colonnes=("low",)), "AtOpen"),
    ("barre_inversee", dict(inverse=True), "AtClose"),
]


@pytest.mark.parametrize("nom, kw, prix", CAS, ids=[c[0] for c in CAS])
def test_un_prix_nan_ne_fait_plus_paniquer_le_run(tmp_path, monkeypatch, nom, kw, prix):
    store = _store(tmp_path, nom, **kw)
    plein = bt.run(_strategie(), _config(prix), store)
    equite = plein.equity_df()["equity"].to_numpy()
    assert np.isfinite(equite).all(), f"{nom}: l'equite contient des NaN"
    fills = plein.trades_df()
    assert len(fills) > 100, "sans remplissage, le cas ne prouve rien"
    prix_fill = fills["fill_price"].to_numpy()
    assert np.isfinite(prix_fill).all(), f"{nom}: un remplissage s'est fait a NaN"

    rapide = _lite(store, prix, monkeypatch, general=False)
    general = _lite(store, prix, monkeypatch, general=True)
    assert np.isfinite(rapide.metrics["total_return"])
    # Same backtest whichever kernel runs it, NaN bars included.
    assert rapide.metrics["total_return"] == general.metrics["total_return"]
    assert rapide.metrics["total_return"] == pytest.approx(
        plein.metrics["total_return"], rel=1e-12)


@pytest.mark.parametrize("colonne, prix", [("close", "AtClose"), ("open", "AtOpen")])
def test_le_repli_sur_la_derniere_cloture_est_annonce(tmp_path, colonne, prix):
    """A fill on a bar without its price lands on the last known close, and
    the run says how many did."""
    store = _store(tmp_path, colonne, colonnes=(colonne,))
    plein = bt.run(_strategie(), _config(prix), store)
    avis = [w for w in plein.warnings if "no execution price" in w]
    assert len(avis) == 1, plein.warnings

    table = _table(colonnes=(colonne,))
    ts = pd.DatetimeIndex(table.column("timestamp").to_pandas()).as_unit("ns").asi8
    close = table.column("close").to_numpy()
    trou = np.isnan(table.column(colonne).to_numpy())
    derniere = pd.Series(close).ffill().to_numpy()
    fills = plein.trades_df()
    # Signal fills only: a stop exit fills at its own level, not at the bar's price.
    signal = fills[fills["exit_reason"] == 0]
    execution = pd.DatetimeIndex(
        pd.to_datetime(signal["execution_timestamp"], utc=True)).as_unit("ns").asi8
    rang = np.searchsorted(ts, execution)
    assert (ts[rang] == execution).all()
    sur_trou = trou[rang]
    assert sur_trou.sum() > 0, "aucun remplissage ne tombe sur une barre trouee"
    # To the ulp: the trade log carries the order's average price (price * qty
    # / qty), which can land one ulp away from the price itself.
    np.testing.assert_allclose(
        signal["intended_price"].to_numpy()[sur_trou], derniere[rang[sur_trou]],
        rtol=1e-14, atol=0.0)
