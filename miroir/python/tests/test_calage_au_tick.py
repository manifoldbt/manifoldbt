"""`execution.tick_size` cale les niveaux au repos sur la grille de la venue.

Trois choses.

1. La SERIALISATION. Absent reste absent : une config qui ne parle pas de la
   grille produit le meme JSON qu'avant, octet pour octet. C'est ce qui garantit
   qu'aucun run existant ne change.
2. Le REFUS. Un tick nul, negatif ou non fini est refuse par son nom, avant la
   premiere barre.
3. Les NIVEAUX. Un ordre au repos hors grille est ramene dessus dans le sens de
   la passivite (un bid vers le bas, un ask vers le haut), et
   `order_activity["levels_snapped"]` compte ceux qui ont bouge. Un ordre
   agressif n'est pas touche : il s'execute au prix du print qui l'a declenche.

Les valeurs elles-memes sont epinglees par les tests du moteur ; ici
on verifie que le reglage traverse Python et que le compteur remonte.
"""
import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

N = 40


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Des barres journalieres qui descendent a 99.5 et montent a 100.5."""
    ts = pd.date_range("2023-01-01", periods=N, freq="1D", tz="UTC")
    close = np.full(N, 100.0)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(N, 1000.0),
    })
    root = tmp_path_factory.mktemp("calage_au_tick")
    return bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1d",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )


def _config(tick, orders) -> bt.BacktestConfig:
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=int(pd.Timestamp("2024-01-01", tz="UTC").value),
        bar_interval=Interval.days(1),
        initial_capital=10_000.0,
        warmup_bars=0,
        execution=bt.ExecutionConfig(
            signal_delay=1, execution_price="AtClose", max_position_pct=1.0,
            allow_short=True, position_sizing_mode="Units",
            tick_size=tick, orders=orders,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
    )


def _strategie() -> "bt.Strategy":
    return (bt.Strategy.create("quote").signal("un", lit(1.0)).size(col("un")))


def _lance(store, tick, *, price=99.63, trigger="Limit"):
    orders = bt.OrderConfig(
        limit_entry={
            "price": {"Absolute": price},
            "trigger": trigger,
            "time_in_force": "GTC",
        }
    )
    return bt.run(_strategie(), _config(tick, orders), store)


# ---------------------------------------------------------------------------
# Serialisation et refus
# ---------------------------------------------------------------------------


def test_absent_reste_absent_dans_le_json():
    assert "tick_size" not in bt.ExecutionConfig().to_json_dict()
    assert bt.ExecutionConfig(tick_size=0.1).to_json_dict()["tick_size"] == 0.1


@pytest.mark.parametrize("mauvais", [0.0, -0.1, float("nan"), float("inf")])
def test_un_tick_qui_nest_pas_positif_est_refuse_par_son_nom(store, mauvais):
    with pytest.raises(Exception) as leve:
        _lance(store, mauvais)
    assert "tick_size" in str(leve.value)


# ---------------------------------------------------------------------------
# Les niveaux
# ---------------------------------------------------------------------------


def _prix_dentree(res) -> float:
    trades = res.trades_df()
    entrees = trades[trades["side"] == 1]
    assert len(entrees) > 0, "l'entree doit remplir, sinon le test ne dit rien"
    return float(entrees["fill_price"].iloc[0])


def test_un_bid_est_ramene_vers_le_bas_et_le_compteur_le_dit(store):
    sans = _lance(store, None)
    avec = _lance(store, 0.1)

    assert _prix_dentree(sans) == 99.63, "sans grille, le niveau est pose tel quel"
    assert _prix_dentree(avec) == 99.6, "un bid descend sur la grille"

    assert sans.order_activity.get("levels_snapped", 0) == 0
    assert avec.order_activity["levels_snapped"] >= 1


def test_un_niveau_deja_sur_la_grille_ne_bouge_pas(store):
    res = _lance(store, 0.1, price=99.6)
    assert _prix_dentree(res) == 99.6
    assert res.order_activity.get("levels_snapped", 0) == 0


def test_un_ordre_agressif_nest_pas_touche(store):
    avec = _lance(store, 0.1, price=100.37, trigger="Stop")
    sans = _lance(store, None, price=100.37, trigger="Stop")
    assert _prix_dentree(avec) == _prix_dentree(sans)
    assert avec.order_activity.get("levels_snapped", 0) == 0
