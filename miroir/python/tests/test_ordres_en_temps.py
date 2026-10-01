"""Orders that live in TIME, and a level that reads the position held.

Three surfaces are checked here:

* ``time_in_force`` accepts a duration -- ``Interval.seconds(n)``,
  ``Interval.millis(n)``, or the explicit ``{"GTT": Interval.millis(n)}`` -- and
  turns it into the object shape the engine reads;
* ``result.order_activity`` says how many quotes it took to get the fills, next
  to ``result.fill_fragility``, which says what the fills were worth;
* ``position()`` is readable in the level of an entry order and refused by name
  everywhere else.

Everything asserted below is a count of FILLS, a fill PRICE or a count of
QUOTES. Never an intraday position: the free tier returns one row per day, so a
test that read a position between two bars would be red there and green here for
no reason of its own.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt import position  # noqa: E402
from manifoldbt.config import time_in_force  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

# (open, high, low, close), one 1-minute bar each. With signal_delay=1 and a
# constant target the levels work out as:
#   bar1: order #1 rests at close[0] * 0.98 = 98.00, low 99   -> no fill
#   bar2: low 99  -> no fill
#   bar3: low 100 -> no fill
#   bar4: low 98.5 -> an order resting at 98.98 FILLS
#   bar5: low 98.0 -> an order still resting at 98.00 fills only here
ROWS = [
    (100.0, 101.0, 99.0, 100.0),
    (100.0, 102.0, 99.0, 101.0),
    (101.0, 102.0, 99.0, 101.0),
    (101.0, 102.0, 100.0, 101.0),
    (101.0, 102.0, 98.5, 99.0),
    (99.0, 100.0, 98.0, 99.0),
]


def _store(tmp_path):
    o, h, l, c = (np.array([r[i] for r in ROWS], dtype=np.float64) for i in range(4))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2021-03-01", periods=len(ROWS), freq="1min", tz="UTC"
            ),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": np.full(len(ROWS), 1_000.0),
        }
    )
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df,
        symbol="TEST",
        symbol_id=1,
        interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    return store, int(df["timestamp"].iloc[-1].value)


def _config(last_ns, pyramiding=False):
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=last_ns + 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000.0,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            position_sizing_mode="Units",
            pyramiding=pyramiding,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def _strategy(tif):
    return (
        bt.Strategy.create("duree")
        .signal("sig", lit(1.0))
        .size(col("sig"))
        .limit_entry(offset_bps=200, time_in_force=tif)
    )


# ---------------------------------------------------------------------------
# The four shapes a lifetime is written in
# ---------------------------------------------------------------------------


def test_une_duree_devient_la_forme_que_le_moteur_lit():
    assert time_in_force(Interval.seconds(1)) == {"GTT": {"duration_ns": 1_000_000_000}}
    assert time_in_force(Interval.millis(500)) == {"GTT": {"duration_ns": 500_000_000}}
    assert time_in_force({"GTT": Interval.millis(500)}) == {
        "GTT": {"duration_ns": 500_000_000}
    }
    # Already in the engine's shape: carried through untouched.
    assert time_in_force({"GTT": {"duration_ns": 7}}) == {"GTT": {"duration_ns": 7}}
    # Nothing written before durations existed changes meaning.
    for unchanged in ("GTC", "IOC", {"GTB": 5}):
        assert time_in_force(unchanged) == unchanged


def test_une_duree_nulle_ou_une_horloge_sont_refusees_par_leur_nom():
    with pytest.raises(ValueError, match="positive"):
        time_in_force(Interval.seconds(0))
    with pytest.raises(TypeError, match="clock"):
        time_in_force(Interval.trades())
    with pytest.raises(TypeError, match="duration"):
        time_in_force({"GTT": "une seconde"})


def test_le_constructeur_de_strategie_accepte_un_intervalle():
    strategy = _strategy(Interval.seconds(1))
    entry = strategy.to_json_dict()["orders"]["limit_entry"]
    assert entry["time_in_force"] == {"GTT": {"duration_ns": 1_000_000_000}}


# ---------------------------------------------------------------------------
# What a duration does to a run
# ---------------------------------------------------------------------------


def test_une_duree_d_une_barre_recote_a_chaque_barre(tmp_path):
    """Half a second on a one-minute grid: the order dies on every bar, and the
    one posted in its place reads the level of the bar that posts it.

    The same trade comes out of ``{"GTB": 2}``, which requotes half as often --
    so the fill is the control and the QUOTE COUNTS are what separate them."""
    store, last_ns = _store(tmp_path)

    rapide = bt.run(_strategy(Interval.millis(500)), _config(last_ns), store)
    lent = bt.run(_strategy({"GTB": 2}), _config(last_ns), store)

    for res in (rapide, lent):
        assert res.trade_count == 1
        assert float(res.trades_df()["fill_price"].iloc[0]) == pytest.approx(
            98.98, abs=1e-9
        )

    assert rapide.order_activity == {
        "orders_posted": 4,
        "requotes": 3,
        "expired_unfilled": 3,
    }
    assert lent.order_activity == {
        "orders_posted": 2,
        "requotes": 1,
        "expired_unfilled": 1,
    }


def test_un_ordre_qui_ne_meurt_pas_ne_recote_pas(tmp_path):
    """The control: under ``GTC`` the order rests untouched at its first level,
    so it fills lower and later, and nothing is requoted."""
    store, last_ns = _store(tmp_path)
    res = bt.run(_strategy("GTC"), _config(last_ns), store)

    assert res.trade_count == 1
    assert float(res.trades_df()["fill_price"].iloc[0]) == pytest.approx(98.0, abs=1e-9)
    assert res.order_activity == {
        "orders_posted": 1,
        "requotes": 0,
        "expired_unfilled": 0,
    }


# ---------------------------------------------------------------------------
# The level reads the position
# ---------------------------------------------------------------------------


def _maker(skew):
    """Half a unit per bar, quoted at 98.5 and stepping back by `skew` per unit
    already held."""
    return (
        bt.Strategy.create("maker")
        .signal("cote", lit(98.5) - lit(skew) * position())
        .size(lit(0.5))
        .limit_entry(signal="cote", time_in_force=Interval.seconds(60))
    )


def test_le_decalage_d_inventaire_deplace_la_cote(tmp_path):
    """With no skew the quote sits at 98.5 and both bars that trade down to it
    fill there. With a skew of one point per unit, the first fill takes the
    inventory to half a unit, so the next quote sits at 98.5 - 1.0 * 0.5 = 98.0
    -- and the second fill lands exactly there.

    The fill PRICE is the assertion, not a count: it is the level the engine
    computed, read back."""
    store, last_ns = _store(tmp_path)
    config = _config(last_ns, pyramiding=True)

    plat = bt.run(_maker(0.0), config, store)
    decale = bt.run(_maker(1.0), config, store)

    assert plat.trades_df()["fill_price"].tolist() == pytest.approx([98.5, 98.5])
    assert decale.trades_df()["fill_price"].tolist() == pytest.approx([98.5, 98.0])


def test_la_position_est_refusee_partout_ailleurs(tmp_path):
    store, last_ns = _store(tmp_path)

    taille = (
        bt.Strategy.create("interdit")
        .signal("cote", col("close"))
        .size(lit(0.1) * position())
        .limit_entry(signal="cote")
    )
    with pytest.raises(Exception, match="position\\(\\) is not a column"):
        bt.run(taille, _config(last_ns), store)

    fenetre = (
        bt.Strategy.create("interdit")
        .signal("cote", position().rolling_mean(5))
        .size(lit(1.0))
        .limit_entry(signal="cote")
    )
    with pytest.raises(Exception, match="position\\(\\) is not a column"):
        bt.run(fenetre, _config(last_ns), store)


def test_le_niveau_qui_lit_la_position_reste_serialisable():
    """`position()` is a bare name in the JSON, so no payload written before it
    existed changes meaning."""
    expr = (lit(98.5) - lit(1.0) * position()).to_json()
    assert expr["Sub"][1]["Mul"][1] == "Position"
