"""Reading the simulation state at a wake-up, and ``round`` / ``clip``.

What is checked here is the Python surface: the JSON each builder writes is the
one the engine reads (``compile_strategy_json`` deserialises it), every state
reader is refused BY NAME where the evaluation is vectorised (a signal, a size),
and ``round`` / ``clip`` compile and answer as the builtins do on plain numbers.

The wake-up program itself (the batch/residual cut, the null rules, the value to
the bit against the batch evaluator) is tested by the engine's own tests.
"""
import json
import math

import pytest

import manifoldbt as bt
from manifoldbt.expr import col, lit


def _compile(strategy):
    return json.loads(bt.compile_strategy_json(strategy.to_json()))


# ---------------------------------------------------------------------------
# JSON shapes
# ---------------------------------------------------------------------------


def test_les_lectures_d_etat_s_ecrivent_comme_le_moteur_les_lit():
    assert bt.live_qty("bid").to_json() == {"SimState": {"LiveQty": "Bid"}}
    assert bt.order_age("ask").to_json() == {"SimState": {"OrderAge": "Ask"}}
    assert bt.last_fill_px("bid").to_json() == {"SimState": {"LastFillPx": "Bid"}}
    assert bt.queue_ahead("ASK").to_json() == {"SimState": {"QueueAhead": "Ask"}}
    assert bt.cash().to_json() == {"SimState": "Cash"}
    assert bt.order_price("bid").to_json() == {"SimState": {"OrderPrice": "Bid"}}
    assert bt.position_age().to_json() == {"SimState": "PositionAge"}
    assert bt.last_fill_age("ask").to_json() == {"SimState": {"LastFillAge": "Ask"}}
    assert bt.last_cancel_age("bid").to_json() == {"SimState": {"LastCancelAge": "Bid"}}
    # Sans cote : le plus recent des deux, min qui ignore le cote jamais servi.
    assert bt.last_fill_age().to_json() == {
        "Function": [
            "min",
            [{"SimState": {"LastFillAge": "Bid"}}, {"SimState": {"LastFillAge": "Ask"}}],
        ]
    }
    assert bt.book.bid_levels().to_json() == {"SimState": {"BookLevels": "Bid"}}
    assert bt.book.ask_levels().to_json() == {"SimState": {"BookLevels": "Ask"}}
    assert bt.book.bid_price_at(2).to_json() == {
        "SimState": {"BookPriceAt": {"side": "Bid", "level": {"Literal": {"Float64": 2.0}}}}
    }
    # Le niveau est une expression, qui peut lire l'etat.
    niveau = bt.book.ask_price_at(bt.round(bt.position() / 2.0) + 1)
    assert niveau.to_json()["SimState"]["BookPriceAt"]["level"] == {
        "Add": [
            {"Function": ["round", [{"Div": ["Position", {"Literal": {"Float64": 2.0}}]}]]},
            {"Literal": {"Float64": 1.0}},
        ]
    }


def test_round_et_clip_s_ecrivent_en_fonctions():
    assert bt.round(col("x")).to_json() == {"Function": ["round", [{"Column": "x"}]]}
    assert bt.clip(col("x"), -1, 1).to_json() == {
        "Function": [
            "clip",
            [{"Column": "x"}, {"Literal": {"Float64": -1.0}}, {"Literal": {"Float64": 1.0}}],
        ]
    }


def test_un_cote_ou_un_niveau_invalide_est_refuse_en_python():
    with pytest.raises(ValueError, match="'bid' or 'ask'"):
        bt.live_qty("buy")
    for lecture in (bt.order_price, bt.last_fill_age, bt.last_cancel_age):
        with pytest.raises(ValueError, match="'bid' or 'ask'"):
            lecture("sell")
    with pytest.raises(ValueError, match="1 or more"):
        bt.book.bid_price_at(0)
    with pytest.raises(TypeError, match="bool"):
        bt.book.ask_price_at(True)
    with pytest.raises(TypeError, match="round_to"):
        bt.round(col("x"), 2)


# ---------------------------------------------------------------------------
# round / clip on plain numbers: the builtins
# ---------------------------------------------------------------------------


def test_sur_des_nombres_round_est_le_builtin():
    assert bt.round(2.5) == 2 and bt.round(3.5) == 4 and bt.round(-2.5) == -2
    assert bt.round(1.23456, 2) == 1.23
    assert isinstance(bt.round(2.4), int)


def test_sur_des_nombres_clip_garde_le_nan():
    assert bt.clip(5, 0, 1) == 1.0
    assert bt.clip(-5, 0, 1) == 0.0
    assert bt.clip(0.25, 0, 1) == 0.25
    assert bt.clip(0.5, 2, 1) == 1.0  # lo > hi -> hi, comme numpy.clip
    assert math.isnan(bt.clip(float("nan"), 0, 1))
    assert math.isnan(bt.clip(0.5, float("nan"), 1))


# ---------------------------------------------------------------------------
# The engine: round/clip compile, the state is refused by name in batch
# ---------------------------------------------------------------------------


def test_round_et_clip_compilent_dans_un_signal():
    strategy = (
        bt.Strategy.create("arrondi")
        .signal("n", bt.clip(bt.round(col("close") / 10.0), -3, 3))
        .size(lit(0.0))
    )
    summary = _compile(strategy)
    assert "n" in summary["signal_names"]


@pytest.mark.parametrize(
    "lecture, nom",
    [
        (lambda: bt.live_qty("bid"), 'live_qty("bid")'),
        (lambda: bt.order_age("ask"), 'order_age("ask")'),
        (lambda: bt.last_fill_px("bid"), 'last_fill_px("bid")'),
        (lambda: bt.queue_ahead("ask"), 'queue_ahead("ask")'),
        (lambda: bt.cash(), "cash()"),
        (lambda: bt.order_price("bid"), 'order_price("bid")'),
        (lambda: bt.position_age(), "position_age()"),
        (lambda: bt.last_fill_age("ask"), 'last_fill_age("ask")'),
        (lambda: bt.last_fill_age(), 'last_fill_age("bid")'),
        (lambda: bt.last_cancel_age("bid"), 'last_cancel_age("bid")'),
        (lambda: bt.book.bid_price_at(1), "book.bid_price_at(...)"),
        (lambda: bt.book.ask_levels(), "book.ask_levels()"),
    ],
)
def test_chaque_lecture_d_etat_est_refusee_par_son_nom_dans_le_lot(lecture, nom):
    dans_un_signal = (
        bt.Strategy.create("interdit")
        .signal("x", col("close") - lecture())
        .size(lit(0.0))
    )
    with pytest.raises(Exception) as err:
        _compile(dans_un_signal)
    message = str(err.value)
    assert "simulation state is not a column" in message
    assert nom in message

    # Enfouie sous une fenetre : refusee de meme, et le message nomme le signal.
    sous_une_fenetre = (
        bt.Strategy.create("interdit")
        .signal("lisse", (col("close") + lecture()).rolling_mean(5))
        .size(lit(0.0))
    )
    with pytest.raises(Exception, match="simulation state is not a column") as err:
        _compile(sous_une_fenetre)
    assert "lisse" in str(err.value)
