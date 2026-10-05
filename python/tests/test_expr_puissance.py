"""``expr ** n`` : le moteur n'a pas de fonction puissance, donc un exposant
entier litteral se deplie en produits, les memes noeuds que la forme ecrite a
la main ; le reste est refuse en disant quoi ecrire.

Qu'un ``**`` tourne dans une cote (programme de reveil) comme le produit ecrit
a la main est verifie dans ``test_quote_dsl.py``.
"""
import json

import pytest

import manifoldbt as bt
from manifoldbt.expr import col, lit, param


X = col("x")
Y = col("y")


def _j(expr):
    return json.dumps(expr.to_json(), sort_keys=True)


# ---------------------------------------------------------------------------
# Puissance
# ---------------------------------------------------------------------------


def test_puissance_deux_est_le_produit_ecrit_a_la_main():
    assert _j(X ** 2) == _j(X * X)
    assert _j(X ** 2.0) == _j(X * X)
    assert _j((X - Y) ** 2) == _j((X - Y) * (X - Y))


def test_puissance_entiere_se_deplie_de_gauche_a_droite():
    assert _j(X ** 3) == _j(X * X * X)
    assert _j(X ** 4) == _j(X * X * X * X)
    assert X ** 1 is X


def test_puissance_negative_est_un_sur_le_produit():
    assert _j(X ** -1) == _j(lit(1.0) / X)
    assert _j(X ** -2) == _j(lit(1.0) / (X * X))


def test_puissance_avec_param_dans_la_base():
    k = param("k", default=2.0)
    s = bt.Strategy.create("p").signal("v", (X * k) ** 2).size(lit(0.0))
    assert "k" in s.to_json_dict()["parameters"]


@pytest.mark.parametrize(
    "exposant,erreur,mots",
    [
        (0, ValueError, "constant 1"),
        (9, ValueError, "+/-8"),
        (-9, ValueError, "+/-8"),
        (0.5, TypeError, "sqrt(x)"),
        (1.5, TypeError, "exp(p * log(x))"),
        (float("nan"), TypeError, "literal integer"),
        (True, TypeError, "literal integer"),
        ("2", TypeError, "literal integer"),
    ],
)
def test_un_exposant_non_supporte_est_refuse_en_disant_quoi_faire(exposant, erreur, mots):
    with pytest.raises(erreur) as err:
        X ** exposant
    assert mots in str(err.value)


def test_un_exposant_expression_est_refuse():
    with pytest.raises(TypeError, match="no power function"):
        X ** Y
    with pytest.raises(TypeError, match="no power function"):
        X ** param("n", default=2)


def test_une_base_litterale_a_la_puissance_d_une_expression_est_refusee():
    with pytest.raises(TypeError, match=r"2 \*\* expr is not supported"):
        2 ** X


def test_pow_modulaire_refuse():
    with pytest.raises(TypeError, match="mod"):
        pow(X, 2, 3)
