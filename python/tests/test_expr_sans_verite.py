"""Une expression n'a pas de valeur de verite.

Avant : ``bool(expr)`` valait ``True`` (tout objet Python est vrai par
defaut), donc ``max(expr, lit(0.1))`` rendait un des deux operandes selon leur
ORDRE, sans erreur. Un harnais de market making a ainsi envoye 101 570 ordres
au lieu de 49 178. ``if``, ``and``, ``or``, ``not``, ``a < b < c``, ``in`` et
``expr != x`` (qui rendait ``False``) avaient le meme defaut.
"""
import inspect
import json

import pytest

import manifoldbt as bt
from manifoldbt import indicators as ind
from manifoldbt.expr import Expr, col, lit, when


X = col("x")
Y = col("y")


def _j(expr):
    return json.dumps(expr.to_json(), sort_keys=True)


# ---------------------------------------------------------------------------
# Valeur de verite
# ---------------------------------------------------------------------------


def _max_py(a, b):
    return max(a, b)


def _min_py(a, b):
    return min(a, b)


def _if(a, b):
    if a > b:
        return 1
    return 0


def _and(a, b):
    return (a > 0) and (b > 0)


def _or(a, b):
    return (a > 0) or (b > 0)


def _not(a, b):
    return not (a > b)


def _chained(a, b):
    return lit(0.0) < a < b


def _in(a, b):
    return b in [a]


def _assert(a, b):
    assert a > b


def _ternary(a, b):
    return a if a > b else b


@pytest.mark.parametrize(
    "misuse",
    [_max_py, _min_py, _if, _and, _or, _not, _chained, _in, _assert, _ternary],
    ids=["max", "min", "if", "and", "or", "not", "a<b<c", "in", "assert", "ternaire"],
)
@pytest.mark.parametrize("order", ["expr_d_abord", "litteral_d_abord"])
def test_une_expression_refuse_d_etre_lue_comme_un_booleen(misuse, order):
    a, b = (X, lit(0.1)) if order == "expr_d_abord" else (lit(0.1), X)
    with pytest.raises(TypeError, match="no truth value"):
        misuse(a, b)


def test_max_de_python_ne_rend_plus_un_operande_au_hasard():
    # Le cas du harnais, dans les deux ordres : l'un rendait le litteral,
    # l'autre l'expression, et les deux passaient.
    with pytest.raises(TypeError):
        max(X * lit(2.0), lit(0.1))
    with pytest.raises(TypeError):
        max(lit(0.1), X * lit(2.0))
    with pytest.raises(TypeError):
        max(X, 0.1)


def test_le_message_dit_pourquoi_et_quoi_ecrire():
    with pytest.raises(TypeError) as err:
        bool(X > 0)
    msg = str(err.value)
    # pourquoi
    assert "evaluates bar by bar" in msg
    assert "silently pick one operand" in msg
    # quoi ecrire, avec les noms reels du DSL
    for words in ("max_val(a, b)", "min_val(a, b)", "manifoldbt.indicators",
                  "(a > 0) & (b < 1)", "~", "(a < b) & (b < c)",
                  "when(cond, x, y)", "is None"):
        assert words in msg, words
    # et ces noms existent bien
    assert callable(ind.max_val) and callable(ind.min_val) and callable(bt.when)


def test_les_equivalents_proposes_construisent_la_bonne_expression():
    assert _j(ind.max_val(X, lit(0.1))) == _j(Expr("Function", "max", [X, lit(0.1)]))
    assert _j((X > 0) & (Y < 1)) == _j(Expr("And", X > 0, Y < 1))
    assert _j(~(X > 0)) == _j(Expr("Not", X > 0))
    assert _j(when(X > Y, X, Y)) == _j(Expr("IfElse", X > Y, X, Y))


def test_un_multiexpr_reste_un_tuple():
    # Un indicateur a plusieurs sorties est un tuple : sa verite est celle
    # d'un tuple non vide, et le deballage marche comme avant.
    bands = ind.bollinger_bands(X, 20)
    assert bool(bands)
    upper, middle, lower = bands
    assert isinstance(upper, Expr)


def test_l_egalite_de_deux_multiexpr_compare_les_arbres():
    # tuple.__eq__ comparait les membres avec `==`, donc lisait bool(Eq(...)).
    assert ind.bollinger_bands(X, 20) == ind.bollinger_bands(X, 20)
    assert ind.bollinger_bands(X, 20) != ind.bollinger_bands(X, 21)
    assert ind.bollinger_bands(X, 20) != ("upper",)
    bands = ind.bollinger_bands(X, 20)
    assert bands == tuple(bands)
    assert bands in [ind.bollinger_bands(X, 20)]


def test_is_none_et_isinstance_restent_permis():
    e = X > 0
    assert e is not None
    assert isinstance(e, Expr)
    assert [e][0] is e


# ---------------------------------------------------------------------------
# != construit la comparaison au lieu de rendre False
# ---------------------------------------------------------------------------


def test_different_de_construit_non_egal():
    ne = X != 0
    assert isinstance(ne, Expr)
    assert _j(ne) == _j(~(X == 0))
    assert _j(0 != X) == _j(~(X == 0))
    # Utilisable comme condition, la ou il donnait un litteral False.
    assert _j(when(X != Y, 1.0, 0.0)) == _j(Expr("IfElse", ~(X == Y), lit(1.0), lit(0.0)))


# ---------------------------------------------------------------------------
# Colonnes personnalisees des indicateurs : `h or high` lisait bool(h)
# ---------------------------------------------------------------------------


def _with_custom_columns():
    for name, f in sorted(vars(ind).items()):
        if not inspect.isfunction(f) or f.__module__ != ind.__name__ or name.startswith("_"):
            continue
        kws = [p for p in inspect.signature(f).parameters.values()
               if p.kind == p.KEYWORD_ONLY and p.name in ("o", "h", "l", "c", "v")]
        if kws:
            yield name, f, [p.name for p in kws]


@pytest.mark.parametrize("name,f,kws", list(_with_custom_columns()),
                         ids=[n for n, _, _ in _with_custom_columns()])
def test_un_indicateur_lit_les_colonnes_passees(name, f, kws):
    out = f(**{k: col(f"mine_{k}") for k in kws})
    parts = out if isinstance(out, tuple) else (out,)
    text = " ".join(_j(p) for p in parts)
    for k in kws:
        assert f'"mine_{k}"' in text, (name, k)
    native = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    for k in kws:
        assert f'{{"Column": "{native[k]}"}}' not in text, (name, k)
    # Et sans argument, les colonnes natives.
    default = f()
    parts = default if isinstance(default, tuple) else (default,)
    text = " ".join(_j(p) for p in parts)
    for k in kws:
        assert f'{{"Column": "{native[k]}"}}' in text, (name, k)
