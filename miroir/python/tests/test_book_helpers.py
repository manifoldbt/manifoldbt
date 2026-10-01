"""``bt.book``: the names a strategy reads the stored order book under.

Each helper is ``col(<name>)`` and nothing more, so what is pinned here is the
name, which the engine parses, and the refusal of a level that cannot exist.
Reading a real book is covered where a book can be stored.
"""
import pytest

import manifoldbt as bt


def _column(expr):
    doc = expr.to_json()
    assert list(doc) == ["Column"], doc
    return doc["Column"]


@pytest.mark.parametrize(
    "helper, level, name",
    [
        (bt.book.bid_price, 1, "bid_price_1"),
        (bt.book.ask_price, 2, "ask_price_2"),
        (bt.book.bid_size, 3, "bid_size_3"),
        (bt.book.ask_size, 10, "ask_size_10"),
        (bt.book.bid_depth, 5, "bid_depth_5"),
        (bt.book.ask_depth, 20, "ask_depth_20"),
        (bt.book.imbalance, 5, "book_imbalance_5"),
    ],
)
def test_each_helper_names_one_column(helper, level, name):
    assert _column(helper(level)) == name


def test_the_best_level_is_the_default_where_one_makes_sense():
    assert _column(bt.book.bid_price()) == "bid_price_1"
    assert _column(bt.book.ask_size()) == "ask_size_1"
    assert _column(bt.book.imbalance()) == "book_imbalance_1"


def test_a_helper_is_the_column_it_names():
    assert bt.book.imbalance(5).to_json() == bt.col("book_imbalance_5").to_json()


@pytest.mark.parametrize("bad", [0, -1])
def test_a_level_below_one_is_refused(bad):
    with pytest.raises(ValueError, match="1 = best"):
        bt.book.bid_size(bad)


@pytest.mark.parametrize("bad", [1.0, "1", True, None])
def test_a_level_that_is_not_an_int_is_refused(bad):
    with pytest.raises(TypeError, match="level must be an int"):
        bt.book.imbalance(bad)


def test_the_module_is_exported():
    assert "book" in bt.__all__
