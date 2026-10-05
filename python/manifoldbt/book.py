"""The stored order book, as columns a strategy reads.

A book enters the store through :func:`manifoldbt.ingest_book`. Once it is
there, a strategy reads it like any bar column, by name::

    import manifoldbt as bt

    imb = bt.book.imbalance(5)            # col("book_imbalance_5")
    strat = (
        bt.Strategy.create("imbalance")
        .signal("long", bt.when(imb > 0.3, 1.0, 0.0))
        .size(bt.col("long"))
    )

Every function of the table below returns ``col(<name>)``, so
``bt.col("bid_size_3")`` is exactly ``bt.book.bid_size(3)``. The names:

====================  ====================================================
``bid_price_<k>``     price of bid level ``k``, 1 = best
``ask_price_<k>``     price of ask level ``k``
``bid_size_<k>``      size resting at bid level ``k``
``ask_size_<k>``      size resting at ask level ``k``
``bid_depth_<k>``     size resting over bid levels 1 to ``k``
``ask_depth_<k>``     size resting over ask levels 1 to ``k``
``book_imbalance_<k>``  ``(bid_depth_k - ask_depth_k) / (bid_depth_k + ask_depth_k)``
====================  ====================================================

The engine loads the book only when a strategy names one of these, and only as
deep as the deepest ``k`` it names. It refuses by name a run whose store holds
no book for a symbol, or holds one shallower than the level asked.

A run reads every stored level unless ``BacktestConfig.book_levels`` bounds
it, and so do the queue, the fill marks and the book of a quote. Under a bound,
a column deeper than it (``bt.book.bid_price(30)`` with ``book_levels=25``) is
refused before anything is read, with the setting to raise.

When each row reads the book:

* on bars, at the instant the bar closes: the ladder standing one nanosecond
  before the next bar opens. The book and the ``close`` describe the same
  instant, and a row never reads a book its own close could not have seen.
  Resampled bars (``bar_interval`` coarser than the stored bars) read the
  ladder at the close of their last stored bar, like their ``close``;
* on the trade clock (``bar_interval=Interval.trades()``), at the print itself,
  with a book update and a trade sharing a millisecond ordered book first;
* in a strategy that quotes (``Strategy.quote``), at each wake-up, as the
  strategy sees the book: the stored state as of ``execution.latency["feed"]``
  before the wake-up.

A value is null (NaN in a DataFrame) where the book says nothing: before its
first update, on a day the store holds no book for, at a level that does not
exist at that instant (a side thinner than ``k``), and for the imbalance of an
empty book. ``bid_depth_k`` and ``ask_depth_k`` are zero on an empty side.
A null flows through the expression like a NaN close: ``when()`` on a null
condition is null, and a null at the top of the sizing holds the position.

The book at a wake-up
---------------------

:func:`bid_price_at`, :func:`ask_price_at`, :func:`bid_levels` and
:func:`ask_levels` are not columns. They read the book the simulation holds at
the instant the engine wakes up, at a level the expression itself may compute
(``bt.book.bid_price_at(bt.round(bt.position() / 2) + 1)``), and exist only in
the expressions of a quote, which the engine evaluates at each wake-up. A
signal or a size refuses them by name, as it refuses ``position()``.
"""
from __future__ import annotations

from typing import Any

from manifoldbt.expr import Expr, _book_price_at, col

__all__ = [
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "bid_depth",
    "ask_depth",
    "imbalance",
    "bid_price_at",
    "ask_price_at",
    "bid_levels",
    "ask_levels",
]


def _level(level: int) -> int:
    # bool is an int subclass: bid_size(True) would name level 1 by accident.
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f"level must be an int, got {type(level).__name__}")
    if level < 1:
        raise ValueError(f"level must be 1 or more (1 = best), got {level}")
    return level


def bid_price(level: int = 1) -> Expr:
    """Price of bid level ``level``, 1 = the best bid."""
    return col(f"bid_price_{_level(level)}")


def ask_price(level: int = 1) -> Expr:
    """Price of ask level ``level``, 1 = the best ask."""
    return col(f"ask_price_{_level(level)}")


def bid_size(level: int = 1) -> Expr:
    """Size resting at bid level ``level``."""
    return col(f"bid_size_{_level(level)}")


def ask_size(level: int = 1) -> Expr:
    """Size resting at ask level ``level``."""
    return col(f"ask_size_{_level(level)}")


def bid_depth(levels: int) -> Expr:
    """Size resting over the best ``levels`` bid levels."""
    return col(f"bid_depth_{_level(levels)}")


def ask_depth(levels: int) -> Expr:
    """Size resting over the best ``levels`` ask levels."""
    return col(f"ask_depth_{_level(levels)}")


def imbalance(levels: int = 1) -> Expr:
    """Signed imbalance over the best ``levels`` levels, in [-1, 1].

    ``(bid_depth - ask_depth) / (bid_depth + ask_depth)``: +1 when only bids
    rest, -1 when only asks do, null when the book is empty.
    """
    return col(f"book_imbalance_{_level(levels)}")


def bid_price_at(level: Any) -> Expr:
    """Price of bid level ``level`` at the wake-up, 1 = the best bid.

    ``level`` is an int or an expression, which may read the state. The value
    is NaN where the level does not exist at that instant or lies past the
    depth the run reads (``BacktestConfig.book_levels``, when set), and where
    ``level`` is not a whole number of at least 1 (NaN included); null where
    ``level`` is null. Readable only in the expressions of a quote: see the module notes.
    """
    return _book_price_at("bid", level)


def ask_price_at(level: Any) -> Expr:
    """Price of ask level ``level`` at the wake-up, 1 = the best ask. Same
    rules as :func:`bid_price_at`."""
    return _book_price_at("ask", level)


def bid_levels() -> Expr:
    """How many bid levels the book holds at the wake-up, as the strategy
    sees it (``execution.latency["feed"]`` earlier): the ladder counted from
    the best bid down to its first empty level, ``0`` when the side is empty,
    at most ``BacktestConfig.book_levels`` when it bounds the depth read.
    ``bid_price_at(bid_levels())`` is the deepest bid read. Readable only in
    the expressions of a quote."""
    return Expr("SimState", "BookLevels", "Bid")


def ask_levels() -> Expr:
    """How many ask levels the book holds at the wake-up. Same rules as
    :func:`bid_levels`."""
    return Expr("SimState", "BookLevels", "Ask")
