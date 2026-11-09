"""A market whose book cannot be read says so instead of running in silence.

- ``mbt.sim.run`` warns of a market that holds prints and no book state:
  ``sim.book()`` stays empty there, and without ``prints_update_book`` no
  order can fill;
- ``Market.from_arrays`` warns when quotes with a side missing or crossed
  leave book states empty, with their count and the first instant;
- a priced level of no size, missing side columns and a column given as
  ``None`` are refused with words that name the market and the column.
"""
import warnings

import numpy as np
import pytest

import manifoldbt as bt

S = 1_000_000_000
T0 = 1_700_000_000 * S


def _layer_open():
    try:
        bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    except PermissionError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _layer_open(),
    reason="order-level simulation locked (runs in the engine's own debug builds)",
)

TRADES = dict(
    ts_ns=np.array([1, 2, 3]) * S + T0,
    price=np.array([0.50, 0.51, 0.52]),
    qty=np.array([10.0, 10.0, 10.0]),
    aggressor=np.array([1, 1, -1]),
)


def _quotes(bid_px, ask_px=(0.51, 0.52, 0.53), bid_qty=(5.0, 5.0, 5.0)):
    return dict(
        ts_ns=np.array([1, 2, 3]) * S + T0,
        bid_px=np.array(bid_px, dtype=float),
        bid_qty=np.array(bid_qty, dtype=float),
        ask_px=np.array(ask_px, dtype=float),
        ask_qty=np.array([5.0, 5.0, 5.0]),
    )


def _buy_each_print(sim):
    while sim.wait(trade=True).kind != "end":
        sim.post("X", "buy", None, 1.0, tif="IOC")


def test_prints_without_a_book_warn_that_no_order_can_fill():
    m = bt.sim.Market.from_arrays("X", 0.01, trades=TRADES)
    with pytest.warns(UserWarning, match=r"'X' has prints and no book state.*no order can fill"):
        res = bt.sim.run(_buy_each_print, markets=[m])
    assert len(res.fills_df(backend="arrow")["price"]) == 0


def test_prints_updating_the_book_warn_that_the_book_stays_empty():
    m = bt.sim.Market.from_arrays("X", 0.01, trades=TRADES)
    cfg = bt.sim.Config(prints_update_book="both_sides")
    with pytest.warns(UserWarning, match=r"orders fill against the prints.*sim\.book\(\) stays empty"):
        res = bt.sim.run(_buy_each_print, cfg, markets=[m])
    assert len(res.fills_df(backend="arrow")["price"]) > 0


def test_a_market_with_a_book_runs_without_a_warning():
    m = bt.sim.Market.from_arrays("X", 0.01, quotes=_quotes([0.49, 0.50, 0.51]), trades=TRADES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bt.sim.run(_buy_each_print, markets=[m])


@pytest.mark.parametrize(
    "bid, ask",
    [
        ([np.nan, 0.50, 0.51], [0.51, 0.52, 0.53]),  # a side missing
        ([0.0, 0.50, 0.51], [0.51, 0.52, 0.53]),  # a best price of zero
        ([0.49, 0.50, 0.51], [np.nan, 0.52, 0.53]),  # the other side
        ([0.60, 0.50, 0.51], [0.51, 0.52, 0.53]),  # crossed
    ],
)
def test_quotes_that_empty_the_book_are_counted(bid, ask):
    with pytest.warns(UserWarning, match=rf"X: 1 of 3 book states show an empty book, the first at {T0 + S} ns"):
        m = bt.sim.Market.from_arrays("X", 0.01, quotes=_quotes(bid, ask))
    assert m.book_states == 3


def test_valid_quotes_build_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bt.sim.Market.from_arrays("X", 0.01, quotes=_quotes([0.49, 0.50, 0.51]))


def test_a_priced_level_of_no_size_is_refused_by_column():
    with pytest.raises(ValueError, match=r"X: a priced level needs a size above zero, row 0 has bid_qty 0 at bid_px 0.49"):
        bt.sim.Market.from_arrays("X", 0.01, quotes=_quotes([0.49, 0.50, 0.51], bid_qty=(0.0, 5.0, 5.0)))


def test_a_missing_side_says_both_sides_are_needed():
    q = _quotes([0.49, 0.50, 0.51])
    del q["bid_px"], q["bid_qty"]
    with pytest.raises(ValueError, match=r'no column "bid_px".*Quotes need both sides'):
        bt.sim.Market.from_arrays("X", 0.01, quotes=q)


def test_a_column_given_as_none_is_named():
    q = _quotes([0.49, 0.50, 0.51])
    q["bid_px"] = None
    with pytest.raises(TypeError, match=r"column 'bid_px' is None"):
        bt.sim.Market.from_arrays("X", 0.01, quotes=q)
