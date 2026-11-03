"""``bt.sim.Market.from_arrays``: a market from columns.

The same quotes and prints written out by hand (``from_ladders``) and handed
over as columns (pandas, polars, pyarrow, numpy) build the same market, and a
run on one is the run on the other. The engine's own tests pin the rules for
quotes and prints at one instant, invalid quotes and the tick rule; this file
checks that Python reaches them, and the refusals, by name.
"""
import math

import numpy as np
import pyarrow as pa
import pytest

import manifoldbt as bt

MS = 1_000_000


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

# Quotes at 1, 2, 2 (the same instant: the last stands), 4 and 6 ms; prints at
# 3, 3 (the same instant), 5 and 6 ms (with the quote of 6 ms).
QUOTES = {
    "ts_ns": np.array([1, 2, 2, 4, 6], dtype=np.int64) * MS,
    "bid_px": np.array([100.0, 99.9, 100.0, 99.9, 100.0]),
    "bid_qty": np.array([2.0, 8.0, 3.0, 4.0, 1.5]),
    "ask_px": np.array([100.1, 100.1, 100.1, 100.0, 100.2]),
    "ask_qty": np.array([5.0, 5.0, 5.0, 2.0, 7.0]),
}
TRADES = {
    "ts_ns": np.array([3, 3, 5, 6], dtype=np.int64) * MS,
    "price": np.array([100.0, 100.0, 99.9, 100.2]),
    "qty": np.array([1.0, 2.5, 1.0, 0.5]),
    "aggressor": np.array(["sell", "sell", "sell", "buy"], dtype=object),
}


def _ladders():
    q, t = QUOTES, TRADES
    states = [
        (int(q["ts_ns"][i]), [(q["bid_px"][i], q["bid_qty"][i])], [(q["ask_px"][i], q["ask_qty"][i])])
        for i in range(len(q["ts_ns"]))
    ]
    trades = [
        (int(t["ts_ns"][i]), t["price"][i], t["qty"][i], t["aggressor"][i])
        for i in range(len(t["ts_ns"]))
    ]
    return bt.sim.Market.from_ladders("X", 0.1, states, trades)


def _run(market):
    """A buy of 1 at 100.0 posted at 1.5 ms, the run to 10 ms."""

    def strategy(sim):
        sim.elapse(MS + MS // 2 - sim.now())
        sim.post("X", "buy", 100.0, 1.0)
        sim.elapse(10 * MS - sim.now())

    res = bt.sim.run(strategy, markets=[market])
    return (
        res.fills_df(backend="pandas").to_dict("list"),
        res.events_df(backend="pandas").astype(str).to_dict("list"),
        res.equity_df(backend="pandas").to_dict("list"),
        res.position,
        res.cash,
    )


def _shape(m):
    return (m.name, m.tick_size, m.book_states, m.trades, m.span)


def _pandas():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(QUOTES), pd.DataFrame(TRADES)


def _polars():
    pl = pytest.importorskip("polars")
    return pl.DataFrame(QUOTES), pl.DataFrame({**TRADES, "aggressor": list(TRADES["aggressor"])})


def _pyarrow():
    t = dict(TRADES, aggressor=pa.array(list(TRADES["aggressor"])))
    return pa.table(QUOTES), pa.record_batch(t)


def _numpy():
    return dict(QUOTES), dict(TRADES)


@pytest.mark.parametrize("source", [_numpy, _pandas, _polars, _pyarrow])
def test_columns_build_the_market_and_the_run_of_the_ladders(source):
    quotes, trades = source()
    by_hand = _ladders()
    columns = bt.sim.Market.from_arrays("X", 0.1, quotes=quotes, trades=trades)
    assert _shape(columns) == _shape(by_hand) == ("X", 0.1, 4, 4, (1 * MS, 6 * MS))
    fills, *_ = ran = _run(columns)
    assert fills["qty"], "the scenario fills the resting buy"
    assert ran == _run(by_hand)


def test_timestamps_of_another_unit_and_an_index_are_read_in_nanoseconds():
    pd = pytest.importorskip("pandas")
    stamps = pd.to_datetime(QUOTES["ts_ns"], utc=True)
    q = pd.DataFrame({k: v for k, v in QUOTES.items() if k != "ts_ns"}, index=stamps)
    q.index.name = "timestamp"
    m = bt.sim.Market.from_arrays("X", 0.1, quotes=q)
    assert m.span == (1 * MS, 6 * MS)
    us = pa.array(QUOTES["ts_ns"] // 1_000, pa.timestamp("us"))
    m = bt.sim.Market.from_arrays("X", 0.1, quotes=dict(QUOTES, ts_ns=us))
    assert (m.book_states, m.span) == (4, (1 * MS, 6 * MS))


def test_a_tape_without_aggressor_takes_the_tick_rule():
    tape = {k: v for k, v in TRADES.items() if k != "aggressor"}
    no_side = bt.sim.Market.from_arrays("X", 0.1, quotes=QUOTES, trades=tape)
    # 100.0, 100.0, 99.9, 100.2 by the tick rule: buy, buy, sell, buy.
    sides = np.array([1, 1, -1, 1], dtype=np.int8)
    told = bt.sim.Market.from_arrays("X", 0.1, quotes=QUOTES, trades=dict(tape, aggressor=sides))
    nan = np.array([math.nan, math.nan, -1.0, math.nan])
    floats = bt.sim.Market.from_arrays("X", 0.1, quotes=QUOTES, trades=dict(tape, aggressor=nan))
    assert _run(floats) == _run(told)
    assert _run(no_side) == _run(told)
    # Unknown rows (null, NaN) take the rule too; 2 is a sell.
    some = pa.array([None, None, 2, 1], pa.int8())
    partly = bt.sim.Market.from_arrays("X", 0.1, quotes=QUOTES, trades=dict(tape, aggressor=some))
    assert _run(partly) == _run(told)


def test_crossed_or_one_sided_quotes_show_an_empty_book():
    q = {
        "ts_ns": np.array([1, 2, 3], dtype=np.int64) * MS,
        "bid_px": np.array([100.0, 100.2, math.nan]),
        "bid_qty": np.ones(3),
        "ask_px": np.array([100.1, 100.1, 100.1]),
        "ask_qty": np.ones(3),
    }
    seen = []

    def strategy(sim):
        for t in (1, 2, 3):
            sim.elapse(t * MS - sim.now())
            seen.append(sim.book("X").best_bid)

    bt.sim.run(strategy, markets=[bt.sim.Market.from_arrays("X", 0.1, quotes=q)])
    assert seen == [100.0, None, None]


def test_empty_and_absent_tables_make_an_empty_market():
    empty = {k: v[:0] for k, v in QUOTES.items()}
    for m in (
        bt.sim.Market.from_arrays("X", 0.1),
        bt.sim.Market.from_arrays("X", 0.1, quotes=empty, trades={k: v[:0] for k, v in TRADES.items()}),
    ):
        assert (m.book_states, m.trades, m.span) == (0, 0, None)
    prints_only = bt.sim.Market.from_arrays("X", 0.1, trades=TRADES)
    assert (prints_only.book_states, prints_only.trades) == (0, 4)


@pytest.mark.parametrize(
    "quotes, trades, error, match",
    [
        (dict(QUOTES, ts_ns=QUOTES["ts_ns"][::-1].copy()), None, ValueError, "quotes must come in time order"),
        (dict(QUOTES, ts_ns=QUOTES["ts_ns"].astype(float)), None, ValueError, "ts_ns must be integer nanoseconds"),
        (dict(QUOTES, bid_px=QUOTES["bid_px"].astype(str)), None, ValueError, "bid_px must be numbers"),
        ({k: v for k, v in QUOTES.items() if k != "ask_qty"}, None, ValueError, 'no column "ask_qty"'),
        (dict(QUOTES, ask_qty=QUOTES["ask_qty"][:2]), None, ValueError, "columns differ in length"),
        (None, dict(TRADES, aggressor=np.array([True] * 4)), ValueError, "a boolean does not say"),
        (None, dict(TRADES, aggressor=np.array(["sell", "s", "buy", "buy"])), ValueError, 'row 1 is "s"'),
        (None, dict(TRADES, aggressor=np.array([1, 0, 1, 1], dtype=np.uint8)), ValueError, "row 1 is 0"),
        (None, dict(TRADES, qty=np.array([1.0, 0.0, 1.0, 1.0])), ValueError, "qty must be finite and positive"),
        ([1, 2, 3], None, TypeError, "quotes takes a pandas or polars DataFrame"),
    ],
)
def test_what_it_cannot_read_is_refused_by_name(quotes, trades, error, match):
    with pytest.raises(error, match=match):
        bt.sim.Market.from_arrays("X", 0.1, quotes=quotes, trades=trades)
