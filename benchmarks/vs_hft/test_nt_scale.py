"""The raw scale of the NautilusTrader adapter, on whichever build of Nautilus is installed.

    python -m pytest benchmarks/vs_hft/test_nt_scale.py -q

The standard wheel (Windows) keeps prices and quantities on 9 decimals and the
high-precision wheel (Linux) on 16. The adapter builds its data objects from raw integers,
so a scale written for one build is wrong on the other without any error: prices read as
zero and nothing trades. These tests read the objects back.
"""
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("nautilus_trader")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import engine_nt as N  # noqa: E402


def _data():
    return {
        "tr_ts": np.array([2_000], dtype=np.int64), "tr_px": np.array([120101.7]),
        "tr_qty": np.array([0.000123]), "tr_side": np.array([0], dtype=np.int8),
        "q_ts": np.array([1_000], dtype=np.int64), "q_bpx": np.array([1201016], dtype=np.int64),
        "q_apx": np.array([1201017], dtype=np.int64), "q_bqty": np.array([1.5]), "q_aqty": np.array([2.25]),
        "d25_ts": np.array([1_000], dtype=np.int64), "d25_px": np.array([1201016], dtype=np.int64),
        "d25_qty": np.array([0.75]), "d25_side": np.array([0], dtype=np.int8), "d25_last": np.array([1], dtype=np.int8),
    }


def _objects(tier):
    N._imports()
    _, iid, _ = N.instrument()
    (chunk,) = list(N.stream(_data(), tier, iid, 0, [0.0]))
    return chunk


def test_raw_units_follow_the_build():
    N._imports()
    from nautilus_trader.model.objects import FIXED_PRECISION
    assert N.PX_UNIT == 10 ** (FIXED_PRECISION - N.PX_PREC)
    assert N.SZ_UNIT == 10 ** (FIXED_PRECISION - N.SZ_PREC)
    N.check_raw_scale(N.PX_PREC, N.SZ_PREC)
    N.check_raw_scale(4, 0)


def test_best_bid_ask_ticks_and_trades_read_back():
    quote, trade = _objects("bbo")
    assert quote.bid_price.as_double() == pytest.approx(120101.6, rel=1e-12)
    assert quote.ask_price.as_double() == pytest.approx(120101.7, rel=1e-12)
    assert quote.bid_size.as_double() == pytest.approx(1.5, rel=1e-12)
    assert quote.ask_size.as_double() == pytest.approx(2.25, rel=1e-12)
    assert trade.price.as_double() == pytest.approx(120101.7, rel=1e-12)
    assert trade.size.as_double() == pytest.approx(0.000123, rel=1e-12)


def test_book_deltas_read_back():
    delta, trade = _objects("l2")
    assert delta.order.price.as_double() == pytest.approx(120101.6, rel=1e-12)
    assert delta.order.size.as_double() == pytest.approx(0.75, rel=1e-12)
    assert trade.price.as_double() == pytest.approx(120101.7, rel=1e-12)
