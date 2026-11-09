"""``mbt.sim.Market.from_store(..., tape_only=True)``: a market read from the
stored tape alone, for a venue whose book history is not published.

Without it, a symbol with prints and no book is refused by name, as before.
With it, the market holds the prints and no book: ``sim.book()`` stays empty
and orders execute against the prints under ``prints_update_book``. It takes
neither ``levels``, ``by_day`` nor ``device``.

No network: a Bybit archive written by hand in the cache stands in for a day,
as in ``test_ingest_trades.py``.
"""
import gzip

import pytest

pytest.importorskip("pyarrow")

import manifoldbt as bt  # noqa: E402

DAY = "2023-03-01"
T0 = 1677628800  # 2023-03-01T00:00:00Z
S = 10**9
SYMBOL = "TESTUSDT"
HEADER = ("timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
          "grossValue,homeNotional,foreignNotional")


def _open():
    try:
        bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
        bt.ticks.tape_info(__file__)
    except PermissionError:
        return False
    except Exception:
        return True
    return True


pytestmark = pytest.mark.skipif(
    not _open(), reason="tick layer locked (runs in the engine's own debug builds)"
)


def _line(ts, side, size, price):
    return (f"{ts},{SYMBOL},{side},{size},{price},PlusTick,"
            f"00000000-0000-0000-0000-000000000000,1,{size},{size * price!r}")


@pytest.fixture
def store(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    lines = [_line(f"{T0 + 1}", "Buy", 10, 0.50),
             _line(f"{T0 + 2}", "Buy", 10, 0.51),
             _line(f"{T0 + 3}", "Sell", 10, 0.52)]
    (cache / f"{SYMBOL}{DAY}.csv.gz").write_bytes(
        gzip.compress(("\n".join([HEADER, *lines]) + "\n").encode()))
    return bt.ingest_trades(
        "bybit", SYMBOL, symbol_id=7, start=DAY, end=DAY,
        data_root=str(tmp_path / "data"), metadata_db=str(tmp_path / "meta.sqlite"),
        category="linear", cache_dir=str(cache), progress=False,
    )


def _range():
    return T0 * S, (T0 + 86_400) * S


def test_without_tape_only_a_symbol_without_a_book_is_refused(store):
    with pytest.raises(ValueError, match="no stored book"):
        bt.sim.Market.from_store(store, SYMBOL, *_range())


def test_tape_only_reads_the_prints_and_no_book(store):
    m = bt.sim.Market.from_store(store, SYMBOL, *_range(), tape_only=True)
    assert (m.book_states, m.trades) == (0, 3)
    assert m.span == ((T0 + 1) * S, (T0 + 3) * S)


def test_orders_execute_against_the_prints(store):
    m = bt.sim.Market.from_store(store, SYMBOL, *_range(), tape_only=True)
    seen = []

    def strategy(sim):
        while sim.wait(trade=True).kind != "end":
            seen.append(sim.book(SYMBOL).best_ask)
            sim.post(SYMBOL, "buy", None, 1.0, tif="IOC")

    cfg = bt.sim.Config(prints_update_book="both_sides")
    res = bt.sim.run(strategy, cfg, markets=[m])
    assert seen and all(a is None for a in seen)
    assert res.fills_df(backend="arrow")["price"] == [0.51, 0.52]


@pytest.mark.parametrize("extra", [{"levels": 5}, {"by_day": True}, {"device": "cuda"}])
def test_tape_only_takes_no_book_setting(store, extra):
    with pytest.raises(ValueError, match="tape_only=True reads the tape alone"):
        bt.sim.Market.from_store(store, SYMBOL, *_range(), tape_only=True, **extra)


def test_a_range_without_a_print_is_refused(store):
    start, end = (T0 + 2 * 86_400) * S, (T0 + 3 * 86_400) * S
    with pytest.raises(ValueError, match="no stored print"):
        bt.sim.Market.from_store(store, SYMBOL, start, end, tape_only=True)
