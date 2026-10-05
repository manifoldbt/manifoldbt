"""`bt.ingest_mbo` turning a day of Nasdaq ITCH 5.0 into stored order-by-order
events.

No file from Nasdaq: the messages are written here, framed as the day file
frames them, and read back through the store. What is checked is the shape of
what is stored (one file per UTC day, the events of each order), the checks it
makes (an event about an order that is not in the book is refused, the day left
as it was), and that the three ways a day comes in (gzip, ZSTD, plain) read the
same.
"""
import gzip
import struct

import pytest

import manifoldbt as bt

DATE = "2025-12-09"
# 2025-12-09 is in winter: New York is 5 hours behind UTC.
H = 3_600_000_000_000


def _unlocked():
    """Probe the gate, not the banner: a Pro licence does not open this."""
    from manifoldbt._native import py_attach_quotes_arrays

    try:
        py_attach_quotes_arrays("", "", 1, [], [], [])
    except PermissionError:
        return False
    except Exception:
        return True
    return True


def _frame(body: bytes) -> bytes:
    return struct.pack(">H", len(body)) + body


def _head(typ: bytes, locate: int, ts: int) -> bytes:
    return typ + struct.pack(">HH", locate, 0) + ts.to_bytes(6, "big")


def _stock(name: str) -> bytes:
    return name.encode().ljust(8)


def add(locate, ts, ref, buy, qty, sym, px):
    return _frame(_head(b"A", locate, ts) + struct.pack(">Q", ref) + (b"B" if buy else b"S")
                  + struct.pack(">I", qty) + _stock(sym) + struct.pack(">I", px))


def execute(locate, ts, ref, qty, match):
    return _frame(_head(b"E", locate, ts) + struct.pack(">QIQ", ref, qty, match))


def cancel(locate, ts, ref, qty):
    return _frame(_head(b"X", locate, ts) + struct.pack(">QI", ref, qty))


def delete(locate, ts, ref):
    return _frame(_head(b"D", locate, ts) + struct.pack(">Q", ref))


def replace(locate, ts, old, new, qty, px):
    return _frame(_head(b"U", locate, ts) + struct.pack(">QQII", old, new, qty, px))


def directory(locate, sym):
    return _frame(_head(b"R", locate, 0) + _stock(sym) + bytes(20))


def day_stream() -> bytes:
    """AAPL and MSFT in one stream; 09:30 in New York is 14:30 UTC."""
    t = 9 * H + H // 2
    msgs = [
        directory(1, "AAPL"),
        directory(2, "MSFT"),
        add(1, t, 1, True, 300, "AAPL", 2_000_000),
        add(2, t, 2, True, 100, "MSFT", 4_000_000),
        add(1, t + 1, 3, False, 200, "AAPL", 2_000_100),
        execute(1, t + 2, 1, 100, 7),
        cancel(1, t + 3, 3, 20),
        replace(1, t + 4, 3, 4, 150, 2_000_200),
        delete(1, t + 5, 1),
        execute(2, t + 6, 2, 100, 8),
    ]
    return b"".join(msgs)


pytestmark = pytest.mark.skipif(
    not _unlocked(),
    reason="order-by-order layer locked (runs in the engine's own debug builds)",
)


def _ingest(tmp_path, path, symbols):
    return bt.ingest_mbo(
        str(path), symbols, DATE,
        data_root=str(tmp_path / "data"),
        metadata_db=str(tmp_path / "meta.sqlite"),
        progress=False,
    )


@pytest.mark.parametrize("packing", ["plain", "gzip", "zstd"])
def test_a_day_is_stored_order_by_order(tmp_path, packing):
    raw = day_stream()
    path = tmp_path / f"day.itch.{packing}"
    if packing == "gzip":
        path.write_bytes(gzip.compress(raw))
    elif packing == "zstd":
        zstd = pytest.importorskip("zstandard")
        path.write_bytes(zstd.ZstdCompressor().compress(raw))
    else:
        path.write_bytes(raw)
    store = _ingest(tmp_path, path, {"AAPL": 1, "MSFT": 2})
    assert isinstance(store, bt.DataStore)
    for sym in ("AAPL", "MSFT"):
        days = sorted(p.name for p in (tmp_path / "data" / "mega" / "nasdaq" / "mbo" / sym).iterdir())
        assert days == [f"{DATE}.arrow"]
    ids = dict((t, i) for i, t in store.list_symbols())
    assert ids == {"AAPL": 1, "MSFT": 2}


def test_the_events_of_each_order_read_back(tmp_path):
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "day.itch"
    path.write_bytes(day_stream())
    _ingest(tmp_path, path, {"AAPL": 1})
    day = tmp_path / "data" / "mega" / "nasdaq" / "mbo" / "AAPL" / f"{DATE}.arrow"
    with pa.memory_map(str(day)) as src:
        table = pa.ipc.open_file(src).read_all()
    acts = bytes(table.column("action").to_pylist()).decode()
    # The order still resting when the stream ends (4) leaves at its end.
    assert acts == "AAEXUADD"
    assert table.schema.metadata[b"format"] == b"mbo-v1"
    qty = table.column("qty").to_pylist()
    # Replaced: the old order leaves with what was left of it (200 - 20).
    assert qty[4] == 180.0 and qty[5] == 150.0
    ts = table.column("ts_ns").cast(pa.int64()).to_pylist()
    midnight_utc = 1_765_238_400_000_000_000  # 2025-12-09T00:00:00Z
    assert ts[0] == midnight_utc + 5 * H + 9 * H + H // 2
    # Every row says when its order entered the book.
    order_ts = table.column("order_ts").cast(pa.int64()).to_pylist()
    assert order_ts[3] == ts[1]  # the cancellation of order 3
    price = table.column("price").to_pylist()
    assert price[0] == 200 * 100_000_000


def test_an_event_about_an_order_not_in_the_book_is_refused(tmp_path):
    path = tmp_path / "bad.itch"
    t = 10 * H
    path.write_bytes(add(1, t, 1, True, 100, "AAPL", 2_000_000) + execute(1, t + 1, 9, 10, 1))
    with pytest.raises(Exception, match="order 9, not in the book"):
        _ingest(tmp_path, path, {"AAPL": 1})
    day = tmp_path / "data" / "mega" / "nasdaq" / "mbo" / "AAPL"
    assert not list(day.glob("*.part"))


def test_the_arguments_are_checked(tmp_path):
    path = tmp_path / "day.itch"
    path.write_bytes(day_stream())
    with pytest.raises(ValueError, match="at least one ticker"):
        _ingest(tmp_path, path, {})
    with pytest.raises(ValueError, match="given to two tickers"):
        _ingest(tmp_path, path, [("AAPL", 1), ("MSFT", 1)])
    with pytest.raises(Exception, match="not a YYYY-MM-DD date"):
        bt.ingest_mbo(str(path), {"AAPL": 1}, "09/12/2025",
                      data_root=str(tmp_path / "d"), metadata_db=str(tmp_path / "m.sqlite"),
                      progress=False)


def test_ingest_mbo_is_exported():
    assert "ingest_mbo" in bt.__all__
