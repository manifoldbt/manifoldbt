"""`bt.ingest_trades` reading Bybit's derivatives archive into a stored tape.

No network: with ``cache_dir`` an archive already in the cache under its venue
name is read exactly as a downloaded one would be, so a hand-written one stands
in for a real day. What is checked is the conversion, not the numbers of any
real day: the aggressor's side, the stamp to the digit, the venue's order
(newest-first files included), the kind of contract, and one market and one id
per symbol, refused before the network.
"""
import gzip

import pytest

pa = pytest.importorskip("pyarrow")

import manifoldbt as bt  # noqa: E402

DAY = "2023-03-01"
# 2023-03-01T00:00:00Z
T0 = 1677628800
SYMBOL = "TESTUSDT"
HEADER = ("timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
          "grossValue,homeNotional,foreignNotional")


def _tape_unlocked():
    """Probe the gate, not the banner: a Pro licence does not open this."""
    try:
        bt.ticks.tape_info(__file__)
    except PermissionError:
        return False
    except Exception:
        return True
    return True


pytestmark = pytest.mark.skipif(
    not _tape_unlocked(),
    reason="tick layer locked (runs in the engine's own debug builds)",
)


def _line(ts, side, size, price, symbol=SYMBOL, inverse=False):
    """One trade as Bybit writes it: the coin value of an inverse contract is
    size / price, the quote value of a linear one size * price."""
    foreign = size / price if inverse else size * price
    return (f"{ts},{symbol},{side},{size},{price},PlusTick,"
            f"00000000-0000-0000-0000-000000000000,1,{size},{foreign!r}")


def _archive(cache, lines, symbol=SYMBOL, day=DAY):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{symbol}{day}.csv.gz"
    path.write_bytes(gzip.compress(("\n".join([HEADER, *lines]) + "\n").encode()))
    return path


def _ingest(tmp_path, category="linear", symbol=SYMBOL, symbol_id=7, day=DAY, **kw):
    return bt.ingest_trades(
        "bybit", symbol, symbol_id=symbol_id, start=day, end=day,
        data_root=str(tmp_path / "data"), metadata_db=str(tmp_path / "meta.sqlite"),
        category=category, cache_dir=str(tmp_path / "cache"), progress=False, **kw,
    )


def _stored(tmp_path, symbol=SYMBOL, day=DAY):
    path = tmp_path / "data" / "mega" / "bybit" / "ticks" / symbol / f"{day}.arrow"
    with pa.ipc.open_file(str(path)) as f:
        return f.read_all(), {k.decode(): v.decode() for k, v in (f.schema.metadata or {}).items()}


def test_a_linear_day_is_stored_print_by_print_with_its_market(tmp_path):
    _archive(tmp_path / "cache", [
        _line(f"{T0 + 3}.1129", "Sell", 111, 0.3761),
        _line(f"{T0 + 3}.1129", "Buy", 105, 0.3762),
        _line(f"{T0 + 3}.96", "Buy", 360, 0.3762),
    ])
    _ingest(tmp_path)
    table, meta = _stored(tmp_path)
    assert (meta["category"], meta["source"]) == ("linear", "bybit/linear")
    ns = [t.value for t in table.column("timestamp")]
    # The stamp to the digit: `.96` is `.9600`, and no float on the way.
    assert ns == [(T0 + 3) * 10**9 + 112_900_000] * 2 + [(T0 + 3) * 10**9 + 960_000_000]
    assert table.column("qty").to_pylist() == [111.0, 105.0, 360.0]
    # The side is the aggressor's: a sell aggressor means the buyer made.
    assert table.column("is_buyer_maker").to_pylist() == [True, False, False]
    # A UUID has no place in the integer id.
    assert table.column("trade_id").null_count == 3


def test_a_day_listed_newest_first_comes_back_in_the_venue_order(tmp_path):
    """Up to 2021-12-06 Bybit lists a day newest first: reversed, the prints
    sharing a stamp keep the order they were matched in."""
    _archive(tmp_path / "cache", [
        _line(f"{T0 + 9}.5", "Buy", 30, 1.5),
        _line(f"{T0 + 5}.25", "Sell", 21, 1.4),
        _line(f"{T0 + 5}.25", "Sell", 20, 1.4),
        _line(f"{T0 + 1}", "Sell", 18, 1.5),
    ])
    _ingest(tmp_path)
    table, _ = _stored(tmp_path)
    assert table.column("qty").to_pylist() == [18.0, 20.0, 21.0, 30.0]


def test_prints_stamped_outside_the_day_stay_out_of_its_file(tmp_path):
    _archive(tmp_path / "cache", [
        _line(f"{T0 - 1}.5", "Buy", 1, 1.5),
        _line(f"{T0 + 1}", "Buy", 2, 1.5),
        _line(f"{T0 + 86_400}", "Buy", 3, 1.5),
    ])
    _ingest(tmp_path)
    table, _ = _stored(tmp_path)
    assert table.column("qty").to_pylist() == [2.0]


def test_an_inverse_archive_asked_as_linear_is_refused_by_name(tmp_path):
    _archive(tmp_path / "cache", [_line(f"{T0 + 1}", "Buy", 517, 23134.5, inverse=True)],
             symbol="TESTUSD")
    with pytest.raises((ValueError, RuntimeError), match="inverse contract.*category='inverse'"):
        _ingest(tmp_path, symbol="TESTUSD")
    assert not (tmp_path / "data" / "mega" / "bybit" / "ticks" / "TESTUSD").exists()
    # Under its own name it is stored, sizes in contracts.
    _ingest(tmp_path, category="inverse", symbol="TESTUSD")
    table, meta = _stored(tmp_path, symbol="TESTUSD")
    assert meta["category"] == "inverse"
    assert table.column("qty").to_pylist() == [517.0]


def test_one_symbol_is_one_market_refused_before_the_network(tmp_path):
    """The day asked for is in no cache: were the market checked after the
    download, the error would be the network's."""
    _archive(tmp_path / "cache", [_line(f"{T0 + 1}", "Buy", 1, 1.5)])
    _ingest(tmp_path)
    with pytest.raises((ValueError, RuntimeError), match="linear tape of TESTUSDT.*one market"):
        _ingest(tmp_path, category="spot", day="2023-03-02")
    with pytest.raises((ValueError, RuntimeError), match="linear tape of TESTUSDT.*one market"):
        bt.ingest_book(
            "bybit", SYMBOL, symbol_id=7, start="2023-03-02", end="2023-03-02",
            data_root=str(tmp_path / "data"), metadata_db=str(tmp_path / "meta.sqlite"),
            category="spot", cache_dir=str(tmp_path / "cache"), progress=False,
        )


def test_the_tape_and_the_book_of_a_symbol_share_one_id(tmp_path):
    _archive(tmp_path / "cache", [_line(f"{T0 + 1}", "Buy", 1, 1.5)])
    _ingest(tmp_path, symbol_id=7)
    with pytest.raises((ValueError, RuntimeError), match="symbol_id 7.*symbol_id=7"):
        _ingest(tmp_path, symbol_id=8)


@pytest.mark.parametrize("provider, kwargs, match", [
    ("bybit", {"category": "option"}, "no option trade archive"),
    ("binance", {"category": "linear"}, "provider must be 'bybit'"),
    ("binance", {"cache_dir": "x"}, "cache_dir keeps Bybit"),
    ("bybit", {"category": "linear", "exchange": "BINANCE"}, "filed under BYBIT"),
    ("bybit", {"category": "linear", "asset_class": "crypto_spot"}, "perpetual"),
])
def test_what_cannot_be_read_is_refused_before_the_network(tmp_path, provider, kwargs, match):
    with pytest.raises(ValueError, match=match):
        bt.ingest_trades(
            provider, SYMBOL, symbol_id=7, start=DAY, end=DAY,
            data_root=str(tmp_path / "data"), metadata_db=str(tmp_path / "meta.sqlite"),
            progress=False, **kwargs,
        )
    assert not (tmp_path / "data").exists()
