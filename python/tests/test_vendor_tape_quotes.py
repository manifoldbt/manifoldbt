"""Tapes and quotes from the data vendors (Massive, Databento).

No network: what a vendor answers is pinned by the Rust connector tests
(pages, cursor, quoted 64-bit fields, BBO interval stamps, tick rule). Here,
what the Python entry points refuse before any request, and that a missing key
is named rather than sent.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402


def _unlocked(probe):
    try:
        probe()
    except PermissionError:
        return False
    except Exception:
        return True
    return True


def _tape_open():
    return _unlocked(lambda: bt.ticks.tape_info(__file__))


def _quotes_open():
    from manifoldbt._native import py_attach_quotes_arrays

    return _unlocked(lambda: py_attach_quotes_arrays("", "", 1, [], [], []))


tape = pytest.mark.skipif(not _tape_open(), reason="tick layer locked")
quotes = pytest.mark.skipif(not _quotes_open(), reason="quote layer locked")


def _store(tmp_path):
    base_ns = 1_793_059_200_000_000_000  # 2026-10-27T00:00:00Z
    ts = base_ns + np.arange(10, dtype=np.int64) * 60_000_000_000
    closes = np.full(10, 100.0)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="ns", utc=True),
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": np.ones(10),
    })
    return bt.import_dataframe(
        df, symbol="AAPL", symbol_id=1, interval="1m", asset_class="equity",
        data_root=os.path.join(tmp_path, "data"),
        metadata_db=os.path.join(tmp_path, "meta.sqlite"),
    )


@tape
def test_dataset_is_for_databento_only(tmp_path):
    with pytest.raises(ValueError, match="Databento dataset"):
        bt.ingest_trades("binance", "BTCUSDT", symbol_id=1, start="2026-10-27",
                         end="2026-10-27", dataset="XNAS.ITCH", progress=False,
                         data_root=str(tmp_path / "d"),
                         metadata_db=str(tmp_path / "m.sqlite"))


@tape
def test_databento_tape_needs_an_asset_class(tmp_path):
    with pytest.raises(ValueError, match="asset_class"):
        bt.ingest_trades("databento", "ESZ6", symbol_id=1, start="2026-10-27",
                         end="2026-10-27", dataset="GLBX.MDP3", progress=False,
                         data_root=str(tmp_path / "d"),
                         metadata_db=str(tmp_path / "m.sqlite"))


@tape
def test_massive_tape_without_a_key_names_it(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(Exception, match="MASSIVE_API_KEY"):
        bt.ingest_trades("massive", "AAPL", symbol_id=1, start="2026-10-27",
                         end="2026-10-27", progress=False,
                         data_root=str(tmp_path / "d"),
                         metadata_db=str(tmp_path / "m.sqlite"))


def test_an_unknown_quote_provider_is_refused(tmp_path):
    store = _store(str(tmp_path))
    with pytest.raises(ValueError, match="'bybit', 'massive' or 'databento'"):
        bt.attach_quotes(store, "AAPL", "2026-10-27", "2026-10-27",
                         provider="kraken", progress=False)


def test_quote_dataset_is_for_databento_only(tmp_path):
    store = _store(str(tmp_path))
    with pytest.raises(ValueError, match="Databento dataset"):
        bt.attach_quotes(store, "AAPL", "2026-10-27", "2026-10-27",
                         provider="massive", dataset="XNAS.ITCH", progress=False)


def test_vendor_quotes_by_id_need_the_vendor_symbol(tmp_path):
    store = _store(str(tmp_path))
    with pytest.raises(ValueError, match="provider_symbol is required"):
        bt.attach_quotes(store, 1, "2026-10-27", "2026-10-27",
                         provider="massive", progress=False)


@quotes
def test_massive_quotes_without_a_key_names_it(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    store = _store(str(tmp_path))
    with pytest.raises(Exception, match="MASSIVE_API_KEY"):
        bt.attach_quotes(store, "AAPL", "2026-10-27", "2026-10-27",
                         provider="massive", progress=False)
