"""Regression tests for the diagnostics config-preparation path.

Guards the fix for the bug where ``detect_lookahead`` / ``check_exposure_stability``
crashed with a dict ``universe`` (e.g. ``{"binance": ["BTC-USDT:perp"]}``):
they serialized the config without resolving the universe, so ``config.to_json()``
emitted a JSON *map* while the Rust loader expects a *sequence*
(``ValueError: invalid type: map, expected a sequence``).

The fix routes diagnostics through the same preparation as ``run()`` via
``_prepare_for_diagnostics``. These tests assert that helper resolves a dict
universe into a list of integer SymbolIds (so serialization is a JSON array),
without needing a Pro license or real market data.
"""
import json
import sqlite3

import manifoldbt as bt
from manifoldbt.diagnostics import _prepare_for_diagnostics


def _make_metadata_db(path):
    """Create a minimal metadata sqlite with one resolvable symbol (id=1)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE symbols ("
        "id INTEGER PRIMARY KEY, base_currency TEXT, quote_currency TEXT, "
        "asset_class TEXT, exchange TEXT, ticker TEXT)"
    )
    conn.execute(
        "INSERT INTO symbols VALUES (1, 'BTC', 'USDT', 'CryptoPerpetual', "
        "'BINANCE', 'BTC-USDT:perp')"
    )
    conn.commit()
    conn.close()
    return str(path)


class _StubStore:
    """Minimal DataStore stand-in.

    ``_resolve_normalized`` only needs ``metadata_db()`` (+ ``resolve_symbol``
    as a fallback). ``dataset()`` raises so ``_resolve_store`` returns the store
    unchanged instead of trying to swap datasets on disk.
    """

    def __init__(self, db_path):
        self._db = db_path

    def metadata_db(self):
        return self._db

    def dataset(self):
        raise NotImplementedError

    def resolve_symbol(self, name):  # fallback, not expected to be hit here
        return 1


def _simple_strategy():
    return (
        bt.Strategy.create("regression")
        .signal("s", bt.lit(1.0))
        .size(bt.col("s"))
    )


def test_prepare_for_diagnostics_resolves_dict_universe(tmp_path):
    """A dict universe must become a list of ints before serialization."""
    db = _make_metadata_db(tmp_path / "metadata.sqlite")
    store = _StubStore(db)

    config = bt.BacktestConfig(
        universe={"binance": ["BTC-USDT:perp"]},
        time_range_start=0,
        time_range_end=4_000_000_000,
        bar_interval={"Hours": 1},
        initial_capital=1000.0,
    )

    prepared, _ = _prepare_for_diagnostics(config, _simple_strategy(), store)

    # Core invariant: universe is a list of ints, never a dict.
    assert isinstance(prepared.universe, list)
    assert prepared.universe == [1]

    # And the JSON the Rust loader sees is an array, not a map (the crash cause).
    universe_json = json.loads(prepared.to_json())["universe"]
    assert isinstance(universe_json, list)
    assert universe_json == [1]


def test_prepare_for_diagnostics_passes_through_list_universe(tmp_path):
    """An already-resolved list universe is left intact."""
    db = _make_metadata_db(tmp_path / "metadata.sqlite")
    store = _StubStore(db)

    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=4_000_000_000,
        bar_interval={"Hours": 1},
        initial_capital=1000.0,
    )

    prepared, _ = _prepare_for_diagnostics(config, _simple_strategy(), store)

    assert prepared.universe == [1]
    assert json.loads(prepared.to_json())["universe"] == [1]
