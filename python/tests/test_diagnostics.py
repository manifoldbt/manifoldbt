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

import pytest

import manifoldbt as bt
from manifoldbt.diagnostics import _prepare_for_diagnostics
from manifoldbt.exceptions import LicenseError


def test_detect_lookahead_gated_on_community(monkeypatch):
    """Look-ahead detection is Pro: Community gets a clean LicenseError before any
    work (the analysis itself is enforced natively; this is the friendly UX gate).
    """
    monkeypatch.setattr(bt, "_is_pro", lambda: False)
    with pytest.raises(LicenseError):
        # Raises before touching strategy/config/store, so None args are fine.
        bt.diagnostics.detect_lookahead(None, None, None)


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


# ===========================================================================
# Static rule: a resting order cannot be priced off the bar it fills on
# ===========================================================================

def _resting_strategy():
    return (
        bt.Strategy.create("resting")
        .signal("s", bt.lit(1.0))
        .size(bt.col("s"))
        .limit_entry(offset_bps=25)
    )


def test_resting_order_with_delay_zero_is_refused(tmp_path):
    """`plan_entry` prices the order from bar `t - signal_delay` and the gate
    runs on bar `t` in the same pass, so delay 0 reads the fill bar twice."""
    store = _StubStore(_make_metadata_db(tmp_path / "m.sqlite"))
    config = bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=10**12,
        execution=bt.ExecutionConfig(signal_delay=0),
    )
    with pytest.raises(ValueError, match="signal_delay"):
        bt._prepared_config_json(config, _resting_strategy(), store)


def test_market_order_with_delay_zero_is_fine(tmp_path):
    """Delay 0 is exactly what a market fill is for; only resting orders read
    the bar twice."""
    store = _StubStore(_make_metadata_db(tmp_path / "m.sqlite"))
    config = bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=10**12,
        execution=bt.ExecutionConfig(signal_delay=0),
    )
    market = bt.Strategy.create("market").signal("s", bt.lit(1.0)).size(bt.col("s"))
    bt._prepared_config_json(config, market, store)  # must not raise


def test_resting_order_with_delay_one_is_fine(tmp_path):
    store = _StubStore(_make_metadata_db(tmp_path / "m.sqlite"))
    config = bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=10**12,
        execution=bt.ExecutionConfig(signal_delay=1),
    )
    bt._prepared_config_json(config, _resting_strategy(), store)  # must not raise


def test_the_check_runs_before_the_config_memo(tmp_path):
    """The memo keys on the config alone, so a config first seen with a market
    strategy must not let a resting one through on the next call.
    """
    store = _StubStore(_make_metadata_db(tmp_path / "m.sqlite"))
    config = bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=10**12,
        execution=bt.ExecutionConfig(signal_delay=0),
    )
    market = bt.Strategy.create("market").signal("s", bt.lit(1.0)).size(bt.col("s"))
    bt._prepared_config_json(config, market, store)          # populates the memo
    with pytest.raises(ValueError, match="signal_delay"):
        bt._prepared_config_json(config, _resting_strategy(), store)
