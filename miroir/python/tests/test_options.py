"""Tests for the option path from Python: contract terms in, settlement out.

The Rust side already proves the payoff arithmetic and the simulation loop.
What is under test here is the bridge: terms recorded at ingest must reach the
engine, and the one thing the user has to decide (which price series settles the
contract) must fail loudly when it is missing rather than be guessed.
"""
import os

import pytest

import manifoldbt as bt

pd = pytest.importorskip("pandas")

OPTION_ID = 2
UNDERLYING_ID = 1
STRIKE = 50_000.0
N_BARS = 40
# Expiry lands on bar 30 of a 40-bar daily series starting 2020-01-01.
EXPIRY_MS = 1_577_836_800_000 + 30 * 86_400_000


def _daily(prices):
    ts = pd.date_range("2020-01-01", periods=len(prices), freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [100.0] * len(prices),
        }
    )


def _store(tmp_path, underlying_price, premium, option_class="option"):
    """A two-symbol store: a perpetual and a call written against it.

    ``option_class`` exists so a test can write the same series as a plain
    linear instrument, which is a different thing from an option missing its
    terms.
    """
    root = os.path.join(str(tmp_path), "data")
    meta = os.path.join(str(tmp_path), "m.sqlite")
    store = bt.import_dataframe(
        _daily([underlying_price] * N_BARS),
        symbol="BTC-PERPETUAL",
        symbol_id=UNDERLYING_ID,
        interval="1d",
        data_root=root,
        metadata_db=meta,
        asset_class="crypto_perp",
    )
    store = bt.import_dataframe(
        _daily([premium] * N_BARS),
        symbol="BTC-CALL",
        symbol_id=OPTION_ID,
        interval="1d",
        data_root=root,
        metadata_db=meta,
        asset_class=option_class,
    )
    return store, root, meta


def _write_terms(meta_db, settlement="cash_inverse"):
    """Record contract terms the way an option connector would."""
    import sqlite3

    conn = sqlite3.connect(meta_db)
    conn.execute(
        "UPDATE symbols SET option_underlying = ?, option_type = ?, option_strike = ?,"
        " option_expiry = ?, option_contract_size = ?, option_settlement = ?"
        " WHERE id = ?",
        (
            "BTC_USD index",
            "call",
            STRIKE,
            pd.Timestamp(EXPIRY_MS, unit="ms", tz="UTC").isoformat().replace("+00:00", "Z"),
            1.0,
            settlement,
            OPTION_ID,
        ),
    )
    conn.commit()
    conn.close()


def _config(**kwargs):
    from manifoldbt.helpers import time_range, Interval

    start, end = time_range("2020-01-01", "2020-03-01")
    base = dict(
        universe=[UNDERLYING_ID, OPTION_ID],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.days(1),
        initial_capital=10.0,
        currency="BTC",
        execution=bt.ExecutionConfig(position_sizing_mode="Units"),
    )
    base.update(kwargs)
    return bt.BacktestConfig(**base)


def _hold(**per_symbol):
    """Hold a fixed number of units of each named symbol id, every bar.

    Legs are told apart by `col("symbol_id")` rather than by price level. A
    price threshold is a trap: a premium crossing it flips the leg to zero and
    the strategy closes its own position, which is exactly how an earlier
    version of this file broke.
    """
    from manifoldbt.indicators import col

    size = bt.when(col("symbol_id") < 0.0, 0.0, 0.0)  # a typed zero to fold onto
    for symbol_id, units in per_symbol.items():
        size = size + bt.when(col("symbol_id") == float(symbol_id), float(units), 0.0)
    return (
        bt.Strategy.create("hold")
        .signal("leg", col("symbol_id"))
        .size(size)
        .describe("Fixed units per leg, held into expiry")
    )


def _long_one_option():
    return _hold(**{str(OPTION_ID): 1.0})


def test_contract_terms_round_trip_to_python(tmp_path):
    _, _, meta = _store(tmp_path, 60_000.0, 0.05)
    _write_terms(meta)
    store = bt.DataStore(os.path.join(str(tmp_path), "data"), meta)

    terms = store.option_contracts()
    assert OPTION_ID in terms
    assert terms[OPTION_ID]["option_type"] == "call"
    assert terms[OPTION_ID]["strike"] == STRIKE
    assert terms[OPTION_ID]["settlement"] == "cash_inverse"
    assert UNDERLYING_ID not in terms, "a perpetual has no contract terms"


def test_an_option_without_a_declared_underlying_is_refused(tmp_path):
    store, _, meta = _store(tmp_path, 60_000.0, 0.05)
    _write_terms(meta)

    # The public API re-classifies the failure, so catch what a user catches.
    from manifoldbt.exceptions import DataError

    with pytest.raises(DataError) as excinfo:
        bt.run(_long_one_option(), _config(), store)

    message = str(excinfo.value)
    assert "option_underlyings" in message
    assert "own last traded premium" in message


def test_a_call_expiring_in_the_money_settles_at_intrinsic(tmp_path):
    # S = 60k against a 50k strike, inverse settlement: 10000/60000 BTC.
    store, _, meta = _store(tmp_path, 60_000.0, 0.05)
    _write_terms(meta)

    result = bt.run(
        _long_one_option(),
        _config(option_underlyings={OPTION_ID: UNDERLYING_ID}),
        store,
    )

    trades = result.trades.to_pandas()
    settlements = trades[(trades.symbol_id == OPTION_ID) & (trades.exit_reason == 5)]
    assert len(settlements) == 1, f"expected one settlement, got:\n{trades}"
    assert settlements.iloc[0].fill_price == pytest.approx(10_000.0 / 60_000.0, abs=1e-12)


def test_a_call_expiring_out_of_the_money_settles_at_zero(tmp_path):
    store, _, meta = _store(tmp_path, 40_000.0, 0.05)
    _write_terms(meta)

    result = bt.run(
        _long_one_option(),
        _config(option_underlyings={OPTION_ID: UNDERLYING_ID}),
        store,
    )

    trades = result.trades.to_pandas()
    settlements = trades[(trades.symbol_id == OPTION_ID) & (trades.exit_reason == 5)]
    assert len(settlements) == 1
    assert settlements.iloc[0].fill_price == 0.0
    # The premium paid is the whole loss, and it is a loss.
    assert float(result.equity_curve[-1]) < float(result.equity_curve[0])


def test_a_linear_universe_is_untouched_by_the_option_path(tmp_path):
    # Two ordinary linear instruments: the option path must not touch them.
    store, _, _ = _store(tmp_path, 40_000.0, 0.05, option_class="crypto_spot")

    result = bt.run(_long_one_option(), _config(), store)

    trades = result.trades.to_pandas()
    assert (trades.exit_reason != 5).all(), "nothing may settle without contract terms"


def test_an_option_symbol_without_contract_terms_is_refused(tmp_path):
    """The Databento case before this branch: an option that never expires.

    A symbol recorded as an option but carrying no strike or expiration would
    otherwise price, trade and be held forever at its last quoted premium, with
    nothing in the output looking wrong.
    """
    from manifoldbt.exceptions import DataError

    store, _, _ = _store(tmp_path, 60_000.0, 0.05)  # asset_class="option", no terms

    with pytest.raises(DataError) as excinfo:
        bt.run(_long_one_option(), _config(), store)

    message = str(excinfo.value)
    assert "no contract terms" in message
    assert "deribit, databento" in message


def test_a_multiplier_option_costs_and_settles_like_one_contract(tmp_path):
    """A listed-style option: 100 units of premium IS one exchange contract."""
    # Premium 4.70, underlying 490, strike 470 -> one contract pays (490-470)*100.
    store, _, meta = _store(tmp_path, 490.0, 4.70)
    import sqlite3

    conn = sqlite3.connect(meta)
    conn.execute(
        "UPDATE symbols SET option_underlying = ?, option_type = ?, option_strike = ?,"
        " option_expiry = ?, option_contract_size = ?, option_settlement = ? WHERE id = ?",
        (
            "SPY",
            "call",
            470.0,
            pd.Timestamp(EXPIRY_MS, unit="ms", tz="UTC").isoformat().replace("+00:00", "Z"),
            100.0,
            "cash_linear",
            OPTION_ID,
        ),
    )
    conn.commit()
    conn.close()

    # 100 units of the option leg, nothing on the underlying.
    hold_one_contract = _hold(**{str(OPTION_ID): 100.0})
    config = _config(
        initial_capital=100_000.0,
        option_underlyings={OPTION_ID: UNDERLYING_ID},
    )

    result = bt.run(hold_one_contract, config, store)
    trades = result.trades.to_pandas()
    legs = trades[trades.symbol_id == OPTION_ID]

    entry = legs[legs.exit_reason == 0].iloc[0]
    assert entry.quantity * entry.fill_price == pytest.approx(470.0), "what one contract costs"

    settlement = legs[legs.exit_reason == 5].iloc[0]
    assert settlement.fill_price == pytest.approx(20.0), "intrinsic per share, not per contract"
    assert settlement.quantity * settlement.fill_price == pytest.approx(
        2_000.0
    ), "what one contract pays"


PUT_ID = 3


def _store_two_legs(tmp_path, underlying_price, call_premium, put_premium):
    """Underlying + a call + a put, all daily, all the same length."""
    root = os.path.join(str(tmp_path), "data")
    meta = os.path.join(str(tmp_path), "m.sqlite")
    for symbol, symbol_id, price, klass in (
        ("BTC-PERPETUAL", UNDERLYING_ID, underlying_price, "crypto_perp"),
        ("BTC-CALL", OPTION_ID, call_premium, "option"),
        ("BTC-PUT", PUT_ID, put_premium, "option"),
    ):
        store = bt.import_dataframe(
            _daily([price] * N_BARS),
            symbol=symbol,
            symbol_id=symbol_id,
            interval="1d",
            data_root=root,
            metadata_db=meta,
            asset_class=klass,
        )
    return store, meta


def _write_leg_terms(meta_db, symbol_id, option_type, strike):
    import sqlite3

    conn = sqlite3.connect(meta_db)
    conn.execute(
        "UPDATE symbols SET option_underlying = ?, option_type = ?, option_strike = ?,"
        " option_expiry = ?, option_contract_size = ?, option_settlement = ? WHERE id = ?",
        (
            "BTC_USD index",
            option_type,
            strike,
            pd.Timestamp(EXPIRY_MS, unit="ms", tz="UTC").isoformat().replace("+00:00", "Z"),
            1.0,
            "cash_inverse",
            symbol_id,
        ),
    )
    conn.commit()
    conn.close()


def test_a_two_leg_structure_settles_each_leg_on_its_own_terms(tmp_path):
    """A risk reversal: long a call, short a put, both expiring together.

    Each leg settles against the same underlying but on its own strike and
    side, so one finishes in the money and the other worthless.
    """
    # S = 60k at expiry: the 50k call is ITM, the 40k put is worthless.
    store, meta = _store_two_legs(tmp_path, 60_000.0, 0.05, 0.03)
    _write_leg_terms(meta, OPTION_ID, "call", 50_000.0)
    _write_leg_terms(meta, PUT_ID, "put", 40_000.0)

    config = _config(
        universe=[UNDERLYING_ID, OPTION_ID, PUT_ID],
        option_underlyings={OPTION_ID: UNDERLYING_ID, PUT_ID: UNDERLYING_ID},
        option_margin_model="deribit",
        execution=bt.ExecutionConfig(position_sizing_mode="Units", allow_short=True),
    )
    result = bt.run(
        _hold(**{str(OPTION_ID): 1.0, str(PUT_ID): -1.0}),
        config,
        store,
    )

    trades = result.trades.to_pandas()
    settlements = trades[trades.exit_reason == 5]
    assert set(settlements.symbol_id) == {OPTION_ID, PUT_ID}, "both legs must settle"

    call = settlements[settlements.symbol_id == OPTION_ID].iloc[0]
    put = settlements[settlements.symbol_id == PUT_ID].iloc[0]
    assert call.fill_price == pytest.approx(10_000.0 / 60_000.0, abs=1e-12)
    assert put.fill_price == 0.0, "a 40k put is worthless with the underlying at 60k"

    # Long the call, short the put: the short is bought back to close.
    assert call.side == 2 and put.side == 1


def test_per_leg_sizing_leaves_the_other_leg_flat(tmp_path):
    """`col("symbol_id")` must target one leg without disturbing the others."""
    store, meta = _store_two_legs(tmp_path, 60_000.0, 0.05, 0.03)
    _write_leg_terms(meta, OPTION_ID, "call", 50_000.0)
    _write_leg_terms(meta, PUT_ID, "put", 40_000.0)

    config = _config(
        universe=[UNDERLYING_ID, OPTION_ID, PUT_ID],
        option_underlyings={OPTION_ID: UNDERLYING_ID, PUT_ID: UNDERLYING_ID},
    )
    result = bt.run(_hold(**{str(OPTION_ID): 1.0}), config, store)

    trades = result.trades.to_pandas()
    assert (trades.symbol_id == OPTION_ID).all(), (
        f"only the call leg may trade, got:\n{trades}"
    )
