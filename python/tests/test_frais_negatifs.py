"""A negative fee rate is a rebate, on both doors.

`maker_fee_bps` / `taker_fee_bps` below zero used to be accepted and the fee
brought back to zero by the `min_fee` floor. A rebate now gives a negative fee
(the cash is credited), and `min_fee` floors a commission only. A run with
fees of zero or more is unchanged.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, when  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from manifoldbt.indicators import close as close_px, sma  # noqa: E402

MS = 1_000_000
POST, END = 2 * MS, 10 * MS
CAPITAL = 100_000.0
START = pd.Timestamp("2021-03-01", tz="UTC")


def _sim_layer_open():
    try:
        bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    except PermissionError:
        return False
    return True


def _fills(maker_bps, taker_bps):
    market = bt.sim.Market.from_ladders(
        "BTCUSDT",
        0.1,
        [(MS, [(100.0, 5.0), (99.9, 9.0)], [(100.1, 5.0), (100.5, 6.0)])],
        [(3 * MS, 99.9, 10.0, "sell")],
    )

    def strategy(sim):
        sim.elapse(POST - sim.now())
        sim.post("BTCUSDT", "buy", 100.0, 1.0)  # rests, served by the print
        sim.elapse(2 * MS)
        sim.post("BTCUSDT", "sell", None, 2.0, tif="IOC")  # takes at arrival
        sim.elapse(END - sim.now())

    cfg = bt.sim.Config(maker_fee_bps=maker_bps, taker_fee_bps=taker_bps, initial_cash=1_000.0)
    return bt.sim.run(strategy, cfg, markets=[market])


@pytest.mark.skipif(not _sim_layer_open(), reason="order-level simulation locked")
def test_a_rebate_credits_the_cash_in_bt_sim():
    paid = _fills(1.0, 5.0)
    rebated = _fills(-1.0, -5.0)
    fees = [round(f, 6) for f in rebated.fills_df(backend="pandas")["fee"]]
    assert fees == [-0.01, -0.1]
    assert round(rebated.fees, 6) == -0.11
    # The same fills, the fees turned around: the cash moves by twice them.
    assert round(rebated.cash - paid.cash, 6) == round(2 * (0.01 + 0.1), 6)
    # A commission is what it was.
    assert [round(f, 6) for f in paid.fills_df(backend="pandas")["fee"]] == [0.01, 0.1]


def _store(tmp_path, rows=3000):
    rng = np.random.default_rng(5)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 6e-4, rows)))
    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = rng.uniform(0.2, 1.8, rows) * 3e-4 * close
    frame = pd.DataFrame({
        "timestamp": pd.date_range(START, periods=rows, freq="1min"),
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": np.full(rows, 1_000.0),
    })
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    return bt.import_dataframe(
        frame, symbol="REB", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    ), rows


def _run(store, rows, fees):
    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=int(START.value),
        time_range_end=int(START.value) + rows * 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=0, execution_price="AtClose", max_position_pct=1.0,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=fees,
        slippage=Slippage.none(),
        warmup_bars=0,
    )
    sized = when(col("fast") > col("slow"), lit(1.0), lit(0.0))
    strategy = (bt.Strategy.create("reb")
                .signal("fast", sma(close_px, 5))
                .signal("slow", sma(close_px, 20))
                .size(sized))
    return bt.run(strategy, config, store).metrics["total_return"]


def test_a_rebate_pays_in_bt_run_and_the_minimum_floors_a_commission_only(tmp_path):
    store, rows = _store(tmp_path)
    free = _run(store, rows, bt.FeeConfig.zero())
    paid = _run(store, rows, bt.FeeConfig(taker_fee_bps=5.0, maker_fee_bps=5.0))
    rebate = _run(store, rows, bt.FeeConfig(taker_fee_bps=-5.0, maker_fee_bps=-5.0))
    assert paid < free < rebate
    # The minimum is a floor under a commission: it never lifts a rebate.
    floored = _run(store, rows, bt.FeeConfig(taker_fee_bps=-5.0, maker_fee_bps=-5.0, min_fee=500.0))
    assert floored == rebate
    # And a commission below the minimum (50 on a trade of 100 000) is raised to it.
    assert _run(store, rows, bt.FeeConfig(taker_fee_bps=5.0, maker_fee_bps=5.0, min_fee=500.0)) < paid
