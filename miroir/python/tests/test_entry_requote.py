"""An entry order that expires unfilled is posted again.

A position target is a desired STATE, not an event. An entry order with a life
in bars (``time_in_force={"GTB": n}``) that expires without filling leaves that
desire unmet, so the engine posts it again on the next bar, at the level that
bar yields. Without that rule a strategy whose target never moves -- a maker
quoting a level every bar behind ``size(lit(1.0))`` -- placed exactly one order
and then stopped trading, with nothing in the result saying so.

The bars below are hand-built so the requote is visible in the fill PRICE, not
just in the trade count: the same strategy under ``GTC`` (the order never
expires, so it never moves) fills at the first level, and under ``GTB(2)`` at
the level of the bar that posted it again.

Counting fills and reading their prices keeps this test true whatever the
output resolution of the run: it never looks at an intraday position.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

# (open, high, low, close), one 1-minute bar each. With signal_delay=1 and a
# constant target of 1:
#   bar1: order #1 rests at close[0] * 0.98 = 98.00, low 99   -> no fill
#   bar2: low 99 -> no fill, and a GTB(2) order expires here
#   bar3: order #2 rests at close[2] * 0.98 = 98.98, low 100  -> no fill
#   bar4: low 98.5 <= 98.98 -> the requoted order FILLS at 98.98
#   bar5: low 98.0 <= 98.00 -> a GTC order, still resting at its first level,
#         only fills here
ROWS = [
    (100.0, 101.0, 99.0, 100.0),
    (100.0, 102.0, 99.0, 101.0),
    (101.0, 102.0, 99.0, 101.0),
    (101.0, 102.0, 100.0, 101.0),
    (101.0, 102.0, 98.5, 99.0),
    (99.0, 100.0, 98.0, 99.0),
]


def _store(tmp_path):
    o, h, l, c = (np.array([r[i] for r in ROWS], dtype=np.float64) for i in range(4))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2021-03-01", periods=len(ROWS), freq="1min", tz="UTC"
            ),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": np.full(len(ROWS), 1_000.0),
        }
    )
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df,
        symbol="TEST",
        symbol_id=1,
        interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    return store, int(df["timestamp"].iloc[-1].value)


def _config(last_ns):
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=last_ns + 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000.0,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def _strategy(time_in_force):
    return (
        bt.Strategy.create("requote")
        .signal("sig", lit(1.0))
        .size(col("sig"))
        .limit_entry(offset_bps=200, time_in_force=time_in_force)
    )


def test_an_expired_entry_is_posted_again_while_the_target_holds(tmp_path):
    store, last_ns = _store(tmp_path)
    result = bt.run(_strategy({"GTB": 2}), _config(last_ns), store)

    assert result.trade_count == 1, (
        "the order expired on bar 2 and must be posted again while the target "
        f"holds, got {result.trade_count} fills"
    )
    fill = float(result.trades_df()["fill_price"].iloc[0])
    assert fill == pytest.approx(98.98, abs=1e-9), (
        "the order must come back at the level of the bar that posts it "
        f"(98.98), got {fill}"
    )


def test_an_entry_that_never_expires_keeps_its_first_level(tmp_path):
    """The control. Under GTC the order rests untouched at its first level, so
    it fills lower and later. Without it, the test above would also pass on an
    engine that simply never expired anything."""
    store, last_ns = _store(tmp_path)
    result = bt.run(_strategy("GTC"), _config(last_ns), store)

    assert result.trade_count == 1
    fill = float(result.trades_df()["fill_price"].iloc[0])
    assert fill == pytest.approx(98.0, abs=1e-9), (
        f"a GTC order must still rest at its first level (98.00), got {fill}"
    )
