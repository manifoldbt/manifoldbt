"""Fill-fragility: the touch-vs-traverse passive-fill convention.

A bar whose extreme merely touches a resting level (limit entry, take-profit)
cannot say whether the order was actually served: the queue ahead of it may
never have been consumed. The engine books those fills under its default
``touch`` convention; ``fill_model.passive_fill = "traverse"`` books only
fills the bar traded through. Running both conventions side by side says how
much of a verdict rests on the gap (see examples/26_fill_costs.py).

The bar data below is hand-built so the interesting extreme lands EXACTLY on
the order level (the touch-only case) — mirroring the Rust e2e tests in
``crates/bt-core/tests/backtest_orders.rs``.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402


def _store_from_rows(tmp_path, rows):
    """rows: list of (open, high, low, close) 1-minute bars."""
    o, h, l, c = (np.array([r[i] for r in rows], dtype=np.float64) for i in range(4))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2021-03-01", periods=len(rows), freq="1min", tz="UTC"
            ),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": np.full(len(rows), 1_000.0),
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
    last_ns = int(df["timestamp"].iloc[-1].value)
    return store, last_ns


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


def test_limit_entry_touch_only_counted_and_convention_respected(tmp_path):
    """A buy limit at 99.5: bar1 low == 99.5 exactly (touch-only), bar2
    traverses. ``touch`` fills on bar1 and counts it fragile; ``traverse``
    fills one bar later and counts nothing."""
    rows = [
        (100.0, 101.0, 99.8, 100.0),
        (100.0, 101.0, 99.5, 100.0),
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 101.0, 99.8, 100.0),
    ]
    store, last_ns = _store_from_rows(tmp_path, rows)
    strategy = (
        bt.Strategy.create("limit_touch")
        .signal("sig", lit(1.0))
        .size(col("sig"))
        .limit_entry(price=99.5)
    )

    config = _config(last_ns)
    touch = bt.run(strategy, config, store)
    assert touch.fill_fragility == {"maker_fills": 1, "touch_only_fills": 1}
    assert touch.trade_count == 1

    strict = _config(last_ns)
    strict.execution.fill_model = {"passive_fill": "traverse"}
    traverse = bt.run(strategy, strict, store)
    assert traverse.fill_fragility == {"maker_fills": 1, "touch_only_fills": 0}
    assert traverse.trade_count == 1

    # Same level, but the traverse fill happens one bar later.
    t_touch = touch.trades_df()["execution_timestamp"].iloc[0]
    t_trav = traverse.trades_df()["execution_timestamp"].iloc[0]
    assert t_trav > t_touch
