"""Signal-driven execution price: ``ExecutionPrice.custom(<signal name>)``.

The user-facing surface of the band-strategy fix: the engine could always
COMPUTE a level in the DSL (a band around an SMA) but the only reachable fill
was the close of the bar, systematically on the wrong side of a mean-reverting
touch. ``custom()`` now also accepts the name of a signal the strategy
defines, and the fill lands on that series.

The scenario is the minimal honest slice of the real case (short at the touch
of an upper band on native fine bars): the touch bar OPENS below the band and
its HIGH crosses it, so the band level provably traded inside the bar, yet it
equals neither the open nor the close.

Runs on synthetic bars in a tmp store; no license assumptions beyond what the
other python tests already make (trade fills are exact on Community builds).
"""
import pytest

pd = pytest.importorskip("pandas")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, when  # noqa: E402
from manifoldbt.helpers import ExecutionPrice, Interval, Slippage  # noqa: E402

CAPITAL = 10_000.0
BAND = 100.5

# One-minute bars. Bar 1 is the touch bar: open 100.2 < BAND 100.5 <= high
# 100.8, close 100.7. The band level traded inside the bar, but AtClose can
# only fill at 100.7.
BARS = dict(
    o=[100.0, 100.2, 100.7, 100.6],
    h=[100.4, 100.8, 100.9, 100.8],
    l=[99.8, 100.1, 100.5, 100.4],
    c=[100.2, 100.7, 100.6, 100.5],
)


def _frame():
    ts = pd.date_range("2023-01-01", periods=len(BARS["c"]), freq="1min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts,
         "open": list(map(float, BARS["o"])), "high": list(map(float, BARS["h"])),
         "low": list(map(float, BARS["l"])), "close": list(map(float, BARS["c"])),
         "volume": [1000.0] * len(BARS["c"])}
    )


def _strategy(target: float):
    """Enter (long or short) at the touch of the band; fill on its level.

    ``exec_level`` is the docs' composition: the band when touched (clipped to
    the open when the bar opens through it), the close otherwise.
    """
    touched = col("high") >= lit(BAND)
    sig = when(touched, lit(target), lit(float("nan")))
    exec_level = when(
        touched,
        when(col("open") >= lit(BAND), col("open"), lit(BAND)),
        col("close"),
    )
    return (
        bt.Strategy.create("band-touch")
        .signal("position", sig)
        .signal("exec_level", exec_level)
        .size(sig)
    )


def _run(tmp_path, name, strat, execution_price, *, allow_short=False):
    import os

    root = str(tmp_path / name)
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        _frame(), symbol="TEST", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    cfg = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=int(_frame()["timestamp"].iloc[-1].value) + 86_400_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=0, execution_price=execution_price,
            max_position_pct=1.0, allow_short=allow_short,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )
    return bt.run(strat, cfg, store)


def _entry_fill(res) -> float:
    tr = res.trades_df()
    assert len(tr) >= 1, f"expected an entry fill, got:\n{tr}"
    return float(tr.iloc[0]["fill_price"])


def test_long_entry_fills_on_the_band_not_at_the_close(tmp_path):
    at_level = _run(tmp_path, "lvl", _strategy(1.0), ExecutionPrice.custom("exec_level"))
    at_close = _run(tmp_path, "cls", _strategy(1.0), "AtClose")
    assert _entry_fill(at_level) == pytest.approx(BAND), (
        "the fill must land on the band level the DSL computed"
    )
    assert _entry_fill(at_close) == pytest.approx(BARS["c"][1]), (
        "the AtClose control must fill at the touch bar's close"
    )


def test_short_entry_fills_on_the_band_and_reports_the_worse_side(tmp_path):
    at_level = _run(tmp_path, "lvl", _strategy(-1.0),
                    ExecutionPrice.custom("exec_level"), allow_short=True)
    at_close = _run(tmp_path, "cls", _strategy(-1.0), "AtClose", allow_short=True)
    assert _entry_fill(at_level) == pytest.approx(BAND)
    # For a short at the touch of an upper band, the honest band fill (100.5)
    # is WORSE than the close fill (100.7): the fix must be able to move the
    # result down, not just up.
    assert _entry_fill(at_close) > _entry_fill(at_level)


def test_unknown_name_is_rejected_before_the_run(tmp_path):
    with pytest.raises(Exception, match="neither a bar column nor a signal"):
        _run(tmp_path, "bad", _strategy(1.0), ExecutionPrice.custom("nope"))
