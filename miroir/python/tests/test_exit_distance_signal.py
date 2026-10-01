"""An exit distance read from a signal.

``stop_pct`` was one number for the whole run, so a stop "two ATR from the
entry" could not be written: the distance a volatility stop wants moves from
trade to trade. The three exit orders now take ``signal="name"`` instead of
``pct``, read the series on the bar whose signal opened the trade, freeze it
for the life of that trade and apply it to the price the entry actually
filled at.

What is pinned here is the Python surface and the fill log: the levels each
trade got, that they differ from one another, that the exits are counted as
stops, and that a series holding nothing usable arms no bracket and says so.
Parity with the lite and fast kernels is pinned by the Rust suites, which can
see which kernel ran.
"""
import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
from manifoldbt.expr import col, lit, when
from manifoldbt.helpers import Interval, Slippage
from manifoldbt.indicators import atr, close, sma
from manifoldbt._convert import trades_arrays
from manifoldbt._trades import round_trips

N_BARS = 3000
CAPITAL = 10_000.0
EXIT_STOP_LOSS = 1  # ExitReason::StopLoss
ATR_PERIOD = 14


def _bars(seed: int = 7) -> pd.DataFrame:
    """Bars whose range swings widely, so the ATR distance really varies."""
    rng = np.random.default_rng(seed)
    step = rng.normal(0.0, 0.006, N_BARS)
    px = 100.0 * np.exp(np.cumsum(step))
    # A slow volatility cycle: the true range moves by a factor of ~4 over the
    # run, which is what makes a per-trade distance different from a constant.
    wobble = 0.002 + 0.006 * (1.0 + np.sin(np.arange(N_BARS) / 180.0)) / 2.0
    ts = pd.date_range("2023-01-01", periods=N_BARS, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": px,
        "high": px * (1.0 + wobble),
        "low": px * (1.0 - wobble),
        "close": px,
        "volume": 1000.0,
    })


_DF = _bars()
_END_NS = int(_DF["timestamp"].iloc[-1].value) + 86_400_000_000_000


def _atr_wilder(df: pd.DataFrame, period: int) -> np.ndarray:
    """The engine's ATR, transcribed operation for operation.

    Wilder's smoothing seeded on the mean of the first `period` true ranges,
    NaN before that. Recomputing it here rather than reading it back from the
    engine is the point: the level each trade got has to match a number this
    test derived on its own.
    """
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    out = np.full(len(h), np.nan)
    if len(h) <= period:
        return out
    tr_sum = 0.0
    for i in range(1, period + 1):
        tr_sum += max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
    value = tr_sum / period
    out[period] = value
    for i in range(period + 1, len(h)):
        tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
        value = (value * (period - 1) + tr) / period
        out[i] = value
    return out


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("exit_distance_signal")
    return bt.import_dataframe(
        _DF, symbol="TEST", symbol_id=1, interval="1h",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )


def _config() -> bt.BacktestConfig:
    # Sized off initial capital and filled at the close, so a trade's level is
    # a plain function of its own entry price.
    return bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=_END_NS,
        bar_interval=Interval.hours(1), initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=1, execution_price="AtClose", max_position_pct=10.0,
            allow_short=False, position_sizing_mode="FractionOfInitialCapital",
        ),
        fees=bt.FeeConfig(taker_fee_bps=0.0, maker_fee_bps=0.0),
        slippage=Slippage.none(), warmup_bars=60,
    )


def _long_when_trending(name: str) -> bt.Strategy:
    return (
        bt.Strategy.create(name)
        .signal("fast", sma(col("close"), 8))
        .signal("slow", sma(col("close"), 40))
        .size(when(col("fast") > col("slow"), 1.0, 0.0))
    )


# ---------------------------------------------------------------------------
# The user's stop, in three lines
# ---------------------------------------------------------------------------

def test_two_atr_stop_gives_every_trade_its_own_level(store):
    """The stop the request was about, written as three lines of DSL.

    Every trade's stop level is checked against
    ``fill_price * (1 - dist_at_entry / 100)`` one trade at a time, with the
    distance recomputed here from the bars. An average would hide exactly the
    bug this feature could have: a level built from the wrong bar.
    """
    dist = lit(2.0) * atr(ATR_PERIOD) / close * lit(100.0)
    strategy = (
        _long_when_trending("atr_stop")
        .signal("stop_dist", dist)
        .stop_loss(signal="stop_dist")
    )
    result = bt.run(strategy, _config(), store)

    assert result.brackets_not_armed == 0, result.warnings
    stats = result.metrics["trade_stats"]
    assert stats["sl_exits"] > 0, "the stop must fire, else this proves nothing"

    trades = trades_arrays(result)
    signal_ns = trades["signal_timestamp"].astype("datetime64[ns]").view(np.int64)
    bar_ns = _DF["timestamp"].to_numpy("datetime64[ns]").view(np.int64)
    row_of_ns = {int(v): i for i, v in enumerate(bar_ns)}

    expected_dist = 2.0 * _atr_wilder(_DF, ATR_PERIOD) / _DF["close"].to_numpy(float) * 100.0

    rt = round_trips(result, include_open=False)
    stopped = rt["exit_reason"] == EXIT_STOP_LOSS
    assert stopped.sum() == stats["sl_exits"]

    opens = _DF["open"].to_numpy(float)
    exit_ns = rt["exit_timestamp"].view(np.int64)

    applied, gapped = [], 0
    for entry_row, entry_px, exit_px, exit_at in zip(
        rt["entry_row"][stopped],
        rt["entry_price"][stopped],
        rt["exit_price"][stopped],
        exit_ns[stopped],
    ):
        signal_bar = row_of_ns[int(signal_ns[entry_row])]
        d = expected_dist[signal_bar]
        assert np.isfinite(d)
        level = entry_px * (1.0 - d / 100.0)
        # A stop is a market order once touched, so a bar that OPENED below the
        # level fills at the open. That rule predates this feature and is the
        # same one a `pct` stop obeys.
        bar_open = opens[row_of_ns[int(exit_at)]]
        expected_fill = bar_open if bar_open < level else level
        gapped += bar_open < level
        assert exit_px == pytest.approx(expected_fill, rel=1e-9, abs=1e-9), (
            f"trade entered at {entry_px} on bar {signal_bar}: stop filled at "
            f"{exit_px}, expected fill_price * (1 - {d} / 100) = {level}"
        )
        applied.append(d)

    applied = np.asarray(applied)
    assert len(applied) >= 10, f"only {len(applied)} stopped trades to check"
    # Most of them landed ON the level, so the check above is about the level
    # and not about the gap rule.
    assert gapped < len(applied) / 2, f"{gapped} of {len(applied)} stops gapped"
    # The distance really is per-trade, not one number wearing a signal's name.
    assert applied.max() / applied.min() > 1.5, (
        f"distances barely moved: {applied.min():.3f}% to {applied.max():.3f}%"
    )


def test_a_signal_distance_is_not_the_same_backtest_as_a_constant(store):
    """Vacuity guard: the feature has to change the result it is applied to."""
    dist = lit(2.0) * atr(ATR_PERIOD) / close * lit(100.0)
    varying = bt.run(
        _long_when_trending("varying").signal("stop_dist", dist).stop_loss(
            signal="stop_dist"),
        _config(), store,
    )
    constant = bt.run(
        _long_when_trending("constant").stop_loss(pct=2.0), _config(), store)
    assert varying.metrics["total_return"] != pytest.approx(
        constant.metrics["total_return"], rel=1e-9)


def test_a_constant_signal_matches_the_same_pct(store):
    """A signal holding one number is the old order, to the last decimal."""
    by_signal = bt.run(
        _long_when_trending("by_signal").signal("stop_dist", lit(2.0)).stop_loss(
            signal="stop_dist"),
        _config(), store,
    )
    by_pct = bt.run(
        _long_when_trending("by_pct").stop_loss(pct=2.0), _config(), store)

    a, b = trades_arrays(by_signal), trades_arrays(by_pct)
    np.testing.assert_array_equal(a["exit_reason"], b["exit_reason"])
    np.testing.assert_array_equal(a["fill_price"], b["fill_price"])
    assert by_signal.metrics["total_return"] == by_pct.metrics["total_return"]


# ---------------------------------------------------------------------------
# The empty case
# ---------------------------------------------------------------------------

def test_an_unreadable_distance_arms_nothing_and_says_so(store):
    """A warm-up hole is the everyday cause, and it must never pass in silence.

    ``atr(400)`` is NaN for its first 400 bars while ``warmup_bars`` lets the
    strategy trade from bar 60, so the first entries ask for a distance the
    series cannot give.
    """
    dist = lit(2.0) * atr(400) / close * lit(100.0)
    strategy = (
        _long_when_trending("warmup_hole")
        .signal("stop_dist", dist)
        .stop_loss(signal="stop_dist")
    )
    result = bt.run(strategy, _config(), store)

    assert result.brackets_not_armed > 0
    named = [w for w in result.warnings if "NO bracket" in w and "stop_loss" in w]
    assert named, result.warnings


def test_the_counter_is_zero_when_every_trade_gets_its_bracket(store):
    result = bt.run(
        _long_when_trending("clean").signal("stop_dist", lit(2.0)).stop_loss(
            signal="stop_dist"),
        _config(), store,
    )
    assert result.brackets_not_armed == 0


# ---------------------------------------------------------------------------
# What the builder accepts, and what it says when it does not
# ---------------------------------------------------------------------------

def test_the_json_keeps_a_number_a_number_and_a_name_a_name():
    assert bt.Strategy.create("s").stop_loss(pct=2.0).to_json_dict()["orders"] == {
        "stop_loss": {"stop_pct": 2.0}
    }
    orders = (
        bt.Strategy.create("s")
        .stop_loss(signal="d", side="short")
        .take_profit(signal="d")
        .trailing_stop(signal="d", use_high=False)
        .to_json_dict()["orders"]
    )
    assert orders == {
        "stop_loss": {"stop_pct": "d", "side": "Short"},
        "take_profit": {"profit_pct": "d"},
        "trailing_stop": {"trail_pct": "d", "use_high": False},
    }


@pytest.mark.parametrize("method", ["stop_loss", "take_profit", "trailing_stop"])
def test_an_expression_is_refused_by_naming_the_way_out(method):
    """The frequent slip: handing over the expression instead of its name."""
    strategy = bt.Strategy.create("s")
    with pytest.raises(TypeError) as err:
        getattr(strategy, method)(signal=atr(14) / close)
    msg = str(err.value)
    assert "NAME of a signal" in msg
    assert '.signal("stop_dist", <expression>)' in msg
    assert method in msg


@pytest.mark.parametrize("method", ["stop_loss", "take_profit", "trailing_stop"])
@pytest.mark.parametrize("bad", [-2.0, 0.0, float("nan")])
def test_a_distance_that_is_not_a_positive_percentage_is_refused(method, bad):
    strategy = bt.Strategy.create("s")
    with pytest.raises(ValueError) as err:
        getattr(strategy, method)(pct=bad)
    assert "finite positive percentage" in str(err.value)


@pytest.mark.parametrize("method", ["stop_loss", "take_profit", "trailing_stop"])
def test_an_empty_signal_name_is_refused(method):
    strategy = bt.Strategy.create("s")
    with pytest.raises(ValueError, match="non-empty signal name"):
        getattr(strategy, method)(signal="   ")


@pytest.mark.parametrize("method", ["stop_loss", "take_profit", "trailing_stop"])
def test_pct_and_signal_are_exclusive(method):
    strategy = bt.Strategy.create("s")
    with pytest.raises(ValueError, match="exactly one of pct="):
        getattr(strategy, method)(pct=2.0, signal="d")
    with pytest.raises(ValueError, match="exactly one of pct="):
        getattr(strategy, method)()


def test_an_exit_signal_the_strategy_does_not_define_fails_the_run(store):
    strategy = _long_when_trending("missing").stop_loss(signal="nowhere")
    with pytest.raises(Exception) as err:
        bt.run(strategy, _config(), store)
    assert "nowhere" in str(err.value)
