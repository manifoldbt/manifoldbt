"""Exit orders armed on one side only.

A strategy that trades both directions could not stop its shorts without
stopping its longs: ``StopLossConfig`` carried a distance and nothing else,
and the bracket mirrored it on every position. ``side="short"`` (or
``"long"``) now arms the order on that side and leaves the other bare, on
every path that reads the distance. These tests pin the Python surface and
the general loop on the real engine; the lite path and the GPU kernel are
pinned against the general loop by the Rust parity suites.
"""
import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
from manifoldbt.config import OrderConfig
from manifoldbt.expr import col, when
from manifoldbt.helpers import Interval, Slippage
from manifoldbt.indicators import sma
from manifoldbt._trades import round_trips

N_BARS = 3000
CAPITAL = 10_000.0
EXIT_STOP_LOSS = 1  # ExitReason::StopLoss
SIDE_LONG, SIDE_SHORT = 1, 2


def _bars(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.006, N_BARS)))
    ts = pd.date_range("2023-01-01", periods=N_BARS, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close * 1.004,
        "low": close * 0.996, "close": close, "volume": 1000.0,
    })


_DF = _bars()
_END_NS = int(_DF["timestamp"].iloc[-1].value) + 86_400_000_000_000


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("exit_orders_side")
    return bt.import_dataframe(
        _DF, symbol="TEST", symbol_id=1, interval="1h",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )


def _config() -> bt.BacktestConfig:
    # Sized off initial capital, so what a stop does to the equity on one
    # side cannot change the size, hence the trades, of the other.
    return bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=_END_NS,
        bar_interval=Interval.hours(1), initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=1, execution_price="AtOpen", max_position_pct=10.0,
            allow_short=True, position_sizing_mode="FractionOfInitialCapital",
        ),
        fees=bt.FeeConfig(taker_fee_bps=5.0, maker_fee_bps=5.0),
        slippage=Slippage.none(), warmup_bars=45,
    )


_FAST, _SLOW = sma(col("close"), 8), sma(col("close"), 40)


def _both_ways(name: str) -> bt.Strategy:
    """Always in the market: long above the slow average, short below."""
    return (bt.Strategy.create(name).signal("d", col("close"))
            .size(when(_FAST > _SLOW, 0.3, -0.3)))


def _trips(result, side):
    rt = round_trips(result, include_open=False)
    keep = rt["side"] == side
    return {k: v[keep] for k, v in rt.items()}


def _same_trades(a, b):
    """Same bars, same prices.

    Fees are not compared: a flip books the closing fees on the closed leg, a
    fresh entry books its own. Prices are compared to 1e-12: the pairing
    averages the entry price of a position opened by a flip (``price * qty /
    qty``), which lands an ulp away from the plain fill price now and then.
    """
    for key in ("entry_timestamp", "exit_timestamp"):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ("entry_price", "exit_price"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12)


def test_default_side_keeps_the_json_as_it_was():
    orders = bt.Strategy.create("s").stop_loss(pct=2.0).to_json_dict()["orders"]
    assert orders == {"stop_loss": {"stop_pct": 2.0}}

    orders = (
        bt.Strategy.create("s")
        .stop_loss(pct=2.0, side="short")
        .take_profit(pct=4.0, side="Long")
        .trailing_stop(pct=3.0, side="both")
        .to_json_dict()["orders"]
    )
    assert orders == {
        "stop_loss": {"stop_pct": 2.0, "side": "Short"},
        "take_profit": {"profit_pct": 4.0, "side": "Long"},
        "trailing_stop": {"trail_pct": 3.0, "use_high": True},
    }
    assert OrderConfig.stop_loss_only(2.0, side="long").stop_loss == {
        "stop_pct": 2.0, "side": "Long",
    }
    with pytest.raises(ValueError, match="side must be"):
        bt.Strategy.create("s").stop_loss(pct=2.0, side="shorts")


@pytest.mark.parametrize(
    "side, stopped, bare",
    [("short", SIDE_SHORT, SIDE_LONG), ("long", SIDE_LONG, SIDE_SHORT)],
)
def test_stop_on_one_side_leaves_the_other_bare(store, side, stopped, bare):
    free = bt.run(_both_ways("free"), _config(), store)
    one_side = bt.run(
        _both_ways("one_side").stop_loss(pct=2.0, side=side), _config(), store)
    both = bt.run(_both_ways("both").stop_loss(pct=2.0), _config(), store)

    # The stopped side has stop exits; the bare side has none and trades
    # exactly as it did without any stop.
    assert (_trips(one_side, stopped)["exit_reason"] == EXIT_STOP_LOSS).sum() > 0
    assert (_trips(one_side, bare)["exit_reason"] == EXIT_STOP_LOSS).sum() == 0
    _same_trades(_trips(one_side, bare), _trips(free, bare))
    # The stopped side trades exactly as it does under a two-sided stop.
    _same_trades(_trips(one_side, stopped), _trips(both, stopped))
    # A two-sided stop does stop the other side: the bare side is a choice.
    assert (_trips(both, bare)["exit_reason"] == EXIT_STOP_LOSS).sum() > 0


def test_swept_stop_distance_keeps_the_side(store):
    """A grid over ``stop_loss`` replaces the distance, not the side."""
    base = _both_ways("swept").stop_loss(pct=1.0, side="short")
    [swept] = bt.run_sweep_lite(
        base, {"stop_loss": [2.0]}, _config(), store, device="cpu")
    direct = bt.run(
        _both_ways("direct").stop_loss(pct=2.0, side="short"), _config(), store)
    two_sided = bt.run(_both_ways("two").stop_loss(pct=2.0), _config(), store)

    assert swept.metrics["total_return"] == pytest.approx(
        direct.metrics["total_return"], rel=1e-12)
    assert swept.metrics["total_return"] != pytest.approx(
        two_sided.metrics["total_return"], rel=1e-9)
