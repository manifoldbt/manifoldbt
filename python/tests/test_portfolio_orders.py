"""run_portfolio honors the orders each strategy carries.

Orders once lived on the shared execution config, and portfolio mode could
not apply them per strategy: ``Portfolio.strategy()`` warned that they were
ignored. They now travel inside the strategy JSON (``StrategyDef.orders``) and
the portfolio runner resolves them per strategy the same way ``run`` does. The
warning outlived the fix and sent a user down a dead end (a long+short
strategy that wanted a stop on its shorts only), so the behaviour is pinned
here on the real engine: no warning, and a stop on one leg changes that leg
and only that leg.

Each leg runs on ``initial_capital * weight``; sized off initial capital, the
portfolio equals the sum of the two legs run alone, bit for bit. That
equality is what lets the per-leg exit reasons be read from separate runs:
``round_trips`` on the combined result pairs fills by symbol and would mix
the legs.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
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
    root = tmp_path_factory.mktemp("portfolio_orders")
    return bt.import_dataframe(
        _DF, symbol="TEST", symbol_id=1, interval="1h",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )


def _config(capital: float, *, sizing: str = "FractionOfInitialCapital",
            max_position_pct: float = 10.0) -> bt.BacktestConfig:
    return bt.BacktestConfig(
        universe=[1], time_range_start=0, time_range_end=_END_NS,
        bar_interval=Interval.hours(1), initial_capital=capital,
        execution=bt.ExecutionConfig(
            signal_delay=1, execution_price="AtOpen",
            max_position_pct=max_position_pct, allow_short=True,
            position_sizing_mode=sizing,
        ),
        fees=bt.FeeConfig(taker_fee_bps=5.0, maker_fee_bps=5.0),
        slippage=Slippage.none(), warmup_bars=45,
    )


_FAST, _SLOW = sma(col("close"), 8), sma(col("close"), 40)


def _leg(name: str, size, stop_pct=None) -> bt.Strategy:
    s = bt.Strategy.create(name).signal("d", col("close")).size(size)
    return s.stop_loss(pct=stop_pct) if stop_pct is not None else s


def _long_leg(size: float = 0.6, stop_pct=None):
    return _leg("long_leg", when(_FAST > _SLOW, size, 0.0), stop_pct)


def _short_leg(size: float = 0.6, stop_pct=None):
    return _leg("short_leg", when(_FAST > _SLOW, 0.0, -size), stop_pct)


def _equity(result) -> np.ndarray:
    return np.asarray(result.equity_curve, dtype=np.float64)


def test_strategy_with_orders_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bt.Portfolio().strategy(_short_leg(stop_pct=2.0), weight=0.5)


def test_stop_on_one_leg_reaches_that_leg_only(store):
    half = _config(CAPITAL * 0.5)
    long_alone = bt.run(_long_leg(), half, store)
    short_stopped = bt.run(_short_leg(stop_pct=2.0), half, store)
    short_free = bt.run(_short_leg(), half, store)

    portfolio = bt.run_portfolio(
        bt.Portfolio()
        .strategy(_long_leg(), weight=0.5)
        .strategy(_short_leg(stop_pct=2.0), weight=0.5),
        _config(CAPITAL), store,
    )

    # The portfolio is the two legs, each run with its own orders.
    np.testing.assert_array_equal(
        _equity(portfolio), _equity(long_alone) + _equity(short_stopped))
    # The stop did something: the stopped short leg is not the free one.
    assert not np.array_equal(_equity(short_stopped), _equity(short_free))

    # Every stop exit sits on the short leg; the long leg has none.
    rt_short = round_trips(short_stopped, include_open=False)
    assert (rt_short["side"] == SIDE_SHORT).all()
    assert (rt_short["exit_reason"] == EXIT_STOP_LOSS).sum() > 0
    rt_long = round_trips(long_alone, include_open=False)
    assert (rt_long["side"] == SIDE_LONG).all()
    assert (rt_long["exit_reason"] == EXIT_STOP_LOSS).sum() == 0


def test_portfolio_equals_single_strategy_without_compounding(store):
    """The split into legs is exact when nothing depends on a leg's equity.

    Weights of 0.5 give each leg half the capital, so the legs size 0.6 to
    hold what the single strategy holds at 0.3; ``max_position_pct`` is
    raised because it clamps on the leg's equity.
    """
    both = _leg("both", when(_FAST > _SLOW, 0.3, -0.3))
    split = (bt.Portfolio()
             .strategy(_long_leg(0.6), weight=0.5)
             .strategy(_short_leg(0.6), weight=0.5))

    single = bt.run(both, _config(CAPITAL), store)
    portfolio = bt.run_portfolio(split, _config(CAPITAL), store)
    # Same fills, same P&L; only the rounding order differs (two legs of
    # 5000 + x and 5000 + y summed, against one accumulation of 10000 + x + y),
    # which shows as a last-ulp gap on a few bars.
    np.testing.assert_allclose(_equity(portfolio), _equity(single), rtol=1e-12)

    # Compounding breaks it: each leg then compounds on its own equity.
    single_c = bt.run(both, _config(CAPITAL, sizing="FractionOfEquity"), store)
    portfolio_c = bt.run_portfolio(
        split, _config(CAPITAL, sizing="FractionOfEquity"), store)
    assert not np.allclose(_equity(portfolio_c), _equity(single_c), rtol=1e-9)
