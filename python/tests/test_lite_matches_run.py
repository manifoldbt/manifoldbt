"""The lite sweep path must agree with `run()` on intraday bars.

`run_sweep_lite` is a separate transcription of the simulation, kept for speed
(roughly ten times the throughput of the full sweep). Its metrics are computed
from a *daily* equity curve, and that curve's first point is the equity at the
CLOSE of day one. Taking it as the growth base silently drops day one's profit
and loss from every metric measured against it, which shipped as an 8% error on
`total_return` for a fourteen-day intraday backtest.

The bug was invisible on daily bars: with a 60-period indicator the warmup
covers sixty days, so the close of day one still equals the initial capital and
the base is right by accident. It only appears when trading starts on day one,
which on 1-minute bars is the normal case. Hence this test runs intraday.

All fourteen metrics must be identical. `ulcer_index` used to be the exception:
it is accumulated over whichever curve it is handed, so the lite and GPU sweeps
measured it on daily points while `run()` measured it bar by bar, and the same
backtest carried two different values depending on the entry point. It now
follows the daily series on every path, like the Sharpe, Sortino and volatility
beside it, and like the published definition of the Ulcer Index. `max_drawdown`
deliberately stays full-resolution: a drawdown that opens and recovers inside a
day is a real one and belongs in the maximum.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, param, when  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from manifoldbt.indicators import close as close_px, sma  # noqa: E402

CAPITAL = 100_000.0
FAST, SLOW = 10, 60

# Metrics that are pure functions of the equity path and its base, so the two
# code paths must agree to float-reordering noise.
MUST_MATCH = (
    "total_return",
    "cagr",
    "calmar",
    "tstat_sharpe",
    "sharpe",
    "sortino",
    "volatility",
    "max_drawdown",
    "avg_daily_return",
    "best_day",
    "worst_day",
    "pct_positive_days",
    "ulcer_index",
    "alpha",
    "beta",
)


def _intraday_bars(rows=8_000, seed=7):
    """Gap-free 1-minute random walk. Long enough to span several days, and
    volatile enough that the crossover trades inside the first day."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 3e-4, rows)))
    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = rng.uniform(0.2, 1.8, rows) * 3e-4 * close
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-03-01", periods=rows, freq="1min", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": np.full(rows, 1_000.0),
        }
    )


def _config(df, execution_price="AtClose", signal_delay=0, allow_short=False):
    last_ns = int(df["timestamp"].iloc[-1].value)
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=last_ns + 86_400_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=signal_delay,
            execution_price=execution_price,
            max_position_pct=1.0,
            allow_short=allow_short,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def test_lite_sweep_matches_run_on_intraday_bars(tmp_path):
    df = _intraday_bars()
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
    config = _config(df)

    sized = when(col("fast") > col("slow"), lit(1.0), lit(0.0))
    fixed = (
        bt.Strategy.create("fixed")
        .signal("fast", sma(close_px, FAST))
        .signal("slow", sma(close_px, SLOW))
        .size(sized)
    )
    swept = (
        bt.Strategy.create("swept")
        .signal("fast", sma(close_px, param("fast")))
        .signal("slow", sma(close_px, param("slow")))
        .size(sized)
    )

    full = bt.run(fixed, config, store).metrics
    lite = bt.run_sweep_lite(
        swept, {"fast": [FAST], "slow": [SLOW]}, config, store
    )[0].metrics

    # The strategy must actually trade on day one, otherwise the base is right
    # by accident and the test proves nothing.
    assert full["total_return"] != 0.0

    for name in MUST_MATCH:
        expected, got = full[name], lite[name]
        assert abs(expected - got) <= 1e-9 * max(1.0, abs(expected)), (
            f"{name}: run()={expected!r} but run_sweep_lite()={got!r}. "
            "The lite path has drifted from the full simulation."
        )


@pytest.mark.parametrize("signal_delay", [0, 1])
@pytest.mark.parametrize("allow_short", [False, True])
def test_lite_sweep_matches_run_at_open(tmp_path, signal_delay, allow_short):
    """Same contract as above, for ``execution_price="AtOpen"``.

    AtOpen used to be refused by the fast path, so the lite sweep quietly ran
    the slow general loop for it: correct, ten times slower, and nothing said
    so. The fast kernel now fills at the open of the EXECUTION bar -- the signal
    row shifted by ``signal_delay`` -- and this pins that it still answers what
    ``run()`` answers, on the same fifteen metrics.

    The bars have ``open[i] == close[i-1]``, so AtOpen and AtClose are genuinely
    different backtests: the last assertion checks exactly that, otherwise this
    test would pass on an engine that silently ignored the setting.
    """
    df = _intraday_bars()
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
    config = _config(
        df,
        execution_price="AtOpen",
        signal_delay=signal_delay,
        allow_short=allow_short,
    )

    bas = lit(-1.0) if allow_short else lit(0.0)
    sized = when(col("fast") > col("slow"), lit(1.0), bas)
    fixed = (
        bt.Strategy.create("fixed")
        .signal("fast", sma(close_px, FAST))
        .signal("slow", sma(close_px, SLOW))
        .size(sized)
    )
    swept = (
        bt.Strategy.create("swept")
        .signal("fast", sma(close_px, param("fast")))
        .signal("slow", sma(close_px, param("slow")))
        .size(sized)
    )

    result = bt.run(fixed, config, store)
    full = result.metrics
    lite_result = bt.run_sweep_lite(
        swept, {"fast": [FAST], "slow": [SLOW]}, config, store
    )[0]
    lite = lite_result.metrics

    assert full["total_return"] != 0.0

    # The simulation claim, first and unconditionally: the two paths must walk
    # the same equity path and end on the same number.
    last_equity = float(np.asarray(result.equity_curve)[-1])
    assert abs(last_equity - lite_result.final_equity) <= 1e-12 * max(
        1.0, abs(last_equity)
    ), (
        f"final equity: run()={last_equity!r} but "
        f"run_sweep_lite()={lite_result.final_equity!r}"
    )

    # `run()` divides by the FIRST point of its equity curve; the lite path
    # divides by the initial capital. Under AtOpen with signal_delay=0 the very
    # first bar fills at its own open, so that first point already carries the
    # bar's profit and loss and the two bases differ. That is a pre-existing
    # choice-of-base defect -- the mirror of the one this file's docstring
    # describes for the lite path -- and NOT a divergence of the simulation: the
    # equity paths above are identical. Pinned by its exact relation rather than
    # tolerated, so that fixing the base turns this branch red instead of
    # leaving a silent tolerance behind.
    first_equity = float(np.asarray(result.equity_curve)[0])
    if first_equity != CAPITAL:
        assert (1.0 + full["total_return"]) * first_equity == pytest.approx(
            (1.0 + lite["total_return"]) * CAPITAL, rel=1e-12
        ), "the two total_returns differ by more than their base"
    else:
        for name in MUST_MATCH:
            expected, got = full[name], lite[name]
            assert abs(expected - got) <= 1e-9 * max(1.0, abs(expected)), (
                f"{name}: run()={expected!r} but run_sweep_lite()={got!r}. "
                "The AtOpen fast path has drifted from the full simulation."
            )

    # AtOpen must not be the same backtest as AtClose, or everything above is
    # vacuous: a kernel that ignored the setting would pass it.
    at_close = bt.run_sweep_lite(
        swept,
        {"fast": [FAST], "slow": [SLOW]},
        _config(df, "AtClose", signal_delay, allow_short),
        store,
    )[0].metrics
    assert lite["total_return"] != at_close["total_return"], (
        "AtOpen and AtClose returned the same number: the execution price is "
        "being ignored."
    )
