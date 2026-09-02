"""Round-trip pairing and the two trade-level charts.

The pairing is pinned to the engine's own ``trade_stats`` rather than to
hand-written expectations: if ``round_trips()`` ever drifts from
``bt_analytics::build_round_trips``, the count, win rate and expectancy stop
agreeing and these fail. The chart tests assert at the figure-spec level,
the same way ``test_plot_charts.py`` does.
"""
import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt
from manifoldbt.expr import col, when
from manifoldbt.helpers import Interval, Slippage
from manifoldbt.indicators import sma
from manifoldbt._trades import EXIT_REASON_OPEN, round_trips

N_BARS = 600
CAPITAL = 10_000.0


def _bars(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, N_BARS))
    ts = pd.date_range("2023-01-01", periods=N_BARS, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1000.0,
    })


def _run(tmp_path, name, strat, *, allow_short=False, fees=None):
    df = _bars()
    root = tmp_path / name
    root.mkdir()
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1h",
        data_root=str(root / "data"), metadata_db=str(root / "meta.sqlite"),
    )
    cfg = bt.BacktestConfig(
        universe=[1], time_range_start=0,
        time_range_end=int(df["timestamp"].iloc[-1].value) + 86_400_000_000_000,
        bar_interval=Interval.hours(1), initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=1, execution_price="AtOpen", max_position_pct=1.0,
            allow_short=allow_short,
        ),
        fees=fees if fees is not None else bt.FeeConfig(taker_fee_bps=5.0, maker_fee_bps=5.0),
        slippage=Slippage.none(), warmup_bars=20,
    )
    return bt.run(strat, cfg, store)


@pytest.fixture
def bracket_result(tmp_path):
    """Long-only SMA cross with a stop and a take-profit: three exit reasons."""
    fast, slow = sma(col("close"), 5), sma(col("close"), 20)
    strat = (bt.Strategy.create("rt_bracket").signal("d", col("close"))
             .size(when(fast > slow, 1.0, 0.0)).stop_loss(pct=2.0).take_profit(pct=4.0))
    return _run(tmp_path, "bracket", strat)


@pytest.fixture
def flip_result(tmp_path):
    """Always in the market, long or short: every signal change is a flip."""
    fast, slow = sma(col("close"), 5), sma(col("close"), 20)
    strat = (bt.Strategy.create("rt_flip").signal("d", col("close"))
             .size(when(fast > slow, 1.0, -1.0)))
    return _run(tmp_path, "flip", strat, allow_short=True)


def _stats(result):
    ts = result.metrics["trade_stats"]
    return ts["round_trips"], ts["win_rate"], ts["expectancy"], ts["avg_win"], ts["avg_loss"]


def _assert_matches_trade_stats(result):
    n_rt, win_rate, expectancy, avg_win, avg_loss = _stats(result)
    rt = round_trips(result, include_open=False)
    pnl = rt["pnl"]
    assert len(pnl) == n_rt
    assert not rt["is_open"].any()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    assert win_rate == pytest.approx(len(wins) / n_rt, abs=1e-12)
    assert expectancy == pytest.approx(pnl.sum() / n_rt, rel=1e-9)
    if len(wins):
        assert avg_win == pytest.approx(wins.mean(), rel=1e-9)
    if len(losses):
        assert avg_loss == pytest.approx(losses.mean(), rel=1e-9)
    return rt


def test_round_trips_match_trade_stats_with_brackets(bracket_result):
    rt = _assert_matches_trade_stats(bracket_result)
    assert bracket_result.trade_count > 0
    # Fees are attributed: with 5 bps both ways no trade is fee-free.
    assert (rt["fees"] > 0).all()
    # The exit code is the exit fill's, so the bracket reasons come through.
    assert set(np.unique(rt["exit_reason"])) <= {0, 1, 2, 3}
    assert 1 in rt["exit_reason"] or 2 in rt["exit_reason"]
    # Rows point back into the fill log and are ordered.
    assert (rt["exit_row"] > rt["entry_row"]).all()
    assert (rt["holding_seconds"] > 0).all()
    assert (rt["side"] == 1).all()


def test_round_trips_match_trade_stats_through_flips(flip_result):
    rt = _assert_matches_trade_stats(flip_result)
    assert {1, 2} <= set(np.unique(rt["side"]))
    # A flip closes at the flip fill and reopens there: consecutive round
    # trips on one symbol share the boundary timestamp.
    boundary = rt["exit_timestamp"][:-1] == rt["entry_timestamp"][1:]
    assert boundary.all()


def test_open_position_is_marked_to_the_last_bar(flip_result):
    closed = round_trips(flip_result, include_open=False)
    with_open = round_trips(flip_result, include_open=True)
    assert len(with_open) == len(closed)
    n_open = int(with_open["is_open"].sum())
    assert len(with_open["pnl"]) == len(closed["pnl"]) + n_open
    if n_open:
        last = with_open["is_open"]
        assert (with_open["exit_reason"][last] == EXIT_REASON_OPEN).all()
        assert (with_open["exit_row"][last] == -1).all()
        pos = flip_result.positions_df()
        assert with_open["exit_price"][last][0] == pytest.approx(float(pos["close"].iloc[-1]))


def test_round_trips_df_backends(bracket_result):
    df = bracket_result.round_trips_df(backend="pandas")
    assert list(df.columns)[:4] == ["symbol_id", "entry_timestamp", "exit_timestamp", "side"]
    assert len(df) == len(round_trips(bracket_result)["pnl"])
    pl = pytest.importorskip("polars")
    pdf = bracket_result.round_trips_df(backend="polars")
    assert isinstance(pdf, pl.DataFrame) and pdf.shape == df.shape


# ── Charts ───────────────────────────────────────────────────────────────────

pytest.importorskip("plotly")


def _traces(fig):
    return {t.name: t for t in fig.data}


def test_trades_chart_draws_entries_and_outcome_coloured_exits(bracket_result):
    from manifoldbt.plot import trades
    from manifoldbt.plot._theme import GREEN, RED

    fig = trades(bracket_result, show=False)
    tr = _traces(fig)
    assert "Close" in tr and "Entry" in tr
    rt = round_trips(bracket_result)
    assert len(tr["Entry"].x) == len(rt["pnl"])
    n_win = int(((rt["pnl"] > 0) & ~rt["is_open"]).sum())
    n_loss = int(((rt["pnl"] <= 0) & ~rt["is_open"]).sum())
    if n_win:
        assert tr["Exit (profit)"].marker.color == GREEN
        assert len(tr["Exit (profit)"].x) == n_win
    if n_loss:
        assert tr["Exit (loss)"].marker.color == RED
        assert len(tr["Exit (loss)"].x) == n_loss
    # Every trace in the legend carries a name.
    assert all(t.name for t in fig.data)
    # Zones are on by default at this size: one vrect per closed round trip.
    assert len(fig.layout.shapes) == n_win + n_loss + int(rt["is_open"].sum())


def test_trades_chart_zones_can_be_disabled(bracket_result):
    from manifoldbt.plot import trades

    fig = trades(bracket_result, zones=False, show=False)
    assert len(fig.layout.shapes) == 0


def test_trade_pnl_scatter_scales_markers_and_switches_axis(bracket_result):
    from manifoldbt.plot import trade_pnl

    rt = round_trips(bracket_result)
    fig = trade_pnl(bracket_result, show=False)
    ys = np.concatenate([np.asarray(t.y, dtype=float) for t in fig.data])
    assert np.allclose(np.sort(ys), np.sort(rt["pnl"]))
    sizes = np.concatenate([np.asarray(t.marker.size, dtype=float) for t in fig.data])
    assert sizes.min() >= 6.0 - 1e-9 and sizes.max() <= 14.0 + 1e-9
    assert sizes.max() == pytest.approx(14.0)
    assert fig.layout.yaxis.title.text.startswith("PnL")

    fig_pct = trade_pnl(bracket_result, pct_scale=True, show=False)
    ys = np.concatenate([np.asarray(t.y, dtype=float) for t in fig_pct.data])
    assert np.allclose(np.sort(ys), np.sort(rt["return_pct"]))
    assert fig_pct.layout.yaxis.title.text == "Return"
    assert fig_pct.layout.yaxis.tickformat.endswith("%")


def test_result_plot_dispatch_knows_the_trade_charts(bracket_result):
    assert bracket_result.plot("trades", show=False) is not None
    assert bracket_result.plot("trade_pnl", show=False, pct_scale=True) is not None
