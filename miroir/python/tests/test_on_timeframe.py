"""Tests for tf(..).apply(..) — indicators evaluated ON the higher timeframe.

The defect this feature fixes, pinned by `test_apply_differs_from_staircase`:
an indicator over a step-held `tf()` column counts its period in SIMULATION
bars, so `sma(tf("1h").close, 20)` on a 1m simulation is a 20-MINUTE smoothing
of an hourly staircase — mid-hour it equals the previous hourly close exactly.
`tf("1h").apply(sma(close, 20))` is the true 20-HOUR average.

The reference implementation (`_hand_band`) is the exo-column recipe users had
to build by hand before this feature: resample to 1h in pandas, indicator on
the 1h grid, shift(1) (a closed bar is readable from the next bar on), ffill
onto the 1m grid. `test_apply_matches_hand_rolled_exo` demands bit-identical
metrics against it.
"""
import os

import pytest

import manifoldbt as bt
from manifoldbt.indicators import close, sma

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

N_BARS = 20_000  # ~13.9 days of 1m bars -> ~333 hourly bars
PERIOD = 20


def _bars_df():
    ts = pd.date_range("2022-01-01", periods=N_BARS, freq="1min", tz="UTC")
    rng = np.random.default_rng(3)
    px = 100.0 + np.cumsum(np.sin(np.arange(N_BARS) / 700.0) * 0.5 + rng.normal(0, 0.4, N_BARS)) * 0.01
    px = np.maximum(px, 1.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": px,
            "high": px * 1.0005,
            "low": px * 0.9995,
            "close": px,
            "volume": [1000.0] * N_BARS,
        }
    )


def _store(tmp_path, df):
    root = tmp_path / "otf_store"
    return bt.import_dataframe(
        df,
        symbol="ZT",
        symbol_id=1,
        interval="1m",
        asset_class="equity",
        exchange="TEST",
        data_root=str(root / "data"),
        metadata_db=str(root / "metadata.sqlite"),
    ), str(root / "data")


def _hand_band(df, period):
    """The pre-feature recipe: hourly SMA built by hand, no lookahead."""
    h1 = df.set_index("timestamp")["close"].resample("1h").last()
    band = h1.rolling(period).mean().shift(1)
    return band.reindex(df["timestamp"], method="ffill")


def _config(tmp_path=None, exo=False):
    start, end = bt.time_range("2022-01-01", "2022-01-14")
    cfg = bt.BacktestConfig(
        universe=[1],
        time_range_start=start,
        time_range_end=end,
        initial_capital=10_000.0,
        provider="TEST",
        bar_interval=bt.Interval.minutes(1),
        symbol_names={"ZT": 1},
        extra_timeframes={} if exo else {"1h": bt.Interval.hours(1)},
        exo_data=["hand_band"] if exo else [],
    )
    cfg.warmup_bars = 0
    return cfg


def _strategy(band_expr, name):
    return (
        bt.Strategy.create(name)
        .signal("band", band_expr)
        .size(bt.when(close > bt.col("band"), 1.0, 0.0))
    )


def test_serializes_as_on_timeframe():
    e = bt.tf("1h").apply(sma(close, PERIOD))
    payload = e.to_json()
    assert list(payload) == ["OnTimeframe"]
    label, inner = payload["OnTimeframe"]
    assert label == "1h"
    assert list(inner) == ["RollingMean"]


def test_apply_matches_hand_rolled_exo(tmp_path):
    """The money test: tf("1h").apply(sma(close, 20)) must be bit-identical to
    the hand-precomputed hourly-SMA exo column it replaces."""
    df = _bars_df()
    store, data_root = _store(tmp_path, df)

    band = _hand_band(df, PERIOD)
    bt.register_exo(
        "hand_band",
        pd.DataFrame({"timestamp": df["timestamp"], "value": band.values}),
        data_root=data_root,
    )

    native = bt.run(
        _strategy(bt.tf("1h").apply(sma(close, PERIOD)), "native"),
        _config(),
        store,
    )
    hand = bt.run(
        _strategy(bt.exo("hand_band", "value"), "hand"),
        _config(exo=True),
        store,
    )

    for key in ("total_return", "sharpe", "max_drawdown", "volatility"):
        assert native.metrics.get(key) == hand.metrics.get(key), (
            f"{key}: native {native.metrics.get(key)} != hand {hand.metrics.get(key)}"
        )
    assert len(native.trades_df()) == len(hand.trades_df())


def test_apply_differs_from_staircase(tmp_path):
    """Guard against regressing to the old semantics: the staircase version
    (indicator over the step-held tf() column) must NOT equal apply()."""
    df = _bars_df()
    store, _ = _store(tmp_path, df)

    applied = bt.run(
        _strategy(bt.tf("1h").apply(sma(close, PERIOD)), "applied"),
        _config(),
        store,
    )
    staircase = bt.run(
        _strategy(sma(bt.tf("1h").close, PERIOD), "staircase"),
        _config(),
        store,
    )
    assert applied.metrics.get("total_return") != staircase.metrics.get("total_return"), (
        "apply() and the staircase smoothing agreed; the coarse-grid evaluation "
        "is not actually happening"
    )


def test_swept_period_matches_fixed_runs(tmp_path):
    """param() INSIDE apply(): each combo must equal the fixed-period run."""
    df = _bars_df()
    store, _ = _store(tmp_path, df)
    periods = [10, 20, 40]

    sweep = bt.run_sweep_lite(
        _strategy(bt.tf("1h").apply(sma(close, bt.param("len"))), "swept"),
        {"len": periods},
        _config(),
        store,
        device="cpu",
    )
    assert len(sweep) == len(periods)

    for got, period in zip(sweep, periods):
        ref = bt.run_sweep_lite(
            _strategy(bt.tf("1h").apply(sma(close, period)), f"fixed_{period}"),
            {},
            _config(),
            store,
            device="cpu",
        )[0]
        for key in ("total_return", "sharpe", "max_drawdown"):
            assert got.metrics.get(key) == ref.metrics.get(key), (
                f"len={period}: {key} diverged"
            )


def test_lite_matches_run(tmp_path):
    df = _bars_df()
    store, _ = _store(tmp_path, df)
    strat = _strategy(bt.tf("1h").apply(sma(close, PERIOD)), "parity")

    full = bt.run(strat, _config(), store)
    lite = bt.run_sweep_lite(strat, {}, _config(), store, device="cpu")[0]
    for key in ("total_return", "sharpe"):
        assert full.metrics.get(key) == lite.metrics.get(key), f"{key} diverged"
    # max_drawdown carries a pre-existing ~1e-16 run-vs-lite float-noise gap
    # (measured on a plain sma(close, 20) strategy with no OnTimeframe on this
    # same data), so exact equality would pin the wrong thing here.
    a, b = full.metrics["max_drawdown"], lite.metrics["max_drawdown"]
    assert a == pytest.approx(b, rel=1e-12), f"max_drawdown diverged: {a} vs {b}"


def test_missing_extra_timeframe_is_a_clear_error(tmp_path):
    df = _bars_df()
    store, _ = _store(tmp_path, df)
    cfg = _config()
    cfg.extra_timeframes = {}
    with pytest.raises(Exception, match="extra_timeframes"):
        bt.run(_strategy(bt.tf("1h").apply(sma(close, PERIOD)), "no_tf"), cfg, store)
