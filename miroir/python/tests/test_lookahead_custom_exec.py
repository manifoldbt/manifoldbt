"""Anti-look-ahead tests for `ExecutionPrice.custom(...)` with `signal_delay=0`.

This is the configuration of `examples/21_fill_at_computed_level.py`, and the
one with the most room for leakage in the whole engine: the fill price is read
from a strategy signal, the order acts on the same bar it was computed on, and
the level itself comes from a higher timeframe. Three chances for a bar to be
priced with information it could not have had.

Two independent methods, because a single one can pass for the wrong reason:

  * **future perturbation** — corrupt every bar after K, re-run, and require
    the equity of bars 0..K to be *bit-identical*. Any decision that read a
    future bar moves the prefix.
  * **the engine's own detector** — `detect_lookahead`, which compares the
    trades of truncated runs against the full run.

Both carry an anti-vacuity guard. A look-ahead test that compares nothing
passes just as loudly as one that compares everything, which is exactly how
the built-in detector reports PASS when its splits fall outside the data.
"""
import os
import tempfile

import pytest

import manifoldbt as bt

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from manifoldbt.helpers import ExecutionPrice, Interval, Slippage  # noqa: E402

# The look-ahead detector is a Pro feature, and the perturbation
# assertions count sub-daily bars: coarsened to the Community daily
# floor they compare 4 bars and conclude nothing.
pytestmark = pytest.mark.skipif(
    bt.license_info()[0] != "Pro",
    reason="requires Pro: skipped on a Community wheel",
)


# Three days of 1-minute bars: enough for the hourly SMA to have a history,
# small enough not to weigh on the suite.
N_BARS = 3 * 1440
SPLIT = 2 * 1440  # perturb everything after this bar


def _mean_reverting_bars(seed=7):
    """The construction of example 21, at a size a test can afford."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.0010, N_BARS)
    level = np.cumsum(steps) * 0.85
    px = 100.0 * np.exp(level - np.linspace(0, level[-1], N_BARS))
    o = px
    c = np.roll(px, -1)
    c[-1] = px[-1]
    amp = np.abs(rng.normal(0.0, 0.0012, N_BARS))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=N_BARS, freq="1min", tz="UTC"),
        "open": o,
        "high": np.maximum(o, c) * (1 + amp),
        "low": np.minimum(o, c) * (1 - amp),
        "close": c,
        "volume": rng.uniform(1_000, 5_000, N_BARS),
    })


def _band_strategy():
    """Short the upper band, cover the lower one, filling ON the band."""
    from manifoldbt.indicators import close, high, low, open as open_px, sma

    h1 = bt.tf("1h")
    band_up = sma(h1.close, 8) * 1.004
    band_dn = sma(h1.close, 8) * 0.997
    touch_up = high >= band_up
    touch_dn = low <= band_dn
    target = bt.when(touch_dn, 0.0, bt.when(touch_up, -1.0))
    exec_level = bt.when(
        touch_dn, bt.when(open_px <= band_dn, open_px, band_dn),
        bt.when(touch_up, bt.when(open_px >= band_up, open_px, band_up), close),
    )
    return (
        bt.Strategy.create("band_touch_short")
        .signal("position", target)
        .signal("exec_level", exec_level)
        .size(target)
        .stop_loss(pct=25.0)
    )


def _store(frame, tmp_path, tag):
    root = os.path.join(str(tmp_path), tag)
    return bt.import_dataframe(
        frame, symbol="SYNTH", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )


def _config(frame):
    """A time range bounded by the DATA, not by the epoch.

    `time_range_start=0` would stretch the range over 54 years, which is what
    makes the built-in detector split outside the data and pass vacuously.
    """
    ts = pd.DatetimeIndex(frame["timestamp"])
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=int(ts[0].value),
        time_range_end=int(ts[-1].value) + 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000,
        execution=bt.ExecutionConfig(
            signal_delay=0,
            execution_price=ExecutionPrice.custom("exec_level"),
            max_position_pct=0.4,
            allow_short=True,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=bt.FeeConfig.binance_perps(),
        slippage=Slippage.fixed_bps(2),
        warmup_bars=60 * 4,
        extra_timeframes={"1h": Interval.hours(1)},
    )


def _equity(result):
    return np.array([float(x) for x in result.equity_curve])


def test_future_bars_cannot_move_the_past(tmp_path):
    """The decisive test: corrupt the future, the past must not budge."""
    frame = _mean_reverting_bars()
    strategy = _band_strategy()
    config = _config(frame)

    reference = _equity(bt.run(strategy, _config(frame), _store(frame, tmp_path, "ref")))

    # Same multiplicative factor on all four price columns, so the bars stay
    # valid (high >= max(open, close), low <= min(open, close)). Small enough
    # that the short strategy survives it: a 3x future turns the equity
    # negative and the run refuses, which would prove nothing.
    rng = np.random.default_rng(1234)
    corrupted = frame.copy()
    tail = slice(SPLIT + 1, None)
    factor = 1.0 + rng.uniform(-0.005, 0.005, N_BARS - SPLIT - 1)
    for col in ("open", "high", "low", "close"):
        corrupted.loc[corrupted.index[tail], col] = corrupted[col].to_numpy()[tail] * factor

    perturbed = _equity(bt.run(strategy, config, _store(corrupted, tmp_path, "pert")))

    # Anti-vacuity: if the corruption changed nothing at all, an identical
    # prefix would be meaningless.
    assert abs(reference[-1] - perturbed[-1]) > 1e-6, (
        "the perturbation left the future untouched; the test would be vacuous"
    )

    n = min(len(reference), len(perturbed), SPLIT + 1)
    assert n > 1000, f"only {n} bars compared, too few to conclude"
    delta = np.abs(reference[:n] - perturbed[:n])
    first = int(np.argmax(delta > 0)) if delta.max() > 0 else -1
    assert delta.max() == 0.0, (
        f"future data leaked into the past: bar {first} differs by {delta.max():.3e}"
    )


def test_builtin_detector_agrees_and_is_not_vacuous(tmp_path):
    """The engine's own detector, plus a check that it compared something.

    `trades=0, mismatched=0` is reported as PASS. Asserting only on `.passed`
    would accept that empty verdict.
    """
    from manifoldbt.diagnostics import detect_lookahead

    frame = _mean_reverting_bars()
    result = detect_lookahead(
        _band_strategy(), _config(frame), _store(frame, tmp_path, "det"), mode="all"
    )

    assert result.passed, f"look-ahead reported: {result}"
    compared = sum(r.total_trades_overlap for r in result.reports)
    assert compared > 0, (
        f"the detector compared no trade at all, its PASS is empty: {result.reports}"
    )


def test_every_fill_lands_on_a_level_known_before_the_bar(tmp_path):
    """No fill may be priced at its own bar's close.

    A fill at the close is only knowable once the bar is over. It is also what
    the example would produce if `custom(...)` silently fell back to AtClose,
    which would make the whole feature a no-op.
    """
    frame = _mean_reverting_bars()
    result = bt.run(_band_strategy(), _config(frame), _store(frame, tmp_path, "fills"))

    trades = result.trades_df()
    assert len(trades) > 20, f"only {len(trades)} trades, too few to conclude"

    bars = frame.set_index(pd.DatetimeIndex(frame["timestamp"]))
    at = bars.reindex(pd.DatetimeIndex(pd.to_datetime(trades["execution_timestamp"], utc=True)))
    intended = trades["intended_price"].to_numpy()

    on_close = np.isclose(intended, at["close"].to_numpy(), rtol=0, atol=1e-12)
    assert not on_close.any(), (
        f"{int(on_close.sum())} fill(s) landed on their bar's close, "
        "which is AtClose behaviour, not a computed level"
    )


def _leaking_strategy():
    """The clean strategy, with one deliberate leak: the entry reads ahead."""
    from manifoldbt.indicators import close, low, open as open_px, sma

    h1 = bt.tf("1h")
    base = sma(h1.close, 8)
    band_up = base * 1.004
    band_dn = base * 0.997
    touch_up = close.lead(5) >= band_up  # <- the leak
    touch_dn = low <= band_dn
    target = bt.when(touch_dn, 0.0, bt.when(touch_up, -1.0))
    exec_level = bt.when(
        touch_dn, bt.when(open_px <= band_dn, open_px, band_dn),
        bt.when(touch_up, bt.when(open_px >= band_up, open_px, band_up), close),
    )
    return (
        bt.Strategy.create("leaky")
        .signal("position", target)
        .signal("exec_level", exec_level)
        .size(target)
        .stop_loss(pct=25.0)
    )


def test_the_perturbation_method_catches_a_real_leak(tmp_path):
    """A look-ahead test that cannot fail proves nothing.

    Same data, same perturbation, same comparison as
    :func:`test_future_bars_cannot_move_the_past` -- only the strategy reads
    five bars ahead. The prefix MUST diverge, or the method above is blind and
    its PASS is worthless.
    """
    frame = _mean_reverting_bars()
    strategy = _leaking_strategy()
    config = _config(frame)

    reference = _equity(bt.run(strategy, config, _store(frame, tmp_path, "leak_ref")))

    rng = np.random.default_rng(1234)
    corrupted = frame.copy()
    tail = slice(SPLIT + 1, None)
    factor = 1.0 + rng.uniform(-0.005, 0.005, N_BARS - SPLIT - 1)
    for col in ("open", "high", "low", "close"):
        corrupted.loc[corrupted.index[tail], col] = corrupted[col].to_numpy()[tail] * factor

    perturbed = _equity(bt.run(strategy, config, _store(corrupted, tmp_path, "leak_pert")))

    n = min(len(reference), len(perturbed), SPLIT + 1)
    delta = np.abs(reference[:n] - perturbed[:n])
    assert delta.max() > 0.0, (
        "a strategy reading 5 bars ahead went undetected: the perturbation "
        "method is blind and every PASS in this file is meaningless"
    )
    # The divergence must sit just before the split, where the lead reaches
    # into the corrupted tail -- not somewhere unrelated.
    first = int(np.argmax(delta > 0))
    assert SPLIT - 60 <= first <= SPLIT, (
        f"divergence at bar {first}, expected it near the split at {SPLIT}"
    )


def test_detector_splits_on_the_data_not_on_the_configured_range(tmp_path):
    """A config starting at the epoch must not empty the detector.

    `time_range_start=0` is what `examples/21_fill_at_computed_level.py`
    writes, and it stretches the period over five decades: both split points
    used to land before the first bar, so both truncated runs saw no data and
    the detector announced PASS having compared nothing.
    """
    from manifoldbt.diagnostics import detect_lookahead

    frame = _mean_reverting_bars()
    config = _config(frame)
    config.time_range_start = 0  # the epoch, as the example does

    result = detect_lookahead(
        _band_strategy(), config, _store(frame, tmp_path, "epoch"), mode="all"
    )

    compared = sum(r.total_trades_overlap for r in result.reports)
    assert compared > 0, (
        "the detector compared no trade: its splits fell outside the data again"
    )
    assert result.passed, f"look-ahead reported: {result}"
