"""Golden test that any installed wheel can run, on any tier.

The engine's other golden fixture is four bars at one-second spacing and asserts
one-second output. That resolution is not available on every tier, so its Python
mirror only runs on a development build: useful to the team, unverifiable by
anyone else.

This one is deliberately ordinary so that it is reproducible by whoever installs
the package: daily bars, daily output, one asset, long only, no leverage. Bars
come from a committed CSV and the expected result from a committed JSON, both
readable in a diff. Run it against a released wheel and it either reproduces the
recorded numbers or it does not.

The definitions below are the single source for both sides of the comparison:
`scripts/gen_golden_daily_fixture.py` imports them to write the fixture, so the
run under test and the run that produced the expectation cannot drift apart.

Golden fixtures are frozen on purpose. If this test fails after an intentional
engine change, regenerate with `python scripts/gen_golden_daily_fixture.py`,
read the diff, and commit it only if every line of it is explainable.
"""
import json
import os
import tempfile

import pytest

import manifoldbt as bt

pd = pytest.importorskip("pandas")

from manifoldbt.helpers import Interval, Slippage  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fixtures", "golden_daily")

# Floats are compared with a tight relative tolerance rather than for exact
# equality: the fixture is generated on one platform and the test runs on
# others, and an assertion that depends on the last bit of a double would fail
# for reasons that have nothing to do with the engine's behaviour. 1e-9 is far
# below any change in logic and far above cross-platform rounding.
RTOL = 1e-9


def load_bars() -> "pd.DataFrame":
    """The committed bars, as the engine will receive them."""
    frame = pd.read_csv(os.path.join(FIXTURE_DIR, "bars.csv"))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def build_config(frame) -> "bt.BacktestConfig":
    """Available on every tier: daily output, one asset, long only."""
    ts = pd.DatetimeIndex(frame["timestamp"])
    one_day_ns = 86_400_000_000_000
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=int(ts[0].value),
        time_range_end=int(ts[-1].value) + one_day_ns,
        bar_interval=Interval.days(1),
        output_resolution=Interval.days(1),
        initial_capital=10_000.0,
        currency="USD",
        risk_free_rate=0.02,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            allow_fractional=True,
            position_sizing_mode="FractionOfEquity",
        ),
        slippage=Slippage.fixed_bps(0),
        warmup_bars=0,
        data_version="golden_daily_v1",
        rng_seed=7,
    )


def build_strategy():
    """A 3/10 mean cross: exercises indicators, sizing and exits, not just holding."""
    from manifoldbt.indicators import close

    fast = close.rolling_mean(3)
    slow = close.rolling_mean(10)
    return (
        bt.Strategy.create("golden_daily")
        .signal("edge", fast - slow)
        .size(bt.when(fast > slow, 1.0, 0.0))
    )


def run_fixture(frame, tmp_dir):
    store = bt.import_dataframe(
        frame,
        symbol="GOLD",
        symbol_id=1,
        interval="1d",
        data_root=os.path.join(tmp_dir, "data"),
        metadata_db=os.path.join(tmp_dir, "meta.sqlite"),
    )
    return bt.run(build_strategy(), build_config(frame), store)


def _plain(value):
    """JSON-safe scalar, matching what the fixture stores."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def snapshot(result) -> dict:
    """The frozen contract: what the engine must keep producing.

    ``result.trades`` is an Arrow RecordBatch. It is converted column-wise on
    purpose: iterating it yields columns rather than rows, which is a quiet way
    to end up asserting nothing.
    """
    trades = {
        name: [_plain(v) for v in values]
        for name, values in result.trades.to_pydict().items()
    }
    manifest = result.manifest
    return {
        "equity_curve": [float(x) for x in result.equity_curve],
        "trade_count": result.trades.num_rows,
        "trades": trades,
        "metrics": {
            k: float(v)
            for k, v in dict(result.metrics).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        "manifest": {
            "strategy_name": manifest["strategy_name"],
            "config": manifest["config"],
            "data_version": manifest.get("data_versions", {}).get("bars_1m", ""),
        },
    }


@pytest.fixture(scope="module")
def expected():
    with open(os.path.join(FIXTURE_DIR, "expected.json"), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def actual():
    tmp = tempfile.mkdtemp(prefix="golden_daily_")
    return snapshot(run_fixture(load_bars(), tmp))


def _close(a, b):
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


def test_equity_curve_matches(expected, actual):
    exp, got = expected["equity_curve"], actual["equity_curve"]
    assert len(got) == len(exp), f"{len(got)} equity points, fixture has {len(exp)}"
    bad = [(i, g, e) for i, (g, e) in enumerate(zip(got, exp)) if not _close(g, e)]
    assert not bad, f"equity diverges at {len(bad)} point(s), first: {bad[0]}"


def test_trades_match(expected, actual):
    exp, got = expected["trades"], actual["trades"]
    assert actual["trade_count"] == expected["trade_count"], (
        f"{actual['trade_count']} trades, fixture has {expected['trade_count']}"
    )
    assert set(got) == set(exp), (
        f"trade columns changed: only in run {sorted(set(got) - set(exp))}, "
        f"only in fixture {sorted(set(exp) - set(got))}"
    )
    for column in sorted(exp):
        for i, (g, e) in enumerate(zip(got[column], exp[column])):
            if isinstance(e, float) and isinstance(g, float):
                assert _close(g, e), f"trades[{column}][{i}]: {g} != {e}"
            else:
                assert g == e, f"trades[{column}][{i}]: {g!r} != {e!r}"


def test_metrics_match(expected, actual):
    exp, got = expected["metrics"], actual["metrics"]
    missing = sorted(set(exp) - set(got))
    assert not missing, f"metrics disappeared from the result: {missing}"
    bad = {k: (got[k], exp[k]) for k in exp if not _close(got[k], exp[k])}
    assert not bad, f"metrics diverge: {bad}"


def test_manifest_matches(expected, actual):
    """The run must still describe itself the same way: this is the provenance record."""
    exp, got = expected["manifest"], actual["manifest"]
    assert got["strategy_name"] == exp["strategy_name"]
    assert got["data_version"] == exp["data_version"]
    assert got["config"] == exp["config"], "the resolved config drifted from the fixture"


def test_manifest_identifies_the_engine(actual):
    """Provenance fields that must be populated, but whose values are not frozen."""
    tmp = tempfile.mkdtemp(prefix="golden_daily_id_")
    manifest = run_fixture(load_bars(), tmp).manifest
    for field in ("run_id", "engine_version", "strategy_code_hash", "platform"):
        assert manifest.get(field), f"manifest.{field} is empty"
