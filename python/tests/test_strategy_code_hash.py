"""``manifest["strategy_code_hash"]``: the hash a run records of its strategy.

The same strategy hashes the same from one run to the next and from one
process to the next. Its signals and parameters are maps, read by name: the
order they were declared in does not count. Its quotes are a sequence, which a
wake-up acts on in order: their order counts. Two strategies that run
differently hash differently, and a combination of a sweep hashes as the same
strategy run alone with those values as its defaults.

The strategies are those of the guide and the examples, on a synthetic store.
The engine's own tests hold the same proofs on the definitions themselves.
"""
import json
import os
import subprocess
import sys

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt import _native  # noqa: E402
from manifoldbt.helpers import Interval  # noqa: E402
from manifoldbt.indicators import atr, close, ema, rsi, sma, volume  # noqa: E402

from test_quote_dsl import _layer_open, walk_market  # noqa: E402
from test_quote_sweep import config as quote_config, maker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# The strategies: a fresh object at every call
# ---------------------------------------------------------------------------


def trend_following():
    """Example 01."""
    fast, slow = ema(close, 12), ema(close, 26)
    trend = fast - slow
    vol_ma = volume.rolling_mean(20)
    return (
        bt.Strategy.create("trend_following")
        .signal("fast", fast)
        .signal("slow", slow)
        .signal("trend", trend)
        .signal("vol_filter", volume > vol_ma)
        .size(bt.when((trend > 0.0) & (volume > vol_ma), 0.5, 0.0))
        .stop_loss(pct=3.0)
        .describe("EMA(12/26) crossover, volume filter, 3% stop-loss")
    )


def ema_crossover():
    """Example 02."""
    fast, slow = ema(close, 12), ema(close, 26)
    return (
        bt.Strategy.create("ema_crossover")
        .signal("fast", fast)
        .signal("slow", slow)
        .size(bt.when(fast > slow, 1.0, -1.0) * 0.25)
    )


def multi_timeframe():
    """Example 14: two signals on a coarser grid."""
    h12 = bt.tf("12h")
    entry_rsi = rsi(close, 14)
    return (
        bt.Strategy.create("multi_tf_trend_dip")
        .signal("bullish", h12.apply(ema(close, 20)) > h12.apply(ema(close, 50)))
        .signal("entry_rsi", entry_rsi)
        .signal("dip", entry_rsi < 35.0)
        .size(bt.when(bt.col("bullish") & bt.col("dip"), 0.5, 0.0))
        .stop_loss(pct=3.0)
    )


def entry_orders(stop=3.0):
    """Example 20, the entry resting on a level the DSL computes."""
    fast, slow = ema(close, 12), ema(close, 50)
    return (
        bt.Strategy.create("entry_orders")
        .signal("fast", fast)
        .signal("slow", slow)
        .signal("pullback", close - atr(14))
        .signal("stop_dist", bt.lit(2.0) * atr(14) / close * bt.lit(100.0))
        .size(bt.when(fast > slow, 1.0, 0.0))
        .stop_loss(pct=stop)
        .limit_entry(signal="pullback", time_in_force={"GTB": 5})
    )


def swept(fast=10, slow=40, k=1.0, reverse=False):
    """The guide's sweep: two periods and a threshold, all parameters. With
    ``reverse`` the same strategy declares its signals and parameters in the
    opposite order."""
    signals = [
        ("fast", sma(close, bt.param("fast", default=fast))),
        ("slow", sma(close, bt.param("slow", default=slow))),
        ("spread", bt.col("fast") - bt.col("slow")),
        ("go", bt.col("spread") > bt.param("k", default=k)),
    ]
    if reverse:
        # Parameters are declared by the first signal that reads them.
        signals = [signals[i] for i in (3, 2, 1, 0)]
    s = bt.Strategy.create("swept")
    for name, expr in signals:
        s = s.signal(name, expr)
    return s.size(bt.when(bt.col("go"), 1.0, 0.0))


BAR_STRATEGIES = {
    "trend_following": trend_following,
    "ema_crossover": ema_crossover,
    "multi_timeframe": multi_timeframe,
    "entry_orders": entry_orders,
    "swept": swept,
}


def avellaneda(swap=False):
    """The Avellaneda-Stoikov maker of the order-level harness: two signals,
    two parameters, a bid and an ask. ``swap`` declares the ask first."""
    from manifoldbt import book
    from manifoldbt import indicators as ind

    gamma, k = bt.param("gamma", default=0.02), bt.param("k", default=4.0)
    mid = (book.bid_price(1) + book.ask_price(1)) * bt.lit(0.5)
    sigma = mid.diff(1).rolling_std(Interval.seconds(30))
    pos = bt.position()
    s2g = bt.col("s2") * gamma
    r = bt.col("mid") - pos / bt.lit(0.01) * s2g
    delta = ind.max_val(bt.lit(0.1), s2g * bt.lit(0.5) + ind.log(bt.lit(1.0) + gamma / k) / gamma)
    quotes = [
        ("buy", (r - delta).floor_to(0.1), pos < 0.05),
        ("sell", (r + delta).ceil_to(0.1), pos > -0.05),
    ]
    if swap:
        quotes.reverse()
    s = bt.Strategy.create("avellaneda").signal("mid", mid).signal("s2", sigma * sigma)
    for side, price, enabled in quotes:
        s = s.quote(side, price, 0.01, tif="GTX", enabled=enabled)
    return s


QUOTE_STRATEGIES = {"maker": maker, "avellaneda": avellaneda}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def bars(rows=3_000, seed=7):
    rng = np.random.default_rng(seed)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, rows)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC"),
        "open": px, "high": px * 1.001, "low": px * 0.999, "close": px,
        "volume": rng.uniform(1.0, 10.0, rows),
    })


def open_store(root):
    df = bars()
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1m",
        data_root=os.path.join(root, "data"), metadata_db=os.path.join(root, "meta.sqlite"),
    )
    config = bt.BacktestConfig(
        universe=[1], time_range_start=0,
        time_range_end=int(df["timestamp"].iloc[-1].value) + 86_400_000_000_000,
        bar_interval=Interval.minutes(1), initial_capital=10_000.0,
        # The coarser grid of example 14.
        extra_timeframes={"12h": Interval.hours(12)},
        # A resting entry is placed from a bar that has closed (example 20).
        execution=bt.ExecutionConfig(signal_delay=1),
    )
    return store, config


def bar_hash(strategy, store, config):
    return bt.run(strategy, config, store).manifest["strategy_code_hash"]


_MARKET = []


def quote_hash(strategy):
    if not _MARKET:
        _MARKET.append(walk_market(seed=3, n=300))
    raw, _ = _native.run_quotes(strategy.to_json(), quote_config().to_json(), None, _MARKET)
    return raw.manifest["strategy_code_hash"]


def all_hashes(root):
    """Every strategy's hash, each from a fresh object: what a process sees."""
    store, config = open_store(root)
    out = {name: bar_hash(build(), store, config) for name, build in BAR_STRATEGIES.items()}
    if _layer_open():
        out.update({name: quote_hash(build()) for name, build in QUOTE_STRATEGIES.items()})
    return out


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return open_store(str(tmp_path_factory.mktemp("code_hash")))


needs_layer = pytest.mark.skipif(
    not _layer_open(),
    reason="order-level simulation locked (runs in the engine's own debug builds)",
)


# ---------------------------------------------------------------------------
# The same strategy, the same hash
# ---------------------------------------------------------------------------


def test_the_hash_is_the_same_in_every_process(tmp_path):
    here = all_hashes(str(tmp_path / "parent"))
    child = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); "
        "import test_strategy_code_hash as t; print(json.dumps(t.all_hashes(sys.argv[2])))"
    )
    for i in range(3):
        got = subprocess.run(
            [sys.executable, "-c", child, HERE, str(tmp_path / f"child{i}")],
            capture_output=True, text=True, timeout=600, env=os.environ.copy(),
        )
        assert got.returncode == 0, got.stderr[-2000:]
        line = [x for x in got.stdout.splitlines() if x.startswith("{")][-1]
        assert json.loads(line) == here, f"process {i}"
    assert all(len(h) == 64 for h in here.values())


def test_the_hash_is_the_same_run_after_run(store):
    s, config = store
    for name, build in BAR_STRATEGIES.items():
        assert len({bar_hash(build(), s, config) for _ in range(4)}) == 1, name


@needs_layer
def test_a_strategy_that_quotes_hashes_the_same_run_after_run():
    for name, build in QUOTE_STRATEGIES.items():
        assert len({quote_hash(build()) for _ in range(6)}) == 1, name


def test_the_order_signals_and_parameters_are_declared_in_does_not_count(store):
    s, config = store
    forward, backward = swept(), swept(reverse=True)
    assert forward.to_json() != backward.to_json()
    assert bar_hash(forward, s, config) == bar_hash(backward, s, config)


@needs_layer
def test_the_order_of_the_quotes_counts():
    assert quote_hash(avellaneda()) != quote_hash(avellaneda(swap=True))


def test_a_combination_of_a_sweep_hashes_as_the_strategy_run_alone(store):
    s, config = store
    grid = {"fast": [10, 20], "k": [1.0, 2.0]}
    sweep = bt.run_sweep(swept(), grid, config, s)
    hashes = [r.manifest["strategy_code_hash"] for r in sweep]
    assert len(set(hashes)) == 4
    for r, h in zip(sweep, hashes):
        p = r.manifest["parameters"]
        alone = swept(fast=_value(p["fast"]), k=_value(p["k"]))
        assert bar_hash(alone, s, config) == h, p


def _value(scalar):
    """``{"Int64": 20}`` -> 20."""
    (v,) = scalar.values()
    return v


# ---------------------------------------------------------------------------
# Different strategies, different hashes
# ---------------------------------------------------------------------------


def test_strategies_that_run_differently_hash_differently(store):
    s, config = store
    variants = dict(BAR_STRATEGIES)
    variants["a parameter's value"] = lambda: swept(k=1.5)
    variants["a period"] = lambda: swept(slow=41)
    variants["the stop"] = lambda: entry_orders(stop=4.0)
    variants["one more signal"] = lambda: ema_crossover().signal("extra", close)
    # Each variant keeps its own defaults whatever was built before it (see
    # test_param_defaut_par_strategie.py).
    hashes = {name: bar_hash(build(), s, config) for name, build in variants.items()}
    assert len(set(hashes.values())) == len(hashes), hashes
