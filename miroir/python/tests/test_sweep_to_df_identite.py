"""A sweep's table labels its rows from ``run_parameters``, not the manifest.

``SweepResult.to_df()`` read each run's whole manifest (config included) to
take its parameters, and read ``metrics`` twice per row. It reads
``run_parameters`` now, the manifest's own ``parameters`` map, and the metrics
once. Held here: ``run_parameters`` is ``manifest["parameters"]``, key for key
and type for type, and the table is the one the manifest road built, to the
bit, for pandas and polars, for full and lite sweeps, and for objects that are
not engine results at all.
"""
import os

import numpy as np
import pytest

import manifoldbt as bt
from manifoldbt import dataframe as D

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")


def same(a, b):
    """The same value of the same type; a float to the bit, NaN to NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a.hex() == b.hex() or (a != a and b != b)
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="module")
def sweep_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sweep")
    n = 3 * 1440
    rng = np.random.default_rng(5)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-02-01", periods=n, freq="1min", tz="UTC"),
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": np.full(n, 5.0),
    })
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1m",
        data_root=os.path.join(tmp, "data"), metadata_db=os.path.join(tmp, "meta.sqlite"),
    )
    cfg = bt.BacktestConfig(
        universe=[1], time_range_start=0,
        time_range_end=int(df["timestamp"].iloc[-1].value) + 60_000_000_000,
        bar_interval=bt.Interval.minutes(1), initial_capital=10_000.0,
    )
    f = bt.indicators.ema(bt.col("close"), bt.param("fast"))
    s = bt.indicators.ema(bt.col("close"), bt.param("slow"))
    strat = (bt.Strategy.create("etiquettes").signal("f", f).signal("s", s)
             .size(bt.when(f > s, bt.param("poids"), 0.0)))
    grid = {"slow": [30, 45, 60], "fast": [3, 8], "poids": [0.25, 1.0]}
    return strat, grid, cfg, store


def test_run_parameters_is_the_manifest_s_own_map(sweep_setup):
    strat, grid, cfg, store = sweep_setup
    sw = bt.run_sweep(strat, grid, cfg, store)
    assert len(sw) == 12
    for r in sw:
        assert same(r.raw.run_parameters, r.raw.manifest["parameters"])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_the_table_is_the_one_the_manifest_road_built(sweep_setup, monkeypatch, backend):
    strat, grid, cfg, store = sweep_setup
    sw = bt.run_sweep(strat, grid, cfg, store)
    lite = bt.run_sweep_lite(strat, grid, cfg, store, device="cpu")
    # Reversed, the rows are out of the engine's order: only a label read off
    # each run's own parameters puts the right values on them.
    backwards = [r.raw for r in sw][::-1]
    new_full = sw.to_df(backend=backend)
    new_raw = D.results_to_df([r.raw for r in sw], grid, backend=backend)
    new_back = D.results_to_df(backwards, grid, backend=backend)
    new_lite = D.results_to_df(lite, grid, backend=backend)
    # The manifest road: no engine type is recognised, every row reads its
    # manifest as before.
    monkeypatch.setattr(D, "_native_parameters", lambda result: None)
    old_full = sw.to_df(backend=backend)
    old_raw = D.results_to_df([r.raw for r in sw], grid, backend=backend)
    old_back = D.results_to_df(backwards, grid, backend=backend)
    old_lite = D.results_to_df(lite, grid, backend=backend)
    assert list(old_back["param_slow"])[0] == 60, "the reversed rows keep their own labels"
    for new, old in ((new_full, old_full), (new_raw, old_raw), (new_back, old_back),
                     (new_lite, old_lite)):
        assert list(new.columns) == list(old.columns)
        if backend == "pandas":
            assert list(map(str, new.dtypes)) == list(map(str, old.dtypes))
            pd.testing.assert_frame_equal(new, old, check_exact=True)
        else:
            assert new.schema == old.schema
            from polars.testing import assert_frame_equal

            assert_frame_equal(new, old, check_exact=True)
        for c in old.columns:
            assert same(list(new[c]), list(old[c])), c
    assert {"param_fast", "param_slow", "param_poids"} <= set(new_full.columns)


class NoMetrics:
    pass


class MetricsRaises:
    @property
    def metrics(self):
        raise AttributeError("not computed")


class PlainMetrics:
    def __init__(self, v):
        self.metrics = {"sharpe": v, "trade_stats": {"total_trades": 3}}


def test_objects_that_are_not_engine_results_read_as_before():
    rows = [NoMetrics(), MetricsRaises(), PlainMetrics(1.5), PlainMetrics(float("nan"))]
    out = D.results_to_df(rows, backend="arrow")
    old = []
    for r in rows:
        m = r.metrics if hasattr(r, "metrics") else {}
        row = {}
        for k, v in m.items():
            if isinstance(v, dict):
                row.update(v)
            else:
                row[k] = v
        old.append(row)
    assert same(out, old)
