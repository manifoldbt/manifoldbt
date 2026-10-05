"""A lite sweep with an execution grid labels every block with its parameters.

``results_to_df`` labels a lite result by its position: combination
``i % len(grid)`` of the parameters, and the execution combination of its
block. The parameter label was only given while ``i < len(grid)``, so every
block after the first came out with its ``param_*`` columns empty (NaN), and
those columns became floats. Held here: each block carries the parameters, the
rows of a full sweep of the same grids carry the same labels, and a list
longer than the grids still gets no invented label.
"""
import os

import numpy as np
import pytest

import manifoldbt as bt
from manifoldbt import dataframe as D

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("etiquettes")
    n = 3 * 1440
    rng = np.random.default_rng(9)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-03-01", periods=n, freq="1min", tz="UTC"),
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
    strat = bt.Strategy.create("blocs").signal("f", f).signal("s", s).size(
        bt.when(f > s, 0.5, 0.0))
    grid = {"fast": [3, 8], "slow": [30, 60, 90]}
    exec_grid = {"max_position_pct": [0.5, 1.0]}
    return strat, grid, exec_grid, cfg, store


def test_every_block_of_a_lite_sweep_carries_its_parameters(setup):
    strat, grid, exec_grid, cfg, store = setup
    lite = bt.run_sweep_lite(strat, grid, cfg, store, device="cpu", execution_grid=exec_grid)
    assert len(lite) == 12
    df = D.results_to_df(lite, grid, backend="pandas", execution_grid=exec_grid)
    assert not df[["param_fast", "param_slow"]].isna().any().any()
    assert str(df["param_fast"].dtype) == "int64"
    combos = D.grid_combos(grid)
    for i in range(12):
        assert df["param_fast"][i] == combos[i % 6]["fast"]
        assert df["param_slow"][i] == combos[i % 6]["slow"]
        assert df["exec_max_position_pct"][i] == [0.5, 1.0][i // 6]
    full = bt.run_sweep(strat, grid, cfg, store, execution_grid=exec_grid).to_df(backend="pandas")
    for c in ("param_fast", "param_slow", "exec_max_position_pct"):
        assert list(df[c]) == list(full[c]), c


def test_a_list_longer_than_the_grids_gets_no_invented_label(setup):
    strat, grid, _exec_grid, cfg, store = setup
    lite = bt.run_sweep_lite(strat, grid, cfg, store, device="cpu")
    rows = list(lite) + list(lite)[:2]
    df = D.results_to_df(rows, grid, backend="pandas")
    assert not df["param_fast"][:6].isna().any()
    assert df["param_fast"][6:].isna().all()
