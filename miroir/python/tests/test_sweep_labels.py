"""``to_df()`` must label each row with the parameters its run actually used.

The engine enumerates a grid with its axes sorted by parameter NAME, the last
axis varying fastest. ``results_to_df`` used to expand the grid in the dict's
insertion order instead, so on any grid whose keys were not alphabetical the
values were right and the rows were wrong: on a 3x2x2 grid, between 2 and 6
of the 12 rows kept the right label depending on the order the dict was
written in, and nothing failed.

Two layers: the labelling alone (no engine, any tier), then a real sweep in
which every row is re-run on its own to prove its label.
"""
import itertools
import os

import pytest

import manifoldbt as bt
from manifoldbt._serde import scalar_value_to_json
from manifoldbt.dataframe import grid_combos, results_to_df

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

# Keys deliberately NOT alphabetical: the engine runs fast, risk, slow.
GRID = {"slow": [20, 30, 40], "fast": [5, 10], "risk": [0.5, 1.0]}


def _engine_order(grid):
    axes = sorted(grid)
    return [dict(zip(axes, c)) for c in itertools.product(*(grid[a] for a in axes))]


def _dict_order(grid):
    return [dict(zip(grid, c)) for c in itertools.product(*grid.values())]


class _Fake:
    """A metrics holder, with a manifest when the engine would provide one."""

    def __init__(self, metrics, manifest=None):
        self.metrics = metrics
        if manifest is not None:
            self.manifest = manifest


def test_the_grid_is_one_where_the_two_orders_differ():
    """Anti-vacuity: on an alphabetical grid the old code was right by luck."""
    assert _dict_order(GRID) != _engine_order(GRID)


def test_grid_combos_follow_the_engine_order():
    combos = grid_combos(GRID)
    assert len(combos) == 12
    # Columns in the caller's order, rows in the engine's.
    assert all(list(c) == list(GRID) for c in combos)
    assert [{k: c[k] for k in sorted(c)} for c in combos] == _engine_order(GRID)
    assert grid_combos({}) == []


def test_rows_without_manifest_follow_the_engine_order():
    results = [_Fake({"sharpe": float(i)}) for i in range(12)]
    df = results_to_df(results, GRID, backend="pandas")
    assert list(df.columns[:3]) == ["param_slow", "param_fast", "param_risk"]
    for i, combo in enumerate(grid_combos(GRID)):
        for name, value in combo.items():
            assert df.loc[i, f"param_{name}"] == value, (i, name)
        assert df.loc[i, "sharpe"] == float(i)


def test_rows_with_manifest_are_labelled_from_it():
    """The manifest names the row, whatever order the results come in."""
    shuffled = grid_combos(GRID)[::-1]
    results = []
    for i, combo in enumerate(shuffled):
        params = {k: scalar_value_to_json(v) for k, v in combo.items()}
        params["untouched"] = {"Int64": 3}  # a default the sweep did not vary
        results.append(_Fake({"sharpe": float(i)}, manifest={"parameters": params}))
    df = results_to_df(results, GRID, backend="pandas")
    for i, combo in enumerate(shuffled):
        for name, value in combo.items():
            assert df.loc[i, f"param_{name}"] == value, (i, name)
    assert "param_untouched" not in df.columns


def test_a_manifest_missing_a_swept_name_falls_back_to_position():
    results = [
        _Fake({"sharpe": float(i)}, manifest={"parameters": {"slow": {"Int64": 99}}})
        for i in range(12)
    ]
    df = results_to_df(results, GRID, backend="pandas")
    for i, combo in enumerate(grid_combos(GRID)):
        assert df.loc[i, "param_slow"] == combo["slow"]


# -- The engine itself -------------------------------------------------------

def _daily_bars(rows=400, seed=3):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, rows)))
    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = rng.uniform(0.2, 1.0, rows) * 0.005 * close
    return pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=rows, freq="1D", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": np.full(rows, 1_000.0),
    })


def _store(frame, tmp_path):
    root = str(tmp_path)
    return bt.import_dataframe(
        frame, symbol="SYNTH", symbol_id=1, interval="1d",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )


def _config(frame):
    from manifoldbt.helpers import Interval, Slippage

    ts = pd.DatetimeIndex(frame["timestamp"])
    one_day_ns = 86_400_000_000_000
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=int(ts[0].value),
        time_range_end=int(ts[-1].value) + one_day_ns,
        bar_interval=Interval.days(1),
        output_resolution=Interval.days(1),
        initial_capital=10_000.0,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            allow_fractional=True,
            position_sizing_mode="FractionOfEquity",
        ),
        slippage=Slippage.fixed_bps(0),
    )


def _strategy(fast=5, slow=20, risk=0.5):
    """Every swept name changes the result: two MA periods and a size."""
    from manifoldbt.indicators import close, sma

    fast_ma = sma(close, bt.param("fast", default=fast))
    slow_ma = sma(close, bt.param("slow", default=slow))
    return (
        bt.Strategy.create("labels")
        .signal("fast_ma", fast_ma)
        .signal("slow_ma", slow_ma)
        .size(bt.when(fast_ma > slow_ma, bt.param("risk", default=risk), 0.0))
    )


def test_every_row_of_a_real_sweep_is_labelled_with_its_own_parameters(tmp_path):
    """Each row, re-run alone with its label, must reproduce its metrics."""
    frame = _daily_bars()
    store = _store(frame, tmp_path)
    config = _config(frame)

    df = bt.run_sweep(_strategy(), GRID, config, store).to_df(backend="pandas")
    assert len(df) == 12

    # Anti-vacuity: if several combos gave the same return, a wrong label
    # could still pass the re-run below.
    returns = df["total_return"].to_numpy()
    assert len(set(np.round(returns, 12))) == 12, returns

    for i, row in df.iterrows():
        alone = bt.run(
            _strategy(fast=int(row["param_fast"]), slow=int(row["param_slow"]),
                      risk=float(row["param_risk"])),
            config, store,
        )
        assert alone.metrics["total_return"] == pytest.approx(row["total_return"], rel=1e-12), (
            f"row {i} labelled {dict(row[[c for c in df.columns if c.startswith('param_')]])} "
            f"does not reproduce when run alone"
        )
