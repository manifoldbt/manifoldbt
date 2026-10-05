"""`execution_grid=`: the execution settings as sweep axes.

Two halves, the split `test_fill_marks.py` already makes.

The half that runs everywhere needs no tape: the shape a grid is written in,
the enumeration order, the labels, and the refusals a wrong path hits first --
which are the ones that matter, because an axis that silently does nothing
publishes a flat surface, and a flat surface reads as "this setting does not
matter".

The end-to-end half needs a store holding a tape and skips by name without one.
The per-combination semantics are pinned by the engine's own tests, which run
each combination of an execution axis end to end.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.dataframe import exec_grid_combos, grid_combos  # noqa: E402
from manifoldbt.exceptions import BacktesterError, ConfigError  # noqa: E402
from manifoldbt.expr import col, lit  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

SYMBOL_ID = 1


def _locked():
    try:
        bt.ticks.tape_info(__file__)
    except PermissionError:
        return True
    except Exception:
        return False
    return False


def _store(tmp_path):
    n = 24
    close = 100.0 + np.arange(n) % 5
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000.0),
        }
    )
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df,
        symbol="TEST",
        symbol_id=SYMBOL_ID,
        interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    return store, int(df["timestamp"].iloc[-1].value)


def _config(last_ns, **execution):
    return bt.BacktestConfig(
        universe=[SYMBOL_ID],
        time_range_start=0,
        time_range_end=last_ns + 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000.0,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            allow_short=False,
            position_sizing_mode="Units",
            **execution,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def _strategy():
    return (
        bt.Strategy.create("grid")
        .signal("pos", bt.when(col("close") > lit(101.0), lit(1.0), lit(0.0)))
        .size(col("pos"))
    )


# ---------------------------------------------------------------------------
# The order the combinations come in
# ---------------------------------------------------------------------------


def test_the_axes_are_sorted_by_path_and_the_last_one_varies_fastest():
    # Written out of order on purpose: the engine sorts, and a table built on
    # the dict's insertion order would put the right values on the wrong rows.
    grid = {"latency.order": [0, 20], "fill_model.queue.assumed_queue": [1, 2, 5]}
    combos = exec_grid_combos(grid)
    assert len(combos) == 6
    assert [(c["fill_model.queue.assumed_queue"], c["latency.order"]) for c in combos] == [
        (1, 0), (1, 20), (2, 0), (2, 20), (5, 0), (5, 20),
    ]
    # Each combo keeps the caller's key order, so a table built from them keeps
    # the caller's columns while its rows follow the engine.
    assert list(combos[0]) == list(grid)


def test_no_grid_is_no_combination_and_one_axis_is_its_own_length():
    assert exec_grid_combos({}) == []
    assert exec_grid_combos(None or {}) == []
    assert len(exec_grid_combos({"signal_delay": [1, 2, 3]})) == 3


# ---------------------------------------------------------------------------
# The sweep, on bars
# ---------------------------------------------------------------------------


def test_a_sweep_runs_one_block_of_the_parameter_surface_per_execution_combo(tmp_path):
    store, last_ns = _store(tmp_path)
    sweep = bt.run_sweep(
        _strategy(), {}, _config(last_ns), store,
        execution_grid={"signal_delay": [1, 2, 3]},
    )
    assert len(sweep) == 3
    df = sweep.to_df()
    assert list(df["exec_signal_delay"]) == [1, 2, 3]
    # And the label comes from the run's own manifest, not from a count.
    for i, result in enumerate(sweep):
        assert result.manifest["config"]["execution"]["signal_delay"] == i + 1


def test_the_execution_axes_are_slower_than_the_parameter_ones(tmp_path):
    store, last_ns = _store(tmp_path)
    strategy = (
        bt.Strategy.create("grid")
        .signal("pos", bt.when(col("close") > lit(101.0),
                               lit(1.0) * bt.param("size", default=1.0), lit(0.0)))
        .size(col("pos"))
    )
    sweep = bt.run_sweep(
        strategy, {"size": [1.0, 2.0]}, _config(last_ns), store,
        execution_grid={"signal_delay": [1, 2]},
    )
    assert len(sweep) == 4
    df = sweep.to_df()
    assert list(df["exec_signal_delay"]) == [1, 1, 2, 2]
    assert list(df["param_size"]) == [1.0, 2.0, 1.0, 2.0]
    # The two enumerations, stated by the two helpers, agree with the table.
    assert len(grid_combos({"size": [1.0, 2.0]})) == 2
    assert len(exec_grid_combos({"signal_delay": [1, 2]})) == 2


def test_an_empty_grid_is_the_sweep_the_engine_always_ran(tmp_path, same_metrics):
    store, last_ns = _store(tmp_path)
    before = bt.run_sweep(_strategy(), {}, _config(last_ns), store)
    after = bt.run_sweep(_strategy(), {}, _config(last_ns), store, execution_grid={})
    assert len(before) == len(after) == 1
    assert same_metrics(before[0].metrics, after[0].metrics)
    assert "exec_signal_delay" not in after.to_df().columns


def test_a_batch_runs_every_strategy_under_every_execution_combination(tmp_path):
    store, last_ns = _store(tmp_path)
    a = _strategy()
    b = (
        bt.Strategy.create("grid-2")
        .signal("pos", bt.when(col("close") > lit(102.0), lit(1.0), lit(0.0)))
        .size(col("pos"))
    )
    out = bt.run_batch(
        [a, b], _config(last_ns), store, execution_grid={"signal_delay": [1, 2]}
    )
    assert len(out) == 4
    names = [r.manifest["strategy_name"] for r in out]
    assert names == ["grid", "grid-2", "grid", "grid-2"]
    delays = [r.manifest["config"]["execution"]["signal_delay"] for r in out]
    assert delays == [1, 1, 2, 2]


def test_a_lite_sweep_takes_an_axis_that_needs_no_tape(tmp_path):
    store, last_ns = _store(tmp_path)
    out = bt.run_sweep_lite(
        _strategy(), {}, _config(last_ns), store, device="cpu",
        execution_grid={"signal_delay": [1, 2, 3]},
    )
    assert len(out) == 3


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------


def test_a_path_that_names_no_setting_is_refused_by_name(tmp_path):
    store, last_ns = _store(tmp_path)
    with pytest.raises((BacktesterError, ConfigError), match="qeue"):
        bt.run_sweep(
            _strategy(), {}, _config(last_ns), store,
            execution_grid={"fill_model.qeue.assumed_queue": [1.0, 2.0]},
        )


def test_a_path_already_rooted_at_execution_says_what_to_write_instead(tmp_path):
    store, last_ns = _store(tmp_path)
    with pytest.raises((BacktesterError, ConfigError), match="latency.order"):
        bt.run_sweep(
            _strategy(), {}, _config(last_ns), store,
            execution_grid={"execution.latency.order": [0]},
        )


def test_a_value_out_of_the_models_domain_is_refused_with_its_own_message(tmp_path):
    store, last_ns = _store(tmp_path)
    with pytest.raises((BacktesterError, ConfigError), match="cancel_ahead_rate"):
        bt.run_sweep(
            _strategy(), {}, _config(last_ns), store,
            execution_grid={"fill_model.queue.cancel_ahead_rate": [-1.0]},
        )


def test_an_axis_with_no_value_is_refused_before_anything_runs():
    with pytest.raises(ValueError, match="no value"):
        bt.run_sweep(
            _strategy(), {}, bt.BacktestConfig(universe=[SYMBOL_ID]), None,
            execution_grid={"signal_delay": []},
        )


def test_a_grid_that_is_not_a_mapping_of_lists_is_refused():
    with pytest.raises(TypeError, match="mapping of dotted config paths"):
        bt.run_sweep(_strategy(), {}, bt.BacktestConfig(universe=[SYMBOL_ID]), None,
                     execution_grid=[("signal_delay", [1])])
    with pytest.raises(TypeError, match="list of values"):
        bt.run_sweep(_strategy(), {}, bt.BacktestConfig(universe=[SYMBOL_ID]), None,
                     execution_grid={"signal_delay": 1})


def test_a_lite_sweep_refuses_a_tape_setting_on_any_combination(tmp_path):
    store, last_ns = _store(tmp_path)
    with pytest.raises(BacktesterError, match="execution.latency"):
        bt.run_sweep_lite(
            _strategy(), {}, _config(last_ns), store, device="cpu",
            execution_grid={"latency.order": [0, Interval.millis(20)]},
        )
    with pytest.raises(BacktesterError, match="fill_model.queue"):
        bt.run_sweep_lite(
            _strategy(), {}, _config(last_ns), store, device="cpu",
            execution_grid={"fill_model.queue.assumed_queue": [1.0]},
        )


def test_the_cuda_sweep_refuses_an_execution_grid_rather_than_ignoring_it(tmp_path):
    store, last_ns = _store(tmp_path)
    # The CUDA path answers to the GPU grant before it reads the grid: a
    # licence without it is refused by name first (see test_gpu_gate.py),
    # and this test is about what comes after that door.
    from manifoldbt._native import _grant_couvre

    if not _grant_couvre("gpu_sweep"):
        pytest.skip("device='cuda' asks for the GPU grant first")
    with pytest.raises((ValueError, ConfigError), match="gpu-sweep-unsupported"):
        bt.run_sweep_lite(
            _strategy(), {}, _config(last_ns), store, device="cuda",
            execution_grid={"signal_delay": [1, 2]},
        )


def test_an_interval_written_as_a_duration_reaches_the_engine_as_nanoseconds():
    from manifoldbt import _execution_grid_json
    import json

    grid = json.loads(_execution_grid_json({"latency.order": [Interval.millis(20)]}))
    assert grid == {"latency.order": [20_000_000]}
    # And a plain number passes through untouched.
    grid = json.loads(_execution_grid_json({"signal_delay": [0, 1]}))
    assert grid == {"signal_delay": [0, 1]}


# ---------------------------------------------------------------------------
# The columns a sweep of execution settings is read through
# ---------------------------------------------------------------------------


def test_the_execution_columns_are_refused_off_a_lite_sweep(tmp_path):
    store, last_ns = _store(tmp_path)
    out = bt.run_sweep_lite(_strategy(), {}, _config(last_ns), store, device="cpu")
    with pytest.raises(ValueError, match="need a full sweep"):
        bt.sweep_columns(out, "service_rate")
    # The metric columns still work off the same list.
    assert bt.sweep_columns(out, "sharpe").shape == (1,)


def test_the_execution_columns_read_off_a_full_sweep(tmp_path):
    store, last_ns = _store(tmp_path)
    sweep = bt.run_sweep(
        _strategy(), {}, _config(last_ns), store,
        execution_grid={"signal_delay": [1, 2]},
    )
    cols = bt.sweep_columns(list(sweep), [
        "sharpe", "service_rate", "queue_decided_fills", "stale_fills",
        "adverse_1s_bps", "adverse_10s_bps", "half_spread_captured_bps",
    ])
    assert set(cols) == {
        "sharpe", "service_rate", "queue_decided_fills", "stale_fills",
        "adverse_1s_bps", "adverse_10s_bps", "half_spread_captured_bps",
    }
    for name, values in cols.items():
        assert values.shape == (2,)
    # This run configured no queue, no latency and no marks, so each of those
    # columns is NaN rather than a zero that would read like a measurement.
    for name in ("queue_decided_fills", "stale_fills", "adverse_1s_bps",
                 "adverse_10s_bps", "half_spread_captured_bps"):
        assert np.isnan(cols[name]).all(), name


def test_an_unknown_column_names_both_families(tmp_path):
    store, last_ns = _store(tmp_path)
    sweep = bt.run_sweep(_strategy(), {}, _config(last_ns), store)
    with pytest.raises(ValueError, match="service_rate"):
        bt.sweep_columns(list(sweep), "not_a_column")
