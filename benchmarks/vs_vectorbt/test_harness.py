"""Guards for the parts of the harness that decide what gets published.

These run without any engine installed: they exercise the gate, the ceilings and
the sweep plumbing, which are plain Python. Run with
``pytest benchmarks/vs_vectorbt/test_harness.py``.
"""
from __future__ import annotations

import argparse

import pytest

import bench
import parity
import report
import sweep_child
from engines import ENGINES, REFERENCE, ceiling


# --------------------------------------------------------------------------- #
# The gate, for an engine that has no drawdown of its own
# --------------------------------------------------------------------------- #
def _metrics(**over):
    base = {
        "total_return": 0.01, "final_equity": 101_000.0, "round_trips": 10,
        "fills": 20, "total_fees": 0.0,
        "max_drawdown": -0.02, "sharpe": 1.0, "sortino": 2.0, "volatility": 0.1,
    }
    base.update(over)
    return base


def test_an_engine_without_a_drawdown_is_gated_on_money_and_round_trips():
    verdict = parity.compare(
        _metrics(), _metrics(max_drawdown=None, sharpe=9.0), "sma_cross_metrics",
        "nautilus_trader")

    assert verdict["status"] == "exact"
    assert verdict["diffs"]["max_drawdown_absent"] is True
    assert "max_drawdown_rel" not in verdict["diffs"]
    # Its ratios sit on their own basis, so they are not subtracted from ours.
    assert verdict["diffs"]["ratio_basis"] == "native"
    assert "advisory_ratio_rel" not in verdict["diffs"]


def test_an_engine_without_a_drawdown_still_fails_on_money():
    verdict = parity.compare(
        _metrics(), _metrics(max_drawdown=None, final_equity=101_050.0),
        "sma_cross_metrics", "nautilus_trader")

    assert verdict["status"] == "failed"


def test_a_drawdown_that_is_present_is_still_gated():
    verdict = parity.compare(
        _metrics(), _metrics(max_drawdown=-0.03), "sma_cross_metrics", "raptorbt")

    assert verdict["status"] == "failed"
    assert verdict["diffs"]["max_drawdown_rel"] > parity.REL_TOL


def test_the_documented_bracket_divergence_is_expected_for_nautilus():
    verdict = parity.compare(
        _metrics(), _metrics(final_equity=101_000.7, max_drawdown=None),
        "bracket_sl_tp", "nautilus_trader")

    assert verdict["expected"] == "documented"
    assert verdict["status"] == "documented"


# --------------------------------------------------------------------------- #
# Ceilings
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_overrides():
    bench.CEILING_OVERRIDES.clear()
    bench.SWEEP_CEILING_OVERRIDES.clear()
    yield
    bench.CEILING_OVERRIDES.clear()
    bench.SWEEP_CEILING_OVERRIDES.clear()


def test_only_the_event_driven_engine_has_a_ceiling_by_default():
    limited = {n for n in ENGINES if ceiling(n) is not None}

    assert limited == {"nautilus_trader"}
    assert REFERENCE not in limited


def test_an_engine_runs_up_to_its_ceiling_and_not_beyond():
    limit = ceiling("nautilus_trader")

    assert bench._within_ceiling("nautilus_trader", limit)
    assert not bench._within_ceiling("nautilus_trader", limit + 1)
    assert bench._within_ceiling("vectorbt", 10_000_000)


def test_an_override_moves_a_ceiling_and_zero_removes_it():
    bench.CEILING_OVERRIDES["nautilus_trader"] = 1_000_000
    assert bench._within_ceiling("nautilus_trader", 1_000_000)
    assert not bench._within_ceiling("nautilus_trader", 1_000_001)

    bench.CEILING_OVERRIDES["nautilus_trader"] = 0
    assert bench._within_ceiling("nautilus_trader", 100_000_000)


def test_a_held_back_engine_gets_a_reason_where_the_site_already_looks_for_one():
    """`unsupported` is the field the website prints under "Not run, and why"."""
    active = ["manifoldbt", "vectorbt", "nautilus_trader"]
    long_run = bench._entry("sma_cross", 1_000_000, frame=_Frame(1_000_000), verdicts={},
                            engines_run=["manifoldbt", "vectorbt"], active=active)
    short_run = bench._entry("sma_cross", 100_000, frame=_Frame(100_000), verdicts={},
                             engines_run=active, active=active)

    assert "nautilus_trader" in long_run["unsupported"]
    assert long_run["unsupported"]["nautilus_trader"].startswith("Not run above 100,000 bars")
    assert long_run["beyond_ceiling"] == {"nautilus_trader": 100_000}
    assert "unsupported" not in short_run
    assert "beyond_ceiling" not in short_run


def test_a_ceiling_is_not_reported_for_a_workload_the_engine_cannot_run_anyway():
    """raptorbt's blocker is the reason on those rows, not a ceiling."""
    entry = bench._entry("sma_cross_costs", 1_000_000, frame=_Frame(1_000_000), verdicts={},
                         engines_run=["manifoldbt", "vectorbt"],
                         active=["manifoldbt", "vectorbt", "raptorbt"])

    assert set(entry["unsupported"]) == {"raptorbt"}
    assert "beyond_ceiling" not in entry


class _Frame:
    """Stands in for a DataFrame where only the length and a digest are read."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return self.rows

    def __getitem__(self, column):
        import numpy as np

        return _Column(np.zeros(self.rows))


class _Column:
    def __init__(self, values):
        self.values = values

    def to_numpy(self, dtype=None):
        return self.values


# --------------------------------------------------------------------------- #
# Sweep points
# --------------------------------------------------------------------------- #
def test_a_plain_point_runs_everyone():
    assert bench.parse_point("20000:5000") == (20_000, 5_000, ())


def test_a_bare_oos_still_means_vectorbt():
    assert bench.parse_point("20000:20000:oos") == (20_000, 20_000, ("vectorbt",))


def test_oos_can_name_several_engines():
    assert bench.parse_point("20000:5000:oos=vectorbt,nautilus_trader") == (
        20_000, 5_000, ("vectorbt", "nautilus_trader"))


def test_the_reference_and_strangers_cannot_be_declared_out_of_scope():
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_point("20000:5000:oos=manifoldbt")
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_point("20000:5000:oos=zipline")
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_point("20000:5000:maybe")


ALL = ["manifoldbt", "vectorbt", "raptorbt", "nautilus_trader"]


def test_a_big_grid_holds_back_the_looped_engine_without_being_told_to():
    """The recipe on the website only says `:oos`; it must not run Nautilus for hours."""
    assert bench._sweep_out_of_scope(5_000, (), ALL) == ["nautilus_trader"]
    assert bench._sweep_out_of_scope(5_000, ("vectorbt",), ALL) == ["vectorbt", "nautilus_trader"]


def test_a_small_grid_is_run_by_everyone():
    assert bench._sweep_out_of_scope(36, (), ALL) == []
    assert bench._sweep_out_of_scope(64, (), ALL) == []
    assert bench._sweep_out_of_scope(65, (), ALL) == ["nautilus_trader"]


def test_the_sweep_ceiling_can_be_moved_or_lifted():
    bench.SWEEP_CEILING_OVERRIDES["nautilus_trader"] = 500
    assert bench._sweep_out_of_scope(500, (), ALL) == []
    bench.SWEEP_CEILING_OVERRIDES["nautilus_trader"] = 0
    assert bench._sweep_out_of_scope(50_000, (), ALL) == []


def test_an_engine_that_is_not_running_is_not_listed_as_out_of_scope():
    assert bench._sweep_out_of_scope(5_000, ("vectorbt",), ["manifoldbt", "raptorbt"]) == []


def test_the_nautilus_anchor_spans_the_grid_it_stands_in_for():
    values = list(range(100))

    picked = sweep_child.spread(values, 4)

    assert picked[0] == 0 and picked[-1] == 99
    assert len(picked) == 4
    assert picked == sorted(set(picked))


def test_a_small_grid_is_its_own_anchor():
    assert sweep_child.spread([5, 7], 4) == [5, 7]


def test_the_reason_for_a_looped_engine_carries_what_was_measured():
    text = sweep_child.oos_reason("nautilus_trader", 20_000, 5_000,
                                  anchor_cells=16, anchor_s=16.0)

    assert "1.00 s per backtest at 20,000 bars" in text
    assert "5,000 combinations" in text
    assert "about 83 minutes" in text


def test_a_grid_held_back_by_the_sweep_ceiling_says_so_even_when_it_is_cheap():
    text = sweep_child.oos_reason("nautilus_trader", 5_000, 100, anchor_cells=16,
                                  anchor_s=3.2, loop_limit=64)

    assert "about 20 seconds per call" in text
    assert "held to 64 combinations" in text
    assert "past the time budget" not in text


def test_the_vectorbt_reason_is_the_memory_one():
    text = sweep_child.oos_reason("vectorbt", 20_000, 20_000, anchor_cells=250, anchor_s=1.0)

    assert text == sweep_child.OOS_VBT_REASON


@pytest.mark.parametrize("seconds,expected", [
    (42.0, "42 seconds"), (600.0, "10 minutes"), (9_000.0, "2.5 hours"),
])
def test_durations_read_in_the_unit_a_person_would_use(seconds, expected):
    assert sweep_child._duration(seconds) == expected


# --------------------------------------------------------------------------- #
# The report, for an engine that was held back
# --------------------------------------------------------------------------- #
def _row(workload, bars, engines, **extra):
    row = {
        "workload": workload, "title": workload, "bars": bars, "status": "exact",
        "engines": engines,
        "timings": {e: {"median_s": 0.01, "iqr_over_median": 0.01} for e in engines},
        "parity": {e: {"status": "exact", "publishable": True, "diffs": {}}
                   for e in engines if e != "manifoldbt"},
        "speedup": {e: {"median_of_ratios": 2.0, "min": 1.0, "max": 3.0}
                    for e in engines if e != "manifoldbt"},
    }
    row.update(extra)
    return row


def _payload(rows, **extra):
    engines = ["manifoldbt", "nautilus_trader"]
    payload = {
        "schema_version": 2, "reference": "manifoldbt", "engines": engines, "reps": 3,
        "generated_at": "2026-10-22T00:00:00+00:00",
        "environment": {
            "os": "Linux", "arch": "x86_64", "cpu": "a cpu", "logical_cores": 4,
            "ram_gb": 16.0, "python": "3.12.0", "pinned_cores": None,
            "versions": {"manifoldbt": "0.28.0", "nautilus_trader": "1.231.0",
                         "numpy": "2.4.3", "numba": "0.64.0", "pandas": "2.3.3"},
        },
        "results": rows,
    }
    payload.update(extra)
    return payload


WHY = "Not run above 100,000 bars. Event-driven."


def test_one_reason_is_printed_once_for_all_the_workloads_it_covers():
    rows = [
        _row("sma_cross", 1_000_000, ["manifoldbt"], unsupported={"nautilus_trader": WHY}),
        _row("multi_asset", 1_000_000, ["manifoldbt"], unsupported={"nautilus_trader": WHY}),
    ]

    text = report.render(_payload(rows))

    assert text.count(WHY) == 1
    assert "`sma_cross`, `multi_asset`" in text


def test_an_engine_without_a_drawdown_is_said_not_to_have_one():
    summarised = _row("sma_cross_metrics", 100_000, ["manifoldbt", "nautilus_trader"],
                      paired_with=["sma_cross"])
    summarised["parity"]["nautilus_trader"]["diffs"] = {
        "max_drawdown_absent": True, "ratio_basis": "native"}
    plain = _row("sma_cross", 100_000, ["manifoldbt", "nautilus_trader"],
                 paired_with=["sma_cross_metrics"])
    # The longest row is one the engine did not run on: its verdict is read from
    # the longest row it did.
    longer = _row("sma_cross_metrics", 1_000_000, ["manifoldbt"],
                  paired_with=["sma_cross"], unsupported={"nautilus_trader": WHY})
    longer_plain = _row("sma_cross", 1_000_000, ["manifoldbt"],
                        paired_with=["sma_cross_metrics"], unsupported={"nautilus_trader": WHY})

    text = report.render(_payload([plain, summarised, longer_plain, longer]))

    assert "reports no maximum drawdown" in text
    assert "was not compared" not in text


def test_a_memory_figure_taken_at_the_ceiling_says_so():
    memory = {
        "workload": "sma_cross", "bars": 2_000_000, "engines": ["manifoldbt", "nautilus_trader"],
        "manifoldbt": {"added_mb_per_million_bars": 0.006, "bars": 2_000_000},
        "nautilus_trader": {"added_mb_per_million_bars": 4.8, "bars": 100_000},
    }

    text = report.render(_payload([_row("sma_cross", 100_000, ["manifoldbt"])], memory=memory))

    assert "5 MB (at 100,000 bars)" in text
    assert "0 MB |" in text


def test_the_cpu_row_survives_an_engine_that_did_not_run_the_longest_series():
    rows = [
        _row("sma_cross", 100_000, ["manifoldbt", "nautilus_trader"],
             cpu_over_wall={"manifoldbt": 1.0, "nautilus_trader": 0.99}),
        _row("sma_cross", 10_000_000, ["manifoldbt"], cpu_over_wall={"manifoldbt": 3.5}),
    ]

    text = report.render(_payload(rows, memory=None, cold_start=None))

    assert "CPU over wall time, at the longest series each ran" in text
    assert "3.50 (at 10,000,000)" in text
    assert "0.99 (at 100,000)" in text


def test_a_reentry_that_differs_only_in_quantity_reports_how_far():
    row = _row("bracket_sl_tp", 100_000, ["manifoldbt", "nautilus_trader"], status="documented")
    row["parity"]["nautilus_trader"] = {"status": "documented", "publishable": False,
                                        "diffs": {"final_equity_vs_capital": 7e-6}}
    row["divergence_scale"] = {
        "manifoldbt": {"reentries_on_exit_bar": 900, "round_trips": 2300,
                       "share_of_round_trips": 0.39, "sl_exits": 400, "tp_exits": 300},
        "nautilus_trader": {"round_trips": 2300, "reentries_on_exit_bar": 900,
                            "reentry_quantity_gap_median": 3e-4,
                            "reentry_quantity_gap_max": 1e-3},
    }

    text = report.render(_payload([row]))

    assert "nautilus_trader books 2300." in text
    assert "a median of 3.0e-04" in text
