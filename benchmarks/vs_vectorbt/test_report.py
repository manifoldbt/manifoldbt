"""Rendering tests for report.py, on payloads built by hand.

No engine is imported and nothing is timed: these only check what the
published Markdown says about a given results file.

    python -m pytest benchmarks/vs_vectorbt/test_report.py -q
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import report  # noqa: E402

ENGINES = ["manifoldbt", "vectorbt", "raptorbt"]


def _timing(seconds: float) -> Dict[str, Any]:
    return {"median_s": seconds, "min_s": seconds, "max_s": seconds, "iqr_s": 0.0}


def _bracket_point(bars: int, reentries: int, trips: int, raptorbt_trips: int,
                   timed: bool = True) -> Dict[str, Any]:
    """A documented ``bracket_sl_tp`` point, with or without a measured size."""
    point: Dict[str, Any] = {
        "workload": "bracket_sl_tp",
        "bars": bars,
        "status": "documented",
        "engines": list(ENGINES),
        "parity": {},
        "divergence_scale": {
            "manifoldbt": {
                "reentries_on_exit_bar": reentries,
                "round_trips": trips,
                "share_of_round_trips": reentries / trips,
                "sl_exits": trips // 2,
                "tp_exits": trips - trips // 2,
            },
            "raptorbt": {"round_trips": raptorbt_trips},
        },
    }
    if timed:
        point["timings"] = {e: _timing(0.01 * (i + 1)) for i, e in enumerate(ENGINES)}
        point["speedup"] = {
            "vectorbt": {"median_of_ratios": 2.0},
            "raptorbt": {"median_of_ratios": 3.0},
        }
    return point


def _results_file(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    versions = {e: "1.0" for e in ENGINES}
    versions.update(numpy="2.0", numba="0.60", pandas="2.2")
    return {
        "schema_version": 2,
        "reference": "manifoldbt",
        "engines": list(ENGINES),
        "reps": 3,
        "generated_at": "2026-01-01T00:00:00Z",
        "environment": {
            "os": "Linux", "arch": "x86_64", "cpu": "test cpu", "logical_cores": 4,
            "ram_gb": 16, "python": "3.12", "versions": versions,
        },
        "results": points,
    }


def _size_lines(text: str) -> List[str]:
    return [line for line in text.splitlines() if "re-enter on the exit bar" in line]


def test_each_documented_point_states_its_own_size():
    points = [
        _bracket_point(10_000, 90, 230, 140),
        _bracket_point(100_000, 925, 2309, 1384),
        _bracket_point(1_000_000, 9100, 23000, 21000),
    ]

    lines = _size_lines(report.render(_results_file(points)))

    assert len(lines) == 3
    for bars, line in zip(("10,000", "100,000", "1,000,000"), lines):
        assert "at {} bars".format(bars) in line


def test_the_sizes_are_list_items_under_one_cause():
    points = [_bracket_point(10_000, 90, 230, 140), _bracket_point(100_000, 925, 2309, 1384)]

    text = report.render(_results_file(points))

    assert all(line.startswith("- `bracket_sl_tp`") for line in _size_lines(text))
    assert text.count("Cause in ") == 1


def test_a_point_without_a_size_adds_no_line():
    sized = _bracket_point(10_000, 90, 230, 140)
    unsized = _bracket_point(100_000, 925, 2309, 1384)
    del unsized["divergence_scale"]

    text = report.render(_results_file([sized, unsized]))

    assert len(_size_lines(text)) == 1
    assert text.count("| bracket_sl_tp |") == 2


def test_no_size_at_all_means_no_cause_line():
    point = _bracket_point(10_000, 90, 230, 140)
    del point["divergence_scale"]

    assert "Cause in " not in report.render(_results_file([point]))


def test_a_divergence_on_an_untimed_point_stays_out_of_the_annex():
    """The annex explains its own table.

    A documented point whose timing was never published appears nowhere in the
    report, so a measured size for it would be an explanation with nothing to
    explain. Feeding render() `documented` instead of `timed` produced exactly
    that: four lines under a three-row table.
    """
    points = [
        _bracket_point(10_000, 90, 230, 140),
        _bracket_point(100_000, 925, 2309, 1384),
        _bracket_point(1_000_000, 9100, 23000, 21000),
        _bracket_point(50_000, 500, 1200, 700, timed=False),
    ]

    text = report.render(_results_file(points))

    assert "at 50,000 bars" not in text
    assert len(_size_lines(text)) == 3
