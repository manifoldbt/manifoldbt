"""Rendering guards for the divergence annex.

`README.md` says a `documented` verdict is published as "the timing, in an
annex, with the cause and **its measured size**". The size is a property of the
point: the same divergence is 0.92% of capital at 100,000 bars and 10.07% at
1,000,000. These tests hold `render()` to that promise for every published
divergent timing, not just the first one.

They drive `render()` rather than the helper underneath it, because the defect
they guard against lived in `render()`: an earlier version of these tests
exercised only the helper, and restoring the bug at its real site left all of
them green.

Run with ``pytest benchmarks/vs_vectorbt/test_report.py``. Nothing here imports
an engine, so no wheel and no competitor is needed.
"""
from __future__ import annotations

import report


def _row(bars: int, reentries: int, round_trips: int, challenger_trips: int,
         *, scale: bool = True, workload: str = "bracket_sl_tp"):
    row = {
        "workload": workload,
        "title": workload,
        "bars": bars,
        "status": "documented",
        "engines": ["manifoldbt", "raptorbt"],
        "timings": {"seconds": {"manifoldbt": 0.007, "raptorbt": 0.007},
                    "ratio": {"raptorbt": 1.0}},
        "parity": {"raptorbt": {"status": "documented", "publishable": False,
                                "diffs": {"final_equity_vs_capital": 0.1,
                                          "round_trips_delta": reentries}}},
    }
    if scale:
        row["divergence_scale"] = {
            "manifoldbt": {
                "reentries_on_exit_bar": reentries,
                "round_trips": round_trips,
                "share_of_round_trips": reentries / round_trips,
                "sl_exits": reentries // 2,
                "tp_exits": reentries // 3,
            },
            "raptorbt": {"round_trips": challenger_trips},
        }
    return row


ENVIRONMENT = {
    "os": "Linux", "os_release": "6.17.0", "arch": "x86_64", "cpu": "a cpu",
    "logical_cores": 4, "ram_gb": 16.8, "python": "3.12.14", "ci": True,
    "pinned_cores": None, "engines": ["manifoldbt", "raptorbt"],
    "versions": {"manifoldbt": "0.20.0", "raptorbt": "0.9.0",
                 "numpy": "2.4.3", "numba": "0.64.0", "pandas": "2.3.3"},
}


def _payload(*rows):
    return {
        "schema_version": 2,
        "reference": "manifoldbt",
        "engines": ["manifoldbt", "raptorbt"],
        "reps": 7,
        "generated_at": "2026-08-25T17:18:43.779130+00:00",
        "environment": dict(ENVIRONMENT),
        "results": list(rows),
    }


THREE_POINTS = (_row(100_000, 925, 2309, 1384),
                _row(1_000_000, 9330, 23075, 13745),
                _row(10_000_000, 93295, 231082, 137787))


def test_every_published_divergent_timing_carries_its_measured_size():
    """Three timings in the annex, three measured sizes under it.

    Rendering only the first explained the mildest point in the table and left
    the 1,000,000-bar row -- the one where raptorbt crosses to the far side of
    the reference -- published with no size beside it.
    """
    text = report.render(_payload(*THREE_POINTS))

    assert text.count("re-enter on the exit bar") == 3
    for bars in ("100,000", "1,000,000", "10,000,000"):
        assert "at {} bars".format(bars) in text, bars


def test_each_line_carries_the_challenger_count_from_its_own_point():
    text = report.render(_payload(*THREE_POINTS))

    for count in ("1384", "13745", "137787"):
        assert "raptorbt books {}.".format(count) in text, count


def test_the_cause_is_stated_once_not_once_per_point():
    """The other direction: the naive fix reprints the link under every row."""
    text = report.render(_payload(*THREE_POINTS))

    assert text.count("Cause in ") == 1


def test_a_documented_point_nobody_measured_invents_no_size():
    text = report.render(_payload(_row(100_000, 0, 1, 0, scale=False)))

    assert "re-enter on the exit bar" not in text
    assert "Cause in " not in text


def test_the_annex_still_lists_the_timings_it_explains():
    text = report.render(_payload(*THREE_POINTS))

    assert "Results differ, kept out of the headline" in text
    assert text.count("| bracket_sl_tp |") == 3


def test_a_divergence_on_an_untimed_point_stays_out_of_the_annex():
    """The annex explains its own table.

    A documented point whose timing was never published appears nowhere in the
    report, so a measured size for it would be an explanation with nothing to
    explain. Feeding render() `documented` instead of `timed` produced exactly
    that: four bullets under a three-row table.
    """
    untimed = _row(50_000, 500, 1200, 700)
    del untimed["timings"]

    text = report.render(_payload(*(THREE_POINTS + (untimed,))))

    assert "at 50,000 bars" not in text
    assert text.count("re-enter on the exit bar") == 3
