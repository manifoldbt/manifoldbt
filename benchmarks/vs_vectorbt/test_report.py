"""Guards for the divergence annex: what is counted, and how it is rendered.

`README.md` says a `documented` verdict is published as "the timing, in an
annex, with the cause and **its measured size**". The size is a property of the
point: the same divergence is 0.92% of capital at 100,000 bars and 10.07% at
1,000,000. These tests hold `render()` to that promise for every published
divergent timing, not just the first one.

They drive `render()` rather than the helper underneath it, because the defect
they guard against lived in `render()`: an earlier version of these tests
exercised only the helper, and restoring the bug at its real site left all of
them green.

The counting half lives in ``divergence.py`` rather than inside an adapter for
the same reason: an adapter imports its engine, and the job that runs this file
installs pytest and nothing else. A measurement no CI can reach is back to being
a claim.

Run with ``pytest benchmarks/vs_vectorbt/test_report.py``. Nothing here imports
an engine, so no wheel and no competitor is needed.
"""
from __future__ import annotations

import itertools
import re

import divergence
import report

# The rendered deferral clause, as (lost, deferred). One place, so a wording change
# is one edit rather than a hunt through three tests.
_DEFERRAL_RE = r"vectorbt books \d+, (\d+) fewer than manifoldbt; (\d+) of them opened"


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


def _deferring(row, *, round_trips, deferred, on_exit_bar=0):
    """Add a vectorbt-shaped scale to a row: an engine that defers rather than skips."""
    row = dict(row)
    row["divergence_scale"] = dict(row["divergence_scale"])
    row["divergence_scale"]["vectorbt"] = {
        "round_trips": round_trips,
        "final_equity": 93097.73,
        "reentries_deferred": deferred,
        "reentries_on_exit_bar": on_exit_bar,
    }
    return row


def test_a_deferring_engine_is_measured_by_the_delay_not_the_population():
    """raptorbt's measure does not transfer to vectorbt.

    vectorbt books 2215 round-trips against the reference's 2309, so counting
    the population would say it diverges on 94 where raptorbt diverges on 925 --
    ten times less by count, while sitting further from the reference in capital
    at that point. The line has to say what it deferred, not only what it lost.

    Both figures are attributed to the engine they were measured on. "831 of
    them a bar late and 94 not at all" reads as a claim that those 831 are the
    reference's own re-entries, and that correspondence does not exist -- see
    the comment in `_divergence_line`.
    """
    text = report.render(_payload(
        _deferring(_row(100_000, 925, 2309, 1384), round_trips=2215, deferred=831)))

    assert ("vectorbt books 2215, 94 fewer than manifoldbt; 831 of them opened on the "
            "bar after vectorbt's own exit with the level still true.") in text


def test_the_two_halves_sum_to_the_reference_count():
    """831 deferred + 94 lost = the 925 the reference re-enters on the exit bar.

    Printed from two different sources on purpose -- the deferred count is
    measured on the challenger, what it lost is the round-trip delta -- so the
    identity is a check rather than a restatement.
    """
    text = report.render(_payload(
        _deferring(_row(100_000, 925, 2309, 1384), round_trips=2215, deferred=831),
        _deferring(_row(1_000_000, 9330, 23075, 13745), round_trips=21983, deferred=8238)))

    found = re.findall(r"re-enter on the exit bar.*?" + _DEFERRAL_RE, text)
    assert len(found) == 2, found
    for (lost, deferred), reentries in zip(found, (925, 9330)):
        assert int(deferred) + int(lost) == reentries, (deferred, lost, reentries)


def test_a_broken_identity_is_visible_in_the_report_not_papered_over():
    """The failure mode of the line above, and the reason `lost` is not derived
    from the reference's own count. If vectorbt ever stops deferring exactly the
    population the reference re-enters on, the printed halves must stop summing
    in front of the reader. A renderer that computes one half from the other has
    no way to show that, because its arithmetic is true by construction.
    """
    # 700 deferred against a 925 re-entry count and a 94 round-trip delta: the
    # mechanism has changed and the numbers no longer close.
    text = report.render(_payload(
        _deferring(_row(100_000, 925, 2309, 1384), round_trips=2215, deferred=700)))

    lost, deferred = re.search(_DEFERRAL_RE, text).groups()
    assert int(deferred) + int(lost) != 925


def test_a_skipping_engine_keeps_its_own_shorter_line():
    """raptorbt does not defer, so it gets no deferral clause. The engine that
    was already measured must not acquire a second measure it did not report."""
    text = report.render(_payload(
        _deferring(_row(100_000, 925, 2309, 1384), round_trips=2215, deferred=831)))

    assert "raptorbt books 1384." in text
    assert "raptorbt books 1384," not in text


def test_a_challenger_that_re_enters_on_the_exit_bar_says_so():
    """`reentries_on_exit_bar` is zero for vectorbt and the workload note says
    why: one order per bar. Reporting it makes that note falsifiable -- if the
    count is ever non-zero the annex says so rather than dropping it."""
    text = report.render(_payload(
        _deferring(_row(100_000, 925, 2309, 1384),
                   round_trips=2215, deferred=831, on_exit_bar=7)))

    assert "re-enters on the exit bar itself 7 times" in text

    quiet = report.render(_payload(
        _deferring(_row(100_000, 925, 2309, 1384), round_trips=2215, deferred=831)))
    assert "re-enters on the exit bar itself" not in quiet


# --------------------------------------------------------------------------- #
# What is counted
# --------------------------------------------------------------------------- #

def test_a_deferred_reentry_is_counted():
    """Exit on bar 10 with the level still true, entry on bar 11: the reference
    would have entered at the close of bar 10, and this engine took it late."""
    counts = divergence.reentry_counts(
        entry_idx=[0, 11], exit_idx=[10, 20], level=[True] * 21)

    assert counts["reentries_deferred"] == 1
    assert counts["reentries_on_exit_bar"] == 0


def test_an_ordinary_next_bar_reentry_is_not_a_deferral():
    """The one that makes the published number 831 instead of 864.

    Same shape -- exit on bar 10, entry on bar 11 -- but the level was FALSE at
    bar 10, so the reference does not re-enter there either and both engines
    open on bar 11 for the same reason. Nothing has diverged, and counting it
    inflates the measure by the ordinary re-entries in the run.
    """
    level = [True] * 21
    level[10] = False
    counts = divergence.reentry_counts(entry_idx=[0, 11], exit_idx=[10, 20], level=level)

    assert counts["reentries_deferred"] == 0


def test_a_same_bar_reentry_is_counted_separately():
    """What the reference does, and what this engine is claimed not to do."""
    counts = divergence.reentry_counts(
        entry_idx=[0, 10], exit_idx=[10, 20], level=[True] * 21)

    assert counts["reentries_on_exit_bar"] == 1
    assert counts["reentries_deferred"] == 0


def test_a_later_reentry_is_neither():
    counts = divergence.reentry_counts(
        entry_idx=[0, 14], exit_idx=[10, 20], level=[True] * 21)

    assert counts == {"reentries_deferred": 0, "reentries_on_exit_bar": 0}


def test_the_trades_are_ordered_before_they_are_paired():
    """Records come back in whatever order the engine wrote them. Pairing an
    entry with the wrong previous exit measures nothing at all.

    Every permutation, not one shuffle. The first version of this test picked a
    single reordering and it happened to be one the unsorted code got right by
    luck, so it passed with the sort removed -- a guard that guarded nothing.
    """
    trades = [(0, 10), (11, 20), (30, 40), (41, 50)]
    expected = {"reentries_deferred": 2, "reentries_on_exit_bar": 0}

    for order in itertools.permutations(trades):
        counts = divergence.reentry_counts(
            entry_idx=[e for e, _ in order], exit_idx=[x for _, x in order],
            level=[True] * 51)
        assert counts == expected, order


def test_a_run_with_nothing_to_pair_counts_nothing():
    """One trade has no previous exit; none has no trades. Neither is an error."""
    assert divergence.reentry_counts([0], [10], [True] * 11) == {
        "reentries_deferred": 0, "reentries_on_exit_bar": 0}
    assert divergence.reentry_counts([], [], []) == {
        "reentries_deferred": 0, "reentries_on_exit_bar": 0}


# --------------------------------------------------------------------------- #
# The identity, across a swept series length
# --------------------------------------------------------------------------- #

def test_the_identity_holds_at_lengths_nobody_publishes():
    """The two counts must close at series lengths the results matrix does not contain.

    This is the one test here that needs the engines, and it exists because the
    checks that did not sweep a length broke. Three published sizes agreed with a version of this
    work that was wrong twice over, and only a fourth length showed it.

    The lengths are chosen to discriminate, not to pass. 110,000 and 120,000 on the
    default seed both end holding a position opened on the same bar as the exit before
    it, which is the configuration both defects here lived in; 100,000 is the control
    that passes under either of them, and is here so a failure at the other two
    has a passing control beside it.

    It skips where no engine is installed, which is the `bench-report` job on
    every push, so it prints a visible `skipped` line there. Nothing in this
    repository's CI runs it with the wheels present -- that is worth knowing
    rather than assuming, and it is why the numbers in the pull request were
    produced by hand against `requirements-lock.txt`.
    """
    import pytest
    pytest.importorskip("vectorbt")
    pytest.importorskip("manifoldbt")
    pytest.importorskip("raptorbt")
    import tempfile

    import data as data_mod
    import engine_mbt
    import engine_rbt
    import engine_vbt

    for bars in (100_000, 110_000, 120_000):
        frame = data_mod.make_ohlcv(bars)
        reference = engine_mbt.diagnose("bracket_sl_tp", frame, tempfile.mkdtemp())
        measured = engine_vbt.diagnose("bracket_sl_tp", frame, tempfile.mkdtemp())
        lost = reference["round_trips"] - measured["round_trips"]

        assert measured["reentries_deferred"] + lost == reference["reentries_on_exit_bar"], (
            "deferred + lost must be the reference's own re-entry count",
            bars, measured["reentries_deferred"], lost, reference["reentries_on_exit_bar"])
        assert measured["reentries_on_exit_bar"] == 0, bars

        # And the reference's own promise, which is what `engine_rbt.diagnose`'s
        # docstring says the pair of counts exists for: raptorbt does not re-arm, so
        # its round-trip count is the reference's minus that population EXACTLY.
        # Counting an unclosed entry broke this on 26 of 90 swept
        # (length, seed) pairs, 110,000 and 120,000 among them.
        skipping = engine_rbt.prepare("bracket_sl_tp", frame, tempfile.mkdtemp())()
        assert reference["round_trips"] - reference["reentries_on_exit_bar"] == \
            skipping["round_trips"], (
                "the reference's re-entry count must subtract exactly to raptorbt's",
                bars, reference["round_trips"], reference["reentries_on_exit_bar"],
                skipping["round_trips"])
