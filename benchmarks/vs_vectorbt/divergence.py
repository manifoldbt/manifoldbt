"""Counting the bracket divergence from a trade list.

Kept in a module of its own, with no engine import and no third-party import,
for one reason: it is the measurement, and the measurement is what a reader has
least reason to take on faith. `test_report.py` runs in a job that installs
pytest and nothing else, so anything that needs numpy or a wheel is out of reach
there. A few hundred thousand trades in plain Python is untimed work
either way: bench.py calls this outside the timed region.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence


def reentry_counts(entry_idx: Sequence[int], exit_idx: Sequence[int],
                   level: Sequence[bool]) -> Dict[str, Any]:
    """How an engine handles the re-entries the reference takes on the exit bar.

    ``level`` is the entry condition per bar, so ``level[exit]`` is what the
    reference sees at the close of the bar it just exited on: true means it
    re-enters right there.

    Two counts come back:

    ``reentries_deferred``
        entries that land exactly one bar after the previous exit *while the
        level was already true at that exit bar*. The qualifier is the whole
        measurement. Without it the count also picks up ordinary re-entries --
        the level went false on the exit bar and true again on the next one, so
        both engines enter on that bar for the same reason and nothing has
        diverged. There are 33 of those at 100,000 bars against 831 real
        deferrals, and counting them would put the published figure at 864.

    ``reentries_on_exit_bar``
        entries on the exit bar itself. For an engine that processes one order
        per bar this is zero, which is the point of reporting it: the claim
        becomes a measurement that can fail rather than a sentence in a note.
    """
    pairs = sorted(zip(entry_idx, exit_idx))
    deferred = 0
    same_bar = 0
    for (entry, _), (_, previous_exit) in zip(pairs[1:], pairs):
        gap = entry - previous_exit
        if gap == 0:
            same_bar += 1
        elif gap == 1 and level[previous_exit]:
            deferred += 1
    return {"reentries_deferred": deferred, "reentries_on_exit_bar": same_bar}
