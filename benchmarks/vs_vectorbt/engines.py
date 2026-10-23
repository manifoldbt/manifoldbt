"""The engine registry: who is in the comparison, and what each one can do.

One backtester is the *reference* and the others are *challengers*. Every parity
check joins a challenger to the reference, never two challengers to each other:
with four engines there are six pairs, and reporting all of them turns a
speed benchmark into a matrix nobody reads. manifoldbt is the reference because
this harness ships with it, which is a statement about the plumbing and not a
claim about the engine: the reference gets no advantage from the position, it is
simply the one every timing is divided by.

A challenger that is not installed is skipped with a printed line rather than
crashing the run. The CI lock file pins all of them, so a skip on a runner is a
finding; a skip on a laptop is somebody who did not want to install a competitor
to read their own numbers.

Ratio basis
-----------
``ratio_basis`` records how an engine annualises Sharpe and Sortino. manifoldbt
and the vectorbt adapter both compute them on daily returns annualised by
sqrt(365), so their ratios are directly comparable and small differences are
worth reporting. raptorbt and NautilusTrader return their own, on their own
bases, and comparing those numbers to manifoldbt's would report a units mismatch
as a disagreement. The parity gate never depends on this either way: it gates on
money, round-trips and, where an engine offers one, drawdown, all basis-free.

Ceilings
--------
``max_bars`` is the longest series an engine is run on by default, and it exists
for one reason: an event-driven engine replays every bar through a simulated
exchange and a Python callback, so its cost is seconds per 100,000 bars, where
the others spend milliseconds. An interleaved repetition loop at 10 million bars
would run for hours inside a job that has to finish. The ceiling is a time
budget, not a verdict, and it is printed in the report with its reason rather
than left as an empty cell. ``bench.py --engine-max-bars`` lifts it for a local
run. ``max_sweep_combos`` is the same idea for a parameter grid: an engine with
no sweep API loops a full backtest per cell, and a grid of thousands of cells at
seconds each is hours.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Engine:
    name: str
    # Short code used by the out-of-process probes, which take argv strings.
    code: str
    module: str
    # "daily_365" ratios are comparable with the reference's; "native" are not.
    ratio_basis: str
    # PyPI distribution name, for the version stamp in the result envelope.
    distribution: str
    # Longest series (in bars) this engine is run on by default; None for no
    # ceiling. See the module docstring.
    max_bars: int | None = None
    # What the report says where the ceiling stopped a run.
    ceiling_why: str = ""
    # Largest parameter grid (in combinations) this engine is looped over by
    # default, or None. Only an engine with no sweep API needs one: its grid is a
    # loop of full backtests, so the cost is combinations times one backtest.
    max_sweep_combos: int | None = None


REFERENCE = "manifoldbt"

ENGINES: Dict[str, Engine] = {
    e.name: e
    for e in (
        Engine("manifoldbt", "mbt", "engine_mbt", "daily_365", "manifoldbt"),
        Engine("vectorbt", "vbt", "engine_vbt", "daily_365", "vectorbt"),
        Engine("raptorbt", "rbt", "engine_rbt", "native", "raptorbt"),
        Engine(
            "nautilus_trader", "nt", "engine_nt", "native", "nautilus_trader",
            max_bars=100_000,
            max_sweep_combos=64,
            ceiling_why=(
                "Not run above {limit:,} bars. NautilusTrader is event-driven: every "
                "bar goes through a simulated exchange and a Python strategy "
                "callback, so one call costs seconds per 100,000 bars and grows "
                "linearly with the series. Repeated and interleaved, a longer series "
                "does not fit the time budget of the job. The measured time at the "
                "ceiling is in the table, and `bench.py --engine-max-bars "
                "nautilus_trader=N` lifts the ceiling for a local run."
            ),
        ),
    )
}

CHALLENGERS: List[str] = [n for n in ENGINES if n != REFERENCE]

BY_CODE: Dict[str, Engine] = {e.code: e for e in ENGINES.values()}


def adapter(name: str):
    """Import an adapter module on demand.

    Lazy on purpose: the cold-start probe measures a process that has imported
    exactly one engine, and a registry that imported all three at module scope
    would make that measurement impossible to take.
    """
    return importlib.import_module(ENGINES[name].module)


def installed(name: str) -> bool:
    try:
        adapter(name)
        return True
    except ImportError:
        return False


def present(names: List[str]) -> List[str]:
    """Filter to the engines actually importable here, preserving order."""
    return [n for n in names if installed(n)]


def sweep_ceiling(name: str, overrides: Dict[str, int] | None = None) -> int | None:
    """The largest grid (combinations) this engine is looped over; 0 removes it."""
    if overrides and name in overrides:
        return overrides[name] or None
    return ENGINES[name].max_sweep_combos


def ceiling(name: str, overrides: Dict[str, int] | None = None) -> int | None:
    """The longest series this engine is run on, honouring a command-line override.

    An override of 0 removes the ceiling.
    """
    if overrides and name in overrides:
        return overrides[name] or None
    return ENGINES[name].max_bars
