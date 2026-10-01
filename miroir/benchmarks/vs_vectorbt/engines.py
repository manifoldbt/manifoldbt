"""The engine registry: who is in the comparison, and what each one can do.

One backtester is the *reference* and the others are *challengers*. Every parity
check joins a challenger to the reference, never two challengers to each other:
with three engines there are three pairs, and reporting all of them turns a
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
worth reporting. raptorbt returns its own, on its own basis, and comparing those
numbers to manifoldbt's would report a units mismatch as a disagreement. The
parity gate never depends on this either way: it gates on money, round-trips and
drawdown, all three basis-free.
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


REFERENCE = "manifoldbt"

ENGINES: Dict[str, Engine] = {
    e.name: e
    for e in (
        Engine("manifoldbt", "mbt", "engine_mbt", "daily_365", "manifoldbt"),
        Engine("vectorbt", "vbt", "engine_vbt", "daily_365", "vectorbt"),
        Engine("raptorbt", "rbt", "engine_rbt", "native", "raptorbt"),
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
