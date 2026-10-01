"""The gate: no speed number is published for a workload the engines disagree on.

A benchmark between backtesters is only a benchmark if they did the same work.
This module compares what a challenger produced against the reference and
classifies the result, and ``bench.py`` refuses to report a timing for anything
it classifies as a failure.

Every comparison is one challenger against the reference. Challengers are never
joined to each other: three engines make three pairs, and a table of pairs is a
matrix, not a benchmark. It also would not add anything, since agreement with
the reference is transitive enough for the only question being asked here, which
is whether a published timing describes the same simulation.

Three verdicts:

``exact``
    Agreement down to float-reordering noise. The timing is publishable.
``documented``
    The two disagree, the workload declared it in advance for this specific
    engine, and the reason is written down. The timing goes to the annex with
    the reason attached.
``failed``
    They disagree and nobody predicted it. That is a finding about the engines,
    not about their speed: the timing is withheld.
"""
from __future__ import annotations

from typing import Any, Dict

from engines import ENGINES, REFERENCE
from workloads import CAPITAL, expectation, why

# Float reordering across two implementations of the same arithmetic lands
# around 1e-13 of the account on a million bars. Anything above this is a
# different decision somewhere, not a different summation order.
REL_TOL = 1e-9


def _rel(a: float, b: float) -> float:
    """Plain relative difference, for quantities that are not money."""
    return abs(a - b) / max(1e-12, abs(b))


def _vs_capital(a: float, b: float) -> float:
    """Difference as a fraction of the money at risk, not of the result itself.

    Anchoring on the result breaks exactly when the result is interesting: a
    strategy that ends near zero equity, or near zero return, turns a difference
    of a hundredth of a cent into a 1% relative error and fails a comparison the
    engines actually passed. The account size is the stable yardstick."""
    return abs(a - b) / CAPITAL


def compare(ref: Dict[str, Any], other: Dict[str, Any], key: str, engine: str) -> Dict[str, Any]:
    expected = expectation(key, engine)

    diffs = {
        "final_equity_vs_capital": _vs_capital(ref["final_equity"], other["final_equity"]),
        "round_trips_delta": ref["round_trips"] - other["round_trips"],
        "total_fees_vs_capital": _vs_capital(ref["total_fees"], other["total_fees"]),
    }
    agrees = (
        diffs["final_equity_vs_capital"] <= REL_TOL
        and diffs["round_trips_delta"] == 0
        and diffs["total_fees_vs_capital"] <= REL_TOL
    )

    # Workloads that also produce a performance summary are gated on the
    # drawdown, which every engine here computes at full bar resolution and
    # which must match. The ratios are reported but not gated: the reference
    # buckets its daily returns slightly differently from vectorbt, a difference
    # worth stating rather than hiding, and worth nothing at all as an argument
    # about speed.
    if other.get("max_drawdown") is not None and ref.get("max_drawdown") is not None:
        diffs["max_drawdown_rel"] = _rel(ref["max_drawdown"], other["max_drawdown"])
        agrees = agrees and diffs["max_drawdown_rel"] <= REL_TOL
        # Only between engines that annualise the same way. raptorbt returns its
        # ratios on its own basis, and subtracting those from the reference's
        # would publish a units mismatch as a disagreement (measured: Sharpe
        # 0.21 against 8.14 on a run whose equity curve is bit-identical).
        if ENGINES[engine].ratio_basis == ENGINES[REFERENCE].ratio_basis:
            diffs["advisory_ratio_rel"] = {
                name: _rel(ref[name], other[name])
                for name in ("sharpe", "sortino", "volatility")
                if ref.get(name) is not None and other.get(name) is not None
            }
        else:
            diffs["ratio_basis"] = ENGINES[engine].ratio_basis

    if agrees:
        status = "exact"
    elif expected == "documented":
        status = "documented"
    else:
        status = "failed"

    return {
        "status": status,
        "expected": expected,
        "publishable": status == "exact",
        "diffs": diffs,
        "metrics": {REFERENCE: ref, engine: other},
        "note": why(key, engine) if status == "documented" else "",
    }
