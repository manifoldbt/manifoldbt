"""Compact reprs for the big containers returned to notebooks.

A sweep returns one object per combo and a walk-forward carries a full
equity curve per fold, so echoing either in a Jupyter cell used to print
thousands of lines. These wrappers subclass ``list``/``dict`` so every
existing access keeps working (indexing, iteration, ``.keys()``, JSON
round-trips); only the repr changes.
"""
from __future__ import annotations

from typing import Any, Dict, List

_MAX_SCAN = 100_000  # cap the repr's own cost on million-combo sweeps


def _fmt(v: float) -> str:
    """Compact number: 3 significant-ish digits, thousands as k."""
    if v is None:
        return "?"
    a = abs(v)
    if a >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"{v / 1_000:.2f}k"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.4g}"


def _span(values) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    return _fmt(lo) if lo == hi else f"{_fmt(lo)}..{_fmt(hi)}"


class SweepLiteResults(list):
    """``run_sweep_lite`` output: a list, with a one-line repr.

    Printing 400 combos used to emit 400 lines of ``BatchResultLite(...)``.
    """

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return "SweepLiteResults(empty)"
        head = self[:_MAX_SCAN]
        eq = _span([getattr(r, "final_equity", None) for r in head])
        sharpes = []
        for r in head:
            m = getattr(r, "metrics", None)
            if isinstance(m, dict):
                sharpes.append(m.get("sharpe"))
        name = getattr(self[0], "strategy_name", "?")
        parts = [f"{n:,} combos", f"strategy {name!r}", f"final_equity {eq}"]
        if any(s is not None for s in sharpes):
            parts.append(f"sharpe {_span(sharpes)}")
        if n > _MAX_SCAN:
            parts.append(f"(range over first {_MAX_SCAN:,})")
        return ("<SweepLiteResults: " + " | ".join(parts) +
                "\n  r[i] for one combo, mbt.sweep_columns(r, 'sharpe') for arrays>")


class WalkForwardResult(dict):
    """``run_walk_forward`` output: a dict, with a one-line repr.

    The raw dict carries a full IS and OOS equity curve per fold, so echoing
    it in a cell used to print tens of thousands of floats.
    """

    def __repr__(self) -> str:
        folds = self.get("folds") or []
        if not folds:
            return "<WalkForwardResult: no folds>"
        metric = self.get("optimize_metric", "sharpe")

        def _m(fold, key):
            v = fold.get(key)
            return v.get(metric) if isinstance(v, dict) else v

        is_v = [_m(f, "is_metrics") for f in folds]
        oos_v = [_m(f, "oos_metrics") for f in folds]
        lines = [
            f"<WalkForwardResult: {len(folds)} folds | metric {metric!r} | "
            f"IS {_span(is_v)} | OOS {_span(oos_v)}"
        ]
        for f in folds:
            best = f.get("best_params") or {}
            flat = {k: (list(v.values())[0] if isinstance(v, dict) else v)
                    for k, v in best.items()}
            i, o = _m(f, "is_metrics"), _m(f, "oos_metrics")
            lines.append(
                f"  fold {f.get('fold_index', '?')}: "
                f"IS {_fmt(i) if i is not None else '?':>8}  "
                f"OOS {_fmt(o) if o is not None else '?':>8}  {flat}"
            )
        lines.append("  keys: " + ", ".join(sorted(self.keys())) + ">")
        return "\n".join(lines)


def wrap_sweep_lite(results: List[Any]) -> "SweepLiteResults":
    return SweepLiteResults(results)


def wrap_walk_forward(result: Dict[str, Any]) -> "WalkForwardResult":
    return WalkForwardResult(result) if isinstance(result, dict) else result
