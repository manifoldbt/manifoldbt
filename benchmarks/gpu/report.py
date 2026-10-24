"""Render a GPU sweep result as Markdown, and into the GitHub job summary.

    python report.py results-gpu.json

The speedup in the table is GPU against the CPU of the same machine, on the cores
named in the header. It is not a comparison with the CPU sweeps job, which runs
on another machine: that job's numbers for the same grids are in
../vs_vectorbt/results/latest.json, and they are read side by side, not divided
by these.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

METHOD_LINK = (
    "https://github.com/manifoldbt/manifoldbt/blob/master/benchmarks/gpu/README.md"
)


def _card(env: Dict[str, Any]) -> str:
    gpus = env.get("gpus") or []
    if not gpus:
        return "no GPU visible"
    g = gpus[0]
    extra = ", ".join(x for x in (g.get("memory.total"), "driver " + g["driver_version"]
                                   if g.get("driver_version") else "") if x)
    return "{} ({})".format(g.get("name", "?"), extra) if extra else g.get("name", "?")


def _seconds(value: float) -> str:
    return "{:.3f} s".format(value) if value >= 0.1 else "{:.1f} ms".format(value * 1e3)


def render(payload: Dict[str, Any]) -> str:
    env = payload.get("environment") or {}
    versions = env.get("versions") or {}
    lines: List[str] = [
        "## GPU parameter sweeps",
        "",
        "manifoldbt {} on {}. CPU side: {}, {} threads. {}".format(
            versions.get("manifoldbt", "?"), _card(env), env.get("cpu", "?"),
            env.get("cpu_threads", "?"), env.get("platform", "")).rstrip(),
        "",
    ]
    if payload.get("status") != "exact" and payload.get("reason"):
        lines += ["**Failed: {}**".format(payload["reason"]), ""]

    rows = payload.get("sweeps") or []
    if rows:
        lines += [
            "| bars | combos | GPU | CPU | GPU vs CPU | per combo (GPU) | first GPU call "
            "| parity |",
            "|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
        for row in rows:
            if row.get("status") != "exact" and "timings" not in row:
                lines.append("| {:,} | {:,} | | | | | | failed: {} |".format(
                    row["bars"], row.get("requested_combos", 0), row.get("reason", "")))
                continue
            t = row["timings"]
            noisy = " ~" if t.get("noisy") else ""
            parity = row["parity"]
            verdict = "{} ({:,}/{:,} bit-identical)".format(
                row["status"], parity.get("identical_cells", 0), row["combos"])
            if row.get("reason"):
                verdict += ": " + row["reason"]
            lines.append(
                "| {:,} | {:,} | {}{} | {}{} | x{:.1f} | {:.2f} us | {} | {} |".format(
                    row["bars"], row["combos"],
                    _seconds(t["seconds"]["gpu"]), noisy,
                    _seconds(t["seconds"]["cpu"]), noisy,
                    t["speedup"], t["per_combo_us"]["gpu"],
                    _seconds(t["first_call_s"]["gpu"]), verdict))
        lines.append("")

    lines += [
        "---",
        "",
        "Medians of {} interleaved repetitions per side, after an untimed first call "
        "(reported apart: it compiles the kernels) and a calibration of the card. "
        "`~` = IQR above 15% of the median, indicative only. Parity: every cell of the "
        "GPU grid against the CPU grid, keyed by parameter pair. Method: {}".format(
            payload.get("reps", "?"), METHOD_LINK),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="render a GPU sweep result")
    parser.add_argument("results", nargs="?", default="results-gpu.json")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        payload = json.load(fh)

    text = render(payload)
    sys.stdout.write(text + "\n")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
