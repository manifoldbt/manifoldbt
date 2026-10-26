"""Render a GPU sweep result as Markdown, and into the GitHub job summary.

    python report.py results-gpu.json [--cpu-results ../vs_vectorbt/results/latest.json]

The CPU column is not measured here. It is the CPU sweeps job's published number
for the same grid, read from its results file and matched on bars, combinations
and the digest of the data, so both columns are about the same work. The two
come from different machines (a GitHub runner and a Modal GPU), which is what
the ratio says: what the sweep costs on the runner the CPU numbers are published
from, against what it costs on the card. It is only computed when both runs
measured the same manifoldbt version.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_RESULTS = os.path.join(HERE, "..", "vs_vectorbt", "results", "latest.json")
METHOD_LINK = (
    "https://github.com/manifoldbt/manifoldbt/blob/master/benchmarks/gpu/README.md"
)

Key = Tuple[int, int, str]


def cpu_reference(published: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The CPU sweeps job's points, keyed by (bars, combos, data digest)."""
    payload = (published or {}).get("sweeps") or {}
    env = payload.get("environment") or {}
    points: Dict[Key, float] = {}
    for row in payload.get("sweeps") or []:
        seconds = ((row.get("timings") or {}).get("seconds") or {}).get("manifoldbt")
        if seconds is not None:
            points[(row["bars"], row["combos"], row.get("data_digest", ""))] = seconds
    return {
        "points": points,
        "version": (env.get("versions") or {}).get("manifoldbt"),
        "cores": env.get("logical_cores"),
        "cpu_model": env.get("cpu"),
        "run_url": (published or {}).get("run_url", ""),
    }


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


def render(payload: Dict[str, Any], cpu: Optional[Dict[str, Any]] = None) -> str:
    cpu = cpu or cpu_reference(None)
    env = payload.get("environment") or {}
    version = (env.get("versions") or {}).get("manifoldbt", "?")
    same_version = cpu["version"] == version
    lines: List[str] = ["## GPU parameter sweeps", "",
                        "manifoldbt {} on {}. {}".format(
                            version, _card(env), env.get("platform", "")).rstrip()]
    if cpu["points"]:
        lines.append(
            "CPU column: the CPU sweeps job's published run{}, manifoldbt {} on a "
            "{}-vCPU GitHub runner{}. GitHub does not always hand out the same "
            "processor, so the ratio moves with it.{}".format(
                " ({})".format(cpu["run_url"]) if cpu["run_url"] else "",
                cpu["version"] or "?", cpu["cores"] or "?",
                " ({})".format(cpu["cpu_model"]) if cpu.get("cpu_model") else "",
                "" if same_version else
                " Not the version measured here, so no ratio is given."))
    lines.append("")
    if payload.get("status") != "exact" and payload.get("reason"):
        lines += ["**Failed: {}**".format(payload["reason"]), ""]

    rows = payload.get("sweeps") or []
    if rows:
        lines += [
            "| bars | combos | GPU | per combo | first GPU call | CPU (runner) "
            "| GPU vs runner CPU | parity |",
            "|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
        for row in rows:
            if "timings" not in row:
                lines.append("| {:,} | {:,} | | | | | | failed: {} |".format(
                    row["bars"], row.get("requested_combos", 0), row.get("reason", "")))
                continue
            t = row["timings"]
            cpu_s = cpu["points"].get((row["bars"], row["combos"], row.get("data_digest", "")))
            ratio = "x{:.1f}".format(cpu_s / t["seconds"]) if cpu_s and same_version else ""
            parity = row["parity"]
            verdict = "{} ({:,}/{:,} bit-identical)".format(
                row["status"], parity.get("identical_cells", 0), row["combos"])
            if row.get("reason"):
                verdict += ": " + row["reason"]
            lines.append("| {:,} | {:,} | {}{} | {:.2f} us | {} | {} | {} | {} |".format(
                row["bars"], row["combos"], _seconds(t["seconds"]),
                " ~" if t.get("noisy") else "", t["per_combo_us"],
                _seconds(t["first_call_s"]), _seconds(cpu_s) if cpu_s else "",
                ratio, verdict))
        lines.append("")

    lines += [
        "---",
        "",
        "GPU: medians of {} repetitions, after an untimed first call (reported apart: "
        "it compiles the kernels) and a calibration of the card. `~` = IQR above 15% of "
        "the median, indicative only. Parity: every cell of the GPU grid against the "
        "same grid run once on the CPU of the GPU machine, keyed by parameter pair. "
        "Method: {}".format(payload.get("reps", "?"), METHOD_LINK),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="render a GPU sweep result")
    parser.add_argument("results", nargs="?", default="results-gpu.json")
    parser.add_argument("--cpu-results", default=CPU_RESULTS,
                        help="the CPU benchmark's published results file")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as fh:
        payload = json.load(fh)
    published = None
    if os.path.exists(args.cpu_results):
        with open(args.cpu_results, encoding="utf-8") as fh:
            published = json.load(fh)

    text = render(payload, cpu_reference(published))
    sys.stdout.write(text + "\n")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
