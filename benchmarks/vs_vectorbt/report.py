"""Turn a results JSON into the Markdown a human reads.

Prints to stdout and, when running under GitHub Actions, appends the same text
to the job summary so the numbers are visible without downloading an artifact.

    python report.py results.json

This is a run output, not an article: tables, and only the glue needed to read
them. Every "why" belongs in README.md, which is written once instead of being
reprinted underneath every single run.

Two result schemas are accepted. Version 2 keys everything by engine name;
version 1, written while the harness compared exactly two engines, is normalised
into that shape on load so the results archived under ``results/`` keep
rendering after the third engine arrived.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

METHOD_LINK = "benchmarks/vs_vectorbt/README.md"

# What version 1 files were, before the shape became a map.
V1_REFERENCE = "manifoldbt"
V1_CHALLENGER = "vectorbt"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def normalise(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Bring a version 1 payload up to the per-engine shape used below."""
    if payload.get("schema_version", 1) >= 2:
        return payload

    payload["reference"] = V1_REFERENCE
    payload["engines"] = [V1_REFERENCE, V1_CHALLENGER]
    for row in payload.get("results", []):
        verdict = row.get("parity") or {}
        row["status"] = verdict.get("status", "exact")
        row["parity"] = {V1_CHALLENGER: verdict}
        row["engines"] = [V1_REFERENCE, V1_CHALLENGER]
        if row.get("speedup"):
            row["speedup"] = {V1_CHALLENGER: row["speedup"]}
        if row.get("divergence_scale"):
            row["divergence_scale"] = {V1_REFERENCE: row["divergence_scale"]}

    cold = payload.get("cold_start")
    if cold:
        cold["engines"] = [V1_REFERENCE, V1_CHALLENGER]
        if not isinstance(cold.get("ratio"), dict):
            cold["ratio"] = {V1_CHALLENGER: cold.get("ratio")}
    mem = payload.get("memory")
    if mem:
        mem["engines"] = [V1_REFERENCE, V1_CHALLENGER]
    for point in payload.get("sweeps") or []:
        timings = point.get("timings")
        if timings and "seconds" not in timings:
            point["timings"] = {
                "seconds": {
                    V1_REFERENCE: timings.get("manifoldbt_s"),
                    V1_CHALLENGER: timings.get("vectorbt_s"),
                },
                "ratio": ({V1_CHALLENGER: timings["ratio"]}
                          if timings.get("ratio") is not None else {}),
            }
        memory = point.get("memory")
        if memory and "manifoldbt_added_mb" in memory:
            point["memory"] = {
                V1_REFERENCE: memory.get("manifoldbt_added_mb"),
                V1_CHALLENGER: memory.get("vectorbt_added_mb"),
            }
        if point.get(V1_CHALLENGER) and "status" in point[V1_CHALLENGER]:
            point["out_of_scope"] = {V1_CHALLENGER: point[V1_CHALLENGER]}
        if point.get("parity") and "status" in point["parity"]:
            point["parity"] = {V1_CHALLENGER: point["parity"]}
    return payload


def _engines(payload: Dict[str, Any]) -> List[str]:
    return payload.get("engines") or [V1_REFERENCE, V1_CHALLENGER]


def _reference(payload: Dict[str, Any]) -> str:
    return payload.get("reference", V1_REFERENCE)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _ms(seconds: float) -> str:
    if seconds < 1.0:
        return "{:.1f} ms".format(seconds * 1e3)
    return "{:.2f} s".format(seconds)


def _header(payload: Dict[str, Any]) -> List[str]:
    env = payload["environment"]
    versions = env["versions"]
    cores = str(env["logical_cores"])
    if env.get("pinned_cores"):
        cores += " (pinned to {})".format(env["pinned_cores"])
    lines = [
        "# " + " vs ".join(
            "{} {}".format(name, versions.get(name, "?")) for name in _engines(payload)),
        "",
        "`{os} {arch}` | {cpu} | {cores} cores | {ram} GB | python {py} | "
        "numpy {np} / numba {nb} / pandas {pd} | {reps} interleaved reps | {when}".format(
            os=env["os"], arch=env["arch"], cpu=env["cpu"], cores=cores, ram=env["ram_gb"],
            py=env["python"], np=versions["numpy"], nb=versions["numba"],
            pd=versions["pandas"], reps=payload["reps"], when=payload["generated_at"][:19],
        ),
    ]
    if env.get("run_url"):
        lines += ["", env["run_url"]]
    lines.append("")
    return lines


def _speed_table(rows: List[Dict[str, Any]], title: str, payload: Dict[str, Any]) -> List[str]:
    """One row per workload and size, one column per engine, then the ratios.

    Columns are taken from the run rather than hardcoded, and a cell is only
    empty when that engine was withheld or sat the workload out. Those cases get
    a marker and a sentence of their own further down, because a blank in a
    speed table reads as a defeat.
    """
    if not rows:
        return []
    reference = _reference(payload)
    engines = _engines(payload)
    challengers = [e for e in engines if e != reference]

    head = "| Workload | Bars | " + " | ".join(engines) + " | "
    head += " | ".join("vs " + c for c in challengers) + " |"
    lines = [
        "## " + title,
        "",
        head,
        "|---|---:|" + "---:|" * (len(engines) + len(challengers)),
    ]
    for row in rows:
        timings = row["timings"]
        cells = [
            _ms(timings[e]["median_s"]) if e in timings else "-" for e in engines
        ]
        speedup = row.get("speedup") or {}
        # The noise flag rides on the last ratio, where a reader's eye already
        # is when deciding whether to believe the number.
        mark = " ~" if row.get("noisy") else ""
        ratios = [
            "**x{:.1f}**".format(speedup[c]["median_of_ratios"]) if c in speedup else "-"
            for c in challengers
        ]
        # On the last ratio that actually has a number: hung on a "-" it would
        # look like a comment on the engine that did not run.
        present = [i for i, cell in enumerate(ratios) if cell != "-"]
        if mark and present:
            ratios[present[-1]] += mark
        lines.append("| {w} | {b:,} | {cells} |".format(
            w=row["workload"], b=row["bars"], cells=" | ".join(cells + ratios)))
    lines.append("")
    return lines


def _summary_cost(exact: List[Dict[str, Any]], payload: Dict[str, Any]) -> List[str]:
    plain = {r["bars"]: r for r in exact if r["workload"] == "sma_cross"}
    summarised = {r["bars"]: r for r in exact if r["workload"] == "sma_cross_metrics"}
    # Only subtract timings that were measured in the same interleaved loop.
    shared = sorted(
        b for b in set(plain) & set(summarised)
        if "sma_cross_metrics" in (plain[b].get("paired_with") or [])
    )
    if not shared:
        return []

    lines = [
        "## Cost of the performance summary",
        "",
        "Same simulation, with and without max drawdown / Sharpe / Sortino / volatility.",
        "",
        "| Bars | Engine | Without | With | Delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for bars in shared:
        for engine in _engines(payload):
            without = (plain[bars]["timings"] or {}).get(engine)
            with_ = (summarised[bars]["timings"] or {}).get(engine)
            if not (without and with_):
                continue
            a, c = without["median_s"], with_["median_s"]
            lines.append(
                "| {b:,} | {e} | {a} | {c} | {d} |".format(
                    b=bars, e=engine, a=_ms(a), c=_ms(c),
                    d=("+" + _ms(c - a)) if c > a else "none measurable",
                )
            )
    lines.append("")

    top = summarised[shared[-1]]
    advisory = {}
    basis_differs = {}
    for engine, verdict in top["parity"].items():
        diffs = verdict.get("diffs", {})
        if diffs.get("advisory_ratio_rel"):
            advisory[engine] = max(diffs["advisory_ratio_rel"].values())
        elif diffs.get("ratio_basis"):
            basis_differs[engine] = diffs.get("max_drawdown_rel")
    if advisory:
        lines += [
            "Gated on total return, round-trips and max drawdown (exact). Sharpe, "
            "Sortino and volatility agree to {:.1e}.".format(max(advisory.values())),
            "",
        ]
    for engine, drawdown_rel in basis_differs.items():
        if drawdown_rel is None:
            agreement = "was not compared"
        elif drawdown_rel == 0.0:
            agreement = "is identical"
        else:
            agreement = "agrees to {:.1e}".format(drawdown_rel)
        lines += [
            "{e} annualises its ratios on its own basis, so only the drawdown is "
            "compared there: it {a}.".format(e=engine, a=agreement),
            "",
        ]
    return lines


def _side_measures(payload: Dict[str, Any], results: List[Dict[str, Any]]) -> List[str]:
    cold = payload.get("cold_start")
    mem = payload.get("memory")
    engines = _engines(payload)
    threading_rows = [
        r for r in results
        if r.get("cpu_over_wall")
        and all(r["cpu_over_wall"].get(e) is not None for e in engines)
    ]
    if not (cold or mem or threading_rows):
        return []

    lines = [
        "## Cold start, memory, threads",
        "",
        "| | " + " | ".join(engines) + " |",
        "|---|" + "---:|" * len(engines),
    ]
    if cold:
        medians, share = cold["median_s"], cold["engine_share_s"]
        lines.append("| Fresh process to first backtest | " + " | ".join(
            "{:.2f} s".format(medians[e]) if e in medians else "-" for e in engines) + " |")
        lines.append("| ... minus the {:.2f} s python baseline | ".format(medians["baseline"])
                     + " | ".join("{:.2f} s".format(share[e]) if e in share else "-"
                                  for e in engines) + " |")
    if mem:
        lines.append("| RAM added by the run, per 1M bars | " + " | ".join(
            "{:.0f} MB".format(mem[e]["added_mb_per_million_bars"]) if e in mem else "-"
            for e in engines) + " |")
    if threading_rows:
        biggest = max(threading_rows, key=lambda r: r["bars"])
        ratio = biggest["cpu_over_wall"]
        lines.append("| CPU over wall time at {:,} bars | ".format(biggest["bars"])
                     + " | ".join("{:.2f}".format(ratio[e]) for e in engines) + " |")
    lines.append("")
    return lines


def _not_run(payload: Dict[str, Any], results: List[Dict[str, Any]]) -> List[str]:
    """Workloads an engine sits out, and why.

    Kept as prose rather than a table because the reason is the content. An
    engine that cannot express a workload has told you something about itself,
    and compressing that into an empty cell would throw away the only part worth
    reading.
    """
    seen: Dict[str, Dict[str, str]] = {}
    for row in results:
        for engine, why in (row.get("unsupported") or {}).items():
            if engine in _engines(payload):
                seen.setdefault(engine, {})[row["workload"]] = why
    if not seen:
        return []

    lines = ["## Not run, and why", ""]
    for engine, entries in seen.items():
        for workload, why in entries.items():
            lines += ["- **{e} on `{w}`.** {why}".format(e=engine, w=workload, why=why), ""]
    return lines


def _gb(mb: float) -> str:
    return "{:.1f} GB".format(mb / 1024) if mb >= 1024 else "{:.0f} MB".format(mb)


def _sweep_section(payload: Dict[str, Any]) -> List[str]:
    """The parameter-grid table, plus the memory that decides what is runnable.

    Memory is reported next to the timings rather than in its own annex because
    on a grid it is not a footnote: it is the first thing to run out. A machine
    that cannot hold the grid does not produce a slow number, it produces no
    number, and a reader sizing a job needs both columns side by side.
    """
    sweeps = payload.get("sweeps")
    if not sweeps:
        return []

    reference = _reference(payload)
    engines = _engines(payload)
    challengers = [e for e in engines if e != reference]
    timed = [s for s in sweeps if s.get("timings")]
    untimed = [s for s in sweeps if not s.get("timings")]
    oos = [s for s in timed if s.get("out_of_scope")]
    lines = ["## Parameter sweeps", ""]

    if timed:
        header = "| Bars | Combinations | " + " | ".join(engines) + " | "
        header += " | ".join("vs " + c for c in challengers) + " | "
        header += " | ".join("RAM " + e for e in engines) + " |"
        lines += [
            header,
            # bars, combinations, one column per engine, one ratio per
            # challenger, then one RAM column per engine.
            "|---:|---:|" + "---:|" * (2 * len(engines) + len(challengers)),
        ]
        for s in timed:
            mem = s.get("memory") or {}
            seconds = s["timings"]["seconds"]
            ratios = s["timings"].get("ratio") or {}
            cells = [_ms(seconds[e]) if seconds.get(e) is not None else "not run"
                     for e in engines]
            cells += ["**x{:.1f}**".format(ratios[c]) if ratios.get(c) is not None else "-"
                      for c in challengers]
            cells += [_gb(mem[e]) if mem.get(e) is not None else "-" for e in engines]
            lines.append("| {b:,} | {c:,} | {cells} |".format(
                b=s["bars"], c=s["combos"], cells=" | ".join(cells)))
        lines += [
            "",
            "Each grid is checked cell by cell before any of it is timed: the "
            "engines are joined on the parameter pair they actually ran, not on "
            "position, and the worst disagreement in the grid is what the gate "
            "sees. A single wrong cell among thousands is exactly the failure a "
            "sweep can have and a single backtest cannot.",
            "",
        ]

    if oos:
        lines += ["### Where a challenger was not run", ""]
        for s in oos:
            for engine, detail in s["out_of_scope"].items():
                verdict = (s.get("parity") or {}).get(engine) or {}
                lines.append(
                    "- **{e}, {c:,} combinations at {b:,} bars.** {why} The "
                    "cross-engine check for this point therefore covers the same "
                    "code path at {a:,} combinations, not this grid: agreement "
                    "was `{st}`.".format(
                        e=engine, c=s["combos"], b=s["bars"], why=detail["reason"],
                        a=verdict.get("checked_at_combos", 0),
                        st=verdict.get("status", "?"),
                    )
                )
        lines.append("")

    if untimed:
        lines += ["No timing for these points:", ""]
        for s in untimed:
            lines.append("- {b:,} bars x {c:,} combinations: {why}".format(
                b=s["bars"], c=s["combos"],
                why=s.get("reason") or s.get("note") or "no result",
            ))
        lines.append("")
    return lines


def _divergence_line(row: Dict[str, Any], payload: Dict[str, Any]) -> str:
    """The measured size of one documented divergence, engine by engine."""
    scales = row.get("divergence_scale") or {}
    reference = _reference(payload)
    view = scales.get(reference)
    if not view:
        return ""
    line = (
        "- `{w}` at {b:,} bars: {n} of {t} {ref} round-trips re-enter on the exit "
        "bar ({p:.0%}); the run books {sl} stop and {tp} target exits in all.".format(
            w=row["workload"], b=row["bars"], n=view["reentries_on_exit_bar"],
            t=view["round_trips"], p=view["share_of_round_trips"],
            sl=view["sl_exits"], tp=view["tp_exits"], ref=reference,
        )
    )
    for engine, scale in scales.items():
        if engine == reference or "round_trips" not in scale:
            continue
        line += " {e} books {n}".format(e=engine, n=scale["round_trips"])
        if "reentries_deferred" in scale:
            # BOTH NUMBERS ARE ABOUT THIS ENGINE'S OWN TRADES, and the sentence
            # says so. An earlier draft read "N of them a bar late and M not at
            # all", which forces "them" to be the REFERENCE's re-entries and so
            # claims a trade-for-trade correspondence. There is none: entering a
            # bar later means entering at a different price, so the brackets that
            # follow fire on different bars and the two streams drift. At 100,000
            # bars only 649 of vectorbt's 831 deferrals sit on a bar the
            # reference also re-entered on, and 276 of the reference's 925 have
            # no deferral on their bar. The counts still sum -- that is a
            # population identity, not a matching -- and printing them from
            # different sources keeps the sum a check rather than a restatement.
            line += ", {lost} fewer than {ref}; {d} of them opened on the bar " \
                    "after {e}'s own exit with the level still true".format(
                        d=scale["reentries_deferred"], e=engine, ref=reference,
                        lost=view["round_trips"] - scale["round_trips"],
                    )
        if scale.get("reentries_on_exit_bar"):
            line += ", and re-enters on the exit bar itself {n} times".format(
                n=scale["reentries_on_exit_bar"])
        line += "."
    return line


def _divergence_note(rows: List[Dict[str, Any]], payload: Dict[str, Any]) -> List[str]:
    """Every measured divergence, one line each, then the cause once.

    One line per row rather than one for the whole section, because README.md
    promises a documented timing is published "with the cause and its measured
    size", and the size is a property of the point rather than of the workload.
    In the run this was written against, raptorbt sits 0.92% of capital from the
    reference at 100,000 bars and 10.07% at 1,000,000, where it also crosses from
    below the reference to above it. Rendering the first row and stopping
    published the other two as timings with no measured size beside them.
    """
    lines = [line for line in
             (_divergence_line(row, payload) for row in rows) if line]
    if not lines:
        return []
    return lines + ["", "Cause in {}.".format(METHOD_LINK), ""]


def render(payload: Dict[str, Any]) -> str:
    payload = normalise(payload)
    results = payload["results"]
    exact = [r for r in results if r.get("status") == "exact" and r.get("timings")]
    documented = [r for r in results if r.get("status") == "documented"]
    failed = [r for r in results if r.get("status") == "failed"]

    lines = _header(payload)
    lines += _speed_table(exact, "Same results, every engine", payload)
    lines += _summary_cost(exact, payload)
    lines += _side_measures(payload, results)
    lines += _not_run(payload, results)
    lines += _sweep_section(payload)

    timed = [r for r in documented if r.get("timings")]
    if timed:
        lines += _speed_table(timed, "Results differ, kept out of the headline", payload)
        lines += _divergence_note(timed, payload)

    if failed:
        lines += ["## Timing withheld", ""]
        for row in failed:
            for engine, verdict in row["parity"].items():
                if verdict["status"] != "failed":
                    continue
                diffs = verdict["diffs"]
                lines.append(
                    "- `{w}` at {b:,} bars, {e}: final equity differs by {d:.2e} of "
                    "capital, round-trips by {t}.".format(
                        w=row["workload"], b=row["bars"], e=engine,
                        d=diffs["final_equity_vs_capital"],
                        t=diffs["round_trips_delta"],
                    )
                )
        lines.append("")

    lines += [
        "---",
        "",
        "`~` = IQR above 15% of the median, indicative only. Ratios are medians of "
        "per-repetition ratios against {ref}, with the engines interleaved; data loading "
        "excluded, warmup discarded. Method and caveats: {link}".format(
            ref=_reference(payload), link=METHOD_LINK),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="render a benchmark result")
    parser.add_argument("results", nargs="?", default="results.json")
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
