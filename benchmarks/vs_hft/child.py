"""One engine, one workload, one fresh process.

    python child.py --engine hftbacktest --workload mm_ladder_l2 --day 2025-08-23 \
        [--window 36000 39600] [--mode strategy|replay] [--warm 1] [--journal out.npz]

Prints a single JSON line. A fresh process per engine is what makes the cold
start and the memory honest: nothing else is loaded, nothing is left over from
the previous engine.

Phases reported (seconds):
  prepare   the engine's own input built from the neutral data (hftbacktest's
            event array, a manifoldbt store opened, ...). Not part of the run.
  cold      the first run in the process (numba compiles here for hftbacktest)
  warm      the following runs; the benchmark takes its medians over these
"""
import argparse
import json
import os
import sys
import time
import traceback


def peak_rss_mb():
    try:
        import psutil

        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)       # Windows
        if peak:
            return peak / 1e6
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3     # Linux: KiB
    except Exception:
        return float("nan")


def rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--day", required=True)
    ap.add_argument("--window", nargs=2, type=int)
    ap.add_argument("--mode", default="strategy")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--journal")
    ap.add_argument("--itch")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import harness

    out = {"engine": args.engine, "workload": args.workload, "mode": args.mode, "status": "ok"}
    try:
        ctx = harness.Ctx(args.day, tuple(args.window) if args.window else None, itch_extract=args.itch)
        wl = harness.all_workloads()[args.workload]
        # load the neutral data first: its memory is common to every engine and is
        # subtracted, so the figure reported is what the engine added
        if wl["kind"] == "mbo":
            ctx.m
        else:
            ctx.d
        base = rss_mb()
        out["baseline_rss_mb"] = round(base, 1)
        inputs = harness.prepare(args.engine, args.workload, ctx)
        out["prepare_s"] = round(inputs["prepare_s"], 3)
        out["after_prepare_rss_mb"] = round(rss_mb(), 1)
        runs = []
        summary = journal = None
        for i in range(1 + args.warm):
            summary, journal, timing = harness.run(args.engine, args.workload, ctx, inputs, args.mode)
            runs.append(timing)
        out["cold_s"] = round(runs[0]["seconds"], 3)
        out["warm_s"] = [round(r["seconds"], 3) for r in runs[1:]]
        for key in ("engine_seconds", "convert_seconds"):
            if key in runs[-1]:
                out[key] = runs[-1][key]
        out["summary"] = {k: v for k, v in (summary or {}).items()}
        out["peak_rss_mb"] = round(peak_rss_mb(), 1)
        if args.journal and journal is not None:
            import numpy as np

            np.savez(args.journal, **{k: v for k, v in journal.items() if hasattr(v, "dtype")})
    except BaseException as exc:             # a refused call is a result, not a crash
        name = type(exc).__name__
        out["status"] = "blocked" if name in ("PermissionError", "LicenseError") else "failed"
        out["error"] = f"{name}: {str(exc)[:400]}"
        if out["status"] == "blocked" and os.environ.get("BENCH_PUBLIC") == "1":
            out["error"] = "refused by the environment"
        if out["status"] == "failed":
            out["traceback"] = traceback.format_exc()[-1500:]
    sys.stdout.write("\n@@RESULT@@" + json.dumps(out, default=str) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
