"""manifoldbt vs hftbacktest vs NautilusTrader on market making and HFT strategies.

    python bench.py --day 2025-08-23 --reps 3 --out results.json

Prepares one pinned day of public data, runs every workload in every engine (one
fresh process per run, engines interleaved inside each repetition), and reports how
long each took together with how far the engines did the same work.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
import harness  # noqa: E402
import parity  # noqa: E402
import spec  # noqa: E402

SCHEMA = 3
ENGINE_ORDER = ["manifoldbt", "hftbacktest", "nautilus"]


def versions():
    from importlib import metadata

    out = {}
    for name in ("manifoldbt", "hftbacktest", "nautilus_trader", "numba", "numpy", "pandas", "pyarrow"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def wheel_sha():
    try:
        import manifoldbt._native as nat
        h = hashlib.sha256()
        with open(nat.__file__, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None


def environment():
    import psutil

    return {
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(), "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "versions": versions(),
        "manifoldbt_native_sha256": wheel_sha(),
    }


def spawn(engine, workload, day, mode, window, warm, journal, itch):
    cmd = [sys.executable, os.path.join(HERE, "child.py"), "--engine", engine, "--workload", workload,
           "--day", day, "--mode", mode, "--warm", str(warm)]
    if window:
        cmd += ["--window", str(window[0]), str(window[1])]
    if journal:
        cmd += ["--journal", journal]
    if itch:
        cmd += ["--itch", itch]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * 3600)
    wall = time.perf_counter() - t0
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            r = json.loads(line[len("@@RESULT@@"):])
            r["process_wall_s"] = round(wall, 2)
            return r
    return {"engine": engine, "workload": workload, "status": "failed",
            "error": "no result line", "stderr_tail": p.stderr[-800:], "process_wall_s": round(wall, 2)}


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return None
    med = statistics.median(xs)
    if len(xs) >= 4:
        q = statistics.quantiles(xs, n=4)
        iqr = q[2] - q[0]
    else:
        iqr = xs[-1] - xs[0]
    return {"n": len(xs), "min": xs[0], "median": med, "max": xs[-1], "iqr": iqr,
            "noisy": bool(med > 0 and iqr / med > 0.15)}


def load_journal(path):
    if not path or not os.path.exists(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2025-08-23")
    ap.add_argument("--workloads", nargs="*", default=["mm_ladder_l2", "mm_touch_bbo", "flow_taker_bbo", "probe_l2_zero"])
    ap.add_argument("--engines", nargs="*", default=ENGINE_ORDER)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--probe-window", nargs=2, type=int, default=[36000, 39600])
    ap.add_argument("--itch", default=None, help="the extract of one symbol (ITCH message stream)")
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--fail-on-blocked", action="store_true",
                    help="exit non-zero when an engine could not run a workload (used in CI: nothing partial is published)")
    ap.add_argument("--tmp", default=os.path.join(data.DATA_ROOT, "tmp"))
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)

    result = {"schema_version": SCHEMA, "kind": "vs_hftbacktest_nautilus", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "environment": environment(), "args": vars(args), "workloads": {}}
    if args.itch is None and "mbo_join_mbo" in args.workloads:
        args.workloads.remove("mbo_join_mbo")

    # ---- data ---------------------------------------------------------------
    wls = harness.all_workloads()
    mark_px = 0.0
    if any(wls[n]["kind"] != "mbo" for n in args.workloads):
        path, meta = data.prepare(args.day)
        result["data"] = {"day": args.day, "prepared_npz": os.path.basename(path), "meta": meta}
        d_full = data.load(path)
        mark_px = float((d_full["q_bpx"][-1] + d_full["q_apx"][-1]) / 2 / spec.TICK_INV)
    if args.itch:
        result.setdefault("data", {})["itch_extract"] = {
            "path": os.path.basename(args.itch), "bytes": os.path.getsize(args.itch),
            "sha256": data.sha256_file(args.itch)}

    for name in args.workloads:
        wl = wls[name]
        window = args.probe_window if wl.get("short") else None
        entry = {"doc": wl["doc"], "kind": wl["kind"], "tier": wl["tier"], "latency": wl["latency"],
                 "window_s": window, "engines": {}}
        runs = {e: [] for e in args.engines}
        replays = {e: [] for e in args.engines}
        journals = {}
        for rep in range(args.reps):                       # engines alternate inside each repetition
            for e in args.engines:
                jp = os.path.join(args.tmp, f"{name}_{e}.npz") if rep == 0 else None
                # numba compiles in the first run of hftbacktest's process, so it gets a second one;
                # Nautilus has nothing to warm and its runs are long.
                r = spawn(e, name, args.day, "strategy", window, 0 if e == "nautilus" else 1, jp, args.itch)
                runs[e].append(r)
                if rep == 0:
                    journals[e] = load_journal(jp)
                print(f"[{name}] rep {rep} {e}: {r.get('status')} warm={r.get('warm_s')} cold={r.get('cold_s')}", flush=True)
            if not args.no_replay and wl["kind"] in ("mm", "mbo") and not wl.get("short"):
                for e in args.engines:
                    replays[e].append(spawn(e, name, args.day, "replay", window, 0 if e == "nautilus" else 1, None, args.itch))
        for e in args.engines:
            rs = runs[e]
            ok = [r for r in rs if r.get("status") == "ok"]
            if not ok:
                entry["engines"][e] = {"status": rs[0].get("status", "failed"), "error": rs[0].get("error")}
                continue
            entry["engines"][e] = {
                "status": "ok",
                "strategy_s": stats([x for r in ok for x in (r["warm_s"] or [r["cold_s"]])]),
                "cold_s": stats([r["cold_s"] for r in ok]),
                "prepare_s": stats([r["prepare_s"] for r in ok]),
                "engine_only_s": stats([r["engine_seconds"] for r in ok if "engine_seconds" in r]),
                "convert_s": stats([r["convert_seconds"] for r in ok if "convert_seconds" in r]),
                "replay_s": stats([x for r in replays[e] if r.get("status") == "ok" for x in (r["warm_s"] or [r["cold_s"]])]),
                "memory_mb": {"baseline_data": ok[0]["baseline_rss_mb"], "peak": max(r["peak_rss_mb"] for r in ok),
                              "added_by_engine": round(max(r["peak_rss_mb"] for r in ok) - ok[0]["baseline_rss_mb"], 1)},
                "result": ok[0]["summary"],
            }
        # ---- parity ---------------------------------------------------------
        par = {"reference": "manifoldbt" if journals.get("manifoldbt") is not None else None, "pairs": {}}
        ref = "manifoldbt" if par["reference"] else ("hftbacktest" if journals.get("hftbacktest") is not None else None)
        par["reference_used"] = ref
        if ref:
            for e in args.engines:
                if e == ref or journals.get(e) is None:
                    continue
                if wl["kind"] == "taker":
                    par["pairs"][f"{e}_vs_{ref}"] = parity.compare_taker(journals[e], journals[ref])
                else:
                    c = parity.compare_quotes(journals[e], journals[ref], e, ref)
                    for eng in (e, ref):
                        j = journals[eng]
                        if wl["kind"] == "mm":
                            c.setdefault("pnl_common", {})[eng] = parity.pnl_common(j, mark_px)
                    par["pairs"][f"{e}_vs_{ref}"] = c
        entry["parity"] = par
        result["workloads"][name] = entry
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=str)
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print("wrote", args.out)
    if args.fail_on_blocked:
        bad = [(n, e) for n, w in result["workloads"].items() for e, r in w["engines"].items() if r.get("status") != "ok"]
        if bad:
            print("engines that did not run:", bad, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
