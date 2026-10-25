"""GPU parameter sweeps: the CPU sweeps job's grids, timed on a card.

The sweeps job of ``bench-vs-vectorbt.yml`` times manifoldbt's parameter sweep on
a four-vCPU GitHub runner. Those runners have no GPU, so that job can only ever
measure the CPU path. This one times the *same* sweep -- same generator, same
seed, same grid, same strategy, imported from ``../vs_vectorbt`` rather than
copied -- on a GPU. The CPU number it is read against is that job's, published
from the runner: the CPU path runs here once per point, untimed, only so that
every GPU cell can be checked against it.

    python bench.py --sweep 20000:5000 20000:20000 200000:10000 --reps 7 \\
        --out results-gpu.json

It runs on any machine with an NVIDIA card, its driver, ``manifoldbt[gpu]`` and a
licence that covers GPU acceleration. In CI it runs on Modal (``modal_app.py``),
because GitHub's hosted runners have no GPU and its GPU runners are paid.

What it refuses to publish
--------------------------
* A GPU number that was not computed on the GPU. ``device="cuda"`` falls back to
  the CPU with a ``UserWarning`` when the card, the build or the strategy cannot
  take the sweep, so that a user's results are never affected. That is the right
  behaviour for a user and the wrong one for a benchmark: here a ``UserWarning``
  raised during a GPU sweep fails the point.
* A timing for work that differs. Every cell of the GPU grid is compared with the
  CPU grid, keyed by its parameter pair, through the gate the CPU job applies to
  vectorbt (``sweep_child.compare``). The engine documents the two paths as
  identical in fp64, so the result also counts the cells that are not identical
  to the bit, and a repetition that does not reproduce the first call fails the
  point.
* A timing measured across a licence change, as in the CPU job.

Timing
------
The card is calibrated first (``calibrate_gpu``, untimed, a few seconds: it
brings the card to its steady clocks and measures the costs the sweep uses to
pick its path), and every point is run once untimed before it is timed, so that
compiling kernels is not counted as sweep time. That first call is still
reported on its own, because it is what a user's first sweep costs.

The parity run on the CPU is given the cores the machine lets this process use,
read from its CPU affinity and cgroup quota rather than from ``os.cpu_count()``,
or ``--cpu-threads``. It is not timed: the CPU number is the runner's.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_BENCH = os.path.join(os.path.dirname(HERE), "vs_vectorbt")
sys.path.insert(0, CPU_BENCH)

import data as data_mod  # noqa: E402
from sweep_child import compare, grid, run_mbt, tier  # noqa: E402

SCHEMA_VERSION = 1
# Same flag as the CPU benchmark: a point whose interquartile range exceeds 15%
# of its median is indicative only.
NOISE_THRESHOLD = 0.15
# The CPU sweeps job's three points: the GPU number of each is read against the
# CPU number that job publishes for the same grid on the same data.
DEFAULT_POINTS = ("20000:5000", "20000:20000", "200000:10000")


# --------------------------------------------------------------------------- #
# The machine
# --------------------------------------------------------------------------- #
def usable_cores() -> Dict[str, Any]:
    """How many cores this process may actually run on, and how that was read."""
    seen = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        affinity = seen
    quota = cgroup_quota()
    usable = affinity if quota is None else max(1, min(affinity, math.floor(quota)))
    return {"os_cpu_count": seen, "affinity": affinity, "cgroup_quota": quota,
            "usable": usable}


def cgroup_quota(path: str = "/sys/fs/cgroup/cpu.max") -> Optional[float]:
    """The cgroup v2 CPU quota in cores, or None when there is none to read."""
    try:
        with open(path, encoding="ascii") as fh:
            fields = fh.read().split()
    except OSError:
        return None
    if len(fields) < 2 or fields[0] == "max":
        return None
    try:
        return float(fields[0]) / float(fields[1])
    except ValueError:
        return None


def cpu_model() -> str:
    """The CPU's name, from whichever source this kernel fills in.

    A sandboxed kernel (Modal runs gVisor) can leave ``model name`` out of
    /proc/cpuinfo, so lscpu and the vendor/family/model numbers are tried next.
    """
    info: Dict[str, str] = {}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                key, sep, value = line.partition(":")
                if sep and key.strip() not in info:
                    info[key.strip()] = value.strip()
    except OSError:
        pass
    if info.get("model name"):
        return info["model name"]
    exe = shutil.which("lscpu")
    if exe:
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
            for line in out.splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "Model name" and value.strip():
                    return value.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if info.get("vendor_id"):
        return "{} family {} model {}".format(
            info["vendor_id"], info.get("cpu family", "?"), info.get("model", "?"))
    return platform.processor() or "unknown"


GPU_FIELDS = ("name", "driver_version", "memory.total", "compute_cap")


def gpu_info() -> List[Dict[str, str]]:
    """What nvidia-smi says about each card. Empty when there is no driver."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    for fields in (GPU_FIELDS, GPU_FIELDS[:-1]):  # compute_cap needs driver >= 510
        try:
            out = subprocess.run(
                [exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        return [dict(zip(fields, (v.strip() for v in line.split(","))))
                for line in out.splitlines() if line.strip()]
    return []


def environment(platform_label: str, threads: int) -> Dict[str, Any]:
    import numpy
    import pandas

    import manifoldbt as mbt

    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        ram_gb = None
    return {
        "platform": platform_label,
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "cpu": cpu_model(),
        "cores": usable_cores(),
        "cpu_threads": threads,
        "ram_gb": ram_gb,
        "gpus": gpu_info(),
        "python": platform.python_version(),
        "versions": {"manifoldbt": mbt.__version__, "numpy": numpy.__version__,
                     "pandas": pandas.__version__},
    }


# --------------------------------------------------------------------------- #
# Measuring
# --------------------------------------------------------------------------- #
def summarise(samples: List[float]) -> Dict[str, Any]:
    """Min, median, max and IQR, computed as the CPU benchmark computes them."""
    ordered = sorted(samples)
    median = statistics.median(ordered)
    if len(ordered) >= 4:
        mid = len(ordered) // 2
        iqr = statistics.median(ordered[-mid:]) - statistics.median(ordered[:mid])
    else:
        iqr = ordered[-1] - ordered[0]
    return {
        "samples_s": samples,
        "min_s": ordered[0],
        "median_s": median,
        "max_s": ordered[-1],
        "iqr_s": iqr,
        "iqr_over_median": iqr / median if median else 0.0,
    }


def timed(call: Callable[[], Dict]) -> Tuple[float, Dict, List[Tuple[str, str]]]:
    """Run once; return the wall time, the results and every warning raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        out = call()
        elapsed = time.perf_counter() - start
    return elapsed, out, [(w.category.__name__, str(w.message)) for w in caught
                          if issubclass(w.category, UserWarning)]


def identical_cells(reference: Dict, other: Dict) -> int:
    """Cells equal to the bit (NaN counts as equal to NaN)."""
    def same(a, b):
        return a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))

    count = 0
    for pair, ref in reference.items():
        got = other.get(pair)
        if got is None:
            continue
        refs = ref if isinstance(ref, tuple) else (ref,)
        gots = got if isinstance(got, tuple) else (got,)
        if all(same(a, b) for a, b in zip(refs, gots)):
            count += 1
    return count


def measure_point(bars: int, combos: int, reps: int, threads: int) -> Dict[str, Any]:
    df = data_mod.make_ohlcv(bars)
    fast_vals, slow_vals = grid(combos)
    entry: Dict[str, Any] = {
        "bars": bars,
        "combos": len(fast_vals) * len(slow_vals),
        "requested_combos": combos,
        "grid": {"fast": len(fast_vals), "slow": len(slow_vals)},
        "data_digest": data_mod.digest(df),
        "tier_before": tier(),
    }
    work = tempfile.mkdtemp(prefix="mbt-gpu-")
    try:
        gpu = run_mbt(df, fast_vals, slow_vals, os.path.join(work, "gpu"),
                      metrics=True, device="cuda")
        cpu = run_mbt(df, fast_vals, slow_vals, os.path.join(work, "cpu"),
                      metrics=True, device="cpu", max_parallelism=threads)

        # The untimed first GPU call: kernels compile here, and the results it
        # returns are the ones every repetition must reproduce.
        gpu_first, gpu_ref, fallbacks = timed(gpu)
        # The CPU path, once and untimed: the reference every GPU cell is
        # checked against. Its timing is the CPU job's, not this machine's.
        _, cpu_ref, _ = timed(cpu)

        samples: List[float] = []
        unstable = 0
        for _rep in range(reps):
            elapsed, out, caught = timed(gpu)
            samples.append(elapsed)
            fallbacks.extend(caught)
            if len(out) != len(gpu_ref) or identical_cells(gpu_ref, out) != len(gpu_ref):
                unstable += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    parity = compare(cpu_ref, gpu_ref)
    parity["reference"] = "cpu"
    parity["reference_threads"] = threads
    parity["identical_cells"] = identical_cells(cpu_ref, gpu_ref)
    stats = summarise(samples)
    gpu_s = stats["median_s"]
    entry.update({
        "parity": parity,
        "timings": {
            "seconds": gpu_s,
            "per_combo_us": gpu_s / entry["combos"] * 1e6,
            "first_call_s": gpu_first,
            "stats": stats,
            "noisy": stats["iqr_over_median"] > NOISE_THRESHOLD,
            "reps": reps,
        },
        "fallbacks": [{"category": c, "message": m} for c, m in fallbacks],
        "unstable_reps": unstable,
        "tier_after": tier(),
    })
    entry["status"], entry["reason"] = verdict(entry)
    return entry


def verdict(entry: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """'exact' or 'failed', with the first reason a reader needs."""
    if entry["fallbacks"]:
        return "failed", "the GPU sweep fell back to the CPU: " + entry["fallbacks"][0]["message"]
    if entry["tier_before"] != entry["tier_after"]:
        return "failed", "the licence tier changed during the point"
    if entry["parity"]["status"] != "exact":
        return "failed", "GPU and CPU results differ beyond the gate"
    if entry["unstable_reps"]:
        return "failed", "a repetition did not reproduce the first call"
    return "exact", None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_point(text: str) -> Tuple[int, int]:
    """'bars:combos'. A third field (the CPU job's ':oos') is accepted and ignored."""
    parts = text.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"{text!r}: expected bars:combos")
    try:
        bars, combos = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: expected bars:combos") from exc
    if bars <= 0 or combos <= 0:
        raise argparse.ArgumentTypeError(f"{text!r}: both must be positive")
    return bars, combos


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", nargs="+", type=parse_point,
                        default=[parse_point(p) for p in DEFAULT_POINTS],
                        metavar="BARS:COMBOS", help="grid points (default: the CPU job's)")
    parser.add_argument("--reps", type=int, default=7,
                        help="timed GPU repetitions per point (default 7)")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="threads for the untimed CPU parity run "
                             "(default: the usable cores)")
    parser.add_argument("--platform", default="",
                        help="where this ran, as the report should name it")
    parser.add_argument("--out", required=True, help="results JSON")
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    return args


def write(path: str, result: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    import manifoldbt as mbt

    threads = args.cpu_threads or usable_cores()["usable"]
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(args.platform, threads),
        "reps": args.reps,
        "calibration": None,
        "sweeps": [],
        "status": "failed",
        "reason": None,
    }
    env = result["environment"]
    card = env["gpus"][0]["name"] if env["gpus"] else "no GPU visible"
    print(f"manifoldbt {env['versions']['manifoldbt']} | {card} | "
          f"CPU parity run on {threads} threads", flush=True)

    if not tier()["pro"]:
        result["reason"] = "no licensed tier: GPU sweeps are not run unlicensed"
        write(args.out, result)
        print(result["reason"], file=sys.stderr)
        return 1
    try:
        result["calibration"] = mbt.calibrate_gpu()
    except Exception as exc:  # noqa: BLE001 - reported, then the run stops
        result["reason"] = f"calibrate_gpu failed: {type(exc).__name__}: {exc}"
        write(args.out, result)
        print(result["reason"], file=sys.stderr)
        return 1

    for bars, combos in args.sweep:
        try:
            entry = measure_point(bars, combos, args.reps, threads)
        except Exception as exc:  # noqa: BLE001 - one point failing is reported
            entry = {"bars": bars, "requested_combos": combos, "status": "failed",
                     "reason": f"{type(exc).__name__}: {exc}"}
        result["sweeps"].append(entry)
        write(args.out, result)  # a crash further on still leaves this point
        if entry["status"] == "exact":
            t = entry["timings"]
            print("{:>9,} bars x {:>6,} combos | GPU {:8.3f} s | first call {:.3f} s | "
                  "{}/{} cells bit-identical to the CPU{}".format(
                      bars, entry["combos"], t["seconds"], t["first_call_s"],
                      entry["parity"]["identical_cells"], entry["combos"],
                      " | NOISY" if t["noisy"] else ""), flush=True)
        else:
            print(f"{bars:>9,} bars x {combos:>6,} combos | FAILED: {entry['reason']}",
                  flush=True)

    failed = [e for e in result["sweeps"] if e["status"] != "exact"]
    result["status"] = "failed" if failed else "exact"
    result["reason"] = f"{len(failed)} point(s) failed" if failed else None
    write(args.out, result)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
