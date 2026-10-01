"""The runner: interleaved A/B timing with a parity gate in front of it.

Design decisions that are the whole point of this harness
---------------------------------------------------------
*Parity first.* Every workload runs once per engine before any measurement, and
each challenger's result is compared against the reference. A workload an engine
disagrees on gets no published timing for that engine (see ``parity.py``).

*Interleaved repetitions.* The engines alternate within each repetition rather
than running in blocks. A cloud runner that slows down halfway through penalises
all of them equally instead of whichever one happened to be last.

*Ratios are the headline, milliseconds are context.* The per-repetition ratio is
computed from measurements taken seconds apart on the same machine, so it
survives the noise that absolute timings on shared hardware do not.

*Dispersion is published.* Every point carries min, median, max and IQR. A point
whose IQR exceeds 15% of its median is flagged noisy, and a flagged point is not
headline material no matter how good it looks.

*An engine that cannot run a workload says so.* It is dropped from that workload
with its reason recorded, never left as a blank cell: a missing number in a
speed table reads as a loss, and "this engine has no fixed-quantity sizing" is
not a loss, it is a different fact that deserves its own sentence.

Usage
-----
    python bench.py --bars 10000 100000 1000000 --reps 7 --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod  # noqa: E402
import engines as engines_mod  # noqa: E402
import parity as parity_mod  # noqa: E402
from engines import CHALLENGERS, ENGINES, REFERENCE  # noqa: E402
from workloads import (  # noqa: E402
    DEFAULT_KEYS,
    SCOPE_PAIR,
    WORKLOADS,
    supported,
    unsupported_by,
)


def _in_range(key: str, bars: int) -> bool:
    """Is this workload still a comparison at this series length?

    A ceiling is not a performance limit, it is a validity one. The fee workload
    bankrupts its account past a certain length, and two engines agreeing that a
    dead account is worth zero is not a measurement. Skipping is loud in the run
    output rather than silent, because a table that is short by one row reads
    like a choice.
    """
    ceiling = WORKLOADS[key].max_bars
    return ceiling is None or bars <= ceiling

# 2: timings, parity and speedups became per-engine maps when the harness grew
# past two engines. `report.py` reads version 1 files as well, so the results
# archived under results/ stay readable.
SCHEMA_VERSION = 2
NOISE_THRESHOLD = 0.15


# --------------------------------------------------------------------------- #
# Environment: what a reader needs to know before trusting a number
# --------------------------------------------------------------------------- #
def _cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif sys.platform.startswith("win"):
            return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb():
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal"):
                        return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except Exception:
        pass
    return None


def _versions() -> Dict[str, str]:
    """Distribution versions, engines first then the stack underneath them.

    Read from the installed metadata rather than from each package's own
    ``__version__``, which is a hand-maintained constant and can lag: raptorbt
    0.4.1 still declares 0.4.0 in its ``__init__``.
    """
    import importlib.metadata as md

    names = [ENGINES[n].distribution for n in ENGINES]
    names += ["numpy", "numba", "pandas"]
    out = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "absent"
    return out


def environment() -> Dict[str, Any]:
    run_id = os.environ.get("GITHUB_RUN_ID")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_url = None
    if run_id and repo:
        run_url = server + "/" + repo + "/actions/runs/" + run_id
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python": platform.python_version(),
        "versions": _versions(),
        "runner": os.environ.get("RUNNER_NAME") or "local",
        "ci": bool(run_id),
        "run_url": run_url,
        "commit": os.environ.get("GITHUB_SHA"),
    }


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _time_once(fn: Callable[[], Dict[str, Any]]):
    start = time.perf_counter()
    metrics = fn()
    return time.perf_counter() - start, metrics


def _cpu_seconds() -> float:
    """Process CPU time, user plus system. Returns 0.0 if psutil is missing."""
    try:
        import psutil

        times = psutil.Process().cpu_times()
        return times.user + times.system
    except Exception:
        return 0.0


class _Parallelism:
    """CPU time over wall time, accumulated across every repetition.

    This is the number that decides whether a result measured on a 20-core
    workstation transposes to a 4-vCPU cloud runner. Close to 1.0 means the
    engine used one thread and the core count barely matters; well above 1.0
    means the measurement is a function of the machine it ran on, and a
    reader on smaller hardware should expect less. Accumulating over all
    repetitions rather than timing each call keeps the coarse resolution of
    the OS accounting clocks from dominating short runs.

    Below `MIN_WALL_S` of accumulated work the answer is withheld rather than
    guessed. Process CPU accounting advances in scheduler ticks of roughly 15 ms,
    so a sub-millisecond run lands either on zero ticks or on one whole tick, and
    the ratio comes out as 0.00 or as 11.0 for work that is plainly single
    threaded. A benchmark that prints "this engine used 11 cores" once has
    spent its credibility on a rounding artefact.
    """

    MIN_WALL_S = 0.5

    def __init__(self) -> None:
        self.cpu = 0.0
        self.wall = 0.0

    def record(self, fn: Callable[[], Dict[str, Any]]):
        cpu_before = _cpu_seconds()
        elapsed, metrics = _time_once(fn)
        self.cpu += _cpu_seconds() - cpu_before
        self.wall += elapsed
        return elapsed, metrics

    @property
    def ratio(self):
        if self.wall < self.MIN_WALL_S:
            return None
        return self.cpu / self.wall


def _summarise(samples: List[float]) -> Dict[str, Any]:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    if len(ordered) >= 4:
        mid = len(ordered) // 2
        lower = statistics.median(ordered[:mid])
        upper = statistics.median(ordered[-mid:])
        iqr = upper - lower
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


def _roll_up(verdicts: Dict[str, Dict[str, Any]]) -> str:
    """One status for the whole entry: the worst any engine came back with."""
    statuses = {v["status"] for v in verdicts.values()}
    for worst in ("failed", "documented"):
        if worst in statuses:
            return worst
    return "exact"


def _prepare_all(key: str, frame, workdir: str, active: List[str]):
    """Build one timed closure per engine that can run this workload."""
    return {
        name: engines_mod.adapter(name).prepare(key, frame, workdir)
        for name in active
        if supported(key, name)
    }


def _warm_and_gate(key: str, runners: Dict[str, Callable]) -> Dict[str, Any]:
    """One discarded call per engine, then the parity verdicts it feeds.

    The warmup is not only there to be thrown away: it is the run the gate
    reads. numba compiles on vectorbt's first call and the engines warm their
    own caches, so the same call cannot be both the first measurement and a fair
    one, but it is a perfectly good sample of what each engine *computed*.
    """
    warm = {name: _time_once(run)[1] for name, run in runners.items()}
    reference = warm[REFERENCE]
    return {
        name: parity_mod.compare(reference, metrics, key, name)
        for name, metrics in warm.items()
        if name != REFERENCE
    }


def _entry(key: str, bars: int, frame, verdicts: Dict[str, Any], engines_run: List[str]):
    entry: Dict[str, Any] = {
        "workload": key,
        "title": WORKLOADS[key].title,
        "bars": bars,
        "data_digest": data_mod.digest(frame),
        "engines": engines_run,
        "parity": verdicts,
        "status": _roll_up(verdicts),
    }
    skipped = unsupported_by(key)
    if skipped:
        entry["unsupported"] = skipped
    return entry


def _collect(entry: Dict[str, Any], runners: Dict[str, Callable],
             samples: Dict[str, List[float]], threading_use: Dict[str, _Parallelism]) -> None:
    """Turn raw per-engine samples into the published shape, in place."""
    published = [
        name for name in runners
        if name == REFERENCE or entry["parity"][name]["status"] != "failed"
    ]
    # A solo run has nothing to compare and nothing to withhold: the gate exists
    # to stop a *comparison* being published on mismatched work. When challengers
    # were present and the gate dropped them all, though, the reference's timing
    # is the leftover of a comparison, and printing it alone would read as one.
    solo = len(runners) == 1
    if REFERENCE not in published or (len(published) == 1 and not solo):
        entry["timings"] = None
        entry["note"] = "timing withheld: unexplained disagreement with the reference"
        return

    stats = {name: _summarise(samples[name]) for name in published}
    reference = samples[REFERENCE]
    entry["timings"] = stats
    entry["speedup"] = {}
    for name in published:
        if name == REFERENCE:
            continue
        ratios = [
            other / ref if ref > 0 else float("nan")
            for ref, other in zip(reference, samples[name])
        ]
        entry["speedup"][name] = {
            "median_of_ratios": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
        }
    entry["noisy"] = any(s["iqr_over_median"] > NOISE_THRESHOLD for s in stats.values())
    entry["cpu_over_wall"] = {name: threading_use[name].ratio for name in published}

    withheld = [name for name in runners if name not in published]
    if withheld:
        entry["withheld"] = withheld


def measure_pair(keys: List[str], bars: int, reps: int, workdir: str,
                 active: List[str]) -> List[Dict[str, Any]]:
    """Measure two workloads inside ONE interleaved loop.

    The report puts these two side by side to show what a performance summary
    costs each engine. That subtraction is only legitimate if every timing comes
    from the same stretch of machine time: measured in separate blocks, the drift
    in absolute timings is larger than the difference being reported, and the
    table ends up claiming the version doing more work is the faster one.
    """
    frame = data_mod.make_ohlcv(bars)
    runners: Dict[str, Dict[str, Callable]] = {}
    entries: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        runners[key] = _prepare_all(key, frame, workdir, active)
        verdicts = _warm_and_gate(key, runners[key])
        entries[key] = _entry(key, bars, frame, verdicts, list(runners[key]))
        entries[key]["paired_with"] = [k for k in keys if k != key]

    samples = {key: {name: [] for name in runners[key]} for key in keys}
    threading_use = {key: {name: _Parallelism() for name in runners[key]} for key in keys}
    for _ in range(reps):
        for key in keys:
            for name, run in runners[key].items():
                elapsed, _ = threading_use[key][name].record(run)
                samples[key][name].append(elapsed)

    for key in keys:
        _collect(entries[key], runners[key], samples[key], threading_use[key])
    return [entries[key] for key in keys]


def measure(key: str, bars: int, reps: int, workdir: str, active: List[str]) -> Dict[str, Any]:
    frame = data_mod.make_ohlcv(bars)
    runners = _prepare_all(key, frame, workdir, active)
    verdicts = _warm_and_gate(key, runners)
    entry = _entry(key, bars, frame, verdicts, list(runners))

    scales = {}
    for name, verdict in verdicts.items():
        if verdict["status"] != "documented":
            continue
        adapter = engines_mod.adapter(name)
        measured = getattr(adapter, "diagnose", lambda *a, **k: {})(key, frame, workdir)
        if measured:
            scales[name] = measured
    # The reference's own view of the divergence: which of its round-trips are
    # the ones the others do not take. Recorded under the reference so the two
    # sides of the subtraction sit next to each other.
    if scales:
        reference_view = engines_mod.adapter(REFERENCE).diagnose(key, frame, workdir)
        if reference_view:
            scales[REFERENCE] = reference_view
        entry["divergence_scale"] = scales

    samples = {name: [] for name in runners}
    threading_use = {name: _Parallelism() for name in runners}
    for _ in range(reps):
        for name, run in runners.items():
            elapsed, _ = threading_use[name].record(run)
            samples[name].append(elapsed)

    _collect(entry, runners, samples, threading_use)
    return entry


# --------------------------------------------------------------------------- #
# Probes that need a fresh process (see probe_child.py)
# --------------------------------------------------------------------------- #
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_child.py")


def _probe(mode: str, engine: str, workload: str, bars: int) -> Dict[str, Any]:
    out = subprocess.run(
        [sys.executable, CHILD, mode, engine, workload, str(bars)],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("PROBE_RESULT "):
            return json.loads(line[len("PROBE_RESULT "):])
    raise RuntimeError("probe produced no result: " + out[-500:])


def cold_start(workload: str, bars: int, reps: int, active: List[str]) -> Dict[str, Any]:
    """Time to a first backtest in a process that has never seen any engine.

    This is the cost the steady-state benchmark throws away as warmup, and the
    one a user actually waits through every time they open a notebook. The
    engines alternate here too, and the interpreter/numpy/pandas baseline is
    measured alongside so each engine's own share is visible.
    """
    order = [name for name in active if supported(workload, name)]
    samples: Dict[str, List[float]] = {name: [] for name in order}
    samples["baseline"] = []
    for _ in range(reps):
        for name in order:
            samples[name].append(
                _probe("coldstart", ENGINES[name].code, workload, bars)["seconds"])
        samples["baseline"].append(_probe("baseline", "none", workload, bars)["seconds"])
    medians = {k: statistics.median(v) for k, v in samples.items()}
    base = medians["baseline"]
    reference = medians[REFERENCE]
    return {
        "workload": workload,
        "bars": bars,
        "engines": order,
        "samples_s": samples,
        "median_s": medians,
        "engine_share_s": {name: medians[name] - base for name in order},
        "ratio": {
            name: (medians[name] / reference if reference else None)
            for name in order if name != REFERENCE
        },
    }


def memory(workload: str, bars: int, active: List[str]) -> Dict[str, Any]:
    """Resident memory each engine adds while running one backtest.

    vectorbt materialises the simulation as arrays, so its footprint grows with
    the series; manifoldbt streams bars out of its store. The number reported is
    what the *run* adds, measured after a warmup, not the process total: the
    one-off cost of building each engine's data representation is a different
    question and is excluded on every side.
    """
    order = [name for name in active if supported(workload, name)]
    out: Dict[str, Any] = {"workload": workload, "bars": bars, "engines": order}
    for name in order:
        out[name] = _probe("memory", ENGINES[name].code, workload, bars)
    return out


SWEEP_CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_child.py")


def parse_point(spec: str) -> tuple:
    """`bars:combos[:oos]`, e.g. `20000:5000` or `20000:100000:oos`.

    The `oos` suffix declares the array-materialising engines out of scope for
    that point: it is for grids whose vectorbt side needs tens of gigabytes,
    where running it anyway would time the swap file rather than the engine.
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not parts[1]:
        raise argparse.ArgumentTypeError(
            f"expected bars:combos or bars:combos:oos, got {spec!r}")
    mode = "run"
    if len(parts) == 3:
        if parts[2] != "oos":
            raise argparse.ArgumentTypeError(
                f"third field must be 'oos', got {parts[2]!r}")
        mode = "oos"
    return int(parts[0]), int(parts[1]), mode


def sweep(points: List[tuple], reps: int, active: List[str]) -> List[Dict[str, Any]]:
    """Parameter-grid comparison, one point per process.

    Each point is spawned rather than run inline. A large grid is the one thing
    in this harness that can exhaust the machine, and a point that dies must
    cost its own result and nothing else: run inline, an out-of-memory kill
    would take the whole benchmark down and lose every number measured before
    it. A dead child is recorded as a crashed point and the run continues.
    """
    out: List[Dict[str, Any]] = []
    challengers = [n for n in active if n != REFERENCE]
    for bars, combos, mode in points:
        proc = subprocess.run(
            [sys.executable, SWEEP_CHILD, "--bars", str(bars),
             "--combos", str(combos), "--reps", str(reps),
             "--engines", *active, "--vectorbt", mode],
            capture_output=True, text=True,
        )
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                payload = json.loads(line)
        if payload is None:
            # No JSON at all: the child died before it could report. Almost
            # always the allocator, on the biggest grid of the matrix.
            payload = {
                "bars": bars,
                "combos": combos,
                "vectorbt_mode": mode,
                "status": "crashed",
                "reason": (proc.stderr or proc.stdout or "no output")[-400:].strip(),
                "exit_code": proc.returncode,
            }
        # Memory comes from dedicated single-engine processes, never from the
        # interleaved timing run above. Timing has to interleave the engines to
        # be fair, and interleaving is exactly what makes a memory reading
        # worthless: from the second repetition on, the peak sampled during one
        # engine's call is the whole process, the other engines' allocations
        # included. Measured both ways, manifoldbt read 6.7 GB interleaved
        # against 94 MB alone, at 5000 combinations.
        payload["memory"] = _sweep_memory(bars, combos, mode, active)
        out.append(payload)
    return out


def _sweep_memory(bars: int, combos: int, mode: str, active: List[str]) -> Dict[str, Any]:
    """Peak each engine adds running the grid once, one engine per process."""
    names = [REFERENCE] if mode == "oos" else list(active)
    added: Dict[str, Any] = {name: None for name in active}
    for name in names:
        proc = subprocess.run(
            [sys.executable, SWEEP_CHILD, "--bars", str(bars),
             "--combos", str(combos), "--memory-only", ENGINES[name].code],
            capture_output=True, text=True,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                added[name] = json.loads(line)["added_mb"]
    return added


def run_sweeps(points: List[tuple], reps: int, active: List[str]) -> tuple:
    """Drive the sweep matrix and print one line per point. Returns (entries, failures).

    Three outcomes count as failures, and the distinction matters:

    * ``failed``  - the engines disagreed on a grid nobody predicted. The gate.
    * ``crashed`` - the point could not be measured at all.
    * ``skipped`` - the point asked for a timing the process was not licensed to
      produce. Silent here would be the worst outcome of all: the matrix would
      simply come back short, and a table missing its largest grid reads like a
      choice rather than a failure.
    """
    entries = sweep(points, reps, active)
    failures = 0
    for e in entries:
        head = "  sweep {b:>9,} bars x {c:>6,} combos ... ".format(
            b=e["bars"], c=e["combos"])
        status = e.get("status") or e.get("parity", {}).get("status", "?")
        ratios = (e.get("timings") or {}).get("ratio") or {}
        if ratios:
            print(head + "{s:10s} ".format(s=status) + ", ".join(
                "{n} x{r:.1f}".format(n=name, r=ratio) for name, ratio in ratios.items()))
        elif e.get("timings"):
            print(head + "{s:10s} {ref} {t:.2f} s, challengers out of scope".format(
                s=status, ref=REFERENCE, t=e["timings"]["seconds"][REFERENCE]))
        else:
            print(head + "{s:10s} {why}".format(
                s=status, why=e.get("reason") or e.get("note") or "no timing"))
            failures += 1
    return entries, failures


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="manifoldbt against other engines")
    parser.add_argument("--bars", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--workloads", nargs="+", default=DEFAULT_KEYS, choices=DEFAULT_KEYS)
    parser.add_argument("--engines", nargs="*", default=CHALLENGERS, choices=CHALLENGERS,
                        help="challengers to run against " + REFERENCE + ". Pass it "
                             "with no value for a solo run: no comparison, no "
                             "ratios, just what the engine costs. That is a "
                             "regression tracker rather than a benchmark, and it "
                             "is the only shape whose wall time is the engine's "
                             "own rather than the slowest challenger's.")
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--workdir", default=None, help="where the engine store is built")
    parser.add_argument("--cold-start-reps", type=int, default=3,
                        help="0 disables the cold-start probe")
    parser.add_argument("--memory-bars", type=int, default=0,
                        help="bars for the memory probe; 0 disables it")
    parser.add_argument("--sweep", type=parse_point, nargs="*", default=[],
                        metavar="BARS:COMBOS",
                        help="parameter-grid points, e.g. 20000:5000. Needs a "
                             "licence: sweep timings are published from a "
                             "licensed run only")
    parser.add_argument("--sweep-reps", type=int, default=3,
                        help="repetitions per sweep point; fewer than --reps "
                             "because a large grid costs seconds, not milliseconds")
    parser.add_argument("--pin-cores", type=int, default=0,
                        help="restrict the process to N logical cores, so a big "
                             "workstation can reproduce what a small cloud runner "
                             "sees; 0 leaves the machine alone")
    args = parser.parse_args()

    if not engines_mod.installed(REFERENCE):
        print("cannot run: the reference engine ({}) is not installed".format(REFERENCE))
        return 2
    active = [REFERENCE] + engines_mod.present(args.engines)
    for name in args.engines:
        if name not in active:
            # Not fatal: somebody benchmarking on their own machine should not
            # have to install every competitor to read their own numbers. It is
            # loud, though, because on a runner it means the lock file did not
            # do its job.
            print("skipping {}: not installed".format(name))

    pinned = None
    if args.pin_cores:
        try:
            import psutil

            everything = list(range(os.cpu_count() or 1))
            psutil.Process().cpu_affinity(everything[: args.pin_cores])
            pinned = args.pin_cores
        except Exception as exc:            # macOS has no affinity API
            print("could not pin to {} cores: {}".format(args.pin_cores, exc))

    workdir = args.workdir or tempfile.mkdtemp(prefix="mbt_bench_")
    env = environment()
    env["pinned_cores"] = pinned
    env["engines"] = active
    versions = env["versions"]
    print(" vs ".join(
        "{} {}".format(name, versions[ENGINES[name].distribution]) for name in active))
    print("{cpu} | {cores} logical cores | {ram} GB | {os} {arch}".format(
        cpu=env["cpu"], cores=env["logical_cores"], ram=env["ram_gb"],
        os=env["os"], arch=env["arch"]))
    if pinned:
        print("pinned to {} logical cores for this run".format(pinned))
    print(str(args.reps) + " interleaved repetitions per point\n")

    def announce(entry: Dict[str, Any]) -> bool:
        print("  {k:18s} {b:>9,} bars ... ".format(k=entry["workload"], b=entry["bars"]),
              end="", flush=True)
        if not entry.get("timings"):
            print("{s:10s} timing withheld".format(s=entry["status"]))
            return True
        speeds = ", ".join(
            "{n} x{v:.1f}".format(n=name, v=s["median_of_ratios"])
            for name, s in entry["speedup"].items()
        ) or "no challenger"
        print("{s:10s} {speeds}{m}".format(
            s=entry["status"], speeds=speeds,
            m=" NOISY" if entry.get("noisy") else ""))
        return entry["status"] == "failed"

    # The scope pair is measured together; everything else one workload at a time.
    paired = [k for k in SCOPE_PAIR if k in args.workloads]
    singles = [k for k in args.workloads if k not in paired]

    def skip(key: str, bars: int) -> None:
        print("  {k:18s} {b:>12,} bars ... skipped    beyond this workload's "
              "ceiling of {c:,} bars".format(
                  k=key, b=bars, c=WORKLOADS[key].max_bars))

    results = []
    failures = 0
    for bars in args.bars:
        runnable = [k for k in paired if _in_range(k, bars)]
        for key in paired:
            if key not in runnable:
                skip(key, bars)
        if len(runnable) > 1:
            for entry in measure_pair(runnable, bars, args.reps, workdir, active):
                results.append(entry)
                failures += announce(entry)
        elif runnable:
            entry = measure(runnable[0], bars, args.reps, workdir, active)
            results.append(entry)
            failures += announce(entry)
    for key in singles:
        for bars in args.bars:
            if not _in_range(key, bars):
                skip(key, bars)
                continue
            entry = measure(key, bars, args.reps, workdir, active)
            results.append(entry)
            failures += announce(entry)

    # The side probes run on the first workload every active engine supports, so
    # a cold-start table cannot come back missing a column because the workload
    # happened to be one somebody sits out.
    probe_workload = next(
        (k for k in args.workloads
         if all(supported(k, n) for n in active) and _in_range(k, 20_000)),
        args.workloads[0],
    )

    cold = None
    if args.cold_start_reps:
        print("")
        print("  cold start ... ", end="", flush=True)
        cold = cold_start(probe_workload, 20_000, args.cold_start_reps, active)
        print(", ".join("{n} {t:.2f} s".format(n=name, t=cold["median_s"][name])
                        for name in cold["engines"]))

    mem = None
    if args.memory_bars:
        print("  memory     ... ", end="", flush=True)
        mem = memory(probe_workload, args.memory_bars, active)
        print(", ".join("{n} +{v:.0f} MB".format(n=name, v=mem[name]["added_mb"])
                        for name in mem["engines"]))

    sweeps = None
    if args.sweep:
        print("")
        sweeps, sweep_failures = run_sweeps(args.sweep, args.sweep_reps, active)
        failures += sweep_failures

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "reference": REFERENCE,
        "engines": active,
        "reps": args.reps,
        "results": results,
        "cold_start": cold,
        "memory": mem,
        "sweeps": sweeps,
        "sweep_reps": args.sweep_reps if args.sweep else None,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote " + args.out)

    # A parity failure is a finding, and CI should go red on it.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
