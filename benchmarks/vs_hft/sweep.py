"""Parameter sweeps: the same grid in every engine.

manifoldbt sweeps a strategy through `run_sweep` / `run_sweep_lite` (the knobs
are `mbt.param`s; the book and the tape are loaded once and the combinations run
in parallel inside the engine, on the CPU).

hftbacktest and NautilusTrader have no sweep call. What a user does is a loop of
runs, and with several cores a pool of processes, each holding its own copy of
the data. That loop is what is timed here, with the processes created and warmed
beforehand (reported apart) so the pool is not charged for its start.

Families
  maker    clip x max_clips x extra levels back on each side   (spec.GRID_AXES: 4, 16, 64, 256)
  latency  the order delay as an axis (0, 1, 5, 10 ms) x inventory bound x levels back; the cancel
           delay is the order delay, in every engine (one value per combination, not a product)
  taker    flow threshold x order size, the flow taker on the trade clock

    python sweep.py --day 2025-08-13 --window 36000 43200 --grids 4 16 64 \
        --engines hftbacktest nautilus manifoldbt --workers 4 --out sweeps.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
import spec  # noqa: E402

_W = {}

GRID_LATENCY = {16: dict(clip=[0.01], max_clips=[3, 5], off_bid=[0, 1], off_ask=[0], lat_order_ms=[0, 1, 5, 10])}
GRID_TAKER = {12: dict(thr=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95], qty=[0.01, 0.02])}


def family_grid(family, n):
    import itertools

    axes = {"maker": spec.GRID_AXES, "latency": GRID_LATENCY, "taker": GRID_TAKER}[family][n]
    names = sorted(axes)
    return axes, names, [dict(zip(names, v)) for v in itertools.product(*[axes[k] for k in names])]


def mbt_latency_grid(lat_ms):
    """The latency axis of the grid, written for manifoldbt: ONE execution axis whose value is the whole
    latency model, so the order delay and the cancel delay move together, as they do in hftbacktest
    (`constant_order_latency`) and in NautilusTrader (`LatencyModel`). Two axes, `latency.order` and
    `latency.cancel`, are swept as a product: N values on each run N x N combinations, N of which have
    the two delays equal.

    The response and the feed delay do not move; the axis restates them, because it replaces the whole
    model. The durations go in as nanoseconds: the wheel converts an `Interval` only when it is the whole
    value of an axis, not when it sits inside a dictionary."""
    return {"latency": [{"order": int(x * spec.MS), "cancel": int(x * spec.MS),
                         "response": spec.LAT_RESPONSE_NS, "feed": spec.LAT_FEED_NS} for x in lat_ms]}


def check_served(n_served, combos, what):
    """A throughput is the combinations announced over the wall time, so it is only one if the engine ran
    those combinations and no others. Refuse to go on when it returned another number."""
    if n_served != len(combos):
        raise RuntimeError(f"{what}: the engine returned {n_served} results for {len(combos)} announced "
                           f"combinations; no throughput is reported for a grid that is not the one announced")


def combo_params(c, tier):
    return dict(clip=c["clip"], max_clips=c["max_clips"], off_bid=c["off_bid"], off_ask=c["off_ask"],
                ladder=(tier == "l2"))


def summarize(out):
    return {k: out[k] for k in ("posts", "fills", "volume", "position", "decisions", "orders") if k in out}


def _noop(_):
    return 0


# ------------------------------------------------------------------ hftbacktest
def _hft_init(npz, tier, family):
    import engine_hft as H
    d = data.load(npz)
    _W.update(d=d, tier=tier, family=family, H=H, window=tuple(int(x) for x in d["window"]))
    feed = spec.LAT_FEED_NS if family != "taker" else 0
    _W["ev"] = H.build_events(d, "bbo" if family == "taker" else tier, feed)
    c = dict(clip=0.01, max_clips=5, off_bid=0, off_ask=0, thr=0.9, qty=0.01)
    _hft_task(c)                                             # compile outside the timing


def _hft_task(c):
    H, d = _W["H"], _W["d"]
    if _W["family"] == "taker":
        p = dict(window_ns=spec.FlowTaker.WINDOW_NS, threshold=c["thr"], qty=c["qty"])
        out, _ = H.run_taker(d, _W["ev"], p, None, 20 * spec.MS, 0)
        return summarize(out)
    o_ns = int(c["lat_order_ms"] * spec.MS) if "lat_order_ms" in c else spec.LAT_ORDER_NS
    out, _ = H.run_mm(d, _W["tier"], _W["ev"], combo_params(c, _W["tier"]), _W["window"],
                      order_ns=o_ns, resp_ns=spec.LAT_RESPONSE_NS)
    return summarize(out)


def hft_serial(npz, tier, family, combos):
    t0 = time.perf_counter()
    _hft_init(npz, tier, family)
    setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = [_hft_task(c) for c in combos]
    return res, {"setup_s": setup, "sweep_s": time.perf_counter() - t0, "workers": 1}


def hft_pool(npz, tier, family, combos, workers):
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    pool = ctx.Pool(workers, initializer=_hft_init, initargs=(npz, tier, family))
    pool.map(_noop, range(workers), chunksize=1)          # wait for every worker to be ready
    setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = pool.map(_hft_task, combos, chunksize=1)
    wall = time.perf_counter() - t0
    pool.close()
    pool.join()
    return res, {"setup_s": setup, "sweep_s": wall, "workers": workers}


# ------------------------------------------------------------------ Nautilus
def _nt_init(npz, tier, family):
    import engine_nt as N
    d = data.load(npz)
    _W.update(d=d, tier=tier, family=family, N=N, window=tuple(int(x) for x in d["window"]))


def _nt_task(c):
    N, d = _W["N"], _W["d"]
    if _W["family"] == "taker":
        p = dict(window_ns=spec.FlowTaker.WINDOW_NS, threshold=c["thr"], qty=c["qty"])
        out, _ = N.run_taker(d, p, None, "nominal")
    else:
        o_ns = int(c["lat_order_ms"] * spec.MS) if "lat_order_ms" in c else None
        out, _ = N.run_mm(d, _W["tier"], combo_params(c, _W["tier"]), _W["window"], "nominal", order_ns=o_ns)
    s = summarize(out)
    s["engine_s"] = out["engine_seconds"]
    s["convert_s"] = out["convert_seconds"]
    return s


def nt_serial(npz, tier, family, combos):
    _nt_init(npz, tier, family)
    t0 = time.perf_counter()
    res = [_nt_task(c) for c in combos]
    return res, {"setup_s": 0.0, "sweep_s": time.perf_counter() - t0, "workers": 1}


def nt_pool(npz, tier, family, combos, workers):
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    pool = ctx.Pool(workers, initializer=_nt_init, initargs=(npz, tier, family))
    pool.map(_noop, range(workers), chunksize=1)
    setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = pool.map(_nt_task, combos, chunksize=1)
    wall = time.perf_counter() - t0
    pool.close()
    pool.join()
    return res, {"setup_s": setup, "sweep_s": wall, "workers": workers}


# ------------------------------------------------------------------ manifoldbt
def mbt_sweep(day, window_ns, tier, family, n, workers, lite=True, root=None):
    """`run_sweep_lite` / `run_sweep` over the grid. Written against the documented API; see DESIGN.md."""
    import manifoldbt as mbt
    import engine_mbt as M

    axes, names, combos = family_grid(family, n)
    store = M.open_store(root or data.DATA_ROOT, day, "bbo" if family == "taker" else tier)
    t0 = time.perf_counter()
    if family == "taker":
        res = mbt.run_sweep(M.flow_taker_strategy(params=True), axes,
                            M.flow_taker_config(day, "nominal", window_ns), store, max_parallelism=workers)
    else:
        grid = {k: v for k, v in axes.items() if k != "lat_order_ms"}
        ex = mbt_latency_grid(axes["lat_order_ms"]) if "lat_order_ms" in axes else None
        strat = M.maker_strategy({"ladder": tier == "l2"}, params=True)
        cfg = M.maker_config(day, window_ns, "nominal")
        fn = mbt.run_sweep_lite if lite else mbt.run_sweep
        res = fn(strat, grid, cfg, store, max_parallelism=workers, execution_grid=ex)
    wall = time.perf_counter() - t0
    check_served(len(list(res)), combos, f"manifoldbt {family} grid {n}")
    return res, {"sweep_s": wall, "workers": workers, "combos": len(combos), "lite": lite}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2025-08-23")
    ap.add_argument("--window", nargs=2, type=int, default=[36000, 43200])
    ap.add_argument("--tier", default="l2")
    ap.add_argument("--family", default="maker", choices=["maker", "latency", "taker"])
    ap.add_argument("--grids", nargs="*", type=int, default=[4, 16, 64])
    ap.add_argument("--engines", nargs="*", default=["hftbacktest", "nautilus"],
                    help="manifoldbt sweeps run in sweep_mbt.py (one process per call, for its own peak memory)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nt-max-combos", type=int, default=16, help="Nautilus runs at most this many combinations per grid")
    ap.add_argument("--pin", type=int, default=0, help="pin this process (and its children) to the first N logical cpus")
    ap.add_argument("--fail-on-blocked", action="store_true")
    ap.add_argument("--out", default="sweeps.json")
    args = ap.parse_args()
    if args.pin:
        import psutil
        psutil.Process().cpu_affinity(list(range(args.pin)))
    window = None if args.family == "taker" else tuple(args.window)
    npz, meta = data.prepare(args.day, window)
    tier = "bbo" if args.family == "taker" else args.tier
    out = {"schema_version": 3, "day": args.day, "window_s": list(window) if window else None, "tier": tier,
           "family": args.family, "workers": args.workers, "pinned_cpus": args.pin or None, "grids": {}}
    for n in args.grids:
        axes, names, combos = family_grid(args.family, n)
        g = {"combos": len(combos), "axes": axes, "engines": {}}
        if "hftbacktest" in args.engines:
            r1, t1 = hft_serial(npz, tier, args.family, combos)
            r2, t2 = hft_pool(npz, tier, args.family, combos, args.workers)
            g["engines"]["hftbacktest"] = {"serial": t1, "pool": t2, "pool_equals_serial": r1 == r2,
                                           "combos_per_s_pool": len(combos) / t2["sweep_s"],
                                           "combos_per_s_serial": len(combos) / t1["sweep_s"], "sample": r1[:3]}
        if "nautilus" in args.engines:
            m = min(len(combos), args.nt_max_combos)
            sub = combos[:: max(1, len(combos) // m)][:m]
            r1, t1 = nt_serial(npz, tier, args.family, sub)
            r2, t2 = nt_pool(npz, tier, args.family, sub, args.workers)
            g["engines"]["nautilus"] = {"combos_run": len(sub), "serial": t1, "pool": t2,
                                        "pool_equals_serial": [summarize(x) for x in r1] == [summarize(x) for x in r2],
                                        "seconds_per_combo_serial": t1["sweep_s"] / len(sub),
                                        "combos_per_s_pool": len(sub) / t2["sweep_s"],
                                        "projected_full_grid_pool_s": t2["sweep_s"] / len(sub) * len(combos),
                                        "sample": [summarize(x) for x in r1[:3]]}
        if "manifoldbt" in args.engines:
            try:
                d = data.load(npz)
                res, t = mbt_sweep(args.day, tuple(int(x) for x in d["window"]), tier, args.family, n, args.workers)
                g["engines"]["manifoldbt"] = {"status": "ok", **t, "combos_per_s": len(combos) / t["sweep_s"]}
            except BaseException as exc:
                refused = type(exc).__name__ in ("PermissionError", "LicenseError")
                g["engines"]["manifoldbt"] = {"status": "blocked" if refused else "failed",
                                              "error": "refused by the environment" if refused and os.environ.get("BENCH_PUBLIC") == "1"
                                              else f"{type(exc).__name__}: {str(exc)[:300]}"}
        out["grids"][str(n)] = g
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print("grid", n, {k: (v.get("combos_per_s_pool") or v.get("combos_per_s") or v.get("status")) for k, v in g["engines"].items()}, flush=True)
    print("wrote", args.out)
    if args.fail_on_blocked:
        bad = [(n, e) for n, g in out["grids"].items() for e, r in g["engines"].items() if r.get("status", "ok") != "ok"]
        if bad:
            print("engines that did not run:", bad, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
