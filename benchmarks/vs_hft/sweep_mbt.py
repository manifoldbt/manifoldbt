"""manifoldbt sweeps of the quoting strategies: throughput, peak memory, and identity with single runs.

    python sweep_mbt.py --day 2025-08-13 --window 36000 43200 --family maker --grids 4 16 64 256 \
        --workers 4 --identity 8 --out results-sweeps-mbt-maker.json

Each (grid, call) runs in a fresh process so that the peak resident memory is the call's own:
  lite   `run_sweep_lite` (metrics only)
  full   `run_sweep` (every combination keeps its order journal)
Identity: a sample of combinations is run alone with `mbt.run` (the parameters written as literals, which
is what "the run `mbt.run` makes alone with those values as its defaults" means) and compared to the
same combination inside the full sweep, bit for bit on the order, fill and event tables, and to the lite
sweep on its metrics.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def peak_rss_mb():
    import psutil

    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)
    if peak:
        return peak / 1e6
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def _df_equal(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f":
            if x.tobytes() != y.tobytes():
                return False
        elif not (x == y).all():
            return False
    return True


def worker(args):
    import numpy as np

    import data
    import engine_mbt as M
    import manifoldbt as mbt
    import spec
    from manifoldbt.helpers import Interval
    from sweep import family_grid

    if args.pin:
        import psutil

        psutil.Process().cpu_affinity(list(range(args.pin)))
    tier = "bbo" if args.family == "taker" else "l2"
    window = None if args.family == "taker" else tuple(args.window)
    path, _ = data.prepare(args.day, window)
    d = data.load(path)
    w = tuple(int(x) for x in d["window"])
    store = M.open_store(data.DATA_ROOT, args.day, tier)
    axes, names, combos = family_grid(args.family, args.grid)
    out = {"grid": args.grid, "combos": len(combos), "call": args.call, "workers": args.workers}
    if args.family == "taker":
        strat = M.flow_taker_strategy(params=True)
        cfg = M.flow_taker_config(args.day, "nominal", w)
        grid, ex = axes, None
    else:
        strat = M.maker_strategy({"ladder": True}, params=True)
        cfg = M.maker_config(args.day, w, "nominal")
        grid = {k: v for k, v in axes.items() if k != "lat_order_ms"}
        ex = None
        if "lat_order_ms" in axes:
            ex = {"latency.order": [Interval.millis(x) for x in axes["lat_order_ms"]],
                  "latency.cancel": [Interval.millis(x) for x in axes["lat_order_ms"]]}
    kw = dict(max_parallelism=args.workers)
    if ex:
        kw["execution_grid"] = ex
    t0 = time.perf_counter()
    if args.call == "lite" and args.family != "taker":
        res = mbt.run_sweep_lite(strat, grid, cfg, store, **kw)
    else:
        res = mbt.run_sweep(strat, grid, cfg, store, **kw)
    out["wall_s"] = round(time.perf_counter() - t0, 3)
    out["combos_per_s"] = round(len(combos) / out["wall_s"], 3)
    out["results"] = len(list(res))
    out["peak_rss_mb"] = round(peak_rss_mb(), 1)
    if args.call == "full" and args.family != "taker":
        sample = []
        for i, r in enumerate(res):
            if i >= 4:
                break
            fl = r.fills_df(backend="pandas")
            sample.append({"posts": int(len(r.orders_df(backend="pandas"))), "fills": int(len(fl)),
                           "volume": float(fl["qty"].sum()) if len(fl) else 0.0,
                           "position": float(r.sim.position[spec.SYMBOL])})
        out["sample"] = sample
    if args.identity and args.call == "full" and args.family == "maker":
        full = list(res)
        # the identity sample is read in the enumeration order: parameters by name, last fastest,
        # execution axes slowest (here none, or the latency axis)
        step = max(1, len(combos) // args.identity)
        idx = list(range(0, len(combos), step))[:args.identity]
        rows = []
        for i in idx:
            c = combos[i]
            p = {k: c[k] for k in ("clip", "max_clips", "off_bid", "off_ask")}
            p["ladder"] = True
            cfg_i = cfg
            if "lat_order_ms" in c:
                cfg_i = M.maker_config(args.day, w, "nominal")
                lat = Interval.millis(c["lat_order_ms"])
                cfg_i = mbt.BacktestConfig(
                    universe=[spec.SYMBOL_ID], time_range_start=w[0], time_range_end=w[1],
                    bar_interval=Interval.millis(spec.WAKE_NS // spec.MS), initial_capital=100_000.0,
                    fees=mbt.FeeConfig(maker_fee_bps=spec.MAKER_FEE_BPS, taker_fee_bps=spec.TAKER_FEE_BPS),
                    execution=mbt.ExecutionConfig(
                        latency={"order": lat, "cancel": lat, "response": Interval.millis(spec.LAT_RESPONSE_NS // spec.MS),
                                 "feed": Interval.millis(spec.LAT_FEED_NS // spec.MS)},
                        fill_model={"queue": {"depth_source": "book", "model": "risk_adverse", "assumed_queue": 2.0}}))
            alone = mbt.run(M.maker_strategy(p, params=False), cfg_i, store)
            # find the combination in the sweep by its labels when it has them, else by position
            sw = full[i]
            rows.append({
                "i": i, **{k: c[k] for k in c},
                "orders": _df_equal(alone.orders_df(backend="pandas"), sw.orders_df(backend="pandas")),
                "fills": _df_equal(alone.fills_df(backend="pandas"), sw.fills_df(backend="pandas")),
                "events": _df_equal(alone.events_df(backend="pandas"), sw.events_df(backend="pandas")),
                "pnl_bits": struct.pack("<d", float(alone.sim.metrics["pnl"])) == struct.pack("<d", float(sw.sim.metrics["pnl"])),
                "n_orders": int(len(alone.orders_df(backend="pandas"))),
            })
        out["identity"] = rows
    if args.identity and args.call == "full" and args.family == "taker":
        full = list(res)
        step = max(1, len(combos) // args.identity)
        rows = []
        for i in list(range(0, len(combos), step))[:args.identity]:
            c = combos[i]
            alone = mbt.run(M.flow_taker_strategy(thr=c["thr"], qty=c["qty"]), cfg, store)
            rows.append({"i": i, **c, "trades": _df_equal(alone.trades_df(), full[i].trades_df()),
                         "n_trades": int(len(alone.trades_df()))})
        out["identity"] = rows
    print("@@SWEEP@@" + json.dumps(out, default=str), flush=True)
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2025-08-13")
    ap.add_argument("--window", nargs=2, type=int, default=[36000, 43200])
    ap.add_argument("--family", default="maker", choices=["maker", "latency", "taker"])
    ap.add_argument("--grids", nargs="*", type=int, default=[4, 16, 64, 256])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pin", type=int, default=0)
    ap.add_argument("--identity", type=int, default=0)
    ap.add_argument("--calls", nargs="*", default=["lite", "full"])
    ap.add_argument("--out", default="sweeps-mbt.json")
    ap.add_argument("--grid", type=int)
    ap.add_argument("--call")
    args = ap.parse_args()
    if args.call:
        worker(args)
        return
    result = {"schema_version": 3, "day": args.day, "window_s": None if args.family == "taker" else args.window,
              "family": args.family, "workers": args.workers, "pinned_cpus": args.pin or None, "grids": {}}
    for n in args.grids:
        g = {}
        for call in args.calls:
            cmd = [sys.executable, os.path.abspath(__file__), "--day", args.day, "--window", str(args.window[0]),
                   str(args.window[1]), "--family", args.family, "--workers", str(args.workers), "--grid", str(n),
                   "--call", call, "--identity", str(args.identity if call == "full" else 0)]
            if args.pin:
                cmd += ["--pin", str(args.pin)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            line = next((l for l in p.stdout.splitlines() if l.startswith("@@SWEEP@@")), None)
            if line:
                g[call] = json.loads(line[len("@@SWEEP@@"):])
            else:
                g[call] = {"status": "failed", "stderr_tail": (p.stderr or p.stdout)[-900:]}
            print(f"grid {n} {call}: {g[call].get('combos_per_s', g[call].get('status'))} combos/s, "
                  f"peak {g[call].get('peak_rss_mb')} MB", flush=True)
        result["grids"][str(n)] = g
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=str)
    print("wrote", args.out)
    bad = [(n, c) for n, g in result["grids"].items() for c, v in g.items() if v.get("status") == "failed"]
    if bad:
        print("calls that failed:", bad, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
