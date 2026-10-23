"""Render a result file as Markdown (and into the GitHub job summary when there is one).

    python report.py results.json
"""
import json
import os
import sys


def ms(x):
    return "-" if x is None else f"{x * 1000:,.0f} ms" if x < 10 else f"{x:,.1f} s"


def med(s):
    return None if not s else s["median"]


def fmt_stats(s):
    if not s:
        return "-"
    v = ms(s["median"])
    return v + ("~" if s.get("noisy") else "")


def ratio(a, b):
    if a is None or b is None or a == 0:
        return "-"
    r = b / a
    return f"x{r:,.1f}" if r >= 10 else f"x{r:.2f}"


def render_books(res, out):
    env = res.get("environment", {})
    out.append(f"Python {env.get('python')} on {env.get('platform')}, {env.get('cpu_count_logical')} logical cpus, {env.get('ram_gb')} GB.")
    out.append("Versions: " + ", ".join(f"{k} {v}" for k, v in (env.get("versions") or {}).items() if v))
    d = res.get("data") or {}
    if d.get("meta"):
        m = d["meta"]
        out.append("")
        out.append(f"Data: Bybit spot BTCUSDT, {d['day']}. " + ", ".join(
            f"{k}: {v['bytes'] / 1e6:.1f} MB (sha256 {v['sha256'][:12]}...)" for k, v in m["raw"].items()))
    for name, w in res["workloads"].items():
        out.append("")
        out.append(f"### {name}")
        out.append(w["doc"] + f" (data: {w['tier']}, latency: {w['latency']})")
        out.append("")
        out.append("| engine | strategy run | replay only | engine / conversion | cold start | memory added | status |")
        out.append("|---|---:|---:|---:|---:|---:|---|")
        ref = w["engines"].get("manifoldbt", {})
        ref_s = med(ref.get("strategy_s")) if ref.get("status") == "ok" else None
        for e, r in w["engines"].items():
            if r.get("status") != "ok":
                out.append(f"| {e} | - | - | - | - | - | {r.get('status')}: {r.get('error', '')[:80]} |")
                continue
            ec = "-"
            if r.get("engine_only_s"):
                ec = f"{ms(med(r['engine_only_s']))} / {ms(med(r['convert_s']))}"
            out.append(f"| {e} | {fmt_stats(r['strategy_s'])} | {fmt_stats(r.get('replay_s'))} | {ec} | "
                       f"{ms(med(r['cold_s']))} | {r['memory_mb']['added_by_engine']:,.0f} MB | ok |")
        par = w.get("parity", {})
        if par.get("pairs"):
            out.append("")
            out.append(f"Parity, reference used: {par.get('reference_used')}"
                       + ("" if par.get("reference") else " (the reference engine did not run: cross-check between challengers)"))
            for pair, c in par["pairs"].items():
                out.append(f"- {pair}: `{json.dumps(c)[:600]}`")


def render_sweeps(res, out):
    out.append(f"Sweeps, {res.get('day')}, window {res.get('window_s')} s, {res.get('workers')} workers.")
    out.append("")
    out.append("| grid | engine | combinations/s | wall | notes |")
    out.append("|---:|---|---:|---:|---|")
    for n, g in res.get("grids", {}).items():
        for e, r in g["engines"].items():
            if "status" in r and r["status"] != "ok":
                out.append(f"| {n} | {e} | - | - | {r['status']} |")
                continue
            cps = r.get("combos_per_s_pool") or r.get("combos_per_s")
            wall = (r.get("pool") or r).get("sweep_s")
            note = ""
            if e == "nautilus":
                note = f"{r['combos_run']} of {g['combos']} run; full grid projected {r['projected_full_grid_pool_s']:.0f} s"
            out.append(f"| {n} | {e} | {cps:,.2f} | {wall:,.1f} s | {note} |")


def main():
    res = json.load(open(sys.argv[1]))
    out = [f"## {os.path.basename(sys.argv[1])}"]
    if "workloads" in res:
        render_books(res, out)
    if "grids" in res:
        render_sweeps(res, out)
    text = "\n".join(out)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
