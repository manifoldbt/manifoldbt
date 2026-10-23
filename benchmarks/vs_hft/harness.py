"""One (engine, workload) execution, the same call for every engine.

`prepare(engine, wl, ctx)` builds what the engine reads from the neutral data
(its event array, its store...) and is timed apart. `run(engine, wl, ctx, inputs,
mode)` is the backtest itself, `mode` being "strategy" or "replay" (same data,
no orders).
"""
from __future__ import annotations

import os
import time

import numpy as np

import data
import itch
import spec
from workloads import WORKLOADS

ENGINES = ("manifoldbt", "hftbacktest", "nautilus")

MBO_WORKLOAD = {
    "mbo_join_mbo": dict(kind="mbo", tier="mbo", params={}, latency="zero",
                         doc="join the best bid and ask with a round lot, stand aside behind a crowded best (Nasdaq ITCH, AAPL)"),
}


def all_workloads():
    d = dict(WORKLOADS)
    d.update(MBO_WORKLOAD)
    return d


class Ctx:
    """Where the data is and which slice of it a run covers."""

    def __init__(self, day, window_s=None, itch_extract=None, root=None):
        self.day = day
        self.window_s = window_s
        self.itch_extract = itch_extract
        self.root = root or data.DATA_ROOT
        self._d = None
        self._m = None

    @property
    def d(self):
        if self._d is None:
            path, _ = data.prepare(self.day, self.window_s)
            self._d = data.load(path)
        return self._d

    @property
    def m(self):
        if self._m is None:
            self._m = itch.events_from_stream(self.itch_extract, spec.MBOJoin.DAY)
        return self._m

    def window(self):
        w0, w1 = (int(x) for x in self.d["window"])
        return w0, w1

    def mbo_bounds(self):
        """(open, close). The open is the first event of the symbol at or after 09:30:00 New York: the
        wake-up clock of every engine starts there (manifoldbt's starts at the first event in range)."""
        sp = spec.MBOJoin
        base = itch.midnight_ny_utc_ns(sp.DAY)
        t_open = base + sp.OPEN_S * spec.SEC
        ts = self.m["ts"]
        t_open = int(ts[int(np.searchsorted(ts, t_open, side="left"))])
        return t_open, base + sp.CLOSE_S * spec.SEC


def _lat(wl):
    """(order_ns, response_ns, feed_ns) of a workload for the engines that take them in ns."""
    if wl["latency"] == "zero":
        return 0, 0, 0
    if wl["kind"] == "taker":
        return 20 * spec.MS, 0, 0
    return spec.LAT_ORDER_NS, spec.LAT_RESPONSE_NS, spec.LAT_FEED_NS


# --------------------------------------------------------------------------- #
def prepare(engine, wlname, ctx):
    wl = all_workloads()[wlname]
    t0 = time.perf_counter()
    inputs = {}
    if engine == "hftbacktest":
        import engine_hft as H
        if wl["kind"] == "mbo":
            inputs["events"] = H.build_events_l3(ctx.m)
        else:
            inputs["events"] = H.build_events(ctx.d, wl["tier"], _lat(wl)[2])
    elif engine == "nautilus":
        if wl["kind"] == "mbo":
            ctx.m      # decoded once; the Nautilus objects are built inside the run
        else:
            ctx.d
    elif engine == "manifoldbt":
        import engine_mbt as M
        import engine_mbt_more as M2
        if wl["kind"] == "mbo":
            inputs["store"] = M2.mbo_store(ctx.root, os.path.join(ctx.itch_extract), spec.MBOJoin.DAY, spec.MBOJoin.SYMBOL)
        else:
            inputs["store"] = M.open_store(ctx.root, ctx.day, wl["tier"],
                                           cache_dir=os.path.join(ctx.root, "raw"))
    inputs["prepare_s"] = time.perf_counter() - t0
    return inputs


def run(engine, wlname, ctx, inputs, mode="strategy"):
    """Returns (summary, journal, timing) for one execution."""
    wl = all_workloads()[wlname]
    kind = wl["kind"]
    t0 = time.perf_counter()
    journal = None
    extra = {}
    if engine == "hftbacktest":
        import engine_hft as H
        if kind == "mm":
            if mode == "replay":
                summary = H.run_replay(ctx.d, inputs["events"])
            else:
                o, r, _ = _lat(wl)
                summary, journal = H.run_mm(ctx.d, wl["tier"], inputs["events"], wl["params"], ctx.window(),
                                            partial=os.environ.get("BENCH_HFT_PARTIAL") == "1",
                                            order_ns=o, resp_ns=r)
        elif kind == "taker":
            o, r, _ = _lat(wl)
            summary, journal = H.run_taker(ctx.d, inputs["events"], wl["params"], None, o, r)
        else:
            t_open, t_close = ctx.mbo_bounds()
            if mode == "replay":
                summary = H.run_replay_l3(inputs["events"])
            else:
                summary, journal = H.run_mbo(inputs["events"], t_open, t_close)
    elif engine == "nautilus":
        import engine_nt as N
        if kind == "mm":
            if mode == "replay":
                summary = N.run_replay(ctx.d, wl["tier"])
            else:
                summary, journal = N.run_mm(ctx.d, wl["tier"], wl["params"], ctx.window(), wl["latency"])
        elif kind == "taker":
            summary, journal = N.run_taker(ctx.d, wl["params"], None, wl["latency"])
        else:
            import engine_nt_l3 as L3
            t_open, t_close = ctx.mbo_bounds()
            summary, journal = L3.run_mbo(ctx.m, t_open, t_close, strategy=(mode != "replay"))
        extra = {k: summary.get(k) for k in ("engine_seconds", "convert_seconds") if k in summary}
    else:
        import engine_mbt as M
        import engine_mbt_more as M2
        import manifoldbt as mbt
        store = inputs["store"]
        if kind == "mm":
            w0, w1 = ctx.window()
            if wl["params"].get("probe"):
                res = M2.probe_run(store, (w0, w1), wl["latency"])
                summary = {"posts": int(res.metrics["orders"]), "fills": int(res.metrics["fills"]),
                           "volume": float(res.metrics["volume"]), "position": float(res.position[spec.SYMBOL])}
                journal = M.journal_of(res, int(ctx.d["anchor"][0]), spec.WAKE_NS)
            else:
                strat = M.maker_strategy(wl["params"], params=False)
                if mode == "replay":
                    strat = _mbt_idle_strategy(M)
                cfg = M.maker_config(ctx.day, (w0, w1), wl["latency"])
                res = mbt.run(strat, cfg, store)
                summary = M.summarize(res)
                if mode != "replay":
                    journal = M.journal_of(res, int(ctx.d["anchor"][0]), spec.WAKE_NS)
        elif kind == "taker":
            res = mbt.run(M.flow_taker_strategy(), M.flow_taker_config(ctx.day, wl["latency"], ctx.window()), store)
            tr = res.trades_df()
            summary = {"fills": int(len(tr)), "orders": int(len(tr)),
                       "position": float(tr["quantity"].where(tr["side"] == 1, -tr["quantity"]).sum()) if len(tr) else 0.0}
            journal = M.journal_taker(res)
        else:
            t_open, t_close = ctx.mbo_bounds()
            strat = M2.mbo_strategy() if mode != "replay" else _mbt_idle_strategy(M)
            res = mbt.run(strat, M2.mbo_config(t_open, t_close), store)
            summary = M.summarize(res, spec.MBOJoin.SYMBOL)
            if mode != "replay":
                journal = M.journal_of(res, t_open, spec.MBOJoin.WAKE_NS, price_scale=100)
    timing = {"seconds": time.perf_counter() - t0}
    timing.update(extra)
    return summary, journal, timing


def _mbt_idle_strategy(M):
    """A strategy that never quotes: the replay cost of the venue alone."""
    mbt = M._mbt()
    from manifoldbt import book
    return (mbt.Strategy.create("idle")
            .quote("buy", book.bid_price_at(1), 1.0, tif="GTX", enabled=mbt.position() > 1e18)
            .quote("sell", book.ask_price_at(1), 1.0, tif="GTX", enabled=mbt.position() > 1e18))
