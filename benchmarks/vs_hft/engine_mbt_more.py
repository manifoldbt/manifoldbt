"""manifoldbt adapter, second half: the open-loop probe (order API) and the
order-by-order (MBO) workload."""
from __future__ import annotations

import datetime as dt
import os

import spec


def _mbt():
    import manifoldbt as mbt
    return mbt


def probe_run(store, window_ns, latency="zero"):
    """The open-loop probe through the order API (`mbt.sim.run`): at every wake-up
    cancel what is live and post one post-only quote a side at the touch. The DSL
    keeps an order that is already at the wanted price, so it cannot re-post the
    same price at every wake-up; the order API can."""
    mbt = _mbt()
    from manifoldbt.helpers import Interval

    ms = Interval.millis
    wake = ms(spec.WAKE_NS // spec.MS)
    live = ("in_flight", "resting", "partially_filled")
    clip = spec.MMLadder.CLIP

    def fn(sim):
        ids = [None, None]
        while sim.elapse(wake):
            bids, asks = sim.book(spec.SYMBOL).levels()
            if not bids or not asks:
                continue
            for i, price in enumerate((bids[0][0], asks[0][0])):
                if ids[i] is not None:
                    o = sim.order(ids[i])
                    if o is not None and o.status in live and not o.cancel_pending:
                        sim.cancel(o.id)
                ids[i] = sim.post(spec.SYMBOL, "buy" if i == 0 else "sell", price, clip, tif="GTX")

    if latency == "nominal":
        lat = {"entry": ms(5), "cancel": ms(5), "response": ms(5), "feed": ms(2)}
    else:
        lat = {"entry": ms(0), "cancel": ms(0), "response": ms(0), "feed": ms(0)}
    cfg = mbt.sim.Config(latency=lat, queue={"depth_source": "book", "model": "risk_adverse", "assumed_queue": 2.0},
                         maker_fee_bps=spec.MAKER_FEE_BPS, taker_fee_bps=spec.TAKER_FEE_BPS)

    # nanoseconds since the epoch: the API takes a date, a datetime or an int, and an RFC 3339 string is refused
    return mbt.sim.run(fn, cfg, store, symbols=[spec.SYMBOL], start=int(window_ns[0]), end=int(window_ns[1]))


def mbo_strategy():
    """The MBO maker (spec.MBOJoin) as quotes: join the best, stand aside when the
    displayed size at the best is above CROWDED or the inventory bound is hit."""
    mbt = _mbt()
    from manifoldbt import book

    sp = spec.MBOJoin
    inv = mbt.position()
    ok = (book.bid_levels() > 0) & (book.ask_levels() > 0)
    return (mbt.Strategy.create("mbo_join")
            .quote("buy", mbt.when(ok, book.bid_price_at(1)), sp.LOT, tif="GTX",
                   enabled=(inv < sp.LOT * sp.MAX_LOTS - 1e-9) & (book.bid_size(1) <= sp.CROWDED))
            .quote("sell", mbt.when(ok, book.ask_price_at(1)), sp.LOT, tif="GTX",
                   enabled=(inv > -sp.LOT * sp.MAX_LOTS + 1e-9) & (book.ask_size(1) <= sp.CROWDED)))


def mbo_config(t_open_ns, t_close_ns):
    mbt = _mbt()
    from manifoldbt.helpers import Interval

    sp = spec.MBOJoin
    z = Interval.millis(0)
    return mbt.BacktestConfig(
        universe=[spec.SYMBOL_ID], time_range_start=int(t_open_ns), time_range_end=int(t_close_ns),
        bar_interval=Interval.millis(sp.WAKE_NS // spec.MS), initial_capital=1_000_000.0,
        execution=mbt.ExecutionConfig(latency={"order": z, "cancel": z, "response": z, "feed": z}),
    )


def mbo_store(root, itch_extract, day, symbol):
    """Ingest the extract of one symbol (a stream of ITCH messages) into a store."""
    mbt = _mbt()
    data_root = os.path.join(root, "store_mbo")
    meta = os.path.join(data_root, "metadata.sqlite")
    arrow = os.path.join(data_root, "mega")
    day_file = os.path.join(arrow, "nasdaq", "mbo", symbol, f"{day}.arrow")
    if os.path.exists(day_file):
        return mbt.DataStore(data_root=data_root, metadata_db=meta, arrow_dir=arrow)
    return mbt.ingest_mbo(itch_extract, {symbol: spec.SYMBOL_ID}, day, data_root=data_root, metadata_db=meta,
                          progress=False)
