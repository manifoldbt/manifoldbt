"""manifoldbt adapter: the strategies of spec.py as `Strategy.quote` programs.

Everything below uses the public API of the PyPI wheel (`import manifoldbt as
mbt`): `ingest_trades`, `ingest_book`, `Strategy.quote`, `mbt.run`,
`mbt.run_sweep`, `mbt.run_sweep_lite`.

Conventions: a run starts at the first book state of the day (`time_range_start`),
the wake-up grid is that instant plus k x 100 ms, and the journal is read back from
`orders_df` / `fills_df` in the same shape the other adapters produce.
"""
from __future__ import annotations

import os

import spec


def _mbt():
    import manifoldbt as mbt
    return mbt


def open_store(root, day, tier, cache_dir=None):
    """Ingest one day of public Bybit spot data into a store under `root` (once).
    Returns the DataStore. `tier` picks the number of book levels stored."""
    mbt = _mbt()
    data_root = os.path.join(root, f"store_{tier}")
    meta = os.path.join(data_root, "metadata.sqlite")
    arrow = os.path.join(data_root, "mega")
    tape = os.path.join(arrow, "bybit", "ticks", spec.SYMBOL, f"{day}.arrow")
    if os.path.exists(tape):
        store = mbt.DataStore(data_root=data_root, metadata_db=meta, arrow_dir=arrow)
    else:
        store = mbt.ingest_trades("bybit", spec.SYMBOL, symbol_id=spec.SYMBOL_ID, start=day, end=day,
                                  data_root=data_root, metadata_db=meta)
    book_done = any(os.path.exists(os.path.join(arrow, "bybit", shape, spec.SYMBOL, f"{day}.arrow"))
                    for shape in ("book_delta", "book"))
    if not book_done:
        store = mbt.ingest_book("bybit", spec.SYMBOL, symbol_id=spec.SYMBOL_ID, start=day, end=day,
                                levels=spec.TIERS[tier], category="spot",
                                data_root=data_root, metadata_db=meta, cache_dir=_seed_cache(root, day))
    return store


def _seed_cache(root, day):
    """The connector keeps raw archives at `cache_dir/<category>/<day>_<SYMBOL>_ob200.data.zip`.
    The archive was already downloaded (and fingerprinted) by data.py: link it there so
    the day is not downloaded twice."""
    import shutil

    import data

    src = os.path.join(data.DATA_ROOT, "raw", f"bybit-spot-ob200-{day}", f"{day}_{spec.SYMBOL}_ob200.data.zip")
    cache = os.path.join(root, "mbt-cache")
    dst_dir = os.path.join(cache, "spot")
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(dst_dir, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return cache


def maker_strategy(p=None, params=False):
    """The two-quote maker. With `params=True` the four knobs are `mbt.param`s
    (sweepable); otherwise they are literals."""
    mbt = _mbt()
    from manifoldbt import book

    p = p or {}
    def knob(name, default):
        v = p.get(name, default)
        return mbt.param(name, default=v) if params else v

    clip = knob("clip", spec.MMLadder.CLIP)
    max_clips = knob("max_clips", spec.MMLadder.MAX_CLIPS)
    off_bid = knob("off_bid", spec.MMLadder.OFF_BID)
    off_ask = knob("off_ask", spec.MMLadder.OFF_ASK)
    ladder = p.get("ladder", True)

    clips = mbt.round(mbt.position() / clip)
    bids, asks = book.bid_levels(), book.ask_levels()
    book_ok = (bids > 0) & (asks > 0)
    if ladder:
        bid = book.bid_price_at(mbt.clip(clips, 0, bids - 1) + 1 + off_bid)
        ask = book.ask_price_at(mbt.clip(-clips, 0, asks - 1) + 1 + off_ask)
    else:
        bid, ask = book.bid_price_at(1), book.ask_price_at(1)
    return (
        mbt.Strategy.create("maker")
        .quote("buy", mbt.when(book_ok, bid), clip, tif="GTX", enabled=clips < max_clips)
        .quote("sell", mbt.when(book_ok, ask), clip, tif="GTX", enabled=clips > -max_clips)
    )


def maker_config(day, window_ns=None, latency="nominal"):
    mbt = _mbt()
    from manifoldbt.helpers import Interval

    d0, d1 = spec.day_bounds_ns(day)
    w0, w1 = window_ns or (d0, d1)
    ms = Interval.millis
    if latency == "nominal":
        lat = {"order": ms(spec.LAT_ORDER_NS // spec.MS), "cancel": ms(spec.LAT_CANCEL_NS // spec.MS),
               "response": ms(spec.LAT_RESPONSE_NS // spec.MS), "feed": ms(spec.LAT_FEED_NS // spec.MS)}
    else:
        lat = {"order": ms(0), "cancel": ms(0), "response": ms(0), "feed": ms(0)}
    return mbt.BacktestConfig(
        universe=[spec.SYMBOL_ID], time_range_start=w0, time_range_end=w1,
        bar_interval=ms(spec.WAKE_NS // spec.MS), initial_capital=100_000.0,
        fees=mbt.FeeConfig(maker_fee_bps=spec.MAKER_FEE_BPS, taker_fee_bps=spec.TAKER_FEE_BPS),
        execution=mbt.ExecutionConfig(
            latency=lat,
            # the queue behind the size the stored book shows; "risk_adverse" is the
            # model a book stored by price gets when none is named
            fill_model={"queue": {"depth_source": "book", "model": "risk_adverse", "assumed_queue": 2.0}},
        ),
    )


def flow_taker_strategy(params=False, thr=None, qty=None):
    """Example 33 on the trade clock, in fixed units: follow a one-sided flow.
    With `params=True` the threshold and the order size are `mbt.param`s (a window
    in time cannot be swept: a duration is a literal)."""
    mbt = _mbt()
    from manifoldbt.expr import col, lit, when
    from manifoldbt.helpers import Interval

    win = Interval.seconds(spec.FlowTaker.WINDOW_NS // spec.SEC)
    bought = col("buy_volume").rolling_sum(win)
    sold = col("sell_volume").rolling_sum(win)
    imbalance = (bought - sold) / (bought + sold + lit(1e-12))
    if params:
        thr = mbt.param("thr", default=spec.FlowTaker.THRESHOLD)
        qty = mbt.param("qty", default=spec.FlowTaker.QTY)
        crossed = (imbalance >= thr) | (imbalance <= -thr)
        side = crossed.value_when(when(imbalance >= thr, lit(1.0), lit(-1.0)))
        size = col("side") * qty
    else:
        thr = spec.FlowTaker.THRESHOLD if thr is None else thr
        qty = spec.FlowTaker.QTY if qty is None else qty
        crossed = (imbalance >= lit(thr)) | (imbalance <= lit(-thr))
        side = crossed.value_when(when(imbalance >= lit(thr), lit(1.0), lit(-1.0)))
        size = col("side") * lit(qty)
    return (mbt.Strategy.create("flow_taker")
            .signal("imbalance", imbalance)
            .signal("side", side)
            .size(size))


def flow_taker_config(day, latency="nominal", window_ns=None):
    mbt = _mbt()
    from manifoldbt.helpers import Interval, Slippage

    d0, d1 = window_ns or spec.day_bounds_ns(day)
    lat = Interval.millis(20) if latency == "nominal" else Interval.millis(0)
    return mbt.BacktestConfig(
        universe=[spec.SYMBOL_ID], time_range_start=d0, time_range_end=d1,
        bar_interval=Interval.trades(), initial_capital=100_000.0,
        fees=mbt.FeeConfig(maker_fee_bps=0.0, taker_fee_bps=spec.TAKER_FEE_BPS, default_fill_type="Taker"),
        slippage=Slippage.none(),
        execution=mbt.ExecutionConfig(
            signal_delay=1, execution_price="AtOpen", allow_short=True,
            position_sizing_mode="Units", max_position_pct=1.0,
            latency={"order": lat, "cancel": lat}, fill_marks=False,
        ),
    )


def _ns(series):
    """A tz-aware timestamp column as int64 nanoseconds, whatever resolution pandas gave it."""
    return series.dt.as_unit("ns").astype("int64").to_numpy()


def journal_of(res, anchor_ns, period_ns, price_scale=None):
    """Quote-level journal of a QuoteResult / SimResult, in the form the other
    adapters produce: k (wake index), side (+1 buy / -1 sell), px (ticks), the
    quantity filled and the time of its first fill. Column names are those of
    `SimResult.orders_df` / `fills_df` (documented in the wheel)."""
    import numpy as np

    orders = res.orders_df(backend="pandas")
    fills = res.fills_df(backend="pandas")
    sent = _ns(orders["sent_at"])
    k = np.rint((sent - anchor_ns) / period_ns).astype(np.int64)
    side = np.where(orders["side"].to_numpy() == "buy", 1, -1).astype(np.int8)
    px = np.round(orders["price"].astype(float).to_numpy() * (price_scale or spec.TICK_INV)).astype(np.int64)
    filled = orders["filled"].astype(float).to_numpy()
    fill_ts = np.zeros(len(orders), dtype=np.int64)
    if len(fills):
        f = fills.assign(ts=_ns(fills["timestamp"]))
        first = f.groupby("order_id")["ts"].min()
        # Integers all the way. `Series.map(first)` puts NaN in the rows of the orders
        # that were never filled, which turns the column into float64, and a float64
        # holds a nanosecond epoch only to the nearest 256 ns: the instant of a first
        # fill would no longer be the instant the engine reported.
        pos = first.index.get_indexer(orders["order_id"])
        fill_ts = np.where(pos >= 0, first.to_numpy(dtype=np.int64)[np.maximum(pos, 0)], 0).astype(np.int64)
    return {"k": k, "side": side, "px": px, "ts": sent, "filled": filled, "fill_ts": fill_ts}


def journal_taker(res):
    """The flow taker's decisions, orders and fills, from the trade log (`trades_df`):
    a decision is the print the signal changed on (`signal_timestamp`), an order is the
    trade that follows it, a fill is its execution."""
    import numpy as np

    tr = res.trades_df()
    sig = _ns(tr["signal_timestamp"])
    ex = _ns(tr["execution_timestamp"])
    side = np.where(tr["side"].to_numpy() == 1, 1, -1).astype(np.int8)
    qty = tr["quantity"].astype(float).to_numpy()
    px = tr["fill_price"].astype(float).to_numpy()
    held = np.cumsum(side * qty)
    return {"dec_ts": sig, "dec_side": np.sign(np.round(held, 9)).astype(np.int8), "dec_imb": np.full(len(sig), np.nan),
            "ord_ts": sig, "ord_side": side, "ord_qty": qty, "fill_px": px, "fill_qty": qty, "fill_ts": ex}


def summarize(res, symbol=None):
    """The same fields the other adapters report, read from a QuoteResult."""
    symbol = symbol or spec.SYMBOL
    m = res.sim.metrics
    act = res.order_activity
    return {
        "posts": int(act["orders_posted"]), "requotes": int(act["requotes"]),
        "fills": int(m["fills"]), "volume": float(m["volume"]), "position": float(res.sim.position[symbol]),
        "fees": float(m["fees"]), "orders_served": int(m["orders_filled"]),
    }
