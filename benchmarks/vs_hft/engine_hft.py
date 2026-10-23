"""hftbacktest 2.4.4 adapter: the strategies of spec.py written as numba kernels.

The kernels are what a user of the library would write for the same rule:
`ROIVectorMarketDepthBacktest` (the fastest depth structure it offers, with the
price range of the day as the region of interest), a risk-adverse queue model,
constant order latency, and the strategy loop inside one `@njit` function.
"""
from __future__ import annotations

import numpy as np
from numba import njit

import spec

# order constants of hftbacktest.order (plain ints so numba treats them as literals)
NONE_, NEW_, EXPIRED_, FILLED_, CANCELED_, PARTIAL_, REJECTED_ = 0, 1, 2, 3, 4, 5, 6
GTC_, GTX_, FOK_, IOC_ = 0, 1, 2, 3
LIMIT_, MARKET_ = 0, 1
I64_MIN = -(1 << 62)
I64_MAX = 1 << 62
MAXLV = 64


def build_events(d, tier, feed_ns):
    """The neutral streams of one tier as hftbacktest's event array."""
    from hftbacktest.data import correct_event_order, validate_event_order
    from hftbacktest.data.validation import correct_local_timestamp
    from hftbacktest.types import BUY_EVENT, DEPTH_EVENT, SELL_EVENT, TRADE_EVENT, event_dtype

    key = "d25" if tier == "l2" else "d1"
    n_d, n_t = len(d[key + "_ts"]), len(d["tr_ts"])
    ev = np.zeros(n_d + n_t, event_dtype)
    side = d[key + "_side"]
    ev["ev"][:n_d] = np.where(side == 0, DEPTH_EVENT | BUY_EVENT, DEPTH_EVENT | SELL_EVENT).astype(np.uint64)
    ev["exch_ts"][:n_d] = d[key + "_ts"]
    ev["px"][:n_d] = d[key + "_px"] / float(spec.TICK_INV)
    ev["qty"][:n_d] = d[key + "_qty"]
    ev["ev"][n_d:] = np.where(d["tr_side"] == 1, TRADE_EVENT | SELL_EVENT, TRADE_EVENT | BUY_EVENT).astype(np.uint64)
    ev["exch_ts"][n_d:] = d["tr_ts"]
    ev["px"][n_d:] = d["tr_px"]
    ev["qty"][n_d:] = d["tr_qty"]
    ev["local_ts"] = ev["exch_ts"] + int(feed_ns)
    ev = correct_local_timestamp(ev, 0)
    out = correct_event_order(ev, np.argsort(ev["exch_ts"], kind="mergesort"),
                              np.argsort(ev["local_ts"], kind="mergesort"))
    validate_event_order(out)
    return out


def make_backtest(data, d, partial, order_ns=None, resp_ns=None):
    from hftbacktest import BacktestAsset, ROIVectorMarketDepthBacktest

    order_ns = spec.LAT_ORDER_NS if order_ns is None else order_ns
    resp_ns = spec.LAT_RESPONSE_NS if resp_ns is None else resp_ns
    a = (BacktestAsset().data([data]).linear_asset(1.0)
         .constant_order_latency(order_ns, resp_ns)
         .risk_adverse_queue_model())
    a = a.partial_fill_exchange() if partial else a.no_partial_fill_exchange()
    a = (a.trading_value_fee_model(spec.MAKER_FEE_BPS / 1e4, spec.TAKER_FEE_BPS / 1e4)
         .tick_size(spec.TICK).lot_size(spec.LOT))
    px = d["tr_px"]
    a = a.roi_lb(float(np.floor(px.min() * 0.97))).roi_ub(float(np.ceil(px.max() * 1.03)))
    return ROIVectorMarketDepthBacktest([a])


# --------------------------------------------------------------------------- #
# Strategy kernels
# --------------------------------------------------------------------------- #
@njit
def _levels_bid(depth, best, out):
    """Fill `out` with the ticks of the non-empty bid levels, best first."""
    n = 0
    x = best
    lo = best - 200_000
    while x > lo and n < out.shape[0]:
        if depth.bid_qty_at_tick(x) > 0.0:
            out[n] = x
            n += 1
        x -= 1
    return n


@njit
def _levels_ask(depth, best, out):
    n = 0
    x = best
    hi = best + 200_000
    while x < hi and n < out.shape[0]:
        if depth.ask_qty_at_tick(x) > 0.0:
            out[n] = x
            n += 1
        x += 1
    return n


@njit
def _record(orders, rec_fill, rec_fill_ts, rec_status, final):
    """Close the books on every order that is no longer live (or on all of them
    when `final`): cumulative filled quantity, last exchange timestamp, status."""
    values = orders.values()
    while values.has_next():
        o = values.get()
        st = o.status
        inactive = st == FILLED_ or st == CANCELED_ or st == EXPIRED_ or st == REJECTED_
        if inactive or final:
            i = o.order_id
            if i < rec_fill.shape[0]:
                filled = o.qty - o.leaves_qty
                if st == FILLED_:
                    filled = o.qty
                rec_fill[i] = filled
                rec_fill_ts[i] = o.exch_timestamp
                rec_status[i] = st


@njit
def mm_kernel(hbt, t_anchor, n_wakes, period, clip, max_clips, off_bid, off_ask, ladder, maxlv, probe,
              post_k, post_side, post_px, post_ts, rec_fill, rec_fill_ts, rec_status, counts, pos_log):
    """The two-quote maker of spec.MMLadder (`ladder` False = quote the touch).

    `probe` turns it into the open-loop probe: inventory is ignored, both quotes
    sit at the touch and are cancelled and posted again at EVERY wake-up, so
    each wake-up yields one quote per side whose fate depends on the market only.

    counts = [wakes, posts, cancels, requotes, submit_errors, last_code, last_wake]
    Returns the code of the last `elapse` (0 ok, 1 end of data, other = blocked).
    """
    lv_b = np.empty(maxlv, np.int64)
    lv_a = np.empty(maxlv, np.int64)
    ids = np.full(2, -1, np.int64)
    oid = 1
    hbt.elapse(1)
    now = hbt.current_timestamp
    k = 1
    t = t_anchor + k * period
    while t <= now:
        k += 1
        t = t_anchor + k * period
    r = hbt.elapse(t - now)
    if r != 0:
        counts[5] = r
        return r
    while True:
        counts[0] += 1
        counts[6] = k
        orders = hbt.orders(0)
        _record(orders, rec_fill, rec_fill_ts, rec_status, False)
        hbt.clear_inactive_orders(0)
        depth = hbt.depth(0)
        bb = depth.best_bid_tick
        ba = depth.best_ask_tick
        tick = depth.tick_size
        pos = hbt.position(0)
        if k < pos_log.shape[0]:
            pos_log[k] = pos
        if bb > I64_MIN and ba < I64_MAX:
            nb = _levels_bid(depth, bb, lv_b)
            na = _levels_ask(depth, ba, lv_a)
            if nb > 0 and na > 0:
                clips = np.round(pos / clip)       # half to even, as Python's round
                if probe:
                    clips = 0.0
                if ladder:
                    ib = int(min(max(clips, 0.0), nb - 1.0)) + off_bid
                    ia = int(min(max(-clips, 0.0), na - 1.0)) + off_ask
                else:
                    ib = 0
                    ia = 0
                on_b = clips < max_clips
                on_a = clips > -max_clips
                for s in range(2):
                    on = on_b if s == 0 else on_a
                    idx = ib if s == 0 else ia
                    n_lv = nb if s == 0 else na
                    oid_s = ids[s]
                    live = False
                    pending = False
                    held = -1
                    if oid_s >= 0:
                        o = orders.get(oid_s)
                        if o is not None:
                            st = o.status
                            rq = o.req
                            live = st == NEW_ or st == PARTIAL_ or (st == NONE_ and rq == NEW_)
                            pending = rq == CANCELED_
                            held = o.price_tick
                    if not on:
                        if live and not pending:
                            hbt.cancel(0, oid_s, False)
                            counts[2] += 1
                        ids[s] = -1
                        continue
                    if idx >= n_lv:
                        continue                    # level not shown: the quote holds
                    tgt = lv_b[idx] if s == 0 else lv_a[idx]
                    if live and held == tgt and not probe:
                        continue                    # keeps its place in the queue
                    if live:
                        if not pending:
                            hbt.cancel(0, oid_s, False)
                            counts[2] += 1
                        counts[3] += 1
                    if s == 0:
                        e = hbt.submit_buy_order(0, oid, tgt * tick, clip, GTX_, LIMIT_, False)
                    else:
                        e = hbt.submit_sell_order(0, oid, tgt * tick, clip, GTX_, LIMIT_, False)
                    if e != 0:
                        counts[4] += 1
                    n = counts[1]
                    if n < post_k.shape[0]:
                        post_k[n] = k
                        post_side[n] = 1 if s == 0 else -1
                        post_px[n] = tgt
                        post_ts[n] = hbt.current_timestamp
                    ids[s] = oid
                    oid += 1
                    counts[1] += 1
        if k >= n_wakes:
            break
        k += 1
        t = t_anchor + k * period
        r = hbt.elapse(t - hbt.current_timestamp)
        if r != 0:
            counts[5] = r
            break
    _record(hbt.orders(0), rec_fill, rec_fill_ts, rec_status, True)
    return counts[5]


def run_mm(d, tier, events, p, window, partial=False, order_ns=None, resp_ns=None):
    """One run of the maker. `p` holds clip, max_clips, off_bid, off_ask, ladder.
    Returns (result dict, journal dict)."""
    w0, w1 = window
    key = "d25" if tier == "l2" else "d1"
    anchor = int(d["anchor"][0]) if "anchor" in d else int(d[key + "_ts"][0])
    period = spec.WAKE_NS
    n_wakes = int(-(-(w1 - anchor) // period)) - 1
    cap = 2 * n_wakes + 16
    post_k = np.zeros(cap, np.int64)
    post_side = np.zeros(cap, np.int8)
    post_px = np.zeros(cap, np.int64)
    post_ts = np.zeros(cap, np.int64)
    rec_fill = np.zeros(cap + 2, np.float64)
    rec_fill_ts = np.zeros(cap + 2, np.int64)
    rec_status = np.zeros(cap + 2, np.int8)
    counts = np.zeros(8, np.int64)
    pos_log = np.full(n_wakes + 2, np.nan)
    hbt = make_backtest(events, d, partial, order_ns, resp_ns)
    code = mm_kernel(hbt, anchor, n_wakes, period, float(p["clip"]), float(p["max_clips"]),
                     int(p["off_bid"]), int(p["off_ask"]), bool(p["ladder"]), spec.TIERS[tier],
                     bool(p.get("probe", False)),
                     post_k, post_side, post_px, post_ts, rec_fill, rec_fill_ts, rec_status, counts, pos_log)
    sv = hbt.state_values(0)
    out = {
        "code_end": int(code), "wakes": int(counts[0]), "last_wake": int(counts[6]), "n_wakes_planned": n_wakes,
        "posts": int(counts[1]), "cancels": int(counts[2]), "requotes": int(counts[3]),
        "submit_errors": int(counts[4]),
        "fills": int(sv.num_trades), "volume": float(sv.trading_volume), "position": float(hbt.position(0)),
        "balance": float(sv.balance), "fee": float(sv.fee),
    }
    hbt.close()
    n = out["posts"]
    # book ids are 1..posts, in posting order
    journal = {
        "k": post_k[:n].copy(), "side": post_side[:n].copy(), "px": post_px[:n].copy(), "ts": post_ts[:n].copy(),
        "filled": rec_fill[1:n + 1].copy(), "fill_ts": rec_fill_ts[1:n + 1].copy(), "status": rec_status[1:n + 1].copy(),
        "pos_log": pos_log,
    }
    return out, journal


# --------------------------------------------------------------------------- #
# Flow taker (spec.FlowTaker), on the trade clock
# --------------------------------------------------------------------------- #
@njit
def taker_kernel(hbt, window_ns, thr, qty, ring_ts, ring_buy, ring_sell,
                 dec_ts, dec_side, dec_imb, ord_ts, ord_side, ord_qty, counts, fill_px, fill_qty, fill_ts):
    """Follow a one-sided flow. counts = [prints, decisions, orders, fills_booked, last_code]."""
    head = 0
    tail = 0
    bought = 0.0
    sold = 0.0
    first_t = -1                  # the window is not full before a whole window of tape has gone by
    side = 0                      # 0 = not armed yet
    sent_pos = 0.0
    oid = 1
    while True:
        r = hbt.wait_next_feed(False, 3_600_000_000_000)
        if r == 1:
            counts[4] = 1
            break
        if r != 2:
            counts[4] = r
            break
        trades = hbt.last_trades(0)
        n = trades.shape[0]
        for j in range(n):
            e = trades[j]
            t = e.exch_ts
            q = e.qty
            is_buy = (e.ev & (1 << 29)) != 0          # BUY_EVENT: the initiator is a buyer
            ring_ts[tail] = t
            ring_buy[tail] = q if is_buy else 0.0
            ring_sell[tail] = 0.0 if is_buy else q
            bought += ring_buy[tail]
            sold += ring_sell[tail]
            tail += 1
            while head < tail and ring_ts[head] <= t - window_ns:
                bought -= ring_buy[head]
                sold -= ring_sell[head]
                head += 1
            counts[0] += 1
            if first_t < 0:
                first_t = t
            imb = (bought - sold) / (bought + sold + 1e-12)
            new_side = side
            if t - first_t >= window_ns:
                if imb >= thr:
                    new_side = 1
                elif imb <= -thr:
                    new_side = -1
            if new_side != side:
                side = new_side
                k = counts[1]
                if k < dec_ts.shape[0]:
                    dec_ts[k] = t
                    dec_side[k] = side
                    dec_imb[k] = imb
                counts[1] += 1
                target = side * qty
                delta = target - sent_pos
                if abs(delta) > 1e-12:
                    if delta > 0:
                        hbt.submit_buy_order(0, oid, 0.0, delta, GTC_, MARKET_, False)
                    else:
                        hbt.submit_sell_order(0, oid, 0.0, -delta, GTC_, MARKET_, False)
                    m = counts[2]
                    if m < ord_ts.shape[0]:
                        ord_ts[m] = hbt.current_timestamp
                        ord_side[m] = 1 if delta > 0 else -1
                        ord_qty[m] = abs(delta)
                    counts[2] += 1
                    oid += 1
                    sent_pos = target
        hbt.clear_last_trades(0)
    # the books: every order is still in the dictionary
    values = hbt.orders(0).values()
    while values.has_next():
        o = values.get()
        if o.status == FILLED_ or o.status == PARTIAL_:
            i = counts[3]
            if i < fill_px.shape[0]:
                fill_px[i] = o.exec_price
                fill_qty[i] = o.exec_qty
                fill_ts[i] = o.exch_timestamp
            counts[3] += 1
    return counts[4]


def run_taker(d, events, p, window, order_ns, resp_ns):
    """One run of the flow taker on hftbacktest. Returns (result, journal)."""
    from hftbacktest import BacktestAsset, ROIVectorMarketDepthBacktest

    n_t = len(d["tr_ts"])
    a = (BacktestAsset().data([events]).linear_asset(1.0)
         .constant_order_latency(order_ns, resp_ns)
         .risk_adverse_queue_model().no_partial_fill_exchange()
         .trading_value_fee_model(spec.MAKER_FEE_BPS / 1e4, spec.TAKER_FEE_BPS / 1e4)
         .tick_size(spec.TICK).lot_size(spec.LOT).last_trades_capacity(4096))
    px = d["tr_px"]
    a = a.roi_lb(float(np.floor(px.min() * 0.97))).roi_ub(float(np.ceil(px.max() * 1.03)))
    hbt = ROIVectorMarketDepthBacktest([a])
    cap = 200_000
    ring_ts = np.zeros(n_t + 8, np.int64)
    ring_buy = np.zeros(n_t + 8)
    ring_sell = np.zeros(n_t + 8)
    dec_ts = np.zeros(cap, np.int64); dec_side = np.zeros(cap, np.int8); dec_imb = np.zeros(cap)
    ord_ts = np.zeros(cap, np.int64); ord_side = np.zeros(cap, np.int8); ord_qty = np.zeros(cap)
    fill_px = np.zeros(cap); fill_qty = np.zeros(cap); fill_ts = np.zeros(cap, np.int64)
    counts = np.zeros(8, np.int64)
    taker_kernel(hbt, int(p["window_ns"]), float(p["threshold"]), float(p["qty"]), ring_ts, ring_buy, ring_sell,
                 dec_ts, dec_side, dec_imb, ord_ts, ord_side, ord_qty, counts, fill_px, fill_qty, fill_ts)
    sv = hbt.state_values(0)
    out = {"prints": int(counts[0]), "decisions": int(counts[1]), "orders": int(counts[2]), "fills": int(counts[3]),
           "position": float(hbt.position(0)), "balance": float(sv.balance), "fee": float(sv.fee),
           "volume": float(sv.trading_volume), "code_end": int(counts[4])}
    hbt.close()
    nd, no, nf = out["decisions"], out["orders"], min(out["fills"], cap)
    journal = {"dec_ts": dec_ts[:nd].copy(), "dec_side": dec_side[:nd].copy(), "dec_imb": dec_imb[:nd].copy(),
               "ord_ts": ord_ts[:no].copy(), "ord_side": ord_side[:no].copy(), "ord_qty": ord_qty[:no].copy(),
               "fill_px": fill_px[:nf].copy(), "fill_qty": fill_qty[:nf].copy(), "fill_ts": fill_ts[:nf].copy()}
    return out, journal


# --------------------------------------------------------------------------- #
# Replay only: the same data, no strategy
# --------------------------------------------------------------------------- #
@njit
def replay_kernel(hbt):
    n = 0
    while hbt.elapse(60_000_000_000) == 0:
        n += 1
    return n


def run_replay(d, events):
    hbt = make_backtest(events, d, False, 0, 0)
    n = replay_kernel(hbt)
    hbt.close()
    return {"steps": int(n), "events": int(len(events))}


# --------------------------------------------------------------------------- #
# Order-by-order market (level 3): the maker that joins the best
# --------------------------------------------------------------------------- #
ADD_ORDER_EVENT, CANCEL_ORDER_EVENT, MODIFY_ORDER_EVENT, FILL_EVENT = 10, 11, 12, 13
EXCH_EVENT, LOCAL_EVENT = 1 << 31, 1 << 30
BUY_FLAG, SELL_FLAG = 1 << 29, 1 << 28


def build_events_l3(m):
    """hftbacktest's order-by-order events from the neutral MBO arrays (itch.py)."""
    from hftbacktest.types import event_dtype

    kind, side = m["kind"], m["side"]
    keep = kind != 4                                   # prints against hidden orders move no displayed order
    ev = np.zeros(int(keep.sum()), event_dtype)
    k = kind[keep]
    code = np.select([k == 0, k == 1, k == 2, k == 3],
                     [ADD_ORDER_EVENT, FILL_EVENT, MODIFY_ORDER_EVENT, CANCEL_ORDER_EVENT]).astype(np.uint64)
    sd = np.where(side[keep] == 0, BUY_FLAG, SELL_FLAG).astype(np.uint64)
    ev["ev"] = np.uint64(EXCH_EVENT | LOCAL_EVENT) | code | sd
    ev["exch_ts"] = m["ts"][keep]
    ev["local_ts"] = m["ts"][keep]
    ev["px"] = m["price"][keep] / 1e4
    ev["qty"] = m["qty"][keep]
    ev["order_id"] = m["oid"][keep]
    return ev


@njit
def mbo_kernel(hbt, t_open, t_close, period, lot, max_pos, crowded, tick,
               post_k, post_side, post_px, post_ts, rec_fill, rec_fill_ts, rec_status, counts, pos_log):
    ids = np.full(2, -1, np.int64)
    oid = 1
    hbt.elapse(1)
    now = hbt.current_timestamp
    if t_open > now:
        hbt.elapse(t_open - now)
    k = 1
    r = hbt.elapse(max(t_open + period - hbt.current_timestamp, 1))
    if r != 0:
        counts[5] = r
        return r
    while True:
        counts[0] += 1
        counts[6] = k
        orders = hbt.orders(0)
        _record(orders, rec_fill, rec_fill_ts, rec_status, False)
        hbt.clear_inactive_orders(0)
        depth = hbt.depth(0)
        bb = depth.best_bid_tick
        ba = depth.best_ask_tick
        pos = hbt.position(0)
        if k < pos_log.shape[0]:
            pos_log[k] = pos
        if bb > -(1 << 62) and ba < (1 << 62):
            bq = depth.best_bid_qty
            aq = depth.best_ask_qty
            for s in range(2):
                on = (pos < max_pos - 1e-9 and bq <= crowded) if s == 0 else (pos > -max_pos + 1e-9 and aq <= crowded)
                tgt = bb if s == 0 else ba
                oid_s = ids[s]
                live = False
                pending = False
                held = -1
                if oid_s >= 0:
                    o = orders.get(oid_s)
                    if o is not None:
                        st = o.status
                        rq = o.req
                        live = st == NEW_ or st == PARTIAL_ or (st == NONE_ and rq == NEW_)
                        pending = rq == CANCELED_
                        held = o.price_tick
                if not on:
                    if live and not pending:
                        hbt.cancel(0, oid_s, False)
                        counts[2] += 1
                    ids[s] = -1
                    continue
                if live and held == tgt:
                    continue
                if live:
                    if not pending:
                        hbt.cancel(0, oid_s, False)
                        counts[2] += 1
                    counts[3] += 1
                if s == 0:
                    e = hbt.submit_buy_order(0, oid, tgt * tick, lot, GTX_, LIMIT_, False)
                else:
                    e = hbt.submit_sell_order(0, oid, tgt * tick, lot, GTX_, LIMIT_, False)
                if e != 0:
                    counts[4] += 1
                n = counts[1]
                if n < post_k.shape[0]:
                    post_k[n] = k
                    post_side[n] = 1 if s == 0 else -1
                    post_px[n] = tgt
                    post_ts[n] = hbt.current_timestamp
                ids[s] = oid
                oid += 1
                counts[1] += 1
        t = t_open + (k + 1) * period
        if t >= t_close:
            break
        k += 1
        r = hbt.elapse(t - hbt.current_timestamp)
        if r != 0:
            counts[5] = r
            break
    _record(hbt.orders(0), rec_fill, rec_fill_ts, rec_status, True)
    return counts[5]


def run_replay_l3(events):
    """Replay of the order-by-order feed alone (level-3 book, no strategy)."""
    from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

    a = (BacktestAsset().data([events]).linear_asset(1.0).constant_order_latency(0, 0)
         .l3_fifo_queue_model().no_partial_fill_exchange()
         .trading_value_fee_model(0.0, 0.0).tick_size(spec.MBOJoin.TICK).lot_size(1.0).last_trades_capacity(0))
    hbt = HashMapMarketDepthBacktest([a])
    n = replay_kernel(hbt)
    hbt.close()
    return {"steps": int(n), "events": int(len(events))}


def run_mbo(events, t_open, t_close, p=None):
    """One run of the MBO maker on hftbacktest (level-3 FIFO queue, no partial fill, zero latency)."""
    from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

    sp = spec.MBOJoin
    a = (BacktestAsset().data([events]).linear_asset(1.0).constant_order_latency(0, 0)
         .l3_fifo_queue_model().no_partial_fill_exchange()
         .trading_value_fee_model(0.0, 0.0).tick_size(sp.TICK).lot_size(1.0).last_trades_capacity(0))
    hbt = HashMapMarketDepthBacktest([a])
    n_wakes = int((t_close - t_open) // sp.WAKE_NS) + 2
    cap = 2 * n_wakes + 16
    post_k = np.zeros(cap, np.int64); post_side = np.zeros(cap, np.int8)
    post_px = np.zeros(cap, np.int64); post_ts = np.zeros(cap, np.int64)
    rec_fill = np.zeros(cap + 2); rec_fill_ts = np.zeros(cap + 2, np.int64); rec_status = np.zeros(cap + 2, np.int8)
    counts = np.zeros(8, np.int64)
    pos_log = np.full(n_wakes + 2, np.nan)
    code = mbo_kernel(hbt, int(t_open), int(t_close), int(sp.WAKE_NS), sp.LOT, sp.LOT * sp.MAX_LOTS, sp.CROWDED, sp.TICK,
                      post_k, post_side, post_px, post_ts, rec_fill, rec_fill_ts, rec_status, counts, pos_log)
    sv = hbt.state_values(0)
    out = {"code_end": int(code), "wakes": int(counts[0]), "posts": int(counts[1]), "cancels": int(counts[2]),
           "requotes": int(counts[3]), "submit_errors": int(counts[4]), "fills": int(sv.num_trades),
           "volume": float(sv.trading_volume), "position": float(hbt.position(0)),
           "balance": float(sv.balance), "fee": float(sv.fee)}
    hbt.close()
    n = out["posts"]
    journal = {"k": post_k[:n].copy(), "side": post_side[:n].copy(), "px": post_px[:n].copy(), "ts": post_ts[:n].copy(),
               "filled": rec_fill[1:n + 1].copy(), "fill_ts": rec_fill_ts[1:n + 1].copy(),
               "status": rec_status[1:n + 1].copy(), "pos_log": pos_log}
    return out, journal
