"""NautilusTrader adapter: the strategies of spec.py as `Strategy` subclasses on a
BacktestEngine (low-level API), fed with the neutral streams.

Book tiers are replayed as `OrderBookDelta` (L2_MBP book); the BBO tier is
replayed as `QuoteTick` (L1_MBP book). Trades are `TradeTick`. Passive orders
queue behind the displayed size (`queue_position=True`) and are served by trades
(`trade_execution=True`).
"""
from __future__ import annotations

import time
from decimal import Decimal

import numpy as np

import spec

PX_PREC, SZ_PREC = 1, 6
PX_UNIT = 10 ** (9 - PX_PREC)      # raw = value x 1e9 at the venue's precision
SZ_UNIT = 10 ** (9 - SZ_PREC)
CHUNK = 400_000


def _imports():
    global BacktestEngine, BacktestEngineConfig, FillModel, LatencyModel, LoggingConfig, StrategyConfig
    global BTC, USDT, OrderBookDelta, QuoteTick, TradeTick, AccountType, AggressorSide, BookAction
    global BookType, OmsType, OrderSide, RecordFlag, TimeInForce, InstrumentId, Symbol, TradeId, Venue
    global CurrencyPair, Money, Price, Quantity, Strategy
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel, LatencyModel
    from nautilus_trader.config import LoggingConfig, StrategyConfig
    from nautilus_trader.model.currencies import BTC, USDT
    from nautilus_trader.model.data import OrderBookDelta, QuoteTick, TradeTick
    from nautilus_trader.model.enums import (AccountType, AggressorSide, BookAction, BookType, OmsType,
                                             OrderSide, RecordFlag, TimeInForce)
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, TradeId, Venue
    from nautilus_trader.model.instruments import CurrencyPair
    from nautilus_trader.model.objects import Money, Price, Quantity
    from nautilus_trader.trading.strategy import Strategy


def instrument():
    venue = Venue("BYBIT")
    iid = InstrumentId(Symbol(spec.SYMBOL), venue)
    return venue, iid, CurrencyPair(
        instrument_id=iid, raw_symbol=Symbol(spec.SYMBOL),
        base_currency=BTC, quote_currency=USDT,
        price_precision=PX_PREC, size_precision=SZ_PREC,
        price_increment=Price.from_str("0.1"), size_increment=Quantity.from_str("0.000001"),
        lot_size=None, max_quantity=None, min_quantity=None, max_notional=None,
        min_notional=None, max_price=None, min_price=None,
        margin_init=Decimal(0), margin_maint=Decimal(0),
        maker_fee=Decimal(str(spec.MAKER_FEE_BPS / 1e4)), taker_fee=Decimal(str(spec.TAKER_FEE_BPS / 1e4)),
        ts_event=0, ts_init=0,
    )


def _merged_order(ts_depth, ts_trade):
    """Indices into [depth..., trades...], by time; at equal times depth first."""
    ts = np.concatenate([ts_depth, ts_trade])
    return np.argsort(ts, kind="stable")


def stream(d, tier, iid, feed_ns, conv_clock):
    """Generator of lists of Nautilus data objects, in `ts_init` order.
    conv_clock[0] accumulates the seconds spent converting."""
    t0 = time.perf_counter()
    n_t = len(d["tr_ts"])
    tr_px = (np.round(d["tr_px"] * 10 ** PX_PREC).astype(np.int64) * PX_UNIT)
    tr_sz = (np.round(d["tr_qty"] * 10 ** SZ_PREC).astype(np.int64) * SZ_UNIT)
    if tier == "l2":
        n_d = len(d["d25_ts"])
        ts_d = d["d25_ts"]
        px_d = d["d25_px"] * PX_UNIT          # ticks (= price x 10^PX_PREC) -> raw
        sz_d = (np.round(d["d25_qty"] * 10 ** SZ_PREC).astype(np.int64) * SZ_UNIT)
        side_d, last_d = d["d25_side"], d["d25_last"]
    else:
        n_d = len(d["q_ts"])
        ts_d = d["q_ts"]
    order = _merged_order(ts_d, d["tr_ts"])
    conv_clock[0] += time.perf_counter() - t0

    Delta, Trade, Quote = OrderBookDelta.from_raw, TradeTick.from_raw, QuoteTick.from_raw
    upd, dele = BookAction.UPDATE, BookAction.DELETE
    buy, sell = OrderSide.BUY, OrderSide.SELL
    a_buy, a_sell = AggressorSide.BUYER, AggressorSide.SELLER
    F_LAST = int(RecordFlag.F_LAST)
    feed = int(feed_ns)
    if tier != "l2":
        q_bp = d["q_bpx"] * PX_UNIT
        q_ap = d["q_apx"] * PX_UNIT
        q_bq = (np.round(d["q_bqty"] * 10 ** SZ_PREC).astype(np.int64) * SZ_UNIT)
        q_aq = (np.round(d["q_aqty"] * 10 ** SZ_PREC).astype(np.int64) * SZ_UNIT)
    for a in range(0, len(order), CHUNK):
        t0 = time.perf_counter()
        out = []
        push = out.append
        for j in order[a:a + CHUNK]:
            j = int(j)
            if j < n_d:
                t = int(ts_d[j])
                if tier == "l2":
                    q = int(sz_d[j])
                    push(Delta(iid, dele if q == 0 else upd, buy if side_d[j] == 0 else sell,
                               int(px_d[j]), PX_PREC, q, SZ_PREC, 0, F_LAST if last_d[j] else 0, 0, t, t + feed))
                else:
                    push(Quote(iid, int(q_bp[j]), int(q_ap[j]), PX_PREC, PX_PREC, int(q_bq[j]), int(q_aq[j]),
                               SZ_PREC, SZ_PREC, t, t + feed))
            else:
                i = j - n_d
                t = int(d["tr_ts"][i])
                push(Trade(iid, int(tr_px[i]), PX_PREC, int(tr_sz[i]), SZ_PREC,
                           a_buy if d["tr_side"][i] == 0 else a_sell, TradeId(str(i)), t, t + feed))
        conv_clock[0] += time.perf_counter() - t0
        yield out


def _make_strategy_classes():
    """Defined lazily: the Strategy base only exists after _imports()."""

    class MakerStrategy(Strategy):
        def __init__(self, iid, tier, p, anchor, period, resp_delay_ns, sink, stop_ns):
            super().__init__(StrategyConfig(strategy_id="MM-001"))
            self.iid, self.tier, self.p = iid, tier, p
            self.anchor, self.period, self.stop_ns = anchor, period, stop_ns
            self.delay = resp_delay_ns
            self.sink = sink                       # dict of result lists
            self.live = [None, None]               # order per side (0 buy, 1 sell)
            self.closed_at = {}                    # client_order_id -> ts the strategy learns it closed
            self.pos_pending = []                  # (visible_ts, signed qty)
            self.pos = 0.0
            self.k = 0
            self.rec = {}                          # client_order_id -> [k, side, px_tick, ts, filled, fill_ts]
            self.posts = self.cancels = self.requotes = 0
            self.rejected = 0
            self.qty = Quantity(float(p["clip"]), SZ_PREC)
            self.wakes = 0
            self.mid_cache = None

        def on_start(self):
            if self.tier == "l2":
                self.subscribe_order_book_deltas(self.iid, book_type=BookType.L2_MBP)
            else:
                self.subscribe_quote_ticks(self.iid)
            # A repeating timer must have a stop time: at the end of the data the
            # engine drains the pending timers, and one with no stop never ends.
            # (Nautilus fires a timer first at start + interval, so start at the anchor.)
            self.clock.set_timer_ns("wake", self.period, self.anchor, self.stop_ns, self.on_wake)

        def on_order_filled(self, event):
            i = event.client_order_id
            r = self.rec.get(i)
            q = event.last_qty.as_double()
            if r is not None:
                r[4] += q
                r[5] = event.ts_event
            sgn = 1.0 if event.order_side == OrderSide.BUY else -1.0
            self.pos_pending.append((event.ts_event + self.delay, sgn * q))
            self.closed_at[i] = event.ts_event + self.delay        # a filled order is closed once whole

        def on_order_canceled(self, event):
            self.closed_at[event.client_order_id] = event.ts_event + self.delay

        def on_order_rejected(self, event):
            self.rejected += 1
            self.closed_at[event.client_order_id] = event.ts_event + self.delay

        def _levels(self):
            if self.tier == "l2":
                book = self.cache.order_book(self.iid)
                if book is None:
                    return None, None
                return book.bids(), book.asks()
            return None, None

        def on_wake(self, event):
            now = event.ts_event
            self.wakes += 1
            self.k = (now - self.anchor) // self.period
            pp = self.pos_pending
            if pp:
                keep = []
                for vis, q in pp:
                    if vis <= now:
                        self.pos += q
                    else:
                        keep.append((vis, q))
                self.pos_pending = keep
            clip = self.p["clip"]
            if self.tier == "l2":
                book = self.cache.order_book(self.iid)
                if book is None:
                    return
                bids, asks = book.bids(), book.asks()
                nb, na = len(bids), len(asks)
            else:
                qt = self.cache.quote_tick(self.iid)
                if qt is None:
                    return
                bids, asks, nb, na = None, None, 1, 1
            if nb == 0 or na == 0:
                return
            clips = round(self.pos / clip)
            probe = bool(self.p.get("probe", False))
            if probe:
                clips = 0
            if self.p["ladder"]:
                ib = int(min(max(clips, 0), nb - 1)) + self.p["off_bid"]
                ia = int(min(max(-clips, 0), na - 1)) + self.p["off_ask"]
            else:
                ib = ia = 0
            on = (clips < self.p["max_clips"], clips > -self.p["max_clips"])
            for s in (0, 1):
                idx = ib if s == 0 else ia
                n_lv = nb if s == 0 else na
                order = self.live[s]
                live = False
                pending = False
                held = None
                if order is not None:
                    cid = order.client_order_id
                    ca = self.closed_at.get(cid)
                    live = not (order.is_closed and ca is not None and ca <= now)
                    pending = order.is_pending_cancel
                    held = int(round(order.price.as_double() * spec.TICK_INV))
                if not on[s]:
                    if live and not pending:
                        self.cancel_order(order)
                        self.cancels += 1
                    self.live[s] = None
                    continue
                if idx >= n_lv:
                    continue
                if self.tier == "l2":
                    lvl = (bids if s == 0 else asks)[idx]
                    tgt = int(round(lvl.price.as_double() * spec.TICK_INV))
                else:
                    tgt = int(round((qt.bid_price if s == 0 else qt.ask_price).as_double() * spec.TICK_INV))
                if live and held == tgt and not probe:
                    continue
                if live:
                    if not pending and not order.is_closed:
                        self.cancel_order(order)
                        self.cancels += 1
                    self.requotes += 1
                new = self.order_factory.limit(
                    self.iid, OrderSide.BUY if s == 0 else OrderSide.SELL, self.qty,
                    Price(tgt / spec.TICK_INV, PX_PREC), time_in_force=TimeInForce.GTC, post_only=True)
                self.rec[new.client_order_id] = [int(self.k), 1 if s == 0 else -1, tgt, now, 0.0, 0]
                self.sink["order_ids"].append(new.client_order_id)
                self.live[s] = new
                self.submit_order(new)
                self.posts += 1

    return MakerStrategy


def run_mm(d, tier, p, window, latency, feed_ns=None, order_ns=None):
    """One backtest of the maker on Nautilus. latency: 'zero' or 'nominal'.
    `order_ns` overrides the order and cancel delay (a sweep axis)."""
    _imports()
    Maker = _make_strategy_classes()
    venue, iid, inst = instrument()
    w0, w1 = window
    key = "d25" if tier == "l2" else "q"
    anchor = int(d["anchor"][0]) if "anchor" in d else int(d[key + "_ts"][0])
    period = spec.WAKE_NS
    nominal = latency == "nominal"
    feed = spec.LAT_FEED_NS if nominal else 0
    lat = None
    delay = 0
    if nominal:
        # The exchange is fed `feed` late, so an order must be sent order+feed
        # before it counts as arriving `order` after the decision. Nautilus
        # has no response latency: the strategy emulates it.
        o_ns = spec.LAT_ORDER_NS if order_ns is None else int(order_ns)
        c_ns = spec.LAT_CANCEL_NS if order_ns is None else int(order_ns)
        lat = LatencyModel(base_latency_nanos=0,
                           insert_latency_nanos=o_ns + feed,
                           update_latency_nanos=o_ns + feed,
                           cancel_latency_nanos=c_ns + feed)
        delay = max(spec.LAT_RESPONSE_NS - feed, 0)
    engine = BacktestEngine(BacktestEngineConfig(trader_id="BENCH-001", logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(
        venue, OmsType.NETTING, AccountType.CASH,
        [Money(100_000_000, USDT), Money(1_000, BTC)],
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=1),
        latency_model=lat,
        book_type=BookType.L2_MBP if tier == "l2" else BookType.L1_MBP,
        trade_execution=True, queue_position=True,
    )
    engine.add_instrument(inst)
    sink = {"order_ids": []}
    strat = Maker(iid, tier, p, anchor, period, delay, sink, int(w1))
    engine.add_strategy(strat)
    conv = [0.0]
    engine.add_data_iterator("flux", stream(d, tier, iid, feed, conv))
    t0 = time.perf_counter()
    engine.run()
    total = time.perf_counter() - t0
    recs = [strat.rec[c] for c in sink["order_ids"]]
    def col(i, dtype):
        return np.array([r[i] for r in recs], dtype=dtype)

    filled = col(4, np.float64)
    fills = int((filled > 0).sum())
    out = {
        "wakes": strat.wakes, "posts": strat.posts, "cancels": strat.cancels, "requotes": strat.requotes,
        "rejected": strat.rejected, "fills": fills, "volume": float(filled.sum()),
        "position": float(strat.pos + sum(q for _, q in strat.pos_pending)),
        "engine_seconds": round(total - conv[0], 3), "convert_seconds": round(conv[0], 3),
        "run_total_seconds": round(total, 3),
    }
    journal = {
        "k": col(0, np.int64), "side": col(1, np.int8), "px": col(2, np.int64),
        "ts": col(3, np.int64), "filled": filled, "fill_ts": col(5, np.int64),
    }
    engine.dispose()
    return out, journal


def _make_taker_class():
    import collections

    class TakerStrategy(Strategy):
        def __init__(self, iid, p):
            super().__init__(StrategyConfig(strategy_id="FT-001"))
            self.iid, self.p = iid, p
            self.win = p["window_ns"]
            self.thr = p["threshold"]
            self.qty_unit = p["qty"]
            self.ring = collections.deque()
            self.bought = self.sold = 0.0
            self.first_t = None
            self.side = 0
            self.sent_pos = 0.0
            self.prints = 0
            self.decisions = []                 # (ts, side, imbalance)
            self.orders = []                    # (ts, side, qty)
            self.fills = []                     # (ts, px, qty, side)
            self.rejected = 0

        def on_start(self):
            self.subscribe_trade_ticks(self.iid)

        def on_trade_tick(self, tick):
            t = tick.ts_event
            q = tick.size.as_double()
            buy = tick.aggressor_side == AggressorSide.BUYER
            b, s = (q, 0.0) if buy else (0.0, q)
            ring = self.ring
            ring.append((t, b, s))
            self.bought += b
            self.sold += s
            lim = t - self.win
            while ring and ring[0][0] <= lim:
                _, ob, os_ = ring.popleft()
                self.bought -= ob
                self.sold -= os_
            self.prints += 1
            if self.first_t is None:
                self.first_t = t
            imb = (self.bought - self.sold) / (self.bought + self.sold + 1e-12)
            new = self.side
            if t - self.first_t >= self.win:        # the window is not full before a whole window of tape has gone by
                if imb >= self.thr:
                    new = 1
                elif imb <= -self.thr:
                    new = -1
            if new != self.side:
                self.side = new
                self.decisions.append((t, new, imb))
                target = new * self.qty_unit
                delta = target - self.sent_pos
                if abs(delta) > 1e-12:
                    o = self.order_factory.market(self.iid, OrderSide.BUY if delta > 0 else OrderSide.SELL,
                                                  Quantity(abs(delta), SZ_PREC))
                    self.orders.append((t, 1 if delta > 0 else -1, abs(delta)))
                    self.submit_order(o)
                    self.sent_pos = target

        def on_order_filled(self, event):
            self.fills.append((event.ts_event, event.last_px.as_double(), event.last_qty.as_double(),
                               1 if event.order_side == OrderSide.BUY else -1))

        def on_order_rejected(self, event):
            self.rejected += 1

    return TakerStrategy


def run_taker(d, p, window, latency):
    """One backtest of the flow taker on Nautilus (BBO tier: quote ticks + trades)."""
    _imports()
    Taker = _make_taker_class()
    venue, iid, inst = instrument()
    nominal = latency == "nominal"
    lat = None
    if nominal:
        n = 20 * spec.MS
        lat = LatencyModel(base_latency_nanos=0, insert_latency_nanos=n, update_latency_nanos=n,
                           cancel_latency_nanos=n)
    engine = BacktestEngine(BacktestEngineConfig(trader_id="BENCH-002", logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(
        venue, OmsType.NETTING, AccountType.CASH, [Money(100_000_000, USDT), Money(1_000, BTC)],
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=1),
        latency_model=lat, book_type=BookType.L1_MBP, trade_execution=True, queue_position=True)
    engine.add_instrument(inst)
    strat = Taker(iid, p)
    engine.add_strategy(strat)
    conv = [0.0]
    engine.add_data_iterator("flux", stream(d, "bbo", iid, 0, conv))
    t0 = time.perf_counter()
    engine.run()
    total = time.perf_counter() - t0
    pos = sum(f[3] * f[2] for f in strat.fills)
    out = {"prints": strat.prints, "decisions": len(strat.decisions), "orders": len(strat.orders),
           "fills": len(strat.fills), "position": float(pos), "rejected": strat.rejected,
           "volume": float(sum(f[2] for f in strat.fills)),
           "engine_seconds": round(total - conv[0], 3), "convert_seconds": round(conv[0], 3),
           "run_total_seconds": round(total, 3)}
    def col(rows, i, dtype):
        return np.array([r[i] for r in rows], dtype=dtype)

    journal = {"dec_ts": col(strat.decisions, 0, np.int64), "dec_side": col(strat.decisions, 1, np.int8),
               "dec_imb": col(strat.decisions, 2, np.float64),
               "ord_ts": col(strat.orders, 0, np.int64), "ord_side": col(strat.orders, 1, np.int8),
               "ord_qty": col(strat.orders, 2, np.float64),
               "fill_px": col(strat.fills, 1, np.float64), "fill_qty": col(strat.fills, 2, np.float64),
               "fill_ts": col(strat.fills, 0, np.int64), "fill_side": col(strat.fills, 3, np.int8)}
    engine.dispose()
    return out, journal


def run_replay(d, tier):
    """The venue and the data, no strategy: what replaying the day costs."""
    _imports()
    venue, iid, inst = instrument()
    engine = BacktestEngine(BacktestEngineConfig(trader_id="BENCH-003", logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(
        venue, OmsType.NETTING, AccountType.CASH, [Money(100_000_000, USDT), Money(1_000, BTC)],
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=1),
        book_type=BookType.L2_MBP if tier == "l2" else BookType.L1_MBP,
        trade_execution=True, queue_position=True)
    engine.add_instrument(inst)
    conv = [0.0]
    engine.add_data_iterator("flux", stream(d, tier, iid, 0, conv))
    t0 = time.perf_counter()
    engine.run()
    total = time.perf_counter() - t0
    engine.dispose()
    return {"engine_seconds": round(total - conv[0], 3), "convert_seconds": round(conv[0], 3),
            "run_total_seconds": round(total, 3)}
