"""NautilusTrader adapter, order-by-order market (level 3, Nasdaq ITCH).

Orders, replacements, cancellations and executions of one symbol become
`OrderBookDelta`s with an order id (L3_MBO book) and `TradeTick`s (the
executions). The book standing at the open is handed over as one batch of ADDs in
arrival order so that the venue does not replay the pre-market.
"""
from __future__ import annotations

import time

import numpy as np

import engine_nt as N
import spec

L3_PX_PREC, L3_SZ_PREC = 4, 0
L3_PX_UNIT = 10 ** (9 - L3_PX_PREC)      # ITCH prices are in 1e-4 units: raw = price x 1e5
L3_SZ_UNIT = 10 ** (9 - L3_SZ_PREC)


def l3_instrument(symbol):
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.instruments import Equity
    from nautilus_trader.model.objects import Price, Quantity

    venue = Venue("XNAS")
    iid = InstrumentId(Symbol(symbol), venue)
    inst = Equity(
        instrument_id=iid, raw_symbol=Symbol(symbol), currency=USD, price_precision=L3_PX_PREC,
        price_increment=Price.from_str("0.0001"), lot_size=Quantity.from_int(1),
        isin=None, ts_event=0, ts_init=0)
    return venue, iid, inst


def opening_book(m, t_open):
    """The orders resting at t_open, in arrival order: ([(oid, side, price, qty)], events consumed)."""
    live = {}
    kinds, oids, sides, prices, qtys, ts = (m[k] for k in ("kind", "oid", "side", "price", "qty", "ts"))
    n = int(np.searchsorted(ts, t_open, side="left"))
    for i in range(n):
        k = kinds[i]
        if k == 0:
            live[int(oids[i])] = (int(sides[i]), int(prices[i]), int(qtys[i]))
        elif k == 2:
            o = int(oids[i])
            if o in live:
                s, p, _ = live[o]
                live[o] = (s, p, int(qtys[i]))
        elif k == 3:
            live.pop(int(oids[i]), None)
    return [(o, s, p, q) for o, (s, p, q) in live.items()], n


def stream(m, iid, t_open, t_close, conv_clock):
    t0 = time.perf_counter()
    opening, n0 = opening_book(m, t_open)
    end = int(np.searchsorted(m["ts"], t_close, side="left"))
    ts, kind, oid, side, price, qty = (m[k] for k in ("ts", "kind", "oid", "side", "price", "qty"))
    conv_clock[0] += time.perf_counter() - t0
    Delta, Trade = N.OrderBookDelta.from_raw, N.TradeTick.from_raw
    ADD_, UPD_, DEL_ = N.BookAction.ADD, N.BookAction.UPDATE, N.BookAction.DELETE
    buy, sell = N.OrderSide.BUY, N.OrderSide.SELL
    a_buy, a_sell = N.AggressorSide.BUYER, N.AggressorSide.SELLER
    F_LAST = int(N.RecordFlag.F_LAST)
    TradeId = N.TradeId
    t0 = time.perf_counter()
    out = []
    for o, s, p, q in opening:
        out.append(Delta(iid, ADD_, buy if s == 0 else sell, p * L3_PX_UNIT, L3_PX_PREC, q * L3_SZ_UNIT, L3_SZ_PREC,
                         o, 0, 0, t_open - 1, t_open - 1))
    conv_clock[0] += time.perf_counter() - t0
    yield out
    n_trade = 0
    for a in range(n0, end, N.CHUNK):
        t0 = time.perf_counter()
        out = []
        push = out.append
        b = min(a + N.CHUNK, end)
        for i in range(a, b):
            k = kind[i]
            t = int(ts[i])
            s = side[i]
            if k == 0:
                push(Delta(iid, ADD_, buy if s == 0 else sell, int(price[i]) * L3_PX_UNIT, L3_PX_PREC,
                           int(qty[i]) * L3_SZ_UNIT, L3_SZ_PREC, int(oid[i]), F_LAST, 0, t, t))
            elif k == 2:
                push(Delta(iid, UPD_, buy if s == 0 else sell, int(price[i]) * L3_PX_UNIT, L3_PX_PREC,
                           int(qty[i]) * L3_SZ_UNIT, L3_SZ_PREC, int(oid[i]), F_LAST, 0, t, t))
            elif k == 3:
                push(Delta(iid, DEL_, buy if s == 0 else sell, int(price[i]) * L3_PX_UNIT, L3_PX_PREC,
                           0, L3_SZ_PREC, int(oid[i]), F_LAST, 0, t, t))
            elif k == 1:
                n_trade += 1
                # an execution against a resting bid is a sale, against an ask a purchase
                push(Trade(iid, int(price[i]) * L3_PX_UNIT, L3_PX_PREC, int(qty[i]) * L3_SZ_UNIT, L3_SZ_PREC,
                           a_sell if s == 0 else a_buy, TradeId(str(n_trade)), t, t))
        conv_clock[0] += time.perf_counter() - t0
        yield out


def _make_maker_class():
    Strategy, StrategyConfig = N.Strategy, N.StrategyConfig
    BookType, OrderSide, TimeInForce = N.BookType, N.OrderSide, N.TimeInForce
    Price, Quantity = N.Price, N.Quantity

    class MBOMaker(Strategy):
        def __init__(self, iid, anchor, stop_ns, sink):
            super().__init__(StrategyConfig(strategy_id="MBO-001"))
            self.iid, self.anchor, self.stop_ns, self.sink = iid, anchor, stop_ns, sink
            sp = spec.MBOJoin
            self.period, self.lot, self.max_pos, self.crowded = sp.WAKE_NS, sp.LOT, sp.LOT * sp.MAX_LOTS, sp.CROWDED
            self.live = [None, None]
            self.pos = 0.0
            self.rec = {}
            self.posts = self.cancels = self.requotes = self.wakes = self.rejected = 0
            self.qty = Quantity(sp.LOT, L3_SZ_PREC)

        def on_start(self):
            self.subscribe_order_book_deltas(self.iid, book_type=BookType.L3_MBO)
            self.clock.set_timer_ns("wake", self.period, self.anchor, self.stop_ns, self.on_wake)

        def on_order_filled(self, event):
            r = self.rec.get(event.client_order_id)
            q = event.last_qty.as_double()
            if r is not None:
                r[4] += q
                r[5] = event.ts_event
            self.pos += q if event.order_side == OrderSide.BUY else -q

        def on_order_rejected(self, event):
            self.rejected += 1
            if len(self.sink.setdefault("reasons", [])) < 5:
                self.sink["reasons"].append(str(event.reason))

        def on_wake(self, event):
            now = event.ts_event
            self.wakes += 1
            k = (now - self.anchor) // self.period
            book = self.cache.order_book(self.iid)
            if book is None:
                return
            bp, ap = book.best_bid_price(), book.best_ask_price()
            if bp is None or ap is None:
                return
            bq, aq = book.best_bid_size().as_double(), book.best_ask_size().as_double()
            for s in (0, 1):
                on = (self.pos < self.max_pos - 1e-9 and bq <= self.crowded) if s == 0 else \
                     (self.pos > -self.max_pos + 1e-9 and aq <= self.crowded)
                tgt_px = bp if s == 0 else ap
                tgt = int(round(tgt_px.as_double() * 100))
                order = self.live[s]
                live = order is not None and not order.is_closed
                pending = live and order.is_pending_cancel
                held = int(round(order.price.as_double() * 100)) if order is not None else None
                if not on:
                    if live and not pending:
                        self.cancel_order(order)
                        self.cancels += 1
                    self.live[s] = None
                    continue
                if live and held == tgt:
                    continue
                if live:
                    if not pending:
                        self.cancel_order(order)
                        self.cancels += 1
                    self.requotes += 1
                new = self.order_factory.limit(
                    self.iid, OrderSide.BUY if s == 0 else OrderSide.SELL, self.qty,
                    Price(tgt / 100.0, L3_PX_PREC), time_in_force=TimeInForce.GTC, post_only=True)
                self.rec[new.client_order_id] = [int(k), 1 if s == 0 else -1, tgt, now, 0.0, 0]
                self.sink["order_ids"].append(new.client_order_id)
                self.live[s] = new
                self.submit_order(new)
                self.posts += 1

    return MBOMaker


def run_mbo(m, t_open, t_close, strategy=True):
    N._imports()
    from nautilus_trader.model.currencies import USD

    Maker = _make_maker_class()
    sp = spec.MBOJoin
    venue, iid, inst = l3_instrument(sp.SYMBOL)
    engine = N.BacktestEngine(N.BacktestEngineConfig(trader_id="BENCH-004",
                                                     logging=N.LoggingConfig(log_level="ERROR")))
    from decimal import Decimal

    # A margin account: a cash account refuses to sell what it does not hold.
    engine.add_venue(venue, N.OmsType.NETTING, N.AccountType.MARGIN, [N.Money(100_000_000, USD)],
                     base_currency=USD, default_leverage=Decimal(10),
                     fill_model=N.FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=1),
                     book_type=N.BookType.L3_MBO, trade_execution=True, queue_position=True)
    engine.add_instrument(inst)
    sink = {"order_ids": []}
    strat = Maker(iid, int(t_open), int(t_close), sink)
    if strategy:
        engine.add_strategy(strat)
    conv = [0.0]
    engine.add_data_iterator("flux", stream(m, iid, int(t_open), int(t_close), conv))
    t0 = time.perf_counter()
    engine.run()
    total = time.perf_counter() - t0
    recs = [strat.rec[c] for c in sink["order_ids"]]

    def col(i, dtype):
        return np.array([r[i] for r in recs], dtype=dtype)

    filled = col(4, np.float64)
    out = {"wakes": strat.wakes, "posts": strat.posts, "cancels": strat.cancels, "requotes": strat.requotes,
           "rejected": strat.rejected, "fills": int((filled > 0).sum()), "volume": float(filled.sum()),
           "position": float(strat.pos), "engine_seconds": round(total - conv[0], 3),
           "convert_seconds": round(conv[0], 3), "run_total_seconds": round(total, 3),
           "reject_reasons": sink.get("reasons", [])}
    journal = {"k": col(0, np.int64), "side": col(1, np.int8), "px": col(2, np.int64), "ts": col(3, np.int64),
               "filled": filled, "fill_ts": col(5, np.int64)}
    engine.dispose()
    return out, journal
