"""Level-3 queue behaviour of Nautilus (L3_MBO) and hftbacktest (l3_fifo_queue_model) on tiny books.

Book: X (id 1) 100 @ 10.00 bid, and an ask. We post a buy of 10 at 10.00 at t=1.1,
behind X (queue ahead: 100). Scenarios 7 to 9 use two orders of 50 ahead of ours.

The point of scenarios 8 and 9: when an execution is fed the way an order-by-order
feed delivers it (a TradeTick AND the deletion / reduction of the executed order),
Nautilus takes it out of the queue ahead twice, and serves an order that an exact
FIFO queue has not served yet. Scenarios 11 and 12 (trades without the book deltas)
show the arithmetic is exact when the execution is counted once.

    python micro_scenarios_l3.py
"""
import os
import sys

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_nt as N  # noqa: E402
import engine_nt_l3 as L3  # noqa: E402

S = 10**9
N._imports()
venue, iid, inst = L3.l3_instrument("TST")
PXU, SZU = L3.L3_PX_UNIT, L3.L3_SZ_UNIT


def delta(act, side, oid, px, qty, t):
    return N.OrderBookDelta.from_raw(iid, act, side, int(round(px * 1e4)) * PXU, L3.L3_PX_PREC, int(qty) * SZU, 0, oid,
                                     int(N.RecordFlag.F_LAST), 0, int(t * S), int(t * S))


def trade(px, qty, aggr_sell, t, i):
    return N.TradeTick.from_raw(iid, int(round(px * 1e4)) * PXU, L3.L3_PX_PREC, int(qty) * SZU, 0,
                                N.AggressorSide.SELLER if aggr_sell else N.AggressorSide.BUYER, N.TradeId(str(i)),
                                int(t * S), int(t * S))


B, A = N.OrderSide.BUY, N.OrderSide.SELL
ADD, UPD, DEL = N.BookAction.ADD, N.BookAction.UPDATE, N.BookAction.DELETE


class One(N.Strategy):
    def __init__(self):
        super().__init__(N.StrategyConfig(strategy_id="ONE-001"))
        self.fills = []

    def on_start(self):
        self.subscribe_order_book_deltas(iid, book_type=N.BookType.L3_MBO)
        self.clock.set_time_alert_ns("post", int(1.1 * S), self.post)

    def post(self, ev):
        o = self.order_factory.limit(iid, B, N.Quantity.from_int(10), N.Price(10.0, L3.L3_PX_PREC),
                                     time_in_force=N.TimeInForce.GTC, post_only=True)
        self.submit_order(o)

    def on_order_filled(self, ev):
        self.fills.append((ev.ts_event / S, ev.last_qty.as_double()))


def base():
    return [delta(ADD, B, 1, 10.0, 100, 1.0), delta(ADD, A, 99, 10.01, 100, 1.0)]


def scen_nt(name, events):
    engine = N.BacktestEngine(N.BacktestEngineConfig(trader_id="M-001", logging=N.LoggingConfig(log_level="ERROR")))
    from decimal import Decimal
    from nautilus_trader.model.currencies import USD
    engine.add_venue(venue, N.OmsType.NETTING, N.AccountType.MARGIN, [N.Money(1_000_000, USD)], base_currency=USD,
                     default_leverage=Decimal(10), fill_model=N.FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=1),
                     book_type=N.BookType.L3_MBO, trade_execution=True, queue_position=True)
    engine.add_instrument(inst)
    s = One()
    engine.add_strategy(s)
    engine.add_data(events)
    engine.run()
    print(f"NT  {name:62s} fills {s.fills}")
    engine.dispose()


n = [0]


def tr(px, q, t):
    n[0] += 1
    return trade(px, q, True, t, n[0])


scen_nt("1 exec of X (100) as trade + delete X (the ITCH pattern)", base() + [tr(10.0, 100, 2.0), delta(DEL, B, 1, 10.0, 0, 2.0)])
scen_nt("2 delete X only (cancel ahead)", base() + [delta(DEL, B, 1, 10.0, 0, 2.0)])
scen_nt("3 trade 100 at the level, no book delta", base() + [tr(10.0, 100, 2.0)])
scen_nt("4 trade 100 + delete X, then trade 5 + reduce of new order Z", base() + [
    tr(10.0, 100, 2.0), delta(DEL, B, 1, 10.0, 0, 2.0),
    delta(ADD, B, 2, 10.0, 50, 2.5), tr(10.0, 5, 3.0), delta(UPD, B, 2, 10.0, 45, 3.0)])
scen_nt("5 half exec of X (50) as trade + update X to 50", base() + [tr(10.0, 50, 2.0), delta(UPD, B, 1, 10.0, 50, 2.0)])
scen_nt("6 half exec of X (50) twice: total 100 consumed", base() + [
    tr(10.0, 50, 2.0), delta(UPD, B, 1, 10.0, 50, 2.0), tr(10.0, 50, 3.0), delta(DEL, B, 1, 10.0, 0, 3.0)])

def base2():
    return [delta(ADD, B, 1, 10.0, 50, 1.0), delta(ADD, B, 3, 10.0, 50, 1.0), delta(ADD, A, 99, 10.01, 100, 1.0)]


scen_nt("7 two orders ahead (50+50): exec of the first (50) + delete", base2() + [tr(10.0, 50, 2.0), delta(DEL, B, 1, 10.0, 0, 2.0)])
scen_nt("8 same, then exec of the second (50) + delete", base2() + [tr(10.0, 50, 2.0), delta(DEL, B, 1, 10.0, 0, 2.0),
                                                                   tr(10.0, 50, 3.0), delta(DEL, B, 3, 10.0, 0, 3.0)])
scen_nt("9 two orders ahead: cancel of the first (50), exec of the second", base2() + [delta(DEL, B, 1, 10.0, 0, 2.0),
                                                                                     tr(10.0, 50, 3.0), delta(DEL, B, 3, 10.0, 0, 3.0)])
scen_nt("10 exec of the first order for 30 (partial) twice", base2() + [tr(10.0, 30, 2.0), delta(UPD, B, 1, 10.0, 20, 2.0),
                                                                        tr(10.0, 30, 3.0)])

scen_nt("11 two orders ahead: two trades of 50, NO book deltas", base2() + [tr(10.0, 50, 2.0), tr(10.0, 50, 3.0)])
scen_nt("12 two orders ahead: trades 50 then 51, NO book deltas", base2() + [tr(10.0, 50, 2.0), tr(10.0, 51, 3.0)])
scen_nt("13 two orders ahead: deletes only (no trades)", base2() + [delta(DEL, B, 1, 10.0, 0, 2.0), delta(DEL, B, 3, 10.0, 0, 3.0)])

# ---- hftbacktest L3
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest  # noqa: E402
from hftbacktest.types import event_dtype  # noqa: E402

EXL = (1 << 31) | (1 << 30)
BUYF, SELLF = 1 << 29, 1 << 28


def hrow(kind, side, oid, px, qty, t):
    return (EXL | kind | side, int(t * S), int(t * S), px, float(qty), oid, 0, 0.0)


@njit
def hrun(hbt, out):
    hbt.elapse(1)
    now = hbt.current_timestamp
    hbt.elapse(max(int(1.1 * 10**9) - now, 1))
    hbt.submit_buy_order(0, 77, 10.0, 10.0, 1, 0, False)
    t = hbt.current_timestamp
    while t < 5 * 10**9:
        r = hbt.elapse(10_000_000)
        t = hbt.current_timestamp
        o = hbt.orders(0).get(77)
        if o is not None and o.status == 3 and out[0] == 0:
            out[0] = 1.0; out[1] = t / 1e9; out[2] = o.exec_qty
        if r != 0:
            break


def scen_hft(name, rows):
    a = np.zeros(len(rows), event_dtype)
    for i, r in enumerate(rows):
        a[i] = r
    asset = (BacktestAsset().data([a]).linear_asset(1.0).constant_order_latency(0, 0).l3_fifo_queue_model()
             .no_partial_fill_exchange().trading_value_fee_model(0.0, 0.0).tick_size(0.01).lot_size(1.0).last_trades_capacity(0))
    hbt = HashMapMarketDepthBacktest([asset])
    out = np.zeros(4)
    hrun(hbt, out)
    hbt.close()
    print(f"HFT {name:62s} fills {(round(out[1], 2), out[2]) if out[0] else []}")


ADDK, CANK, MODK, FILK = 10, 11, 12, 13
base_h = [hrow(ADDK, BUYF, 1, 10.0, 100, 1.0), hrow(ADDK, SELLF, 99, 10.01, 100, 1.0)]
scen_hft("1 exec of X (100): FILL + CANCEL", base_h + [hrow(FILK, BUYF, 1, 10.0, 100, 2.0), hrow(CANK, BUYF, 1, 10.0, 0, 2.0)])
scen_hft("2 cancel X only", base_h + [hrow(CANK, BUYF, 1, 10.0, 0, 2.0)])
scen_hft("4 exec X + cancel, then Z added behind us, exec 5 of Z + modify", base_h + [
    hrow(FILK, BUYF, 1, 10.0, 100, 2.0), hrow(CANK, BUYF, 1, 10.0, 0, 2.0), hrow(ADDK, BUYF, 2, 10.0, 50, 2.5),
    hrow(FILK, BUYF, 2, 10.0, 5, 3.0), hrow(MODK, BUYF, 2, 10.0, 45, 3.0)])
scen_hft("5 half exec of X (50): FILL + MODIFY to 50", base_h + [hrow(FILK, BUYF, 1, 10.0, 50, 2.0), hrow(MODK, BUYF, 1, 10.0, 50, 2.0)])
scen_hft("6 two half execs of X", base_h + [hrow(FILK, BUYF, 1, 10.0, 50, 2.0), hrow(MODK, BUYF, 1, 10.0, 50, 2.0),
                                              hrow(FILK, BUYF, 1, 10.0, 50, 3.0), hrow(CANK, BUYF, 1, 10.0, 0, 3.0)])
