"""The queue behaviour of the engines on tiny books.

One resting buy of 0.01 at 100.0 behind a displayed size of 1.0, and a handful of
market events. For each scenario the table says whether the order was served,
when, and for how much. It is the shortest way to see where two engines' fill
models part, and each row is a case the open-loop probe meets on real data.

    python micro_scenarios.py            # hftbacktest and NautilusTrader

A manifoldbt column is added when the order API is available (`--mbt`).
"""
import sys

import numpy as np

S = 10**9
SCENARIOS = [
    ("A", "prints at the level below the queue (0.5 of 1.0)"),
    ("B", "prints at the level exceed the queue (0.5 then 0.6)"),
    ("C", "one print THROUGH the level (99.9)"),
    ("E", "the level is deleted from the book"),
    ("F", "the best ask drops THROUGH the order (99.9)"),
    ("G", "one print of 1.1 at the level"),
    ("I", "one print of 1.005: 0.005 left for a 0.01 order"),
    ("K", "the displayed size ahead shrinks to 0.3, then a print of 0.35"),
    ("L", "the displayed size ahead shrinks to 0.3, then a print of 0.2"),
]
# (kind, t, price, qty); kinds: bid / ask book levels, sell = print of a selling aggressor
EVENTS = {
    "A": [("sell", 2.0, 100.0, 0.5)],
    "B": [("sell", 2.0, 100.0, 0.5), ("sell", 3.0, 100.0, 0.6)],
    "C": [("sell", 2.0, 99.9, 0.001)],
    "E": [("bid", 2.0, 100.0, 0.0)],
    "F": [("ask", 2.0, 99.9, 1.0)],
    "G": [("sell", 2.0, 100.0, 1.1)],
    "I": [("sell", 2.0, 100.0, 1.005)],
    "K": [("bid", 2.0, 100.0, 0.3), ("sell", 3.0, 100.0, 0.35)],
    "L": [("bid", 2.0, 100.0, 0.3), ("sell", 3.0, 100.0, 0.2)],
}
BASE = [("bid", 1.0, 100.0, 1.0), ("ask", 1.0, 100.1, 1.0)]


def hftbacktest(partial):
    from numba import njit

    from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
    from hftbacktest.data import correct_event_order, validate_event_order
    from hftbacktest.data.validation import correct_local_timestamp
    from hftbacktest.types import BUY_EVENT, DEPTH_EVENT, SELL_EVENT, TRADE_EVENT, event_dtype

    @njit
    def run(hbt, t_post, px, qty, t_end, out):
        hbt.elapse(1)
        now = hbt.current_timestamp
        hbt.elapse(max(t_post - now, 1))
        hbt.submit_buy_order(0, 1, px, qty, 1, 0, False)
        t = hbt.current_timestamp
        while t < t_end:
            r = hbt.elapse(10_000_000)
            t = hbt.current_timestamp
            o = hbt.orders(0).get(1)
            if o is not None and o.status == 3 and out[0] == 0:
                out[0] = 1.0; out[1] = t / 1e9; out[2] = o.exec_qty
            if o is not None and o.status == 5 and out[3] == 0:
                out[3] = 1.0; out[4] = t / 1e9; out[5] = o.exec_qty
            if r != 0:
                break

    res = {}
    for key, _ in SCENARIOS:
        rows = []
        for kind, t, px, q in BASE + EVENTS[key]:
            flag = {"bid": DEPTH_EVENT | BUY_EVENT, "ask": DEPTH_EVENT | SELL_EVENT, "sell": TRADE_EVENT | SELL_EVENT}[kind]
            rows.append((flag, int(t * S), int(t * S), px, q, 0, 0, 0.0))
        a = np.zeros(len(rows), event_dtype)
        for i, r in enumerate(rows):
            a[i] = r
        a = correct_local_timestamp(a, 0)
        a = correct_event_order(a, np.argsort(a["exch_ts"], kind="mergesort"), np.argsort(a["local_ts"], kind="mergesort"))
        validate_event_order(a)
        asset = (BacktestAsset().data([a]).linear_asset(1.0).constant_order_latency(0, 0).risk_adverse_queue_model()
                 .trading_value_fee_model(0.0, 0.0).tick_size(0.1).lot_size(1e-6))
        asset = asset.partial_fill_exchange() if partial else asset.no_partial_fill_exchange()
        hbt = HashMapMarketDepthBacktest([asset])
        out = np.zeros(8)
        run(hbt, int(1.1 * S), 100.0, 0.01, int(5 * S), out)
        hbt.close()
        res[key] = (round(out[2] + out[5], 4), round(out[1] or out[4], 2)) if (out[0] or out[3]) else None
    return res


def nautilus(queue_position=True):
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    import engine_nt as N

    N._imports()
    venue, iid, inst = N.instrument()

    def delta(act, side, px, qty, t):
        return N.OrderBookDelta.from_raw(iid, act, side, int(round(px * 10)) * N.PX_UNIT, N.PX_PREC,
                                         int(round(qty * 1e6)) * N.SZ_UNIT, N.SZ_PREC, 0,
                                         int(N.RecordFlag.F_LAST), 0, int(t * S), int(t * S))

    class One(N.Strategy):
        def __init__(self):
            super().__init__(N.StrategyConfig(strategy_id="ONE-001"))
            self.fills = []

        def on_start(self):
            self.subscribe_order_book_deltas(iid, book_type=N.BookType.L2_MBP)
            self.clock.set_time_alert_ns("post", int(1.1 * S), self.post)

        def post(self, ev):
            o = self.order_factory.limit(iid, N.OrderSide.BUY, N.Quantity.from_str("0.010000"), N.Price(100.0, N.PX_PREC),
                                         time_in_force=N.TimeInForce.GTC, post_only=True)
            self.submit_order(o)

        def on_order_filled(self, ev):
            self.fills.append((ev.ts_event / S, ev.last_qty.as_double()))

    res = {}
    B, A = N.OrderSide.BUY, N.OrderSide.SELL
    U, D = N.BookAction.UPDATE, N.BookAction.DELETE
    for key, _ in SCENARIOS:
        data = []
        for i, (kind, t, px, q) in enumerate(BASE + EVENTS[key]):
            if kind == "bid":
                data.append(delta(D if q == 0 else U, B, px, q, t))
            elif kind == "ask":
                data.append(delta(U, A, px, q, t))
            else:
                data.append(N.TradeTick.from_raw(iid, int(round(px * 10)) * N.PX_UNIT, N.PX_PREC,
                                                 int(round(q * 1e6)) * N.SZ_UNIT, N.SZ_PREC, N.AggressorSide.SELLER,
                                                 N.TradeId(str(i)), int(t * S), int(t * S)))
        engine = N.BacktestEngine(N.BacktestEngineConfig(trader_id="M-001", logging=N.LoggingConfig(log_level="ERROR")))
        engine.add_venue(venue, N.OmsType.NETTING, N.AccountType.CASH, [N.Money(1_000_000, N.USDT), N.Money(10, N.BTC)],
                         fill_model=N.FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=1),
                         book_type=N.BookType.L2_MBP, trade_execution=True, queue_position=queue_position)
        engine.add_instrument(inst)
        s = One()
        engine.add_strategy(s)
        engine.add_data(data)
        engine.run()
        res[key] = (round(sum(q for _, q in s.fills), 4), s.fills[0][0]) if s.fills else None
        engine.dispose()
    return res


def manifoldbt():
    """The same scenarios through the order API on a market written by hand
    (`Market.from_ladders`). Written against the documented API; see DESIGN.md."""
    import manifoldbt as mbt
    from manifoldbt.helpers import Interval

    res = {}
    for key, _ in SCENARIOS:
        bids, asks = [(100.0, 1.0)], [(100.1, 1.0)]
        states = [(int(1.0 * S), list(bids), list(asks))]
        trades = []
        for kind, t, px, q in EVENTS[key]:
            if kind == "bid":
                bids = [(px, q)] if q > 0 else []
                states.append((int(t * S), list(bids), list(asks)))
            elif kind == "ask":
                asks = [(px, q)]
                states.append((int(t * S), list(bids), list(asks)))
            else:
                trades.append((int(t * S), px, q, "sell"))
        states.append((int(6.0 * S), list(bids), list(asks)))     # the run ends at the last state: keep the events inside it
        market = mbt.sim.Market.from_ladders("BTCUSDT", 0.1, states, trades)
        out = {}

        def fn(sim):
            sim.elapse(Interval.millis(100))          # the clock starts at the first state (t = 1.0 s)
            oid = sim.post("BTCUSDT", "buy", 100.0, 0.01, tif="GTX")
            while sim.elapse(Interval.millis(10)):
                o = sim.order(oid)
                if o is not None and o.filled > 0 and "t" not in out:
                    out["t"] = sim.now() / 1e9
                    out["q"] = o.filled

        mbt.sim.run(fn, mbt.sim.Config(queue={"depth_source": "book", "model": "risk_adverse", "assumed_queue": 2.0}),
                    markets=[market])
        res[key] = (round(out["q"], 4), round(out["t"], 2)) if out else None
    return res


def main():
    cols = {"hftbacktest (no partial)": hftbacktest(False), "hftbacktest (partial)": hftbacktest(True),
            "nautilus queue_position=True": nautilus(True), "nautilus queue_position=False": nautilus(False)}
    if "--mbt" in sys.argv:
        try:
            cols["manifoldbt (order API)"] = manifoldbt()
        except Exception as exc:                                # a refusal is a result, not a crash: no column
            print(f"manifoldbt column not produced: {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
    names = list(cols)
    print("| | scenario | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for key, text in SCENARIOS:
        cells = []
        for n in names:
            v = cols[n][key]
            cells.append("not served" if v is None else f"served {v[0]} at t={v[1]}")
        print(f"| {key} | {text} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
