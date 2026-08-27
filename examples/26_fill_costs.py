"""Fills - the same signal, filled four ways.

A backtest's return is half signal, half fill. This runs one ordinary strategy
(buy the dip in an uptrend, leave on a bracket) and changes nothing but how the
entry is filled, so the cost of each execution choice is visible side by side:

    market          take the close, cross the book, pay taker + slippage
    limit -25bps    rest below the market, pay maker, and sometimes miss
    limit -75bps    rest deeper: a better price, filled far less often
    stop +25bps     chase the breakout, pay taker, and gap through on a jump

Then two things worth knowing about a resting order, once you use one:

  * **Unfilled is a result.** A limit that never trades is not a neutral
    outcome, it is a trade you did not take. The engine says how many expired.
  * **Touched is not traded.** When a bar's low stops exactly on your bid, the
    queue ahead of you may have absorbed everything. `fill_fragility` counts
    those fills, and `passive_fill="traverse"` prices them out.

Demonstrates:
  - market, limit and stop entries priced from the same signal
  - maker versus taker: fees paid, slippage paid, fill rate
  - `result.fill_fragility` and the `passive_fill` convention
  - why a resting order needs `signal_delay >= 1`

Data: shared store - real market data from `data/` (see examples/README.md)

Usage:
    python examples/26_fill_costs.py
"""

import copy

import manifoldbt as mbt
from manifoldbt.helpers import Interval, Slippage, time_range
from manifoldbt.indicators import close, ema, rsi

from _bootstrap import open_store

# -- One strategy, four ways to fill it ---------------------------------------
# Long while the trend holds and the pullback is deep enough, out on a bracket.
uptrend = close > ema(close, 200)
dip = rsi(close, 14) < 40


def strategy(entry, name):
    s = (
        mbt.Strategy.create(name)
        .signal("trend", ema(close, 200))
        .signal("rsi", rsi(close, 14))
        .size(mbt.when(uptrend & dip, 1.0, 0.0))
        .take_profit(4.0)
        .stop_loss(3.0)
        .describe("Buy the dip in an uptrend, leave on the bracket")
    )
    return entry(s) if entry else s


ENTRIES = {
    # No entry order at all: a market fill at the execution bar's close.
    "market": None,
    # Passive: rest 25 bps below the close, cancelled if unfilled after 6 bars.
    "limit -25bps": lambda s: s.limit_entry(offset_bps=25, time_in_force={"GTB": 6}),
    # Deeper: a better price when it fills, and it fills far less often.
    "limit -75bps": lambda s: s.limit_entry(offset_bps=75, time_in_force={"GTB": 6}),
    # Breakout: crosses the book, and a bar that gaps through fills at the open.
    "stop +25bps": lambda s: s.stop_entry(offset_bps=-25, time_in_force={"GTB": 6}),
}

# -- Config -------------------------------------------------------------------
# signal_delay=1 is not decoration here: a resting order placed from bar t-1 is
# gated against bar t, so with delay 0 its level would be read from the very bar
# whose high and low decide the fill. The engine now refuses that outright.
start, end = time_range("2023-01-01", "2025-01-01")

config = mbt.BacktestConfig(
    universe={"binance": ["BTC-USDT:perp"]},
    time_range_start=start,
    time_range_end=end,
    bar_interval=Interval.hours(1),
    initial_capital=10_000,
    fees=mbt.FeeConfig.binance_perps(),
    slippage=Slippage.fixed_bps(2),
    warmup_bars=210,
    execution=mbt.ExecutionConfig(
        signal_delay=1,
        max_position_pct=1.0,
        allow_short=False,
        position_sizing_mode="FractionOfEquity",
    ),
)


def expired(result):
    """Entry orders that ran out of time before the price came to them."""
    for w in result.warnings:
        if "expired unfilled" in w:
            return int(w.split()[0])
    return 0


if __name__ == "__main__":
    store = open_store()

    print("Buy the dip in an uptrend, BTC-USDT perp 1h, 2023-2025")
    print("One signal, four ways to fill it.\n")
    print(f"  {'entry':<15}{'trades':>8}{'unfilled':>10}{'fees':>9}"
          f"{'slippage':>10}{'return':>10}{'sharpe':>8}")
    print("  " + "-" * 70)

    results = {}
    for label, entry in ENTRIES.items():
        r = mbt.run(strategy(entry, label.replace(" ", "_")), config, store)
        results[label] = r
        m, ts = r.metrics, r.metrics["trade_stats"]
        # The Arrow column rather than `trades_df()["slippage"]`: that one goes
        # through pandas, which is not a dependency of the engine.
        slip = float(sum(r.trades.column("slippage").to_pylist()))
        print(f"  {label:<15}{r.trade_count:>8}{expired(r):>10}"
              f"{ts['total_fees']:>9.0f}{slip:>10.0f}"
              f"{m['total_return']:>9.1%}{m['sharpe']:>8.2f}")

    print("\n  A resting order trades a worse fill rate for a better price and")
    print("  maker fees. Deeper is not automatically better: the bids that never")
    print("  fill are the pullbacks that kept going, so missing them is not free.")

    # -- What the passive fills rest on ---------------------------------------
    # A maker fill the bar merely touched is one the queue may have absorbed.
    # On BTC that is vanishingly rare (a 0.01 tick against a $40k price), so the
    # count is the reassuring kind of zero; on a cheap alt it is a few percent.
    print("\n  Passive fills, and how many only touched their level:\n")
    print(f"  {'entry':<15}{'maker fills':>13}{'touch-only':>12}{'traverse':>11}")
    print("  " + "-" * 51)
    for label in ("limit -25bps", "limit -75bps"):
        r = results[label]
        f = r.fill_fragility or {}
        maker, touched = f.get("maker_fills", 0), f.get("touch_only_fills", 0)
        strict_cfg = copy.deepcopy(config)
        strict_cfg.execution.fill_model = {"passive_fill": "traverse"}
        strict = mbt.run(strategy(ENTRIES[label], "strict"), strict_cfg, store)
        print(f"  {label:<15}{maker:>13}{touched:>12}"
              f"{strict.metrics['total_return']:>10.1%}")

    print("\n  `traverse` books only the fills the bar traded THROUGH. When the")
    print("  two columns agree, the result does not rest on an assumption about")
    print("  the queue; when they diverge, it does.")
