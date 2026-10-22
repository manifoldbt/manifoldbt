"""An order-flow taker on the trade clock, and the fee it has to beat.

The other high-frequency archetype. Example 28 rests a quote and waits to be
served; this file crosses the spread when the flow is one-sided and asks the
only question a taker has to answer: is the edge per fill larger than the fee
per fill? Everything below exists to measure that number honestly, not to make
it look good.

One BTCUSDT day, one simulation row per print:

  1. the tape          `bt.ingest_trades`   what the venue printed
  2. the book          `bt.ingest_book`     for the mid the marks are read at
  3. the clock         `bar_interval=Interval.trades()`
  4. a window in time  the last 34 SECONDS of aggressor volume, not 34 rows
  5. the latency       the decision reaches the market 20 ms later
  6. the marks         where the mid went after each fill

The strategy is a flow-continuation rule first measured on one-second bars:
signed aggressor volume over the window, divided by total volume, is the
imbalance; it arms long above +0.9 and short below -0.9, and the side stays
armed until the opposite crossing. A market order at the next print takes the
side. There is no exit rule beyond the opposite signal: the position flips.

The same day is run twice, once with the venue's taker fee and once at zero.
The zero-fee run measures the edge per fill in basis points; the fee is the bar
it has to clear. Printing both, side by side, is the whole point of the file.

Demonstrates:
  - `Interval.trades()`: one simulation row per print
  - `rolling_sum(Interval.seconds(34))`: a window that counts time on an
    irregular grid (on this tape 34 rows would be a few milliseconds)
  - `value_when(...)`: an armed state that persists between crossings
  - `execution_price="AtOpen"` under the trade clock: the price of the next
    print, which is what a market order gets
  - `execution.latency`: the order is seen 20 ms after the decision
  - `result.fill_marks`: the mid 100 ms, 1 s and 10 s after each fill
  - `bt.plot.equity(save=...)`, `bt.plot.fill_marks(save=...)`

NEEDS THE RESEARCHER PLAN. Reading a stored tape or a stored book is a
Researcher feature; a Pro licence does not unlock it. Without it the file
exits with the refusal message rather than a traceback; nothing in it is a placeholder.

Data: self-contained (network), and it is not small. Two public Bybit spot
archives for one UTC day are fetched once into `data/tape/`, exactly as
example 28 does: the trades (a few megabytes) and the 200-level book (about
90 MB compressed, streamed, ten levels a side kept). A re-run touches the
network only for what is missing. Real prices: the figures move with the day.

To reuse a store another script already built (same layout: `meta.sqlite` at
the root, Arrow files under `data/mega/`), point `MANIFOLDBT_TAPE_STORE` at its
directory; nothing is fetched then.

Usage:
    python examples/33_trade_clock_order_flow.py
    python examples/33_trade_clock_order_flow.py 2025-08-13   # pin a day
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

import manifoldbt as mbt
from manifoldbt.expr import col, lit, when
from manifoldbt.helpers import Interval, Slippage, time_range

# -- The day and the market ---------------------------------------------------
PROVIDER = "bybit"
CATEGORY = "spot"
SYMBOL = "BTCUSDT"
SYMBOL_ID = 1
LEVELS = 10

_TODAY = datetime.now(timezone.utc).date()
DAY = sys.argv[1] if len(sys.argv) > 1 else str(_TODAY - timedelta(days=3))
NEXT_DAY = str(datetime.strptime(DAY, "%Y-%m-%d").date() + timedelta(days=1))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REUSED = os.environ.get("MANIFOLDBT_TAPE_STORE")
DATA_ROOT = _REUSED or os.path.join(_ROOT, "data", "tape")
METADATA_DB = os.path.join(DATA_ROOT, "meta.sqlite" if _REUSED else "metadata.sqlite")
ARROW_DIR = os.path.join(DATA_ROOT, "data", "mega") if _REUSED else os.path.join(DATA_ROOT, "mega")
ARCHIVE_DIR = os.path.join(DATA_ROOT, "archives")
TAPE_FILE = os.path.join(ARROW_DIR, PROVIDER, "ticks", SYMBOL, f"{DAY}.arrow")
# The day of book, stored as deltas (book_delta/), or as whole ladders (book/)
# by an earlier version.
BOOK_FILES = [
    os.path.join(ARROW_DIR, PROVIDER, shape, SYMBOL, f"{DAY}.arrow")
    for shape in ("book_delta", "book")
]

# -- The signal ---------------------------------------------------------------
FLOW_WINDOW = Interval.seconds(34)  # a window counted in TIME, not in rows
THRESHOLD = 0.9                     # |imbalance| that arms a side
LATENCY = Interval.millis(20)       # decision to market
TAKER_FEE_BPS = 4.0                 # a spot taker fee; a maker rebate is not on offer here
CAPITAL = 100_000.0


def open_store():
    """One UTC day of tape and book, fetched once, reused after that."""
    if _REUSED:
        return mbt.DataStore(data_root=DATA_ROOT, metadata_db=METADATA_DB, arrow_dir=ARROW_DIR)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    if os.path.exists(TAPE_FILE):
        store = mbt.DataStore(data_root=DATA_ROOT, metadata_db=METADATA_DB, arrow_dir=ARROW_DIR)
    else:
        print(f"Fetching {SYMBOL} trades for {DAY} from {PROVIDER} (once)...")
        store = mbt.ingest_trades(
            PROVIDER, SYMBOL, symbol_id=SYMBOL_ID, start=DAY, end=DAY,
            data_root=DATA_ROOT, metadata_db=METADATA_DB,
        )
    if not any(os.path.exists(p) for p in BOOK_FILES):
        print(f"Fetching {SYMBOL} book for {DAY} (once, ~90 MB compressed)...")
        store = mbt.ingest_book(
            PROVIDER, SYMBOL, symbol_id=SYMBOL_ID, start=DAY, end=DAY,
            levels=LEVELS, category=CATEGORY,
            data_root=DATA_ROOT, metadata_db=METADATA_DB,
            cache_dir=ARCHIVE_DIR,
        )
    return store


def build_strategy():
    """Take the side of a one-sided flow, keep it until the flow turns."""
    # 4. A window in time. `buy_volume` and `sell_volume` are, on the trade
    #    clock, the quantity of each print on the aggressor's side and zero on
    #    the other, so the two sums are the aggressor volume of the last 34
    #    seconds whatever number of prints that is.
    bought = col("buy_volume").rolling_sum(FLOW_WINDOW)
    sold = col("sell_volume").rolling_sum(FLOW_WINDOW)
    imbalance = (bought - sold) / (bought + sold + lit(1e-12))

    # The armed side. `value_when` samples its argument on the rows where the
    # condition is true and holds that value until the next such row: +1 from a
    # crossing above +0.9, -1 from a crossing below -0.9, and nothing in between
    # changes it. Before the first crossing it is null, and a null target posts
    # nothing.
    crossed = (imbalance >= lit(THRESHOLD)) | (imbalance <= lit(-THRESHOLD))
    side = crossed.value_when(when(imbalance >= lit(THRESHOLD), lit(1.0), lit(-1.0)))

    return (
        mbt.Strategy.create("flow_taker")
        .signal("imbalance", imbalance)
        .signal("side", side)
        .size(col("side"))
        .describe("Follow a one-sided flow over 34 seconds, flip on the opposite crossing")
    )


def build_config(taker_fee_bps):
    start, end = time_range(DAY, NEXT_DAY)
    return mbt.BacktestConfig(
        universe=[SYMBOL_ID],
        time_range_start=start,
        time_range_end=end,
        # 3. The clock. One simulation row per print; `output_resolution`
        #    defaults to one second under it.
        bar_interval=Interval.trades(),
        initial_capital=CAPITAL,
        fees=mbt.FeeConfig(maker_fee_bps=0.0, taker_fee_bps=taker_fee_bps,
                           default_fill_type="Taker"),
        slippage=Slippage.none(),
        execution=mbt.ExecutionConfig(
            # Decide on a print, act on the next one: under the trade clock a
            # market order is filled at the price of that next print, which is
            # `AtOpen` in the engine's vocabulary.
            signal_delay=1,
            execution_price="AtOpen",
            allow_short=True,
            position_sizing_mode="FractionOfEquity",
            max_position_pct=1.0,
            # 5. The latency. The decision reaches the market 20 ms later, so
            #    the order is filled by the first print at or after that
            #    instant, at its price, not at the print that triggered it.
            latency={"order": LATENCY, "cancel": LATENCY},
            # 6. The marks. Where the mid went after each fill, against the
            #    fill price, signed in the direction of the fill.
            fill_marks=True,
        ),
    )


def horizon(marks, name):
    for h in marks["horizons"]:
        if h["horizon"] == name:
            return h
    return None


def row(label, value):
    print(f"  {label:<44}{value:>16}")


def run(store, taker_fee_bps):
    result = mbt.run(build_strategy(), build_config(taker_fee_bps), store)
    equity = result.equity_df()["equity"]
    fills = len(result.trades_df())
    multiple = float(equity.iloc[-1]) / CAPITAL
    # Log return per fill in basis points: the number a fee compares to.
    edge_bps = math.log(multiple) / fills * 1e4 if fills else float("nan")
    return result, fills, multiple, edge_bps


if __name__ == "__main__":
    try:
        store = open_store()
        free, fills_free, mult_free, edge_free = run(store, 0.0)
        paid, fills_paid, mult_paid, edge_paid = run(store, TAKER_FEE_BPS)
    except PermissionError as exc:
        raise SystemExit(f"\n  {exc}\n") from None

    stats = free.metrics["trade_stats"]
    marks = free.fill_marks
    positions = free.positions_df()
    exposure = float((positions["position"] != 0).mean()) if len(positions) else float("nan")

    print(f"\n{SYMBOL} {DAY}, one simulation row per print.")
    print(f"Flow over 34 seconds, armed at |imbalance| >= {THRESHOLD}, "
          f"market orders 20 ms after the decision.\n")

    print("  The trading")
    row("fills (each one crosses the spread)", f"{fills_free:,}")
    row("average holding time", f"{stats['avg_holding_seconds']:.1f} s")
    row("time in the market (one-second rows)", f"{exposure:.1%}")

    print("\n  The edge, before any fee (log return per fill)")
    row("edge per fill", f"{edge_free:+.3f} bps")
    row("day multiple", f"x{mult_free:.5f}")

    print(f"\n  The same day at {TAKER_FEE_BPS:.0f} bps taker")
    row("net per fill", f"{edge_paid:+.3f} bps")
    row("day multiple", f"x{mult_paid:.5f}")
    row("fees paid", f"{paid.trades_df()['fees'].sum():,.0f} USD")

    print("\n  Where the mid went after a fill (basis points, negative is adverse)")
    for name in ("100ms", "1s", "10s"):
        h = horizon(marks, name)
        if h:
            row(f"markout at +{name}", f"{h['mean_bps']:+.4f} +/- {h['stderr_bps']:.4f}")
    row("marked fills", f"{marks['fills_total']:,}")

    ten = horizon(marks, "10s")
    print(f"\n  A taker pays the fee on every fill, and a flip crosses the spread twice,")
    print(f"  so a {TAKER_FEE_BPS:.0f} bps fee costs up to {2 * TAKER_FEE_BPS:.0f} bps per fill here. Before fees this")
    print(f"  rule earns {edge_free:+.2f} bps per fill: that is the whole fee it could afford.")
    if ten is not None and ten["mean_bps"] > 0 > edge_free:
        print(f"  The markouts say where it goes: the flow does continue after a fill")
        print(f"  ({ten['mean_bps']:+.2f} bps ten seconds on), but the position is held "
              f"{stats['avg_holding_seconds']:.0f} s on")
        print("  average and gives it back before the opposite crossing. The exit rule,")
        print("  not the entry, is what this measurement points at.")
    elif ten is not None and ten["mean_bps"] <= 0:
        print("  The markouts say why: the mid does not keep moving the fill's way.")
    print("  Nothing here is a strategy to trade. It is the measurement a taker")
    print("  needs before believing a fill count.")

    try:
        from manifoldbt import plot
    except ImportError as exc:
        print(f"\n[charts skipped] {exc}", file=sys.stderr)
    else:
        out_dir = os.path.join(_ROOT, "output")
        os.makedirs(out_dir, exist_ok=True)
        eq_png = os.path.join(out_dir, "order_flow_equity.png")
        marks_png = os.path.join(out_dir, "order_flow_marks.png")
        plot.equity(free, title=f"Order-flow taker, trade clock, {SYMBOL} {DAY}, no fees",
                    figsize=(16, 6), save=eq_png, show=False)
        plot.fill_marks(free, save=marks_png, show=False)
        print(f"\n  Charts: {eq_png}, {marks_png}")
