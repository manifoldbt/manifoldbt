"""A one-sided market maker on the trade clock, with a queue, a latency and marks.

A maker does not get filled because the price reached its level. It gets filled
because everything resting in front of it at that level went first, and by then
the price is usually leaving. That sentence is the whole file: every setting
below exists to stop a backtest from booking a half-spread it would never have
captured.

One BTCUSDT day, one simulation row per print, and the seven doors of the tape
layer opened in the order the guide gives them
(`docs/strategy-authoring.md`, "Backtesting on the Tape"):

  1. the tape          `bt.ingest_trades`   what the venue printed
  2. the book          `bt.ingest_book`     what was resting, ten levels a side
  3. the clock         `bar_interval=Interval.trades()`
  4. durations         a flow window in seconds, a quote that lives one second
  5. the queue         `fill_model={"queue": ...}`
  6. the latency       `execution.latency`
  7. the marks         `execution.fill_marks`

The strategy is deliberately plain: rest at the best bid, step back two ticks
per clip already held, and quote only while the last ten seconds of flow were
not one-sided selling. It is a device for reading the counters, not a strategy
to trade. The number to look at is the last one printed: what the quotes
actually captured at the touch, against the drift measured one second later.

Demonstrates:
  - `Interval.trades()`: one simulation row per print
  - `position()` inside an order's level: the quote steps away from inventory
  - `time_in_force=Interval.seconds(1)`: a quote that lives in time, not in rows
  - `fill_model={"queue": ...}`: the volume ahead decides the fill
  - `execution.latency`: the quote is seen late and withdrawn late
  - `result.fill_fragility["queue"]`, `result.order_activity`, `result.fill_marks`
  - `bt.plot.fill_marks(result)`

NEEDS THE RESEARCHER PLAN. Reading a stored tape or a stored book is a
Researcher feature; a Pro licence does not unlock it. Without it the file
exits with the refusal message rather than a traceback; nothing in it is a placeholder.

Data: self-contained (network), and it is not small. Two public Bybit spot
archives for one UTC day are fetched once into `data/tape/`:

  - the trades, `public.bybit.com/spot/BTCUSDT/BTCUSDT_<day>.csv.gz`, a few
    megabytes, stored as one Arrow file under `data/tape/mega/bybit/ticks/`;
  - the 200-level book, `public.bybit.com/.../<day>_BTCUSDT_ob200.data.zip`,
    about 90 MB compressed and several gigabytes inflated. It is streamed, not
    held, and only the top ten levels a side are kept, under
    `data/tape/mega/bybit/book/`. The raw archive is kept in
    `data/tape/archives/` so a second run re-reads it instead of the network.

A re-run touches the network only for what is missing. Real prices: the figures
move with the day.

Usage:
    python examples/28_trade_clock_market_maker.py
    python examples/28_trade_clock_market_maker.py 2025-08-13   # pin a day
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import manifoldbt as mbt
from manifoldbt import position
from manifoldbt.expr import col, lit
from manifoldbt.helpers import Interval, Slippage, time_range

# -- The day and the market ---------------------------------------------------
PROVIDER = "bybit"
CATEGORY = "spot"
SYMBOL = "BTCUSDT"
SYMBOL_ID = 1
LEVELS = 10                     # levels kept per side; the queue reads its own

# The venues keep a rolling archive window of roughly a year, so a hardcoded
# date eventually stops existing and the ingest says so by name. Pass a day on
# the command line to pin a run.
_TODAY = datetime.now(timezone.utc).date()
DAY = sys.argv[1] if len(sys.argv) > 1 else str(_TODAY - timedelta(days=3))
NEXT_DAY = str(datetime.strptime(DAY, "%Y-%m-%d").date() + timedelta(days=1))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(_ROOT, "data", "tape")
METADATA_DB = os.path.join(DATA_ROOT, "metadata.sqlite")
ARCHIVE_DIR = os.path.join(DATA_ROOT, "archives")
TAPE_FILE = os.path.join(DATA_ROOT, "mega", PROVIDER, "ticks", SYMBOL, f"{DAY}.arrow")
# The day of book, stored as deltas (book_delta/), or as whole ladders (book/)
# by an earlier version.
BOOK_FILES = [
    os.path.join(DATA_ROOT, "mega", PROVIDER, shape, SYMBOL, f"{DAY}.arrow")
    for shape in ("book_delta", "book")
]

# -- The quote ----------------------------------------------------------------
# The price grid of the market, and it matters more than it looks. The queue
# reads the depth stored AT THE ORDER'S OWN LEVEL: a level off the grid, or on
# it with nothing resting there, is a level the stored book has no row for. The
# order then joins an `assumed_queue` instead of an observed one, and
# `book_unknown_at_post` counts it. The run below posts about a fifth of its
# orders that way, all of them when the inventory skew has walked the level down
# to a price point nobody was quoting.
#
# So the level is built out of whole ticks only. `fair` is the best bid the
# venue published, which is on the grid by construction; the edge and the
# inventory skew are whole multiples of a tick. The DSL has no `floor`, so
# rounding an arbitrary fair price onto the grid has to happen upstream of the
# engine, or the level has to be one the venue itself published -- which is what
# this file does.
TICK = 0.01
CLIP = 0.01                     # BTC: the unit of inventory the skew is priced per
MAX_INVENTORY = 0.05            # BTC, five clips: the target the orders close towards
EDGE_TICKS = 0                  # how far behind the touch the quote rests
SKEW_TICKS_PER_CLIP = 2         # how much further back, per clip already held

FLOW_WINDOW = Interval.seconds(10)
FLOW_FLOOR = -0.35              # do not quote a bid into a one-sided sell flow
QUOTE_LIFE = Interval.seconds(1)
ROUND_TRIP = Interval.millis(20)


def open_store():
    """One UTC day of tape and one of book, fetched once, reused after that."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    if os.path.exists(TAPE_FILE):
        store = mbt.DataStore(
            data_root=DATA_ROOT,
            metadata_db=METADATA_DB,
            arrow_dir=os.path.join(DATA_ROOT, "mega"),
        )
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
    """Quote behind the touch, step away from inventory, gate on recent flow."""
    # 4. Durations. A window given as an Interval is read on the timestamps, so
    #    it is ten SECONDS of flow whatever number of prints that is. Counted in
    #    rows it would be ten prints, which on this tape is a few milliseconds.
    bought = col("buy_volume").rolling_sum(FLOW_WINDOW)
    sold = col("sell_volume").rolling_sum(FLOW_WINDOW)
    imbalance = (bought - sold) / (bought + sold + lit(1e-12))

    # 2. The book. `bid` is the best bid standing at or before the print, joined
    #    as of it. It is null until the book's first row, and a null level posts
    #    nothing, which is the honest behaviour during the warm-up.
    fair = col("bid")
    # Our own edge behind the touch, in whole ticks. Zero here: the quote rests
    # AT the best bid, the one price the stored ladder is certain to have a row
    # for. Raise it and watch `book_unknown_at_post` climb, because on a 0.01
    # tick the price points between the venue's own levels carry nothing, and a
    # level with no row is a level the book cannot answer for.
    half = lit(EDGE_TICKS * TICK)
    # Price units per unit of inventory. A clip is CLIP BTC, so this is
    # SKEW_TICKS_PER_CLIP ticks of retreat per clip held -- whole ticks, which
    # is what keeps the level on the grid.
    skew = lit(SKEW_TICKS_PER_CLIP * TICK / CLIP)

    return (
        mbt.Strategy.create("tape_maker")
        # `position()` is the signed inventory at the instant the order is
        # posted, not a column: the engine walks the few nodes that read it once
        # per posted order. It is readable in an entry order's LEVEL and nowhere
        # else, so nothing but the order may read this signal.
        .signal("quote", fair - half - skew * position())
        .signal("flow", imbalance)
        .size(mbt.when(imbalance > lit(FLOW_FLOOR), lit(MAX_INVENTORY), lit(0.0)))
        # 4. A quote that lives one SECOND. Expiring is how a maker requotes:
        #    the order is posted again on the next event, at that event's level,
        #    for as long as the target holds. `{"GTB": 1}` here would be one
        #    print, a few milliseconds.
        .limit_entry(signal="quote", time_in_force=QUOTE_LIFE)
        # A maker's exit, in PERCENT: 2 bps of profit, 6 of loss. The
        # take-profit rests, so the queue serves it too -- and because a bracket
        # cannot exit half a position, it only fires once its whole size is
        # served.
        .take_profit(pct=0.02, side="long")
        .stop_loss(pct=0.06, side="long")
        .describe("Rest at the bid, retreat two ticks per clip of inventory")
    )


def build_config():
    start, end = time_range(DAY, NEXT_DAY)
    return mbt.BacktestConfig(
        universe=[SYMBOL_ID],
        time_range_start=start,
        time_range_end=end,
        # 3. The clock: one simulation row per print, nothing aggregated.
        #    `output_resolution` then defaults to one second under this clock,
        #    because a liquid day is millions of events and handing them all
        #    back is a memory bill nobody asked for.
        bar_interval=Interval.trades(),
        initial_capital=100_000.0,
        # Costs are off so the counters read cleanly. Turn them on before
        # believing any P&L: a maker lives or dies on its fee schedule.
        fees=mbt.FeeConfig.zero(),
        slippage=Slippage.none(),
        execution=mbt.ExecutionConfig(
            signal_delay=1,          # decide on a print, act on the next one
            allow_short=False,
            position_sizing_mode="Units",
            max_position_pct=1.0,
            # 5. The queue. The order joins behind the depth the stored book
            #    holds at its own level and instant; a print AT the level burns
            #    that volume first and only the remainder fills. `assumed_queue`
            #    is the declared fallback for an instant the book cannot answer,
            #    in multiples of the order's own size -- without it, such an
            #    instant is refused rather than served from the front.
            fill_model={"queue": {"depth_source": "book", "assumed_queue": 2.0}},
            # 6. The latency. The quote is seen 20 ms after the decision and
            #    withdrawn 20 ms after the cancel, so a requote leaves two orders
            #    live in between and both can fill.
            latency={"order": ROUND_TRIP, "cancel": ROUND_TRIP},
            # 7. The marks. One ordered walk after the loop; the simulation
            #    itself is byte for byte the run without them.
            fill_marks=True,
        ),
    )


def horizon(marks, name):
    for h in marks["horizons"]:
        if h["horizon"] == name:
            return h
    return None


def row(label, value):
    print(f"  {label:<40}{value:>18}")


if __name__ == "__main__":
    try:
        store = open_store()
        result = mbt.run(build_strategy(), build_config(), store)
    except PermissionError as exc:
        raise SystemExit(f"\n  {exc}\n") from None

    activity = result.order_activity
    frag = result.fill_fragility
    queue = frag["queue"]
    marks = result.fill_marks
    served = queue["queue_decided_fills"] + queue["traverse_fills"]

    print(f"\n{SYMBOL} {DAY}, one simulation row per print, "
          f"{result.tape_resolution['bars_on_tape']:,} of them with an order live.")
    print(f"Resting at the best bid, {SKEW_TICKS_PER_CLIP} ticks further back per "
          f"clip held, {CLIP} BTC a clip, one second a quote.\n")

    print("  The quotes")
    row("orders posted", f"{activity['orders_posted']:,}")
    row("of those, requotes", f"{activity['requotes']:,}")
    row("expired unfilled", f"{activity['expired_unfilled']:,}")
    row("service rate", f"{served / max(activity['orders_posted'], 1):.2%}")
    row("live beside a cancelled one", f"{activity['overlapping_live_orders']:,}")
    row("filled after their cancel (stale)", f"{activity['stale_fills']:,}")

    print("\n  How each fill was served")
    row("queue-decided (the volume ahead went)", f"{queue['queue_decided_fills']:,}")
    row("a print through the level", f"{queue['traverse_fills']:,}")
    row("  of those, a book cross", f"{queue['fills_from_book_cross']:,}")
    row("partial fills", f"{queue['partial_fills']:,}")
    row("posted behind an assumed queue", f"{queue['book_unknown_at_post']:,}")
    row("`touch` would have filled", f"{frag['would_fill_touch']:,}")
    row("`traverse` would have filled", f"{frag['would_fill_traverse']:,}")
    print("  The queue decides about one fill in fifteen. The rest is the market")
    print("  walking through the level, the mechanism `traverse` already gave for")
    print("  free -- and the reason `touch` flatters a maker so badly: it grants")
    print("  more than twice the fills, every one of them at the level.")
    print("  The orders posted behind an assumed queue are the ones the inventory")
    print("  skew walked down to a price point nobody was quoting.")

    print("\n  What being quoted cost (basis points, negative is adverse)")
    row("half-spread captured", f"{marks['half_spread_captured_bps']:+.4f}")
    row("the market's own half-spread", f"{marks['book_half_spread_bps']:+.4f}")
    for name in ("100ms", "1s", "10s"):
        h = horizon(marks, name)
        if h:
            row(f"markout at +{name}", f"{h['mean_bps']:+.4f} +/- {h['stderr_bps']:.4f}")
    row("marked fills", f"{marks['fills_total']:,}")

    captured = marks["half_spread_captured_bps"]
    market = marks["book_half_spread_bps"]
    one_sec = horizon(marks, "1s")
    if one_sec is not None and captured is not None:
        print(f"\n  Resting at the touch, the quotes captured no half-spread at all: "
              f"{captured:+.4f} bps,")
        print(f"  against the {market:+.4f} bps the market's own spread was worth at "
              "the same instants.")
        print(f"  One second on, {one_sec['mean_bps']:+.4f} bps. A fill granted by a "
              "print going THROUGH the")
        print("  level is a fill granted because the price was leaving, and nine in "
              "ten of them are.")
        print("  Fees and a fill count never show that. It is what a maker has to "
              "answer for.")

    try:
        from manifoldbt import plot
    except ImportError as exc:
        print(f"\n[chart skipped] {exc}", file=sys.stderr)
    else:
        out = os.path.join(_ROOT, "output", "fill_marks.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot.fill_marks(result, save=out, show=False)
        print(f"\n  Chart: {out}")
