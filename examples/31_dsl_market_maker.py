"""The market maker of example 30, written as quotes in the DSL.

Example 30 hands a Python function a venue and lets it post and cancel orders
itself, one call at a time. This one writes the same maker as two quotes, in
the same DSL as the signals of ``bt.run``, and hands them to ``mbt.run``: the
engine compiles them once and runs the whole day without calling back into
Python. Same day, same venue, same orders: the two runs give the same order
journal, fill for fill.

A quote is three expressions. At each wake-up (every 100 ms here) the engine
evaluates them on what the strategy knows, then acts, quote by quote, in the
order they were declared:

* ``price`` or ``size`` not a number, or ``enabled`` null: hold what is there;
* ``enabled`` false: cancel the live order, if any;
* a live order already at ``price``: keep it, and its place in the queue;
* otherwise: cancel the live order, post a new one at ``price``.

The expressions read the simulation state where example 30 read the ``Sim``:
``mbt.position()`` for ``sim.position``, ``bt.book.bid_price_at(k)`` for
``bids[k - 1][0]`` (the level is 1-based, as in every ``bt.book`` name),
``bt.book.bid_levels()`` for ``len(bids)``. ``mbt.round`` rounds a half to the
even integer, as Python's ``round`` does, so the lean is the same one.

Demonstrates:
  - ``Strategy.quote(side=, price=, size=, tif=, enabled=)``
  - ``bar_interval=Interval.millis(100)``: the wake-up clock of a quoting strategy
  - ``execution.latency`` with its four keys: order, cancel, response, feed
  - ``res.orders_df()``, ``res.fills_df()``, ``res.order_activity``, ``res.fill_marks``

The run starts with 100 000 USDT, where example 30 starts with nothing: the
metrics of ``mbt.run`` are returns, and need a capital to be returns on. The
orders, the fills and the P&L do not depend on it.

Data: the same day and the same files as ``30_order_api_market_maker.py``.

Usage:
    python examples/31_dsl_market_maker.py
    python examples/31_dsl_market_maker.py 2025-08-13   # pin a day
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import manifoldbt as mbt
from manifoldbt import book
from manifoldbt.helpers import Interval, time_range

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
DATA_ROOT = os.path.join(_ROOT, "data", "tape")
METADATA_DB = os.path.join(DATA_ROOT, "metadata.sqlite")
ARCHIVE_DIR = os.path.join(DATA_ROOT, "archives")
TAPE_FILE = os.path.join(DATA_ROOT, "mega", PROVIDER, "ticks", SYMBOL, f"{DAY}.arrow")
BOOK_FILES = [
    os.path.join(DATA_ROOT, "mega", PROVIDER, shape, SYMBOL, f"{DAY}.arrow")
    for shape in ("book_delta", "book")
]

# -- The quotes ---------------------------------------------------------------
CLIP = 0.01             # BTC per quote
MAX_INVENTORY = 0.05    # BTC: no bid above it, no ask below minus it
WAKE = Interval.millis(100)


def build_strategy():
    """One quote a side, stepping back one level per clip of inventory."""
    inventory = mbt.position()
    clips = mbt.round(inventory / CLIP)
    bids, asks = book.bid_levels(), book.ask_levels()
    # With a side of the book empty, both prices are NaN: both quotes hold,
    # as example 30's `continue` holds them.
    book_ok = (bids > 0) & (asks > 0)
    bid = book.bid_price_at(mbt.clip(clips, 0, bids - 1) + 1)
    ask = book.ask_price_at(mbt.clip(-clips, 0, asks - 1) + 1)
    return (
        mbt.Strategy.create("dsl_maker")
        # Post-only: a quote that would execute on arrival is refused, never
        # turned into a taker fill.
        .quote("buy", mbt.when(book_ok, bid), CLIP, tif="GTX",
               enabled=inventory < MAX_INVENTORY - 1e-12)
        .quote("sell", mbt.when(book_ok, ask), CLIP, tif="GTX",
               enabled=inventory > -MAX_INVENTORY + 1e-12)
        .describe("Example 30's maker, as two quotes")
    )


INITIAL_CAPITAL = 100_000.0   # USDT: the metrics are returns on it


def build_config():
    """Example 30's venue: 5 ms each way, 2 ms of feed, the queue behind the
    stored size, a 4 bp taker fee (a post-only maker pays none)."""
    start, end = time_range(DAY, NEXT_DAY)
    return mbt.BacktestConfig(
        universe=[SYMBOL_ID],
        time_range_start=start,
        time_range_end=end,
        bar_interval=WAKE,
        initial_capital=INITIAL_CAPITAL,
        fees=mbt.FeeConfig(maker_fee_bps=0.0, taker_fee_bps=4.0),
        execution=mbt.ExecutionConfig(
            latency={
                "order": Interval.millis(5),     # a post reaches the venue
                "cancel": Interval.millis(5),    # a cancellation does
                "response": Interval.millis(5),  # an answer or a fill comes back
                "feed": Interval.millis(2),      # the book and the prints reach us
            },
            fill_model={"queue": {"depth_source": "book", "assumed_queue": 2.0}},
        ),
    )


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


def main():
    try:
        store = open_store()
        res = mbt.run(build_strategy(), build_config(), store)
    except PermissionError as err:
        print(err)
        return

    orders, fills = res.orders_df(backend="pandas"), res.fills_df(backend="pandas")
    activity = res.order_activity
    print(f"\n{SYMBOL} {DAY}: {len(orders)} orders, {len(fills)} fills")
    print(f"  posted {activity['orders_posted']}, requotes {activity['requotes']}, "
          f"cancelled unfilled {activity['expired_unfilled']}")
    print(f"  final position {res.sim.position[SYMBOL]:+.4f} BTC, "
          f"P&L {res.sim.metrics['pnl']:.2f} after {res.sim.fees:.2f} of fees")
    print(f"  {res.profile['simulation_us'] / 1e6:.2f} s for the day's wake-ups, "
          "without a call to Python")

    if len(fills):
        print("\n  fills by channel:")
        print(fills.groupby("channel")["qty"].agg(["count", "sum"]).to_string())

    marks = res.fill_marks
    print(f"\n  half-spread captured at the touch: {marks['half_spread_captured_bps']:+.3f} bp")
    for h in marks["horizons"]:
        print(f"  markout +{h['horizon']:>5}: {h['mean_bps']:+.3f} bp "
              f"(+/- {h['stderr_bps']:.3f}, {h['marked_fills']} fills)")


if __name__ == "__main__":
    main()
