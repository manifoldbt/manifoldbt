"""A two-sided market maker written against the order API (`bt.sim`).

`bt.run` asks a strategy for a target position per row; `bt.sim` hands it a
venue. The function below wakes every hundred milliseconds, reads the book as it
reaches it, keeps one quote on each side, leaning against its inventory, and
cancels and reposts a quote when its price is no longer the one it wants. Every
order it sends reaches the venue after a latency, every answer comes back after
another, and the market itself is seen late: the four delays between a maker
and its venue.

Each resting quote is served by the same queue model as `bt.run`'s
`fill_model={"queue": ...}`: behind the size the book showed at its price when
it arrived, which only prints at that price burn down. A quote at the touch is
not filled because the price came to it, but because everything in front of it
went first.

The strategy is deliberately plain. The number to read is the last one
printed: what the quotes captured at the touch against how far the market moved
away from them one second later.

Demonstrates:
  - `bt.sim.run(strategy, config, store, symbols, start, end)`
  - `sim.elapse`, `sim.book`, `sim.orders`, `sim.post`, `sim.cancel`
  - `bt.sim.Config(latency=...)`: entry, cancel, response and feed
  - `res.metrics`, `res.fills_df()`, `res.fill_marks`

Data: self-contained (network), the same day as `28_trade_clock_market_maker.py`
and fetched the same way: the trades (a few megabytes) and the 200-level book
(about 90 MB compressed, kept to ten levels a side), once, into `data/tape/`.

Usage:
    python examples/30_order_api_market_maker.py
    python examples/30_order_api_market_maker.py 2025-08-13   # pin a day
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

import manifoldbt as mbt
from manifoldbt.helpers import Interval

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
# The day of book, stored as deltas (book_delta/), or as whole ladders (book/)
# by an earlier version.
BOOK_FILES = [
    os.path.join(DATA_ROOT, "mega", PROVIDER, shape, SYMBOL, f"{DAY}.arrow")
    for shape in ("book_delta", "book")
]

# -- The quotes ---------------------------------------------------------------
# The lean is counted in LEVELS of the book, not in ticks: long two clips, the
# bid rests on the third level down and the ask stays at the touch. Every quote
# is then a price the venue itself published, which is what gives it a queue:
# a price between the venue's levels has nothing stored in front of it.
CLIP = 0.01             # BTC per quote
MAX_INVENTORY = 0.05    # BTC: no bid above it, no ask below minus it
WAKE = Interval.millis(100)
LIVE = ("in_flight", "resting", "partially_filled")   # may still fill

# -- The venue ----------------------------------------------------------------
CONFIG = mbt.sim.Config(
    latency={
        "entry": Interval.millis(5),     # a post or a replacement reaches the venue
        "cancel": Interval.millis(5),    # a cancellation does
        "response": Interval.millis(5),  # an acknowledgement or a fill comes back
        "feed": Interval.millis(2),      # the book and the prints reach us
    },
    # Behind the size stored at the price; the fallback only for a price
    # deeper than the ten stored levels, in multiples of the quote's size.
    queue={"depth_source": "book", "assumed_queue": 2.0},
    maker_fee_bps=0.0,
    taker_fee_bps=4.0,
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


def maker(sim):
    """Keep one quote a side, stepping back one level per clip of inventory."""
    quotes = {"buy": None, "sell": None}   # side -> order id
    while sim.elapse(WAKE):
        book = sim.book(SYMBOL)
        bids, asks = book.levels()
        if not bids or not asks:
            continue
        inventory = sim.position(SYMBOL)
        clips = round(inventory / CLIP)
        bid_level = min(max(clips, 0), len(bids) - 1)
        ask_level = min(max(-clips, 0), len(asks) - 1)
        wanted = {
            "buy": bids[bid_level][0] if inventory < MAX_INVENTORY - 1e-12 else None,
            "sell": asks[ask_level][0] if inventory > -MAX_INVENTORY + 1e-12 else None,
        }
        for side, price in wanted.items():
            oid = quotes[side]
            order = sim.order(oid) if oid is not None else None
            live = order is not None and order.status in LIVE
            if live and price is not None and math.isclose(order.price, price):
                continue            # keep the place in the queue
            if live and not order.cancel_pending:
                sim.cancel(order.id)
            quotes[side] = None
            if price is not None:
                # Post-only: a quote that would execute on arrival is refused,
                # never turned into a taker fill.
                quotes[side] = sim.post(SYMBOL, side, price, CLIP, tif="GTX")


def main():
    try:
        store = open_store()
        res = mbt.sim.run(maker, CONFIG, store, symbols=[SYMBOL], start=DAY, end=NEXT_DAY)
    except PermissionError as err:
        print(err)
        return

    m = res.metrics
    print(f"\n{SYMBOL} {DAY}: {res.summary()}")
    print(f"  final position {m['position'][SYMBOL]:+.4f} BTC, "
          f"max drawdown {m['max_drawdown']:.2f}")

    fills = res.fills_df(backend="pandas")
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
