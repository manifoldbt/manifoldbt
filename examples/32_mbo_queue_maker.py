"""A maker in its exact place in the queue: a day stored order by order.

Examples 30 and 31 read a book stored by price: the queue in front of a quote
is an estimate, because a level-2 book shows a level shrinking, never whose
order left. A feed stored order by order says it. This example ingests a day
of Nasdaq TotalView-ITCH 5.0, then runs one maker twice on it:

* under the exact queue (the default on such a market): the quote waits
  behind the orders that reached its price before it, and is served when the
  market executes one that arrived after it;
* under ``risk_adverse`` reading the same book by price, the model of example
  31, to see what the estimate misses.

The maker joins the best bid and the best ask with one round lot each, and
withdraws a side while more than ``CROWDED`` shares stand ahead of its quote:
``mbt.queue_ahead(side)`` is, on this market, the volume ahead exactly, as the
feed shows it at the strategy's clock.

Data: a day of Nasdaq TotalView-ITCH 5.0, the file Nasdaq publishes as a
sample (``S120925-v50.txt.gz``, 7.9 GB) or a stream of its messages for some
symbols. Ingested once into ``data/mbo``.

Usage:
    python examples/32_mbo_queue_maker.py <itch file> [SYMBOL] [YYYY-MM-DD]
"""

import os
import sys

import manifoldbt as mbt
from manifoldbt import book
from manifoldbt.helpers import Interval

ITCH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ITCH_FILE", "")
SYMBOL = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
DAY = sys.argv[3] if len(sys.argv) > 3 else "2025-12-09"
SYMBOL_ID = 1

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(_ROOT, "data", "mbo")
METADATA_DB = os.path.join(DATA_ROOT, "metadata.sqlite")

LOT = 100.0            # shares per quote
MAX_INVENTORY = 500.0  # shares: no bid above it, no offer below minus it
CROWDED = 20_000.0     # shares ahead of a quote past which its side stands aside
WAKE = Interval.millis(100)


def build_strategy():
    inventory = mbt.position()
    bids, asks = book.bid_levels(), book.ask_levels()
    book_ok = (bids > 0) & (asks > 0)
    # NaN while no quote is live: the comparison is then false, the side quotes.
    crowded_bid = mbt.queue_ahead("bid") > CROWDED
    crowded_ask = mbt.queue_ahead("ask") > CROWDED
    return (
        mbt.Strategy.create("mbo_queue_maker")
        .quote("buy", mbt.when(book_ok, book.bid_price_at(1)), LOT, tif="GTX",
               enabled=(inventory < MAX_INVENTORY) & ~crowded_bid)
        .quote("sell", mbt.when(book_ok, book.ask_price_at(1)), LOT, tif="GTX",
               enabled=(inventory > -MAX_INVENTORY) & ~crowded_ask)
        .describe("Join the best, stand aside behind a crowded queue")
    )


def build_config(queue=None):
    """The regular session, 1 ms each way to the venue and of feed."""
    import datetime as dt

    midnight = dt.datetime.fromisoformat(DAY).replace(tzinfo=dt.timezone.utc)
    midnight_ns = int(midnight.timestamp()) * 1_000_000_000
    # 09:30 to 16:00 in New York.
    offset = _new_york_offset_hours(DAY) * 3_600_000_000_000
    open_ns = midnight_ns + offset + (9 * 60 + 30) * 60_000_000_000
    close_ns = midnight_ns + offset + 16 * 60 * 60_000_000_000
    execution = dict(
        latency={
            "order": Interval.millis(1),
            "cancel": Interval.millis(1),
            "response": Interval.millis(1),
            "feed": Interval.millis(1),
        },
    )
    if queue is not None:
        execution["fill_model"] = {"queue": queue}
    return mbt.BacktestConfig(
        universe=[SYMBOL_ID],
        time_range_start=open_ns,
        time_range_end=close_ns,
        bar_interval=WAKE,
        initial_capital=1_000_000.0,
        execution=mbt.ExecutionConfig(**execution),
    )


def _new_york_offset_hours(day):
    """Hours New York is behind UTC on `day` (summer time since 2007: second
    Sunday of March to first Sunday of November)."""
    import datetime as dt

    d = dt.date.fromisoformat(day)

    def nth_sunday(month, n):
        first = dt.date(d.year, month, 1)
        return first + dt.timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))

    return 4 if nth_sunday(3, 2) <= d < nth_sunday(11, 1) else 5


def open_store():
    day_file = os.path.join(DATA_ROOT, "mega", "nasdaq", "mbo", SYMBOL, f"{DAY}.arrow")
    if os.path.exists(day_file):
        return mbt.DataStore(data_root=DATA_ROOT, metadata_db=METADATA_DB,
                             arrow_dir=os.path.join(DATA_ROOT, "mega"))
    if not ITCH:
        sys.exit(__doc__)
    print(f"Ingesting {SYMBOL} from {ITCH} (once)...")
    return mbt.ingest_mbo(ITCH, {SYMBOL: SYMBOL_ID}, DAY,
                          data_root=DATA_ROOT, metadata_db=METADATA_DB)


def report(name, res):
    fills = res.fills_df(backend="pandas")
    m = res.sim.metrics
    print(f"\n{name}: {len(res.orders_df(backend='pandas'))} orders, {m['fills']} fills, "
          f"P&L {m['pnl']:.2f} USD, final position {m['position'][SYMBOL]:+.0f}")
    if len(fills):
        print(fills.groupby("channel")["qty"].agg(["count", "sum"]).to_string())


def main():
    try:
        store = open_store()
        exact = mbt.run(build_strategy(), build_config(), store)
        estimate = mbt.run(build_strategy(), build_config({"model": "risk_adverse"}), store)
    except PermissionError as err:
        print(err)
        return
    report("exact queue", exact)
    report("risk_adverse on the book by price", estimate)


if __name__ == "__main__":
    main()
