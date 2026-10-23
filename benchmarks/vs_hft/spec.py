"""The single source of truth for what every engine is asked to do.

Nothing in this file imports an engine. Every adapter (engine_mbt.py,
engine_hft.py, engine_nt.py) reads its numbers from here, so a change of a
latency or of a clip is made once and cannot drift between engines.

Venue: Bybit spot BTCUSDT (tick 0.1 USDT, lot 1e-6 BTC), public archives.
"""
from __future__ import annotations

import datetime as dt

SYMBOL = "BTCUSDT"
SYMBOL_ID = 1
TICK = 0.1
TICK_INV = 10            # 1 / TICK, exact: prices are handled as integer ticks
LOT = 1e-6

MS = 1_000_000           # nanoseconds in a millisecond
SEC = 1_000_000_000

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
# Days are pinned. A day that disappears upstream fails the run loudly (the
# fingerprint below is checked when it is set), it is never replaced silently.
DAYS = {
    # day: (sha256 of the book archive, sha256 of the trades archive)
    "2025-08-13": (
        "63e1f91c9b3e28fd0189a0bf1773eadf0e02cc0d63c1608d4403013da848bc38",
        "5a9277a3c73f93af1811917a29cf723fc34b5d4f1a17868dbd90d5fa26dd671c",
    ),
    "2025-08-23": (
        "989b9e3c69cdba8d7a1157e9a7659f93279f342cf55f1c0eff88ab62eebc20b0",
        "aa5fc8f415e5259e7391d49c9a5225c41507ccdbb5d0a95761f6ebda4ca3f101",
    ),
}
# Nasdaq TotalView-ITCH 5.0 day files (4 to 18 GB). 2025-11-28 is the half session after Thanksgiving.
NASDAQ_DAYS = {
    "2025-11-28": {"file": "S112825-v50.txt.gz", "bytes": 4_735_308_661,
                   "sha256": "e965ea6f9779c1b5d7b74de4549836ffe29a5f6ff64bb7894186a6f6404baaa4"},
}
BOOK_URL ="https://quote-saver.bycsi.com/orderbook/spot/{sym}/{day}_{sym}_ob200.data.zip"
TRADES_URL = "https://public.bybit.com/spot/{sym}/{sym}_{day}.csv.gz"

# Data tiers. "depth" is the number of price levels each side that an engine is
# allowed to see; every engine reads the same prepared stream.
TIERS = {
    "bbo": 1,    # best bid and best ask, plus the trade tape
    "l2": 25,    # 25 levels a side, plus the trade tape
}

# --------------------------------------------------------------------------- #
# Venue model, identical for all engines wherever the engine can express it
# --------------------------------------------------------------------------- #
LAT_ORDER_NS = 5 * MS      # a post reaches the venue
LAT_CANCEL_NS = 5 * MS     # a cancellation does
LAT_RESPONSE_NS = 5 * MS   # an acknowledgement or a fill comes back
LAT_FEED_NS = 2 * MS       # the book and the prints reach the strategy
MAKER_FEE_BPS = 0.0
TAKER_FEE_BPS = 4.0

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
WAKE_NS = 100 * MS         # wake-up period of the quoting strategies


class MMLadder:
    """Example 31, parameterised: two post-only quotes, one level back per clip
    of inventory held. The parameters are the sweep axes."""
    name = "mm_ladder"
    tier = "l2"
    CLIP = 0.01            # BTC per quote
    MAX_CLIPS = 5          # inventory bound, in clips
    OFF_BID = 0            # extra levels back on the bid
    OFF_ASK = 0            # extra levels back on the ask


class MMTouch:
    """Touch maker for a BBO feed: join the best price on both sides, withdraw a
    side when the inventory bound is hit. The only data it needs is the best
    bid and ask."""
    name = "mm_touch"
    tier = "bbo"
    CLIP = 0.01
    MAX_CLIPS = 5


class FlowTaker:
    """Example 33, on a fixed grid instead of the trade clock: aggressor
    imbalance over a trailing window arms a side, a market (IOC) order follows
    the armed side, and the position flips on the opposite crossing."""
    name = "flow_taker"
    tier = "bbo"
    WINDOW_NS = 34 * SEC
    THRESHOLD = 0.9
    QTY = 0.01             # BTC per order
    WAKE_NS = 1 * SEC      # decision grid


# --------------------------------------------------------------------------- #
# Sweep grids (nested: each grid contains the smaller ones)
# --------------------------------------------------------------------------- #
GRID_AXES = {
    4:   dict(clip=[0.01, 0.02], max_clips=[3, 5], off_bid=[0], off_ask=[0]),
    16:  dict(clip=[0.005, 0.01, 0.02, 0.03], max_clips=[3, 5], off_bid=[0, 1], off_ask=[0]),
    64:  dict(clip=[0.005, 0.01, 0.02, 0.03], max_clips=[2, 3, 5, 8], off_bid=[0, 1], off_ask=[0, 1]),
    256: dict(clip=[0.005, 0.01, 0.02, 0.03], max_clips=[2, 3, 5, 8], off_bid=[0, 1, 2, 3],
              off_ask=[0, 1, 2, 3]),
}


def grid_combos(n):
    """The combinations, in the order an engine that sorts parameters by name
    and varies the last fastest would enumerate them."""
    import itertools

    axes = GRID_AXES[n]
    names = sorted(axes)
    return names, [dict(zip(names, v)) for v in itertools.product(*[axes[k] for k in names])]


def day_bounds_ns(day: str):
    d0 = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
    t0 = int(d0.timestamp()) * SEC
    return t0, t0 + 86_400 * SEC


class MBOJoin:
    """Order-by-order market (Nasdaq ITCH): join the best bid and ask with one
    round lot, keep the order while it stays at the best, stand a side aside when
    the displayed size at that best price is above CROWDED or the inventory
    bound is reached."""
    name = "mbo_join"
    tier = "mbo"
    LOT = 100.0            # shares per quote
    MAX_LOTS = 5           # inventory bound, in lots
    CROWDED = 5_000.0      # displayed shares at the best price above which the side stands aside
    TICK = 0.01
    WAKE_NS = 100 * MS
    # regular session of the pinned day (New York): 09:30 to 13:00 (half day)
    DAY = "2025-11-28"
    OPEN_S = 9 * 3600 + 30 * 60
    CLOSE_S = 13 * 3600
    SYMBOL = "AAPL"
