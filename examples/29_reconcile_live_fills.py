"""Calibrating an execution model against the fills a broker actually granted.

Every execution setting in the tape layer is an assumption. How long was the
queue in front of the order? How fast did the volume ahead of it cancel? How
late did the quote reach the book? A backtest cannot check any of them against
itself: it can only be internally consistent, and a consistent story about a
queue is still a story.

`bt.reconcile` is the check. Give it a finished run and a journal of the fills
a broker really granted over the same days, and it marks both sides with the
same code against the same tape and the same book, pairs them one to one, and
reports four things: what fraction of the posted quotes each side served, how
adverse each side's fills were at +100 ms, +1 s and +10 s, how far apart the
paired fills were in price and in time, and what the quote captured at the
touch. One factual sentence per quantity, and not one recommendation: choosing
a queue depth from those numbers is the author's job, and `execution_grid` on
`run_sweep` is how the choice gets published as a surface rather than made in
private.

It also measures the one cost no simulation can produce. A backtest replays a
tape that never saw your quotes, so its adverse selection is whatever the
market was going to do anyway; a real journal's fills were served by people who
could see the quote and react to it. The PAIRED difference at each horizon is
that reactive part, and nothing else measures it.

THE JOURNAL BELOW IS SYNTHETIC, and the file says so at every step. It is the
run's own fill log, degraded on purpose: forty per cent of the fills dropped at
random, and every survivor priced half a basis point worse. Nothing here claims
to be a broker's data -- the point is to show what the tables read like when
the two sides disagree, with a disagreement whose size is known in advance. The
reconciliation prints it back: -0.500 bp of extra adverse selection at every
horizon, and a service ratio of six fills in ten.

Demonstrates:
  - `bt.reconcile(result, journal, store=..., symbol_id=...)`
  - reconciling a run with ITSELF: zero everywhere, which is the sanity check
    to run before believing anything a real journal says
  - `rec.verdict`, `rec.by_hour_df()`, `rec.horizons_df()`, `rec.pairs_df()`
  - `bt.plot.reconcile(rec)`

NEEDS THE RESEARCHER PLAN. Reading a stored tape or a stored book is a
Researcher feature; a Pro licence does not unlock it. Without it the file
exits with the refusal message rather than a traceback.

Data: self-contained (network). One UTC day of Bybit BTCUSDT spot, fetched once
into `data/tape/` and shared with example 28: the trades (a few megabytes) and
the 200-level book (about 90 MB compressed, streamed, ten levels a side kept).
One-second bars are then rebuilt from the tape with `bars_from_trades`.

Usage:
    python examples/29_reconcile_live_fills.py
    python examples/29_reconcile_live_fills.py 2025-08-13   # pin a day
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import manifoldbt as mbt
from manifoldbt.expr import col, lit
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
CLIP = 0.01                     # BTC
EDGE_BPS = 1.0                  # how far below the close the bid rests
TP_PCT, SL_PCT = 0.02, 0.06     # percent, so 2 bps of profit against 6 of loss

# -- The synthetic journal ----------------------------------------------------
# Written out here rather than hidden in a helper: these two numbers are what
# the reconciliation has to find again, and a reader who cannot see them cannot
# check that it did.
LOST_SHARE = 0.40               # fills the market never granted
WORSE_BPS = 0.5                 # how much worse the survivors were priced
SEED = 20260908                 # so the file prints the same numbers twice


def open_store():
    """One UTC day of tape, book and one-second bars; fetched once."""
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

    # One-second bars rebuilt from the tape: the simulation grid. The prints
    # themselves still settle every level order (`fill_resolution="ticks"`).
    mbt.bars_from_trades(store, SYMBOL, DAY, NEXT_DAY, interval="1s")
    return store


def build_strategy():
    """A plain one-sided maker: rest a basis point under the close, bracket it."""
    return (
        mbt.Strategy.create("maker")
        .signal("level", col("close") * lit(1.0 - EDGE_BPS * 1e-4))
        .size(lit(CLIP))
        # The quote lives one event and is posted again on the next: requoting
        # is cancelling and posting again, at the back of a fresh queue.
        .limit_entry(signal="level", time_in_force={"GTB": 1})
        .take_profit(pct=TP_PCT, side="long")
        .stop_loss(pct=SL_PCT, side="long")
        .describe("Rest a basis point under the close, one clip at a time")
    )


def build_config():
    start, end = time_range(DAY, NEXT_DAY)
    return mbt.BacktestConfig(
        universe=[SYMBOL_ID],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.seconds(1),
        initial_capital=100_000.0,
        warmup_bars=60,
        fees=mbt.FeeConfig.zero(),
        slippage=Slippage.none(),
        execution=mbt.ExecutionConfig(
            signal_delay=1,
            allow_short=False,
            position_sizing_mode="Units",
            # The prints settle the levels, and the queue decides who is served.
            fill_resolution="ticks",
            fill_model={"queue": {"depth_source": "book", "assumed_queue": 2.0}},
            # The marks: what the reconciliation compares, on both sides.
            fill_marks=True,
        ),
    )


def as_journal(trades, *, degrade):
    """The run's fill log as a broker would have written it.

    `degrade=False` returns it unchanged, which is the identity check: a
    reconciliation of a backtest with itself must be zero everywhere, and if it
    is not, nothing a real journal says can be trusted either.

    `degrade=True` throws away `LOST_SHARE` of the fills at random and prices
    every survivor `WORSE_BPS` worse -- a buy pays more, a sell receives less.
    That is the whole synthesis, and it is four lines so a reader can check
    what the reconciliation is being asked to find.
    """
    kept = trades
    if degrade:
        rng = np.random.default_rng(SEED)
        kept = trades.loc[rng.random(len(trades)) >= LOST_SHARE]
    side = kept["side"].to_numpy()
    price = kept["fill_price"].to_numpy()
    if degrade:
        sign = np.where(side == 1, 1.0, -1.0)
        price = price * (1.0 + sign * WORSE_BPS * 1e-4)
    return pd.DataFrame(
        {
            # The four columns a journal has to carry. A broker export usually
            # spells them otherwise (`time`, `quantity`, `avg_price`, `B`/`S`);
            # reconcile reads the common spellings without a rename.
            "timestamp": kept["execution_timestamp"].to_numpy(),
            "side": np.where(side == 1, "buy", "sell"),
            "price": price,
            "qty": kept["quantity"].to_numpy(),
        }
    )


def row(label, value):
    print(f"  {label:<42}{value:>18}")


if __name__ == "__main__":
    try:
        store = open_store()
        result = mbt.run(build_strategy(), build_config(), store)
    except PermissionError as exc:
        raise SystemExit(f"\n  {exc}\n") from None

    trades = result.trades_df()
    posted = result.order_activity["orders_posted"]
    print(f"\n{SYMBOL} {DAY}: {posted:,} quotes posted, {len(trades):,} fills booked.\n")

    # -- 1. The run against ITSELF ------------------------------------------
    # The check to run before believing anything a real journal says: the same
    # fills, marked by the same code over the same tape, must reconcile to
    # nothing at all.
    same = mbt.reconcile(
        result, as_journal(trades, degrade=False),
        store=store, symbol_id=SYMBOL_ID, tolerance=Interval.seconds(1),
    )
    print("A backtest reconciled with itself")
    row("fills paired", f"{same.matched:,} / {same.live_fills:,}")
    row("real fills with no backtest twin", f"{same.live_only:,}")
    row("backtest fills never granted", f"{same.sim_only:,}")
    for h in same.horizons:
        row(f"paired markout difference at {h['horizon']}",
            f"{h['paired_mean_bps']:+.6f} bp")
    row("price gap, mean", f"{same.price_gap_bps['mean']:+.6f} bp")
    print()

    # -- 2. The degraded journal --------------------------------------------
    rec = mbt.reconcile(
        result, as_journal(trades, degrade=True),
        store=store, symbol_id=SYMBOL_ID, tolerance=Interval.seconds(1),
    )
    print(f"A journal that lost {LOST_SHARE:.0%} of the fills and paid "
          f"{WORSE_BPS} bp more for the rest")
    for line in rec.verdict:
        print(f"  {line}")

    print("\nMarkout, both sides, and the paired difference:")
    print(rec.horizons_df()[[
        "horizon", "sim_mean_bps", "sim_stderr_bps",
        "live_mean_bps", "live_stderr_bps",
        "paired_mean_bps", "paired_stderr_bps", "paired_fills",
    ]].to_string(index=False))

    print("\nService by hour of the UTC day (first six):")
    print(rec.by_hour_df().head(6).to_string(index=False))

    # The two numbers the synthesis put in, read back out of the measurement.
    served = rec.live_fills / rec.sim_fills if rec.sim_fills else float("nan")
    print(f"\nInjected: {1 - LOST_SHARE:.2f} of the fills kept, "
          f"{WORSE_BPS} bp worse.")
    print(f"Measured: {served:.3f} of the fills kept, "
          f"{-rec.price_gap_bps['mean']:.3f} bp worse, "
          f"{-rec.horizons[1]['paired_mean_bps']:.3f} bp more adverse at 1 s "
          f"(+/- {rec.horizons[1]['paired_stderr_bps']:.3f}).")

    print("\nWhat this does NOT say: which queue depth or which latency to "
          "set.\nSweep them and read the surface -- see \"Sweeping execution "
          "parameters\"\nin docs/strategy-authoring.md.")

    try:
        mbt.plot.reconcile(rec)
    except ImportError:
        pass
