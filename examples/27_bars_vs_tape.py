"""Bars and tape - the exit a candle has to guess, and the trades that settle it.

A bracket puts two levels around a position. When one bar's range covers both,
the OHLC cannot say which one traded first: high and low carry no order. The
engine has to assume, and it assumes the stop, because that is the conservative
reading. Assuming is not knowing, and this file measures the gap.

`fill_resolution="ticks"` removes the assumption. Level orders - stop-loss,
take-profit, trailing stop, limit and stop entries - are then resolved against
the individual trades inside each bar, in time order: the level the market
actually printed first is the one that fills, and a stop that gaps fills at the
trade that crossed it. Market orders are unchanged.

Two runs, one per resolution. Same store, same strategy, same bars; the only
difference is the config field. What separates them is the phantom return, the
part of the bar run's result that the trades themselves never gave.

`result.tape_resolution` says how much of the run the tape actually decided:

    bars_on_tape           bars whose level orders were resolved on trades
    bars_fallen_back       bars with no trade in their window (bar rule used)
    contested_bars         bars where the bar rule saw BOTH levels at once
    contested_tp_first     of those, the tape found the take-profit first
    contested_stop_first   of those, the tape found the stop or trailing first
    entries_on_tape        entry orders triggered on a trade

It is `None` in bar mode, which the file prints too: that is how you tell a run
that read the tape from one that could not.

Demonstrates:
  - `fill_resolution="ticks"`: level orders resolved on the bar's own trades
  - `result.tape_resolution`: what the tape decided, and what fell back
  - the return a bar backtest books on the bars it had to guess at

NOT RUNNABLE YET. Reading a stored tape is not unlocked by any licence sold
today, a Pro one included: the code ships ahead of its availability, so this
file is here to be READ. On the gate it exits with the refusal message rather
than a traceback; nothing in it is a placeholder, and the numbers it prints are
the ones it will print when the layer opens.

The strategy is a device, not a claim: fees are zeroed and the bracket is a few
basis points wide, which is what makes single bars cover both levels often
enough to be countable. Read the SIGN of the gap, not its size.

Data: self-contained (network) - one UTC day of a symbol's trades from a public
venue archive, fetched once into `data/tape/`, plus the 1-minute bars rebuilt
from that same tape. The bars ARE the trades, summarised, so the two series
cannot disagree. Real prices: the figures move with the day and the venue.

Usage:
    python examples/27_bars_vs_tape.py
"""

import os
from collections import Counter
from datetime import datetime, timedelta, timezone

import manifoldbt as mbt
from manifoldbt.helpers import Interval, Slippage, time_range
from manifoldbt.indicators import close, sma

# -- The data -----------------------------------------------------------------
# Any venue with a public trade archive will do. The feature reads whatever tape
# the store holds and does not care where it came from; these three constants
# are this file's choice, not the design.
PROVIDER = "bybit"
SYMBOL = "BTCUSDT"
SYMBOL_ID = 1

# The venues keep a rolling archive window, so a hardcoded date eventually stops
# existing and the ingest says so by name. Set DAY to a fixed "YYYY-MM-DD" to
# pin a run.
_TODAY = datetime.now(timezone.utc).date()
DAY = str(_TODAY - timedelta(days=3))
NEXT_DAY = str(_TODAY - timedelta(days=2))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(_ROOT, "data", "tape")
METADATA_DB = os.path.join(DATA_ROOT, "metadata.sqlite")
# A stored tape is one Arrow file per UTC day under `{provider}/ticks/{symbol}/`
# (see `help(mbt.ingest_trades)`). Its presence is what tells this file the day
# has already been fetched.
TAPE_FILE = os.path.join(
    DATA_ROOT, "mega", PROVIDER.lower(), "ticks", SYMBOL, f"{DAY}.arrow"
)

BAR = "1m"
SL_PCT = TP_PCT = 0.05          # symmetric, so neither level is favoured


def open_store():
    """One day of tape, plus the bars rebuilt from that same tape.

    Fetched once; a later run reuses what is on disk. Rebuilding the bars from
    the tape rather than taking the venue's own means every bar is covered, so
    `bars_fallen_back` should come back at zero.
    """
    if os.path.exists(TAPE_FILE):
        store = mbt.DataStore(data_root=DATA_ROOT, metadata_db=METADATA_DB)
    else:
        print(f"Fetching {SYMBOL} trades for {DAY} from {PROVIDER} (once)...")
        store = mbt.ingest_trades(
            PROVIDER, SYMBOL, symbol_id=SYMBOL_ID, start=DAY, end=DAY,
            data_root=DATA_ROOT, metadata_db=METADATA_DB,
        )
    built = mbt.bars_from_trades(store, SYMBOL, DAY, NEXT_DAY, interval=BAR)
    return store, built


# -- The strategy: an ordinary bracket, nothing exotic ------------------------
strategy = (
    mbt.Strategy.create("bracketed")
    .signal("fast", sma(close, 5))
    .signal("slow", sma(close, 30))
    .size(mbt.when(sma(close, 5) > sma(close, 30), 1.0, 0.0))
    .stop_loss(SL_PCT)
    .take_profit(TP_PCT)
    .describe("Long the crossover, leave on whichever bracket level trades first")
)


def config_for(fill_resolution):
    """The same config twice, one field apart."""
    start, end = time_range(DAY, NEXT_DAY)
    return mbt.BacktestConfig(
        universe=[SYMBOL_ID],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000,
        fees=mbt.FeeConfig.zero(),      # costs would only blur the comparison
        slippage=Slippage.none(),
        warmup_bars=40,
        execution=mbt.ExecutionConfig(
            signal_delay=1,
            max_position_pct=1.0,
            allow_short=False,
            position_sizing_mode="FractionOfEquity",
            fill_resolution=fill_resolution,
        ),
    )


REASON = {1: "stop", 2: "target", 3: "trailing"}


def bracket_exits(result):
    """Exits by reason, read off the Arrow trade log (no pandas needed)."""
    codes = result.trades.column("exit_reason").to_pylist()
    return Counter(REASON[c] for c in codes if c in REASON)


def coverage_warning(result):
    """The one warning the engine raises when a bar had no trade to read."""
    for w in result.warnings:
        if "no tape coverage" in w:
            return w
    return None


def row(label, left, right):
    print(f"  {label:<34}{left:>13}{right:>13}")


if __name__ == "__main__":
    try:
        store, built = open_store()
        bar_run = mbt.run(strategy, config_for("bar"), store)
        tape_run = mbt.run(strategy, config_for("ticks"), store)
    except PermissionError as exc:
        raise SystemExit(f"\n  {exc}\n") from None

    print(f"\n{SYMBOL} {DAY}, {built['bars']:,} bars of {BAR} rebuilt from the "
          f"day's own trades.")
    print(f"Bracket -{SL_PCT}% / +{TP_PCT}%, symmetric, fees and slippage off.\n")

    bar_exits, tape_exits = bracket_exits(bar_run), bracket_exits(tape_run)
    tape = tape_run.tape_resolution

    row("", "bar rule", "on the tape")
    print("  " + "-" * 60)
    row("total return",
        f"{bar_run.metrics['total_return']:+.2%}",
        f"{tape_run.metrics['total_return']:+.2%}")
    row("exits on the stop", bar_exits["stop"], tape_exits["stop"])
    row("exits on the target", bar_exits["target"], tape_exits["target"])
    row("tape_resolution", str(bar_run.tape_resolution), "(below)")
    print()
    row("bars resolved on trades", "-", f"{tape['bars_on_tape']:,}")
    row("bars fallen back to the bar rule", "-", f"{tape['bars_fallen_back']:,}")
    row("bars covering BOTH levels", "-", f"{tape['contested_bars']:,}")
    row("  the target traded first", "-", f"{tape['contested_tp_first']:,}")
    row("  the stop traded first", "-", f"{tape['contested_stop_first']:,}")
    row("entries triggered on a trade", "-", f"{tape['entries_on_tape']:,}")

    phantom = bar_run.metrics["total_return"] - tape_run.metrics["total_return"]
    print(f"\n  Phantom return: {phantom:+.2%}")
    print(f"  The bar run minus the tape run, over {tape['contested_bars']} "
          "bars where the bar rule had to")
    print("  choose. It books the stop every time; on "
          f"{tape['contested_tp_first']} of them the target had already")
    print("  traded, so the loss went into the result and never into the trades.")

    missing = coverage_warning(tape_run)
    if missing:
        print(f"\n  [{missing}]")
        print("  Those bars kept the bar rule: the feature degrades, it does "
              "not refuse.")

    print("\n  The bars are not wrong, they are incomplete: a candle records")
    print("  where price went, never in what order. On a strategy whose exits")
    print("  sit inside single bars, that missing order IS the result.")
