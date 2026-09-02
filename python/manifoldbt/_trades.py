"""Round trips rebuilt from the fill log: one row per entry-to-flat cycle.

The trades table the engine returns has one row per FILL. This module pairs
those fills into round trips the way ``bt_analytics::build_round_trips`` does
(FIFO on the average entry price, partial closes, direction flips), so that
the count, win rate and expectancy read off the result agree with
``metrics["trade_stats"]`` to the last digit. Lives outside ``plot/`` for the
same reason as ``_convert``: nothing here needs plotly, and ``Result``
imports it without the plotting extra.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from manifoldbt._convert import positions_arrays, trades_arrays

# Trade-log codes, as documented on the site (Result > Trade Log).
SIDE_LONG = 1
SIDE_SHORT = 2

EXIT_REASON_LABELS: Dict[int, str] = {
    0: "signal",
    1: "stop loss",
    2: "take profit",
    3: "trailing stop",
    4: "limit expiry",
    5: "option expiry",
    6: "margin call",
}

# ``exit_reason`` of a round trip still open at the end of the run.
EXIT_REASON_OPEN = -1

_NS_PER_SECOND = 1_000_000_000


def round_trips(result, *, include_open: bool = True) -> Dict[str, np.ndarray]:
    """Pair the fill log of *result* into round trips.

    Returns a dict of equal-length numpy arrays, one entry per round trip in
    exit order (open positions last, in symbol order):

    ``symbol_id`` (uint32), ``entry_timestamp`` / ``exit_timestamp``
    (datetime64[ns]), ``side`` (1 long, 2 short), ``entry_price``,
    ``exit_price``, ``quantity``, ``fees``, ``pnl`` (net of fees, account
    currency), ``return_pct`` (pnl over the notional at entry),
    ``exit_reason`` (int16, the trade-log code of the exit fill, ``-1`` when
    open), ``holding_seconds``, ``is_open`` (bool), ``entry_row`` /
    ``exit_row`` (row index into ``result.trades``, ``-1`` when open).

    Matching mirrors the Rust analytics exactly: adding to a position
    averages the entry price and accumulates fees; reducing it closes the
    reduced quantity against that average, charging the accumulated entry
    fees plus the exit fill's fees; a partial close resets the entry fees
    (they were attributed to the closed part); a flip closes the whole
    position and opens the remainder at the flip price with zero fees.

    A position still open at the last bar is reported with ``is_open=True``,
    marked to that bar's close and stamped with its timestamp, unless
    ``include_open=False``. The engine's own ``trade_stats`` never counts
    open positions, so compare against ``include_open=False``.
    """
    ta = trades_arrays(result)
    n = len(ta.get("symbol_id", ()))

    sym = np.asarray(ta["symbol_id"], dtype=np.uint32) if n else np.zeros(0, np.uint32)
    exec_ts = ta["execution_timestamp"].astype("datetime64[ns]") if n else np.zeros(0, "datetime64[ns]")
    exec_ns = exec_ts.view(np.int64)
    side = np.asarray(ta["side"], dtype=np.int64) if n else np.zeros(0, np.int64)
    qty = np.asarray(ta["quantity"], dtype=np.float64) if n else np.zeros(0)
    price = np.asarray(ta["fill_price"], dtype=np.float64) if n else np.zeros(0)
    fees = np.asarray(ta["fees"], dtype=np.float64) if n else np.zeros(0)
    reason = np.asarray(ta["exit_reason"], dtype=np.int64) if n else np.zeros(0, np.int64)

    # Per-symbol open position: [entry_ns, side, avg_price, quantity, fees, entry_row]
    open_pos: Dict[int, list] = {}
    out: list = []

    def _close(o, s, i, close_qty, fill_price, fill_fees):
        if o[1] == SIDE_LONG:
            pnl = (fill_price - o[2]) * close_qty - o[4] - fill_fees
        else:
            pnl = (o[2] - fill_price) * close_qty - o[4] - fill_fees
        out.append((
            s, o[0], exec_ns[i], o[1], o[2], fill_price, close_qty,
            o[4] + fill_fees, pnl, reason[i], False, o[5], i,
        ))

    for i in range(n):
        s = int(sym[i])
        signed = qty[i] if side[i] == SIDE_LONG else -qty[i]
        o = open_pos.get(s)
        if o is None:
            open_pos[s] = [exec_ns[i], int(side[i]), price[i], qty[i], fees[i], i]
            continue

        old_pos = o[3] if o[1] == SIDE_LONG else -o[3]
        new_pos = old_pos + signed

        # Same direction: average in.
        if (old_pos > 0.0 and new_pos > old_pos) or (old_pos < 0.0 and new_pos < old_pos):
            total_cost = o[2] * o[3] + price[i] * qty[i]
            o[3] = abs(new_pos)
            o[2] = total_cost / o[3]
            o[4] += fees[i]
            continue

        # Reduced, closed, or flipped.
        close_qty = min(qty[i], o[3])
        _close(o, s, i, close_qty, price[i], fees[i])

        if abs(new_pos) < 1e-12:
            del open_pos[s]
        elif (old_pos > 0.0 > new_pos) or (old_pos < 0.0 < new_pos):
            new_side = SIDE_LONG if new_pos > 0.0 else SIDE_SHORT
            open_pos[s] = [exec_ns[i], new_side, price[i], abs(new_pos), 0.0, i]
        else:
            o[3] = abs(new_pos)
            o[4] = 0.0

    if include_open and open_pos:
        pa = positions_arrays(result)
        p_sym = np.asarray(pa["symbol_id"])
        p_ts = pa["timestamp"].astype("datetime64[ns]").view(np.int64)
        p_close = np.asarray(pa["close"], dtype=np.float64)
        for s in sorted(open_pos):
            o = open_pos[s]
            rows = np.flatnonzero(p_sym == s)
            if len(rows) == 0:
                continue
            last = rows[-1]
            mark, mark_ns = p_close[last], p_ts[last]
            if o[1] == SIDE_LONG:
                pnl = (mark - o[2]) * o[3] - o[4]
            else:
                pnl = (o[2] - mark) * o[3] - o[4]
            out.append((
                s, o[0], mark_ns, o[1], o[2], mark, o[3], o[4], pnl,
                EXIT_REASON_OPEN, True, o[5], -1,
            ))

    m = len(out)
    cols = list(zip(*out)) if m else [()] * 13
    entry_ns = np.asarray(cols[1], dtype=np.int64)
    exit_ns = np.asarray(cols[2], dtype=np.int64)
    entry_price = np.asarray(cols[4], dtype=np.float64)
    quantity = np.asarray(cols[6], dtype=np.float64)
    pnl = np.asarray(cols[8], dtype=np.float64)
    notional = entry_price * quantity
    with np.errstate(divide="ignore", invalid="ignore"):
        return_pct = np.where(notional > 0.0, pnl / notional, 0.0)

    return {
        "symbol_id": np.asarray(cols[0], dtype=np.uint32),
        "entry_timestamp": entry_ns.view("datetime64[ns]"),
        "exit_timestamp": exit_ns.view("datetime64[ns]"),
        "side": np.asarray(cols[3], dtype=np.uint8),
        "entry_price": entry_price,
        "exit_price": np.asarray(cols[5], dtype=np.float64),
        "quantity": quantity,
        "fees": np.asarray(cols[7], dtype=np.float64),
        "pnl": pnl,
        "return_pct": return_pct,
        "exit_reason": np.asarray(cols[9], dtype=np.int16),
        "holding_seconds": (exit_ns - entry_ns) / _NS_PER_SECOND,
        "is_open": np.asarray(cols[10], dtype=bool),
        "entry_row": np.asarray(cols[11], dtype=np.int64),
        "exit_row": np.asarray(cols[12], dtype=np.int64),
    }
