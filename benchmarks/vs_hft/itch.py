"""Nasdaq TotalView-ITCH 5.0: extract the messages of a few symbols from the
day file Nasdaq publishes, and turn them into order-by-order events.

The day file (`S<MMDDYY>-v50.txt.gz`, 4 to 18 GB) is read once, as it inflates.
Every ITCH 5.0 message carries a "stock locate" code in bytes 1-2, so keeping a
symbol is a table lookup per message: no per-order bookkeeping is needed to
filter. The kept messages are written back as they are (2-byte length framing),
which is the extract format `manifoldbt.ingest_mbo` accepts.

`events_from_stream` then replays one symbol's messages through a small state
machine (side, price and remaining size of every live order) and emits one row
per change of the displayed book:

    kind 0 ADD      order enters the book (the new order of a replace too)
    kind 1 EXEC     an execution against the order (E, C): qty executed, at `price`
    kind 2 REDUCE   part of the order is cancelled or executed: new remaining size in `qty`
    kind 3 DELETE   the order leaves the book (D, U, or fully executed / cancelled)
    kind 4 PRINT    a print against an order the book does not show (P)

`price` is an integer in 1e-4 currency units (ITCH's own).
"""
from __future__ import annotations

import os
import struct
import time
import zlib

import numpy as np
from numba import njit, types
from numba.typed import Dict

NS_PER_S = 1_000_000_000

ADD, EXEC, REDUCE, DELETE, PRINT = 0, 1, 2, 3, 4

# message length by type (ITCH 5.0), used only as a sanity check of the framing
LEN = {b"S": 12, b"R": 39, b"H": 25, b"Y": 20, b"L": 26, b"V": 35, b"W": 12, b"K": 28, b"J": 35, b"h": 21,
       b"A": 36, b"F": 40, b"E": 31, b"C": 36, b"X": 23, b"D": 19, b"U": 35, b"P": 44, b"Q": 40, b"B": 19,
       b"I": 50, b"N": 20}


def new_york_offset_hours(day: str) -> int:
    """Hours New York is behind UTC on `day` (US summer time: second Sunday of
    March to first Sunday of November)."""
    import datetime as dt

    d = dt.date.fromisoformat(day)

    def nth_sunday(month, n):
        first = dt.date(d.year, month, 1)
        return first + dt.timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))

    return 4 if nth_sunday(3, 2) <= d < nth_sunday(11, 1) else 5


def midnight_ny_utc_ns(day: str) -> int:
    import datetime as dt

    d0 = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
    return int(d0.timestamp()) * NS_PER_S + new_york_offset_hours(day) * 3600 * NS_PER_S


@njit(cache=True)
def _scan(buf, n, keep_table, out, out_n):
    """Walk the framed messages in buf[:n]. Copy the ones whose locate is in
    keep_table (and every 'S' system event) to `out`. Returns (bytes consumed,
    out length, messages seen)."""
    pos = 0
    seen = 0
    while pos + 2 <= n:
        ln = (np.int64(buf[pos]) << 8) | np.int64(buf[pos + 1])
        if pos + 2 + ln > n:
            break
        t = buf[pos + 2]
        keep = False
        if t == 83:                           # 'S'
            keep = True
        elif ln >= 3:
            loc = (np.int64(buf[pos + 3]) << 8) | np.int64(buf[pos + 4])
            keep = keep_table[loc] != 0
        if keep:
            for i in range(ln + 2):
                out[out_n + i] = buf[pos + i]
            out_n += ln + 2
        pos += 2 + ln
        seen += 1
    return pos, out_n, seen


@njit(cache=True)
def _locates(buf, n, names, found):
    """Fill found[i] with the locate of stock names[i] from 'R' messages."""
    pos = 0
    while pos + 2 <= n:
        ln = (np.int64(buf[pos]) << 8) | np.int64(buf[pos + 1])
        if pos + 2 + ln > n:
            break
        if buf[pos + 2] == 82 and ln >= 20:         # 'R'
            loc = (np.int64(buf[pos + 3]) << 8) | np.int64(buf[pos + 4])
            for i in range(names.shape[0]):
                same = True
                for k in range(8):
                    if buf[pos + 13 + k] != names[i, k]:
                        same = False
                        break
                if same:
                    found[i] = loc
        pos += 2 + ln
    return pos


def _name8(sym):
    b = sym.encode("ascii")
    return np.frombuffer(b.ljust(8, b" "), dtype=np.uint8)


def extract(gz_path, symbols, out_dir, read_block=1 << 24, inflate_block=1 << 26, progress=None):
    """One pass over the day file. Returns a dict with stats and the path of each
    symbol's extract. The directory `out_dir` is created and must be dedicated."""
    os.makedirs(out_dir, exist_ok=True)
    names = np.stack([_name8(s) for s in symbols])
    found = np.full(len(symbols), -1, dtype=np.int64)
    keep_tables = None
    outs = {s: open(os.path.join(out_dir, f"{s}.itch"), "wb") for s in symbols}
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    carry = b""
    total_in = total_out_inflated = msgs = 0
    t_inflate = t_scan = 0.0
    out_buf = np.empty(inflate_block + (1 << 20), dtype=np.uint8)
    t_start = time.perf_counter()
    with open(gz_path, "rb") as f:
        while True:
            raw = f.read(read_block)
            if not raw:
                break
            total_in += len(raw)
            t0 = time.perf_counter()
            data = d.decompress(raw)
            while d.unconsumed_tail:
                data += d.decompress(d.unconsumed_tail)
            t_inflate += time.perf_counter() - t0
            if carry:
                data = carry + data
            total_out_inflated += len(data)
            buf = np.frombuffer(data, dtype=np.uint8)
            n = len(buf)
            t0 = time.perf_counter()
            if keep_tables is None:
                _locates(buf, n, names, found)
                if (found >= 0).all():
                    keep_tables = []
                    for loc in found:
                        t = np.zeros(65536, dtype=np.uint8)
                        t[loc] = 1
                        keep_tables.append(t)
                else:
                    # the directory is not complete yet: keep reading, keep the bytes
                    carry = data
                    t_scan += time.perf_counter() - t0
                    continue
            # one pass per symbol (the tables are disjoint; a day has few symbols)
            consumed = None
            for s, tab in zip(symbols, keep_tables):
                if len(out_buf) < n:
                    out_buf = np.empty(n + (1 << 20), dtype=np.uint8)
                c, out_n, seen = _scan(buf, n, tab, out_buf, 0)
                outs[s].write(out_buf[:out_n].tobytes())
                consumed = c
                msgs = max(msgs, 0)
            msgs += seen
            carry = data[consumed:] if consumed is not None else b""
            t_scan += time.perf_counter() - t0
            if progress:
                progress(total_in, total_out_inflated)
    for fh in outs.values():
        fh.close()
    return {
        "symbols": {s: {"locate": int(l), "path": os.path.join(out_dir, f"{s}.itch"),
                        "bytes": os.path.getsize(os.path.join(out_dir, f"{s}.itch"))} for s, l in zip(symbols, found)},
        "gz_bytes": total_in, "inflated_bytes": total_out_inflated, "messages_seen_last_symbol": msgs,
        "seconds_total": round(time.perf_counter() - t_start, 2), "seconds_inflate": round(t_inflate, 2),
        "seconds_scan": round(t_scan, 2),
    }


# --------------------------------------------------------------------------- #
# Messages -> order-by-order events
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _u48(b, p):
    return ((np.int64(b[p]) << 40) | (np.int64(b[p + 1]) << 32) | (np.int64(b[p + 2]) << 24)
            | (np.int64(b[p + 3]) << 16) | (np.int64(b[p + 4]) << 8) | np.int64(b[p + 5]))


@njit(cache=True)
def _u32(b, p):
    return (np.int64(b[p]) << 24) | (np.int64(b[p + 1]) << 16) | (np.int64(b[p + 2]) << 8) | np.int64(b[p + 3])


@njit(cache=True)
def _u64(b, p):
    v = np.uint64(0)
    for i in range(8):
        v = (v << np.uint64(8)) | np.uint64(b[p + i])
    return v


@njit(cache=True)
def _events(buf, n, ts, kind, oid, side, price, qty, first_ts_of_session):
    slot_of = Dict.empty(key_type=types.uint64, value_type=types.int64)
    s_side = np.zeros(n // 20 + 16, np.int8)
    s_price = np.zeros(n // 20 + 16, np.int64)
    s_left = np.zeros(n // 20 + 16, np.int64)
    ns = 0
    m = 0
    pos = 0
    while pos + 2 <= n:
        ln = (np.int64(buf[pos]) << 8) | np.int64(buf[pos + 1])
        if pos + 2 + ln > n:
            break
        t = buf[pos + 2]
        p = pos + 2                       # message start (type byte)
        if t == 65 or t == 70:            # A, F
            tsv = _u48(buf, p + 5)
            ref = _u64(buf, p + 11)
            sd = 0 if buf[p + 19] == 66 else 1
            sh = _u32(buf, p + 20)
            px = _u32(buf, p + 32)
            slot_of[ref] = ns
            s_side[ns] = sd; s_price[ns] = px; s_left[ns] = sh
            ns += 1
            ts[m] = tsv; kind[m] = 0; oid[m] = ref; side[m] = sd; price[m] = px; qty[m] = sh
            m += 1
        elif t == 69 or t == 67:          # E, C
            tsv = _u48(buf, p + 5)
            ref = _u64(buf, p + 11)
            sh = _u32(buf, p + 19)
            if ref in slot_of:
                k = slot_of[ref]
                px = s_price[k] if t == 69 else _u32(buf, p + 32)
                ts[m] = tsv; kind[m] = 1; oid[m] = ref; side[m] = s_side[k]; price[m] = px; qty[m] = sh
                m += 1
                s_left[k] -= sh
                ts[m] = tsv; oid[m] = ref; side[m] = s_side[k]; price[m] = s_price[k]
                if s_left[k] > 0:
                    kind[m] = 2; qty[m] = s_left[k]
                else:
                    kind[m] = 3; qty[m] = 0
                    del slot_of[ref]
                m += 1
        elif t == 88:                     # X
            tsv = _u48(buf, p + 5)
            ref = _u64(buf, p + 11)
            sh = _u32(buf, p + 19)
            if ref in slot_of:
                k = slot_of[ref]
                s_left[k] -= sh
                ts[m] = tsv; oid[m] = ref; side[m] = s_side[k]; price[m] = s_price[k]
                if s_left[k] > 0:
                    kind[m] = 2; qty[m] = s_left[k]
                else:
                    kind[m] = 3; qty[m] = 0
                    del slot_of[ref]
                m += 1
        elif t == 68:                     # D
            tsv = _u48(buf, p + 5)
            ref = _u64(buf, p + 11)
            if ref in slot_of:
                k = slot_of[ref]
                ts[m] = tsv; kind[m] = 3; oid[m] = ref; side[m] = s_side[k]; price[m] = s_price[k]; qty[m] = 0
                m += 1
                del slot_of[ref]
        elif t == 85:                     # U
            tsv = _u48(buf, p + 5)
            old = _u64(buf, p + 11)
            new = _u64(buf, p + 19)
            sh = _u32(buf, p + 27)
            px = _u32(buf, p + 31)
            if old in slot_of:
                k = slot_of[old]
                sd = s_side[k]
                ts[m] = tsv; kind[m] = 3; oid[m] = old; side[m] = sd; price[m] = s_price[k]; qty[m] = 0
                m += 1
                del slot_of[old]
                slot_of[new] = ns
                s_side[ns] = sd; s_price[ns] = px; s_left[ns] = sh
                ns += 1
                ts[m] = tsv; kind[m] = 0; oid[m] = new; side[m] = sd; price[m] = px; qty[m] = sh
                m += 1
        elif t == 80:                     # P
            tsv = _u48(buf, p + 5)
            sd = 0 if buf[p + 19] == 66 else 1
            sh = _u32(buf, p + 20)
            px = _u32(buf, p + 32)
            ts[m] = tsv; kind[m] = 4; oid[m] = np.uint64(0); side[m] = sd; price[m] = px; qty[m] = sh
            m += 1
        pos += 2 + ln
    return m


def events_from_stream(path, day):
    """Order-by-order events of one symbol's extract (see module docstring)."""
    buf = np.fromfile(path, dtype=np.uint8)
    n = len(buf)
    cap = n // 12 + 1024
    ts = np.empty(cap, np.int64); kind = np.empty(cap, np.int8); oid = np.empty(cap, np.uint64)
    side = np.empty(cap, np.int8); price = np.empty(cap, np.int64); qty = np.empty(cap, np.int64)
    m = _events(buf, n, ts, kind, oid, side, price, qty, 0)
    base = midnight_ny_utc_ns(day)
    return {
        "ts": ts[:m] + base, "kind": kind[:m].copy(), "oid": oid[:m].copy(), "side": side[:m].copy(),
        "price": price[:m].copy(), "qty": qty[:m].astype(np.float64),
    }
