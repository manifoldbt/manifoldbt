"""Download one pinned day of public Bybit spot archives and turn it into the
neutral streams every engine reads.

Source (no key, no account):
  book    quote-saver.bycsi.com/orderbook/spot/BTCUSDT/<day>_BTCUSDT_ob200.data.zip
  trades  public.bybit.com/spot/BTCUSDT/BTCUSDT_<day>.csv.gz

The book archive is a stream of JSON messages: a snapshot, then a delta every
200 ms. It is replayed here exactly once, into a state per millisecond, and from
those states two depth tiers are cut: the best bid and ask ("bbo", 1 level) and
the top 25 levels. Each tier is written as the *difference* between two
successive states (a level that changes or enters the tier carries its new
quantity, a level that leaves it carries 0), which is the only form every engine
can replay without ambiguity.

Rules of the replay (the same ones the manifoldbt connector applies, so the
streams describe the same market):
  * nothing is emitted before the first snapshot; a snapshot replaces the book;
  * a quantity of 0 removes the level;
  * updates that share a millisecond are merged into a single state;
  * only states inside the window are kept; the state in force when the window
    opens is emitted once, at the window's first instant.
"""
from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import json
import os
import time
import zipfile
from array import array
from urllib.request import Request, urlopen

import numpy as np

import spec

# Where the archives, the prepared arrays and the stores live. The workflow points this at a
# directory inside the workspace; the default is a cache directory of the current user.
DATA_ROOT = os.environ.get("BENCH_HFT_DATA") or os.path.join(os.path.expanduser("~"), ".cache", "manifoldbt-bench-hft")


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_arrays(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(a.dtype.str.encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def fetch(url, dest_dir, expected_sha=None):
    """Download `url` into its own directory. Returns a dict with the size, the
    wall time, the sha256 and whether it was already there."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, url.rsplit("/", 1)[1])
    rec = {"url": url, "path": dest, "cached": os.path.exists(dest)}
    t0 = time.perf_counter()
    if not rec["cached"]:
        tmp = dest + ".part"
        req = Request(url, headers={"User-Agent": "manifoldbt-bench/1"})
        with urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dest)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["bytes"] = os.path.getsize(dest)
    rec["sha256"] = sha256_file(dest)
    if expected_sha and rec["sha256"] != expected_sha:
        raise SystemExit(
            f"fingerprint mismatch for {dest}: got {rec['sha256']}, pinned {expected_sha}. "
            "The upstream archive changed; refusing to benchmark a different market."
        )
    return rec


def fetch_day(day):
    book_sha, trades_sha = spec.DAYS.get(day, (None, None))
    root = os.path.join(DATA_ROOT, "raw")
    book = fetch(spec.BOOK_URL.format(sym=spec.SYMBOL, day=day),
                 os.path.join(root, f"bybit-spot-ob200-{day}"), book_sha)
    trades = fetch(spec.TRADES_URL.format(sym=spec.SYMBOL, day=day),
                   os.path.join(root, f"bybit-spot-trades-{day}"), trades_sha)
    return {"book": book, "trades": trades}


# --------------------------------------------------------------------------- #
# Replay of the book archive
# --------------------------------------------------------------------------- #
class _Out:
    """Columns of one depth tier, appended to as compact arrays."""

    def __init__(self):
        self.ts = array("q")
        self.side = array("b")     # 0 bid, 1 ask
        self.px = array("q")       # price in ticks
        self.qty = array("d")
        self.last = array("b")     # 1 on the final event of a state

    def add(self, ts, side, px, qty):
        self.ts.append(ts)
        self.side.append(side)
        self.px.append(px)
        self.qty.append(qty)
        self.last.append(0)

    def close_state(self):
        if len(self.last):
            self.last[-1] = 1

    def arrays(self, prefix):
        return {
            prefix + "_ts": np.frombuffer(self.ts, dtype=np.int64).copy(),
            prefix + "_side": np.frombuffer(self.side, dtype=np.int8).copy(),
            prefix + "_px": np.frombuffer(self.px, dtype=np.int64).copy(),
            prefix + "_qty": np.frombuffer(self.qty, dtype=np.float64).copy(),
            prefix + "_last": np.frombuffer(self.last, dtype=np.int8).copy(),
        }


def _emit_diff(out, ts, prev, cur, side):
    """Removals first (qty 0), then every level whose quantity changed."""
    for k in prev:
        if k not in cur:
            out.add(ts, side, k, 0.0)
    for k, q in cur.items():
        if prev.get(k) != q:
            out.add(ts, side, k, q)


def replay_book(zip_path, w0, w1, depths=(1, 25)):
    """Returns (outs, quotes, stats). outs[n] is the diff stream of the n-level
    tier; quotes is the sequence of best bid/ask changes."""
    inv = spec.TICK_INV
    bids, asks = {}, {}
    bk, ak = [], []                      # sorted price keys (ticks)
    outs = {n: _Out() for n in depths}
    prev = {n: ({}, {}) for n in depths}
    q_ts, q_bp, q_bq, q_ap, q_aq = array("q"), array("q"), array("d"), array("q"), array("d")
    prev_q = None
    seen = False
    pending = None
    started = False                      # have we crossed the window start?
    stats = {"messages": 0, "snapshots": 0, "states": 0, "clamped_ts": 0}
    last_ts = 0

    def top(n):
        tb = {k: bids[k] for k in bk[-n:]}
        ta = {k: asks[k] for k in ak[:n]}
        return tb, ta

    def flush(ts):
        nonlocal started, prev_q
        if ts >= w1:
            return False
        tops = {n: top(n) for n in depths}
        if ts < w0:
            for n in depths:
                prev[n] = tops[n]
            if bk and ak:
                prev_q = (bk[-1], bids[bk[-1]], ak[0], asks[ak[0]])
            return True
        if not started:
            started = True
            # The first REAL state inside the window: manifoldbt's wake-up clock starts there, so
            # every engine anchors its 100 ms grid on it (the state emitted at w0 below is synthetic).
            stats["first_real_ts"] = ts
            # The state in force when the window opens, emitted once at w0.
            for n in depths:
                pb, pa = prev[n]
                if pb or pa:
                    _emit_diff(outs[n], w0, {}, pb, 0)
                    _emit_diff(outs[n], w0, {}, pa, 1)
                    outs[n].close_state()
            if prev_q is not None:
                q_ts.append(w0); q_bp.append(prev_q[0]); q_bq.append(prev_q[1])
                q_ap.append(prev_q[2]); q_aq.append(prev_q[3])
        stats["states"] += 1
        for n in depths:
            pb, pa = prev[n]
            tb, ta = tops[n]
            before = len(outs[n].ts)
            _emit_diff(outs[n], ts, pb, tb, 0)
            _emit_diff(outs[n], ts, pa, ta, 1)
            if len(outs[n].ts) > before:
                outs[n].close_state()
            prev[n] = (tb, ta)
        if bk and ak:
            cq = (bk[-1], bids[bk[-1]], ak[0], asks[ak[0]])
            if cq != prev_q:
                q_ts.append(ts); q_bp.append(cq[0]); q_bq.append(cq[1])
                q_ap.append(cq[2]); q_aq.append(cq[3])
                prev_q = cq
        return True

    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        for raw in f:
            o = json.loads(raw)
            ts = int(o["ts"]) * spec.MS
            if ts < last_ts:
                ts = last_ts
                stats["clamped_ts"] += 1
            last_ts = ts
            if pending is not None and ts != pending:
                if not flush(pending):
                    pending = None
                    break
                pending = None
            stats["messages"] += 1
            data = o["data"]
            if o["type"] == "snapshot":
                stats["snapshots"] += 1
                seen = True
                bids.clear(); asks.clear(); bk.clear(); ak.clear()
            for book, keys, name in ((bids, bk, "b"), (asks, ak, "a")):
                for p, q in data.get(name, ()):
                    k = int(round(float(p) * inv))
                    q = float(q)
                    if q == 0.0:
                        if k in book:
                            del book[k]
                            keys.pop(bisect.bisect_left(keys, k))
                    else:
                        if k not in book:
                            bisect.insort(keys, k)
                        book[k] = q
            if seen:
                pending = ts
        else:
            if pending is not None:
                flush(pending)

    quotes = {
        "q_ts": np.frombuffer(q_ts, dtype=np.int64).copy(),
        "q_bpx": np.frombuffer(q_bp, dtype=np.int64).copy(),
        "q_bqty": np.frombuffer(q_bq, dtype=np.float64).copy(),
        "q_apx": np.frombuffer(q_ap, dtype=np.int64).copy(),
        "q_aqty": np.frombuffer(q_aq, dtype=np.float64).copy(),
    }
    return outs, quotes, stats


def read_trades(path, w0, w1):
    ts, px, qty, side = array("q"), array("d"), array("d"), array("b")
    with gzip.open(path, "rt") as f:
        r = csv.reader(f)
        head = next(r)
        it, ip, iv, isd = (head.index(c) for c in ("timestamp", "price", "volume", "side"))
        for row in r:
            t = int(row[it]) * spec.MS
            if t < w0 or t >= w1:
                continue
            ts.append(t)
            px.append(float(row[ip]))
            qty.append(float(row[iv]))
            side.append(1 if row[isd].lower() == "sell" else 0)    # aggressor side
    return {
        "tr_ts": np.frombuffer(ts, dtype=np.int64).copy(),
        "tr_px": np.frombuffer(px, dtype=np.float64).copy(),
        "tr_qty": np.frombuffer(qty, dtype=np.float64).copy(),
        "tr_side": np.frombuffer(side, dtype=np.int8).copy(),
    }


# --------------------------------------------------------------------------- #
# Prepared dataset
# --------------------------------------------------------------------------- #
def window_tag(w0, w1, day):
    d0, d1 = spec.day_bounds_ns(day)
    if (w0, w1) == (d0, d1):
        return "full"
    return f"w{(w0 - d0) // spec.SEC}-{(w1 - d0) // spec.SEC}"


def prepared_path(day, w0, w1):
    return os.path.join(DATA_ROOT, "prepared", f"{day}_{window_tag(w0, w1, day)}.npz")


def prepare(day, window_s=None):
    """Build (or load) the neutral dataset of a day. `window_s` = (start, end)
    in seconds from the start of the UTC day, or None for the whole day."""
    d0, d1 = spec.day_bounds_ns(day)
    w0, w1 = (d0, d1) if window_s is None else (d0 + window_s[0] * spec.SEC, d0 + window_s[1] * spec.SEC)
    path = prepared_path(day, w0, w1)
    tag = window_tag(w0, w1, day)
    meta_path = path + ".json"
    if os.path.exists(path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            return path, json.load(f)
    t_all = time.perf_counter()
    raw = fetch_day(day)
    t0 = time.perf_counter()
    outs, quotes, stats = replay_book(raw["book"]["path"], w0, w1)
    t_book = time.perf_counter() - t0
    if (w0, w1) == (d0, d1) and len(outs[25].ts):
        # Whole day: the session opens with the first book state. Prints before it
        # have no book to be read against and are left out, for every engine.
        w0 = int(outs[25].ts[0])
    t0 = time.perf_counter()
    trades = read_trades(raw["trades"]["path"], w0, w1)
    t_tr = time.perf_counter() - t0
    arrays = {}
    arrays.update(outs[25].arrays("d25"))
    arrays.update(outs[1].arrays("d1"))
    arrays.update(quotes)
    arrays.update(trades)
    arrays["window"] = np.array([w0, w1], dtype=np.int64)
    arrays["anchor"] = np.array([stats.get("first_real_ts", w0)], dtype=np.int64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    meta = {
        "day": day, "window_ns": [w0, w1], "window_tag": tag,
        "raw": {k: {kk: v[kk] for kk in ("url", "bytes", "seconds", "sha256", "cached")} for k, v in raw.items()},
        "replay": stats,
        "counts": {
            "depth_events_25": int(len(arrays["d25_ts"])), "depth_events_1": int(len(arrays["d1_ts"])),
            "bbo_changes": int(len(arrays["q_ts"])), "trades": int(len(arrays["tr_ts"])),
        },
        "prepare_seconds": {"replay_book": round(t_book, 2), "read_trades": round(t_tr, 2),
                            "total_with_download": round(time.perf_counter() - t_all, 2)},
        "arrays_sha256": sha256_arrays({k: v for k, v in arrays.items()}),
        "npz_bytes": os.path.getsize(path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return path, meta


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
