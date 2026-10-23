"""Who is right on the order-by-order quotes the engines disagree on?

An order-by-order feed says which orders stand ahead of a quote, so the right
answer can be computed without trusting any engine: replay the day in explicit
FIFO queues and apply the rules of an exact queue to every quote.

    python fifo_truth.py <ITCH extract> name=journal.npz [name=journal.npz ...] [--json out.json]

Rules (the queue of an order that reaches the venue at t0):
  1. an execution, a cancellation or a deletion of an order AHEAD takes its size out of the queue ahead;
  2. an execution of an order BEHIND serves the quote (the market went through its place);
  3. an execution on its side at a WORSE price serves it whole;
  4. an order entering the OTHER side at its price or through would have met it first: served.
At one instant the market comes first.

The journals are the ones `bench.py` writes (k, side, px, filled, fill_ts, ts per quote). Only quotes
that every engine sent at the same price are judged. A quote that at least one engine served is
replayed from its posting to the (last) fill instant of the engines that served it; the table says,
per engine, how many of those quotes it served although the exact queue had not served them yet
("spurious"), and how many it did not serve although the exact queue had ("missed", an upper bound:
the engine may have cancelled the quote before that instant).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itch  # noqa: E402
import spec  # noqa: E402


def keyed(j):
    return {(int(k), int(s)): (int(px), float(f), int(t), int(ts)) for k, s, px, f, t, ts in
            zip(j["k"], j["side"], j["px"], j["filled"], j["fill_ts"], j["ts"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("itch", help="the ITCH extract of one symbol")
    ap.add_argument("journals", nargs="+", metavar="name=journal.npz")
    ap.add_argument("--json", default=None, help="also write the table as a JSON file")
    args = ap.parse_args()

    m = itch.events_from_stream(args.itch, spec.MBOJoin.DAY)
    names, keyeds = [], []
    for arg in args.journals:
        name, path = arg.split("=", 1)
        names.append(name)
        keyeds.append(keyed(dict(np.load(path))))
    common = set(keyeds[0])
    for k in keyeds[1:]:
        common &= set(k)
    common = [key for key in common if len({k[key][0] for k in keyeds}) == 1]
    queries = []                                   # (post_ts, book side, px, end_ts, key)
    for key in common:
        served = [k[key] for k in keyeds if k[key][1] > 0]
        if not served:
            continue
        px, _, _, pts = keyeds[0][key]
        end = max(s[2] for s in served) + 1
        queries.append((min(k[key][3] for k in keyeds), 0 if key[1] == 1 else 1, px, end, key))
    queries.sort()
    print("quotes sent at one price by every engine:", len(common), "| served by at least one engine:", len(queries))

    ts, kind, oid, side, price = (m[k] for k in ("ts", "kind", "oid", "side", "price"))
    level, level_of = {}, {}
    active, truth = [], {}
    qi = 0
    for i in range(len(ts)):
        t = int(ts[i])
        while qi < len(queries) and queries[qi][0] < t:                  # the market comes first at an instant
            q = queries[qi]
            active.append([q[3], q[1], q[2], q[4], set(level.get((q[1], q[2]), {}).keys()), False])
            qi += 1
        if active and any(a[0] < t for a in active):
            keep = []
            for a in active:
                if a[0] < t:
                    truth[a[3]] = a[5]
                else:
                    keep.append(a)
            active = keep
        k, o, s, p = kind[i], int(oid[i]), int(side[i]), int(price[i]) // 100
        if k == 0:
            level.setdefault((s, p), {})[o] = 1
            level_of[o] = (s, p)
            for a in active:                                              # rule 4
                if not a[5] and a[1] != s and ((p <= a[2]) if a[1] == 0 else (p >= a[2])):
                    a[5] = True
        elif k == 1:
            for a in active:
                if a[5] or a[1] != s:
                    continue
                if p == a[2]:
                    if o not in a[4]:
                        a[5] = True                                       # rule 2
                elif (p < a[2]) if s == 0 else (p > a[2]):
                    a[5] = True                                           # rule 3
        elif k == 3:
            sp = level_of.pop(o, None)
            if sp is not None:
                level.get(sp, {}).pop(o, None)
    for a in active:
        truth[a[3]] = a[5]

    print("\n| engine | served | exact queue agrees | spurious (served, exact queue had not) | missed (not served, exact queue had) |")
    print("|---|---:|---:|---:|---:|")
    rows = {}
    for name, kd in zip(names, keyeds):
        served = [key for key in truth if kd[key][1] > 0]
        agree = sum(1 for key in served if truth[key])
        spurious = len(served) - agree
        missed = sum(1 for key in truth if kd[key][1] <= 0 and truth[key])
        rows[name] = (len(served), agree, spurious, missed)
        print(f"| {name} | {len(served)} | {agree} | {spurious} ({spurious / max(len(served), 1):.1%}) | {missed} |")
    print("\nquotes judged:", len(truth), "| the exact queue serves", sum(truth.values()))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "schema_version": 3,
                "kind": "exact_queue_replay",
                "quotes_sent_at_one_price_by_every_engine": len(common),
                "quotes_judged": len(truth),
                "exact_queue_serves": int(sum(truth.values())),
                "engines": {name: {"served": r[0], "exact_queue_agrees": r[1], "spurious": r[2], "missed": r[3]}
                            for name, r in rows.items()},
            }, f, indent=1)


if __name__ == "__main__":
    main()
