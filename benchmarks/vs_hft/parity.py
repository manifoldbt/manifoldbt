"""Parity: what must be equal between engines, measured in actions first.

Counting fills alone misleads (an engine that serves in more, smaller pieces
shows more fills for the same executed quantity), so three layers are read:

  actions   the quotes sent: which wake-up, which side, which price
  verdicts  for a quote both engines sent: served or not, how much, how late
  totals    fills, executed quantity, final position, P&L marked at the last mid

A closed-loop strategy feeds its fills back into its next quote, so its actions
part at the first fill the engines disagree on and everything after depends on
that. The open-loop probe (quotes that depend on the market only) is what
isolates the fill model; the closed-loop strategy is what measures how far a
disagreement travels.
"""
from __future__ import annotations

import numpy as np

import spec


def _keyed(j):
    return {(int(k), int(s)): (int(px), float(f), int(t)) for k, s, px, f, t in
            zip(j["k"], j["side"], j["px"], j["filled"], j["fill_ts"])}


def pnl_common(j, mark_px, fee_bps=0.0, price_scale=None):
    """Cash flow of the maker fills at the quote price (a post-only quote is
    served at its own price) plus the position marked at `mark_px`."""
    scale = price_scale or float(spec.TICK_INV)
    side = j["side"].astype(float)
    price = j["px"].astype(float) / scale
    filled = j["filled"]
    cash = -float(np.sum(side * filled * price))
    pos = float(np.sum(side * filled))
    fees = float(np.sum(filled * price)) * fee_bps / 1e4
    return {"pnl": cash + pos * mark_px - fees, "position": pos, "cash": cash}


def compare_quotes(a, b, name_a="a", name_b="b"):
    ka, kb = _keyed(a), _keyed(b)
    keys = sorted(set(ka) | set(kb))
    only_a = sum(1 for k in keys if k not in kb)
    only_b = sum(1 for k in keys if k not in ka)
    common = [k for k in keys if k in ka and k in kb]
    same_px = [k for k in common if ka[k][0] == kb[k][0]]
    first_div = None
    for k in keys:
        if k not in ka or k not in kb or ka[k][0] != kb[k][0]:
            first_div = k
            break
    cat = {"both_served": 0, f"{name_a}_only": 0, f"{name_b}_only": 0, "neither": 0}
    same_qty = 0
    lags = []
    for k in same_px:
        sa, sb = ka[k][1] > 0, kb[k][1] > 0
        if sa and sb:
            cat["both_served"] += 1
            same_qty += int(abs(ka[k][1] - kb[k][1]) < 1e-9)
            lags.append(ka[k][2] - kb[k][2])
        elif sa:
            cat[f"{name_a}_only"] += 1
        elif sb:
            cat[f"{name_b}_only"] += 1
        else:
            cat["neither"] += 1
    out = {
        "quotes": {name_a: len(ka), name_b: len(kb), "common": len(common), "same_price": len(same_px),
                   f"only_{name_a}": only_a, f"only_{name_b}": only_b},
        "first_divergence": {"wake": first_div[0], "side": first_div[1]} if first_div else None,
        "verdicts": cat, "same_executed_qty_when_both_served": same_qty,
    }
    if lags:
        l = np.array(lags, dtype=np.float64) / 1e6
        out["fill_time_lag_ms"] = {"p10": float(np.percentile(l, 10)), "median": float(np.median(l)),
                                   "p90": float(np.percentile(l, 90))}
    n_served_a = sum(1 for k in ka if ka[k][1] > 0)
    n_served_b = sum(1 for k in kb if kb[k][1] > 0)
    out["served_quotes"] = {name_a: n_served_a, name_b: n_served_b}
    return out


def compare_taker(a, b):
    out = {"decisions": {"a": int(len(a["dec_ts"])), "b": int(len(b["dec_ts"]))}}
    same = len(a["dec_ts"]) == len(b["dec_ts"]) and bool(np.all(a["dec_ts"] == b["dec_ts"])) \
        and bool(np.all(a["dec_side"] == b["dec_side"]))
    out["decisions"]["identical_time_and_side"] = same
    if len(a["dec_imb"]) == len(b["dec_imb"]) and len(a["dec_imb"]) and not (
            np.all(np.isnan(a["dec_imb"])) or np.all(np.isnan(b["dec_imb"]))):
        out["decisions"]["max_abs_imbalance_diff"] = float(np.nanmax(np.abs(a["dec_imb"] - b["dec_imb"])))
    oa, ob = len(a["ord_ts"]), len(b["ord_ts"])
    out["orders"] = {"a": oa, "b": ob,
                     "identical": oa == ob and bool(np.all(a["ord_ts"] == b["ord_ts"])) and bool(np.all(a["ord_side"] == b["ord_side"]))
                     and bool(np.allclose(a["ord_qty"], b["ord_qty"]))}
    na = float(np.sum(a["fill_px"] * a["fill_qty"]))
    nb = float(np.sum(b["fill_px"] * b["fill_qty"]))
    out["fills"] = {"a": int(len(a["fill_px"])), "b": int(len(b["fill_px"])),
                    "volume_a": float(np.sum(a["fill_qty"])), "volume_b": float(np.sum(b["fill_qty"])),
                    "notional_diff_bp": (na - nb) / nb * 1e4 if nb else None}
    return out
