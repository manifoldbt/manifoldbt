"""Workloads: which strategy, on which data tier, under which venue settings.

A workload is engine-neutral. `latency` is one of:
  nominal  order 5 ms, cancel 5 ms, response 5 ms, feed 2 ms (the taker: order 20 ms only)
  zero     every delay 0 (the strictest parity setting)
"""
import spec

MM_LADDER = dict(clip=spec.MMLadder.CLIP, max_clips=spec.MMLadder.MAX_CLIPS, off_bid=0, off_ask=0, ladder=True)
MM_TOUCH = dict(clip=spec.MMTouch.CLIP, max_clips=spec.MMTouch.MAX_CLIPS, off_bid=0, off_ask=0, ladder=False)
PROBE = dict(clip=spec.MMLadder.CLIP, max_clips=spec.MMLadder.MAX_CLIPS, off_bid=0, off_ask=0, ladder=False,
             probe=True)
TAKER = dict(window_ns=spec.FlowTaker.WINDOW_NS, threshold=spec.FlowTaker.THRESHOLD, qty=spec.FlowTaker.QTY)

WORKLOADS = {
    "mm_ladder_l2":   dict(kind="mm", tier="l2", params=MM_LADDER, latency="nominal",
                           doc="two post-only quotes, one level back per clip held (example 31), 25 levels"),
    "mm_touch_bbo":   dict(kind="mm", tier="bbo", params=MM_TOUCH, latency="nominal",
                           doc="the same maker on a best-bid/ask feed: it can only join the touch"),
    "flow_taker_bbo": dict(kind="taker", tier="bbo", params=TAKER, latency="nominal",
                           doc="aggressor-imbalance taker on the trade clock (example 33), market orders"),
    # open-loop probe: parity of the queue / fill model, short window
    "probe_l2_zero":  dict(kind="mm", tier="l2", params=dict(PROBE, ladder=False), latency="zero",
                           doc="one quote a side at the touch re-posted at every wake-up, no inventory", short=True),
    "probe_l2_nominal": dict(kind="mm", tier="l2", params=dict(PROBE, ladder=False), latency="nominal",
                             doc="the same probe under the 5/5/5/2 ms venue", short=True),
}
