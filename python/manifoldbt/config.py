"""BacktestConfig helpers matching Rust serde format."""
from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


def entry_price(
    *,
    offset_bps: Optional[float] = None,
    price: Optional[float] = None,
    signal: Optional[str] = None,
) -> dict:
    """Build the price spec for an entry order. Pass exactly one of:

    - ``offset_bps``: distance from the signal-bar close, in bps. Positive is
      more passive (buy lower / sell higher).
    - ``price``: a fixed level, the same on every bar.
    - ``signal``: the name of a strategy signal to read the level from, so the
      order can rest on ``ema(close, 20)``, ``close - 2 * atr(close, 14)``, a
      prior swing low, or anything else the DSL can express.
    """
    given = [x for x in (offset_bps, price, signal) if x is not None]
    if len(given) != 1:
        raise ValueError("entry_price takes exactly one of offset_bps, price, signal")
    if offset_bps is not None:
        return {"OffsetBps": offset_bps}
    if price is not None:
        return {"Absolute": price}
    return {"Signal": signal}


def exit_distance(
    pct: Optional[float],
    signal: Optional[Any],
    where: str,
) -> Union[float, str]:
    """The distance an exit order measures from the entry fill: a constant
    percentage, or the name of a signal holding one.

    Pass exactly one. ``pct`` must be a finite positive number; ``signal`` the
    NAME of a signal the strategy defines, so the distance can be two ATR wide
    on a quiet day and twice that on a violent one.
    """
    accepted = (
        f"{where} takes exactly one of pct= (a finite positive percentage, "
        f"e.g. pct=2.0) or signal= (the name of a signal holding the distance "
        f"in percent, e.g. signal=\"stop_dist\")"
    )
    if (pct is None) == (signal is None):
        raise ValueError(f"{accepted}.")

    if signal is not None:
        # An expression handed over whole is the frequent slip: an exit order
        # reads a NAME, like an entry order does, so the expression has to be
        # named on the strategy first.
        if not isinstance(signal, str):
            got = type(signal).__name__
            raise TypeError(
                f"{where}(signal=...) takes the NAME of a signal, got {got}. "
                f"Name it with .signal(...) first: "
                f'.signal("stop_dist", <expression>).{where}(signal="stop_dist").'
            )
        if not signal.strip():
            raise ValueError(f"{where}(signal=...) takes a non-empty signal name.")
        return signal

    if isinstance(pct, bool) or not isinstance(pct, (int, float)):
        raise TypeError(f"{accepted}, got {type(pct).__name__}.")
    value = float(pct)
    if not (value > 0.0) or value in (float("inf"),) or value != value:
        raise ValueError(f"{accepted}, got {pct!r}.")
    return value


def time_in_force(value: Any) -> Any:
    """Normalise a time-in-force into the shape the engine reads.

    Four policies:

    - ``"GTC"`` (default): rest until filled or until the target moves.
    - ``{"GTB": n}``: expire after ``n`` EVENTS -- bars on a time grid, prints
      under the trade clock.
    - ``"IOC"``: ``{"GTB": 1}`` under another name.
    - a DURATION: ``Interval.seconds(1)``, ``Interval.millis(500)``, or the
      explicit ``{"GTT": Interval.millis(500)}``. The order is cancelled on the
      first event stamped at or after ``posted_at + duration``, before that
      event's prints are offered to it.

    An expired order that filled nothing is posted again while the target still
    holds, so a duration is how a strategy requotes on a clock rather than on a
    count of prints.
    """
    from .expr import _interval_to_nanos

    if isinstance(value, str):
        if value == "Trades":
            raise TypeError(
                "Interval.trades() is a simulation clock, not a lifetime: it "
                "says how often the engine steps, not how long an order lives. "
                "Pass {\"GTB\": n} to count events, or Interval.seconds(n) / "
                "Interval.millis(n) to count time."
            )
        return value
    if isinstance(value, dict) and len(value) == 1:
        key, inner = next(iter(value.items()))
        if key == "GTT":
            if isinstance(inner, dict) and set(inner) == {"duration_ns"}:
                nanos = int(inner["duration_ns"])
            else:
                nanos = _interval_to_nanos(inner)
            if nanos is None:
                raise TypeError(
                    "GTT takes a duration: Interval.millis(n), "
                    "Interval.seconds(n), or {\"duration_ns\": n}"
                )
            if nanos <= 0:
                raise ValueError(
                    "a time_in_force duration must be positive: an order that "
                    "expires on the instant it is posted never reaches the market"
                )
            return {"GTT": {"duration_ns": nanos}}
        nanos = _interval_to_nanos(value)
        if nanos is not None:
            return time_in_force({"GTT": value})
    return value


# Module-level alias: inside the builders below the keyword argument shadows
# the function's own name.
_normalise_tif = time_in_force


def latency(value: Any) -> Any:
    """Normalise a latency into the shape the engine reads.

    ``{"order": Interval.millis(20), "cancel": Interval.millis(20)}`` becomes
    ``{"order": 20000000, "cancel": 20000000}``: the engine counts nanoseconds.
    Both keys are optional and default to zero.

    - ``order``: the decision reaches the market that much later. Only prints
      stamped at or after it can serve the order, the queue is read there, and
      a market order fills at the first print at or after it, at that print's
      price.
    - ``cancel``: a cancellation takes that long to take effect. Until it does
      the stale quote still stands and can still be served, so a requote leaves
      two orders live and the position can overshoot its target by one clip.

    Zero, and an absent setting, are the run the engine always did, byte for
    byte.

    A strategy that quotes (``Strategy.quote``) reads two more keys, and the
    bar engine refuses them non-zero:

    - ``response``: what the venue does (an acknowledgement, a fill, a
      refusal) reaches the strategy that much later;
    - ``feed``: the book and the prints reach the strategy that much later,
      so a wake-up at ``t`` reads the book as it stood at ``t - feed``.

    For a quoting strategy ``cancel`` left out is zero, as for ``bt.run``.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(
            "execution.latency takes a dict: "
            '{"order": Interval.millis(20), "cancel": Interval.millis(20)}'
        )
    unknown = set(value) - set(_LATENCY_KEYS)
    if unknown:
        raise TypeError(
            f"unknown latency key(s) {sorted(unknown)}: execution.latency takes "
            '"order" (decision to visible at the market) and "cancel" '
            "(cancellation decided to cancellation effective), and for a strategy "
            'that quotes "response" and "feed"'
        )
    return {k: _latency_nanos(k, value[k]) for k in _LATENCY_KEYS if k in value}


_LATENCY_KEYS = ("order", "cancel", "response", "feed")


def _latency_nanos(key: str, raw: Any) -> int:
    """One half of a latency, in nanoseconds, or a named refusal."""
    from .expr import _interval_to_nanos

    if raw == "Trades":
        raise TypeError(
            "Interval.trades() is a simulation clock, not a duration: it says "
            "how often the engine steps, not how long a round trip takes. Pass "
            f"Interval.millis(n) or Interval.seconds(n) for {key}."
        )
    nanos = _interval_to_nanos(raw)
    if nanos is None:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(
                f"execution.latency.{key} takes a duration: Interval.millis(n), "
                "Interval.seconds(n), or a number of nanoseconds"
            )
        nanos = raw
    if nanos < 0:
        raise ValueError(
            f"execution.latency.{key} must be a duration of zero or more, got "
            f"{nanos} ns: a decision cannot reach the market before it is taken"
        )
    return int(nanos)


_normalise_latency = latency


# The keys a service rule of the caller's own accepts, and the two it is
# refused for.
_CUSTOM_FILL_KEYS = {"fill", "override_traverse"}

# Keys of `fill_model` whose value is a DURATION the author writes as an
# Interval and the engine reads as nanoseconds. Grouped by the object they sit
# in, so a typo in a model name is refused by the engine under its own name
# rather than silently skipped here.
_FILL_MODEL_DURATIONS = {
    "adverse": ("horizon", "extra_cancel_latency"),
}


def fill_model(value: Any) -> Any:
    """``fill_model`` as the engine reads it.

    Three things happen here and nowhere else.

    Every duration written as an ``Interval`` becomes whole nanoseconds. An
    author writes ``{"adverse": {"model": "conditional",
    "horizon": Interval.millis(100), "slope": 0.5}}`` and this is what turns
    the horizon into ``100000000``.

    A ``custom`` rule's ``fill`` is an ``Expr``, and an ``Expr`` is not JSON: it
    is written out the way every other expression in a strategy is, so the
    engine compiles the same tree from a config as it does from a signal.

    A callable left under ``python`` is refused, by name. It cannot travel in a
    config -- that is the whole difference between the two routes -- so a path
    that serialises one has already lost it, and returning a config without it
    would run a DIFFERENT backtest and say nothing.
    :func:`manifoldbt.run` takes the callable out before this is ever reached;
    every other driver lands here and says so.

    Everything else passes through untouched, including a model name this layer
    has never heard of: naming what a config may contain is the engine's job,
    and it does it by name.
    """
    if not isinstance(value, dict):
        return value
    out = _service_rule(dict(value))
    for group, keys in _FILL_MODEL_DURATIONS.items():
        inner = out.get(group)
        if not isinstance(inner, dict):
            continue
        inner = dict(inner)
        for key in keys:
            if key in inner:
                inner[key] = _fill_model_nanos(group, key, inner[key])
        out[group] = inner
    return out


def _fill_model_nanos(group: str, key: str, raw: Any) -> int:
    """One duration of a fill model, in nanoseconds, or a named refusal."""
    from .expr import _interval_to_nanos

    if raw == "Trades":
        raise TypeError(
            "Interval.trades() is a simulation clock, not a duration: it says "
            "how often the engine steps, not how long a horizon lasts. Pass "
            f"Interval.millis(n) or Interval.seconds(n) for "
            f"fill_model.{group}.{key}."
        )
    nanos = _interval_to_nanos(raw)
    if nanos is None:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(
                f"fill_model.{group}.{key} takes a duration: Interval.millis(n), "
                "Interval.seconds(n), or a number of nanoseconds"
            )
        nanos = raw
    return int(nanos)


_normalise_fill_model = fill_model




def _service_rule(out: dict) -> dict:
    """The ``custom`` and ``python`` halves of :func:`fill_model`."""
    if "python" in out:
        raise TypeError(
            'fill_model={"python": ...} is a callable, and a callable is not part '
            "of a config: it cannot be carried to a sweep, a batch or a lite run, "
            "and it does not describe the run it produced. Use mbt.run for it, or "
            'write the rule in the DSL under fill_model={"custom": {"fill": ...}}, '
            "which travels with the config and sweeps through param()."
        )
    custom = out.get("custom")
    if custom is None:
        return out
    if not isinstance(custom, dict):
        raise TypeError(
            'fill_model["custom"] takes a dict: {"fill": <expression>, '
            '"override_traverse": False}'
        )
    unknown = set(custom) - _CUSTOM_FILL_KEYS
    if unknown:
        raise TypeError(
            f'unknown fill_model["custom"] key(s) {sorted(unknown)}: it takes '
            '"fill" (the rule, an expression of the DSL) and "override_traverse" '
            "(whether the rule answers for prints that went THROUGH the level too)"
        )
    if "fill" not in custom:
        raise TypeError(
            'fill_model["custom"] needs a "fill": the rule itself, an expression '
            "over the fill-context fields, e.g. "
            'min_val(max_val(col("print_qty") - col("queue_ahead"), lit(0.0)), '
            'col("order_size_left"))'
        )
    rule = custom["fill"]
    if not hasattr(rule, "to_json"):
        raise TypeError(
            'fill_model["custom"]["fill"] must be an expression of the DSL, got '
            f"{type(rule).__name__}. Build it with col(), lit(), param(), when() "
            "and the arithmetic, as you would any other expression."
        )
    out["custom"] = {
        "fill": rule.to_json(),
        "override_traverse": bool(custom.get("override_traverse", False)),
    }
    return out

_ORDER_SIDES = {"both": None, "long": "Long", "short": "Short"}


def exit_order(spec: dict, side: str, where: str) -> dict:
    """``spec`` with the engine's name for ``side`` added when it is not the default.

    ``side`` is ``"both"`` (the default: the order arms on longs and shorts
    alike), ``"long"`` or ``"short"``. The JSON carries the key only when it
    departs from the default, so a strategy that never asked for a side
    serializes exactly as it did before the field existed.
    """
    key = side.lower() if isinstance(side, str) else side
    if key not in _ORDER_SIDES:
        raise ValueError(
            f"{where}: side must be 'both', 'long' or 'short', got {side!r}."
        )
    if _ORDER_SIDES[key] is not None:
        spec = {**spec, "side": _ORDER_SIDES[key]}
    return spec


@dataclass
class OrderConfig:
    """Order management configuration for conditional entries, stop-loss,
    take-profit, and trailing stops. All fields are optional — when nothing is
    set the engine uses the legacy market-order path with zero overhead.

    Sub-config dicts:
      limit_entry: where an entry rests instead of taking a market fill.
        price: {"OffsetBps": 10.0} | {"Absolute": 60000.0} | {"Signal": "entry_px"}
               (omit and set offset_bps for the legacy shape)
        trigger: "Limit" (default), "Stop", "StopLimit", "MarketIfTouched"
        limit_price: same shape as price; required by "StopLimit"
        time_in_force: "GTC" (default), {"GTB": 5} (five EVENTS), "IOC", or a
               duration -- Interval.seconds(1) / Interval.millis(500)
        size_at_fill_price: size off the order's own level instead of the close
      stop_loss:  {"stop_pct": 2.0}   — % from entry price
      take_profit: {"profit_pct": 5.0} — % from entry price
      trailing_stop: {"trail_pct": 3.0, "use_high": true}
      Each distance is either a number (a constant percentage) or the name of
      a signal holding one, e.g. {"stop_pct": "stop_dist"}: read on the bar
      whose signal opened the trade and frozen for its life.
      Each of the three also takes "side": "Both" (default), "Long" or
      "Short"; an order armed on one side leaves the other bare.

    Note that a conditional entry runs on the general simulation loop, not the
    fast kernel, so sweeps over one are slower than sweeps over a market entry.
    """

    limit_entry: Optional[dict] = None
    stop_loss: Optional[dict] = None
    take_profit: Optional[dict] = None
    trailing_stop: Optional[dict] = None

    @classmethod
    def bracket(
        cls, stop_pct: float, profit_pct: float, side: str = "both"
    ) -> "OrderConfig":
        """Convenience: create a bracket order (SL + TP)."""
        return cls(
            stop_loss=exit_order(
                {"stop_pct": exit_distance(stop_pct, None, "bracket")}, side, "bracket"
            ),
            take_profit=exit_order(
                {"profit_pct": exit_distance(profit_pct, None, "bracket")},
                side,
                "bracket",
            ),
        )

    @classmethod
    def stop_loss_only(cls, stop_pct: float, side: str = "both") -> "OrderConfig":
        """Convenience: stop-loss only."""
        return cls(
            stop_loss=exit_order(
                {"stop_pct": exit_distance(stop_pct, None, "stop_loss_only")},
                side,
                "stop_loss_only",
            )
        )

    @classmethod
    def trailing(
        cls, trail_pct: float, use_high: bool = True, side: str = "both"
    ) -> "OrderConfig":
        """Convenience: trailing stop only."""
        return cls(trailing_stop=exit_order(
            {
                "trail_pct": exit_distance(trail_pct, None, "trailing"),
                "use_high": use_high,
            },
            side,
            "trailing",
        ))

    @classmethod
    def limit_entry_at(
        cls,
        *,
        offset_bps: Optional[float] = None,
        price: Optional[float] = None,
        signal: Optional[str] = None,
        time_in_force: Any = "GTC",
        size_at_fill_price: bool = False,
    ) -> "OrderConfig":
        """Convenience: a passive limit entry resting at the given level."""
        return cls(
            limit_entry={
                "price": entry_price(
                    offset_bps=offset_bps, price=price, signal=signal
                ),
                "trigger": "Limit",
                "time_in_force": _normalise_tif(time_in_force),
                "size_at_fill_price": size_at_fill_price,
            }
        )

    @classmethod
    def stop_entry_at(
        cls,
        *,
        offset_bps: Optional[float] = None,
        price: Optional[float] = None,
        signal: Optional[str] = None,
        time_in_force: Any = "GTC",
        size_at_fill_price: bool = False,
    ) -> "OrderConfig":
        """Convenience: a breakout entry that fills once price trades through
        the level. Crosses the book, so it pays taker fees and slippage, and a
        bar that gaps through the level fills at the open."""
        return cls(
            limit_entry={
                "price": entry_price(
                    offset_bps=offset_bps, price=price, signal=signal
                ),
                "trigger": "Stop",
                "time_in_force": _normalise_tif(time_in_force),
                "size_at_fill_price": size_at_fill_price,
            }
        )

    def to_json_dict(self) -> dict:
        d: dict = {}
        if self.limit_entry is not None:
            d["limit_entry"] = self.limit_entry
        if self.stop_loss is not None:
            d["stop_loss"] = self.stop_loss
        if self.take_profit is not None:
            d["take_profit"] = self.take_profit
        if self.trailing_stop is not None:
            d["trailing_stop"] = self.trailing_stop
        return d


@dataclass
class ExecutionConfig:
    signal_delay: int = 0
    execution_price: str = "AtClose"
    max_position_pct: float = 1.0
    allow_short: bool = True
    allow_fractional: bool = True
    skip_gap_bars: bool = False
    position_sizing_mode: str = "FractionOfEquity"
    """How position_sizing output is interpreted:
    - "FractionOfEquity": target 1.0 = 100% of equity (default, compounds)
    - "FractionOfInitialCapital": same but uses initial capital (no compounding)
    - "Units": target 1.0 = 1 unit (share/contract/coin)
    """
    pyramiding: bool = False
    """When True, the signal is treated as a delta to ADD to the current position
    each bar (pyramiding), instead of a target position. Works with any sizing mode.
    Signal: 0.0 = go flat, NaN/None = hold, nonzero = add to position."""
    fill_model: Optional[dict] = None
    """Fill model configuration. None = Rust defaults (atomic fill, single point).
    Example: {"max_participation_rate": 0.1, "intra_bar_price": "TypicalPrice"}
    intra_bar_price options: "SinglePoint", "TypicalPrice", "OhlcAverage"
    passive_fill options: "touch" (default: a maker fill books when the bar
      reaches its level) or "traverse" (the bar must trade through the level;
      conservative, and the run stays on the general loop). The result's
      fill_fragility counters say how many maker fills the choice affects.
    queue: the queue in front of a resting maker order, as
      {"depth_source": "book" | "assumed", "assumed_queue": x,
       "cancel_ahead_rate": r}. An order posted at a level joins behind the
      depth stored there at that instant ("book", the default), or behind
      assumed_queue times its OWN SIZE ("assumed"). A trade at the level from
      the side that consumes it burns that volume first, and only the
      remainder fills; a print through the level, or a best opposite price
      crossing it, serves the rest whatever the queue. cancel_ahead_rate is
      the fraction of the volume ahead that cancels per second (0 by default,
      the assumption that nobody ahead ever cancels). Requoting is cancelling
      and posting again, so it starts at the back of a fresh queue.
      assumed_queue is a MULTIPLE OF THE ORDER SIZE because an absolute
      quantity has no scale from one symbol to the next; it and
      cancel_ahead_rate both make the P&L, so they are swept and published as
      cost axes rather than chosen once. Under "book", assumed_queue is the
      declared fallback for an instant the stored book cannot answer (before
      its first line, an empty side, a level deeper than the ladder kept or
      than BacktestConfig.book_levels when it bounds the levels read);
      without it such an instant is refused rather than served from the front
      of the queue. Needs the prints: fill_resolution="ticks", or the trade
      clock. result.fill_fragility["queue"] says how each fill was served.
      queue["model"] picks the rule that moves that volume: "risk_adverse"
      (the default: it comes down when trades execute at the level, and never
      stands above what the level shows), "power" (needs "n") or "log" (the
      probabilistic queues, which take a share of every cancellation from IN
      FRONT of the order). An offer resting at the order's own price serves it
      under every rule. "fifo" is the exact queue of a market stored order by
      order (bt.ingest_mbo): only the order-level simulation serves it
      (bt.sim.run, Strategy.quote), where it is the default on such a market;
      a run on bars or on the trade clock refuses it by name.
    adverse: adverse selection put INTO the fill decision, as
      {"model": "conditional", "horizon": Interval.millis(100), "slope": k} or
      {"model": "snipe", "threshold_ticks": n,
       "extra_cancel_latency": Interval.millis(5)}. The first only serves a
      print at the level when the market then moves against the order, and it
      READS THE FUTURE to decide: it is a stress test, not a simulation of
      anything a venue could have given you. The second slows the
      cancellation of a quote whose level has gone stale, so the stale quote
      is taken more often. Both need fill_model.queue; snipe also needs
      execution.latency and execution.tick_size.
    Every parameter of every named model above is a HYPOTHESIS that makes the
    P&L: sweep it as a cost axis and publish it beside the result, do not pick
    one and hide it. See "Execution models" in the strategy authoring guide.
      See "Queue Model" in the strategy authoring guide.
    custom: a service rule of your OWN, written in the DSL, as
      {"fill": <expression>, "override_traverse": False}. The expression is
      evaluated once per event for each resting order and answers the quantity
      of that event which reaches it; the engine truncates the answer to what
      the event could physically serve and counts every truncation. It does
      not replace the queue, it replaces the queue's SERVICE rule, so
      fill_model.queue must be set beside it: the queue says what volume
      stands ahead of the order, the rule says how much of each print gets
      past it. Its leaves are the fields of the fill context, read as columns
      -- queue_ahead, order_size_left, order_age (seconds), level,
      depth_at_level, best_bid, best_ask, mid, mid_move_since_post_bps,
      mid_move_since_post_ticks, print_price, print_qty, print_aggressor,
      print_qty_eligible, is_book_update, is_traverse, u (a deterministic
      uniform draw) -- plus lit() and param(). A param() inside the rule is a
      sweep axis like any other, provided the strategy declares it with
      .param("name", default=...). Everything outside arithmetic, comparisons,
      the boolean operators, when(), abs, min_val, max_val and round_to /
      floor_to / ceil_to is refused by name at compile time, and a rule that
      answers NaN stops the run naming the event. override_traverse (False by
      default) hands the rule the events that went strictly THROUGH the level,
      which the engine otherwise serves itself because a print past a level
      means everyone resting at it went first.
      result.fill_fragility["queue"] gains custom_decided_fills,
      custom_rule_events, custom_declined_events and custom_truncated_answers:
      how many fills the rule had the last word on, and how often it asked for
      more than an event could give. Needs the prints
      and the general loop, exactly as the queue does.
    python: a service rule given as a CALLABLE instead of an expression, for
      research. Invoked once per event with the seventeen context fields as
      positional arguments, in the order listed above, and returning the
      quantity served. It is not part of a config, so it cannot be swept and
      it does not describe the run it produced: mbt.run routes it, and every
      other driver refuses it by name rather than dropping it. Much slower
      than the DSL route -- the cost is the boundary itself.
    """
    fill_resolution: Optional[str] = None
    """What level orders (stop-loss, take-profit, trailing stop, limit and stop
    entries) are resolved against inside a bar: "bar" (default) uses the bar's
    high and low, "ticks" walks the individual trades of the bar in time order,
    so the order the market actually printed decides which level came first.
    Needs a stored tape for the symbol; a bar with no trade in its window falls
    back to the bar rule, and the result's tape_resolution counters say how
    often that happened. Merged into fill_model when set."""
    fill_marks: bool = False
    """Mark every fill against what the market did next, into
    ``result.fill_marks``: the reference at the fill and at +100 ms, +1 s and
    +10 s, signed in the direction the fill takes the position, in basis points
    of the fill price. Negative = adverse selection, the cost of quoting
    passively, which the fees and the fill count alone never show.
    The mark is anchored on the CLOSE of the bar the fill was booked on (the
    first instant the fill is certainly done) and reads the stored book's mid
    when there is one, the last tape print otherwise; each mark says which.
    Needs the tape: set fill_resolution="ticks", or run on the trade clock.
    Off by default -- it is one more pass and one more field.
    See "Marking a Fill" in the strategy authoring guide."""
    latency: Optional[dict] = None
    """The round trip between a decision and the market, as
    ``{"order": Interval.millis(20), "cancel": Interval.millis(20)}``.

    ``order``: a decision taken on event t reaches the market at t + order.
    Only prints stamped at or after it can serve the order; the queue is read
    there, which is where the order joins it; a duration time_in_force counts
    from there; and a market order fills at the first print at or after it, at
    THAT print's price plus the configured slippage, not at a bar price. An
    aggressive bracket leg (stop-loss, trailing stop) is sent on the print that
    reaches its level and executes at the first print at or after
    trigger + order, at that print's price.

    ``cancel``: a cancellation decided on event t -- an expiry, a target that
    moved, a requote -- takes effect at t + cancel. Until then the stale quote
    still stands, at its own level, and can still be served: that fill is real
    and the engine books it. So a requote leaves TWO orders live for cancel
    nanoseconds and the position can overshoot its target by one clip. The
    engine keeps the overshoot and lets the strategy reduce it on the next
    event; result.order_activity reports stale_fills and
    overlapping_live_orders.

    Absent, and {"order": 0, "cancel": 0}, are the run the engine always did,
    byte for byte. Needs the tape: set fill_resolution="ticks", or run on the
    trade clock. Refused by name on the fast path, the CUDA sweep and the lite
    paths. See "Latency" in the strategy authoring guide."""
    orders: Optional[OrderConfig] = None
    """Order management: limit entries, stop-loss, take-profit, trailing stops.
    When None (default), the engine uses the legacy market-order path."""
    tick_size: Optional[float] = None
    """The venue's tick size, e.g. ``0.1`` for BTCUSDT.

    When set, every level an order RESTS at is brought onto that grid before
    the order is posted, in the direction of passivity: a bid down, an ask up.
    A venue quotes on a grid, and a level off it finds no depth in the book --
    under the queue model the engine has nothing to place the order behind, and
    no venue would have accepted the order to begin with.

    Aggressive legs are NOT touched: a stop entry, a market-if-touched, a
    stop-loss and a take-profit all execute at the price of the print that
    triggered them, not at a level they rest on. A stop-limit's arming price is
    a trigger too; only the limit it then rests at is snapped.

    None (the default) is the run the engine always did, byte for byte.
    ``result.order_activity["levels_snapped"]`` counts the levels that moved,
    and is absent when none did. A value that is not strictly positive is
    refused by name.

    The tick is not read from the store: an ingested book knows its symbol's
    tick in its metadata, and wiring that through is its own step. Authors who
    want the grid inside the expression itself have ``.round_to`` /
    ``.floor_to`` / ``.ceil_to``."""

    def __post_init__(self) -> None:
        """Refuse the values that make the engine panic or lie.

        Two of these reach `f64::clamp` in the core with bounds it cannot
        order: a negative `max_position_pct` makes `min > max`, a non-finite
        one makes both bounds NaN, and either aborts the run with a Rust panic
        that `except Exception` does not catch. Naming the field here costs one
        check and is the only place left that still knows what the user wrote.
        """
        if not isinstance(self.signal_delay, int) or isinstance(self.signal_delay, bool):
            raise ValueError(
                f"signal_delay must be an int, got {self.signal_delay!r}. It "
                "counts bars between the signal bar and the execution bar."
            )
        if self.signal_delay < 0:
            raise ValueError(
                f"signal_delay must be >= 0, got {self.signal_delay}. A "
                "negative delay would execute before the bar that decided."
            )
        if not math.isfinite(self.max_position_pct):
            raise ValueError(
                f"max_position_pct must be a finite number, got "
                f"{self.max_position_pct!r}."
            )
        if self.max_position_pct <= 0.0:
            raise ValueError(
                f"max_position_pct must be > 0, got {self.max_position_pct}. "
                "It caps the position as a fraction of equity, so 1.0 means "
                "100% and 0 or less would forbid every trade."
            )

    def to_json_dict(self) -> dict:
        d = {
            "signal_delay": self.signal_delay,
            "execution_price": self.execution_price,
            "max_position_pct": self.max_position_pct,
            "allow_short": self.allow_short,
            "allow_fractional": self.allow_fractional,
            "skip_gap_bars": self.skip_gap_bars,
            "position_sizing_mode": self.position_sizing_mode,
            "pyramiding": self.pyramiding,
        }
        if self.fill_model is not None:
            d["fill_model"] = _normalise_fill_model(self.fill_model)
        if self.fill_resolution is not None:
            d["fill_model"] = {
                **(_normalise_fill_model(self.fill_model) or {}),
                "fill_resolution": self.fill_resolution,
            }
        # Written out only when asked for, so a config that does not mention
        # the marks serialises byte for byte the way it always did.
        if self.fill_marks:
            d["fill_marks"] = True
        # Same rule for the round trip: absent stays absent.
        if self.latency is not None:
            d["latency"] = _normalise_latency(self.latency)
        if self.orders is not None:
            d["orders"] = self.orders.to_json_dict()
        # Same rule as the marks and the round trip: absent stays absent, so a
        # config that does not mention the grid serialises byte for byte.
        if self.tick_size is not None:
            tick = float(self.tick_size)
            # Named here as well as in the engine: a NaN or an infinity does not
            # survive JSON, so without this check the engine's refusal would
            # reach the caller as a parse error that names nothing.
            if tick != tick or tick in (float("inf"), float("-inf")):
                raise ValueError(
                    f"execution.tick_size must be a finite number, got {tick}"
                )
            if tick <= 0.0:
                raise ValueError(
                    f"execution.tick_size must be strictly positive, got {tick}"
                )
            d["tick_size"] = tick
        return d


@dataclass
class VenueFees:
    """Fee schedule for a single venue (exchange).

    The same fields as a flat (single-venue) :class:`FeeConfig`. Used as the
    value type of :attr:`FeeConfig.per_venue` to express per-exchange fees.
    """

    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    funding_rate_column: Optional[str] = None
    borrow_rate_annual_bps: float = 0.0
    min_fee: float = 0.0
    default_fill_type: str = "Taker"
    """Default fill type for fee calculation: "Maker" or "Taker" (conservative)."""

    def to_json_dict(self) -> dict:
        return {
            "maker_fee_bps": self.maker_fee_bps,
            "taker_fee_bps": self.taker_fee_bps,
            "funding_rate_column": self.funding_rate_column,
            "borrow_rate_annual_bps": self.borrow_rate_annual_bps,
            "min_fee": self.min_fee,
            "default_fill_type": self.default_fill_type,
        }


@dataclass
class FeeConfig:
    """Transaction-cost configuration.

    The flat fields below describe the *default* venue, applied to any symbol
    not present in ``symbol_venue``. Per-venue fees are opt-in via ``per_venue``
    (named fee schedules) plus ``symbol_venue`` (which symbol trades where).
    Single-venue configs are unchanged — leaving ``per_venue``/``symbol_venue``
    empty serializes to the exact same JSON as before.
    """

    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    funding_rate_column: Optional[str] = None
    borrow_rate_annual_bps: float = 0.0
    min_fee: float = 0.0
    default_fill_type: str = "Taker"
    """Default fill type for fee calculation: "Maker" or "Taker" (conservative)."""
    per_venue: Dict[str, VenueFees] = field(default_factory=dict)
    """Named per-venue fee overrides, keyed by venue name (e.g. ``"binance"``)."""
    symbol_venue: Dict[int, str] = field(default_factory=dict)
    """Maps a ``SymbolId`` (integer) to the name of the venue it executes on.
    Symbols absent from this map use the default-venue fields above."""

    def to_json_dict(self) -> dict:
        d: dict = {
            "maker_fee_bps": self.maker_fee_bps,
            "taker_fee_bps": self.taker_fee_bps,
            "funding_rate_column": self.funding_rate_column,
            "borrow_rate_annual_bps": self.borrow_rate_annual_bps,
            "min_fee": self.min_fee,
            "default_fill_type": self.default_fill_type,
        }
        # Emit per-venue keys only when populated so single-venue configs stay
        # byte-identical to the legacy flat shape (matches Rust serde flatten).
        if self.per_venue:
            d["per_venue"] = {
                name: (v.to_json_dict() if isinstance(v, VenueFees) else dict(v))
                for name, v in self.per_venue.items()
            }
        if self.symbol_venue:
            d["symbol_venue"] = {
                str(sid): name for sid, name in self.symbol_venue.items()
            }
        return d

    @classmethod
    def multi_venue(
        cls,
        default: Optional[VenueFees] = None,
        venues: Optional[Dict[str, VenueFees]] = None,
        symbol_venue: Optional[Dict[int, str]] = None,
    ) -> "FeeConfig":
        """Build a per-venue fee config.

        Args:
            default: Fee schedule for symbols without a venue mapping.
            venues: Named per-venue fee schedules (e.g. ``{"binance": VenueFees(...)}``).
            symbol_venue: Maps integer ``SymbolId`` to a venue name in ``venues``.
        """
        d = default or VenueFees()
        return cls(
            maker_fee_bps=d.maker_fee_bps,
            taker_fee_bps=d.taker_fee_bps,
            funding_rate_column=d.funding_rate_column,
            borrow_rate_annual_bps=d.borrow_rate_annual_bps,
            min_fee=d.min_fee,
            default_fill_type=d.default_fill_type,
            per_venue=venues or {},
            symbol_venue=symbol_venue or {},
        )

    @classmethod
    def binance_perps(cls) -> "FeeConfig":
        """Binance USDM perpetual futures defaults (taker fees + funding)."""
        return cls(
            maker_fee_bps=2.0,
            taker_fee_bps=5.0,
            funding_rate_column="funding_rate",
            borrow_rate_annual_bps=0.0,
            min_fee=0.0,
            default_fill_type="Taker",
        )

    @classmethod
    def binance_spot(cls) -> "FeeConfig":
        """Binance spot defaults (taker fees, no funding)."""
        return cls(
            maker_fee_bps=10.0,
            taker_fee_bps=10.0,
            funding_rate_column=None,
            borrow_rate_annual_bps=0.0,
            min_fee=0.0,
            default_fill_type="Taker",
        )

    @classmethod
    def zero(cls) -> "FeeConfig":
        """No fees (for development/debugging)."""
        return cls()



def account_sessions(start, end, tz: str = "UTC", hour: int = 0) -> List[int]:
    """The UTC instants each evaluation session begins at, one per day.

    A prop firm's day is not UTC midnight: an FX firm resets at 00:00 CE(S)T
    and a futures evaluation follows the exchange session, which opens at 17:00
    in Chicago. The engine carries no time-zone database on purpose, so the
    boundary travels as the list itself, built here with ``zoneinfo`` from the
    standard library.

    A list is also the only thing that can say what a (zone, hour) pair cannot:
    a holiday or an early close is simply a start that is not there, and the
    session then runs to the next one.

        rules = AccountRules(
            phases=[AccountPhase(profit_target=0.10, max_daily_loss=0.05,
                                 max_total_loss=0.10, min_trading_days=4)],
            sessions=bt.account_sessions("2024-01-01", "2025-01-01",
                                         tz="Europe/Prague"),
        )

    ``start`` and ``end`` accept nanoseconds, a ``datetime``, or an ISO date.
    A local hour that does not exist on a spring-forward day resolves to the
    instant the clock reaches, which is what an exchange does too.
    """
    from datetime import date as _date, datetime as _dt, time as _time, timedelta, timezone
    from zoneinfo import ZoneInfo

    def _as_utc(v):
        if isinstance(v, _dt):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if isinstance(v, _date):
            return _dt.combine(v, _time(), tzinfo=timezone.utc)
        if isinstance(v, str):
            d = _dt.fromisoformat(v)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        return _dt.fromtimestamp(int(v) / 1e9, tz=timezone.utc)

    if not 0 <= int(hour) <= 23:
        raise ValueError(f"the session hour must be 0..23, got {hour}")
    zone = ZoneInfo(tz)
    shift = timedelta(hours=int(hour))
    first = (_as_utc(start).astimezone(zone) - shift).date()
    last = (_as_utc(end).astimezone(zone) - shift).date()
    out: List[int] = []
    day = first
    while day <= last:
        local = _dt.combine(day, _time(hour=int(hour)), tzinfo=zone)
        out.append(int(local.astimezone(timezone.utc).timestamp() * 1_000_000_000))
        day += timedelta(days=1)
    return out


@dataclass
class AccountPhase:
    """One phase's limits, as fractions of the capital the PHASE starts with."""

    max_total_loss: float = 0.10
    """The floor that closes the account."""
    name: str = "Evaluation"
    """What the firm calls it: "Challenge", "Verification", "Evaluation"."""
    profit_target: Optional[float] = None
    """``None`` == no target: the phase only has to survive."""
    max_daily_loss: Optional[float] = None
    """``None`` == the firm sets no daily limit, which a whole family does."""
    drawdown_type: str = "Static"
    """``"Static"``, ``"TrailingEod"`` (the floor follows the CLOSE) or
    ``"TrailingEquity"`` (it follows the peak equity mark by mark inside the
    session, so a spike handed back can close an account on a day that ends in
    profit)."""
    trailing_lock_at: Optional[float] = None
    """Level at which a trailing floor freezes for good. ``0.0`` == it stops at
    the balance the phase started with, which is what futures evaluations do."""
    min_trading_days: int = 0
    """A trading day is a session in which a position was OPENED, measured on
    the FILLS: a reversal opens the new side and counts. Holding does not count,
    and neither does closing, so a position opened once and held for ten
    sessions is ONE trading day and no `min_trading_days` above 1 can ever be
    met by it."""
    max_calendar_days: Optional[int] = None
    """``None`` == unlimited. Counted in SESSIONS OF THE LIST, not in wall-clock
    days: a calendar of trading days and a limit of 10 buys ten TRADING days,
    which on a five-day week is two calendar weeks."""

    def to_json_dict(self) -> dict:
        # Les non-finis se refusent ICI, la ou l'utilisateur les a tapes. Plus
        # loin, `json.dumps` ecrit les jetons nus `NaN` et `Infinity`, que le
        # desserialiseur Rust rejette par un "invalid json payload: expected
        # value at line 1 column 813" qui ne nomme aucun champ, et les gardes
        # `is_finite` du moteur deviennent inatteignables.
        for champ in ("max_total_loss", "profit_target", "max_daily_loss",
                      "trailing_lock_at"):
            v = getattr(self, champ)
            if v is not None and not math.isfinite(float(v)):
                raise ValueError(
                    f"AccountPhase({self.name!r}).{champ} must be a finite "
                    f"number, got {v!r}."
                )
        return {
            "name": self.name,
            "profit_target": self.profit_target,
            "max_daily_loss": self.max_daily_loss,
            "max_total_loss": self.max_total_loss,
            "drawdown_type": self.drawdown_type,
            "trailing_lock_at": self.trailing_lock_at,
            "min_trading_days": int(self.min_trading_days),
            "max_calendar_days": self.max_calendar_days,
        }


@dataclass
class AccountRules:
    """A prop-firm program, enforced DURING the run.

    The account is judged mark by mark and CLOSED the moment it breaches: it
    liquidates, stops trading, and its equity stays flat to the end of the data.
    That is what the firm does, and it is why the rule lives in the engine
    rather than in a verdict applied to a finished curve, which would read a
    recovery the account never traded.

    A program is a SEQUENCE of phases, and the firm hands out a fresh account
    between them: a Verification does not start already past its own target
    because the Challenge ended at +10%. Each phase therefore measures its
    limits from the equity that phase started at.

        import manifoldbt as bt
        cfg.account_rules = bt.AccountRules(
            phases=[
                bt.AccountPhase(name="Challenge", profit_target=0.10,
                                max_daily_loss=0.05, max_total_loss=0.10,
                                min_trading_days=4),
                bt.AccountPhase(name="Verification", profit_target=0.05,
                                max_daily_loss=0.05, max_total_loss=0.10,
                                min_trading_days=4),
            ],
            sessions=bt.account_sessions(start, end, tz="Europe/Prague"),
            consistency=0.50,
        )
        res = bt.run(strategy, cfg, store)
        res.account   # None while it lives, else how it ended

    ``res.account`` reports ``equity`` and ``floor`` **on the phase's own
    balance**, which restarts at the initial capital for every phase, while
    ``res.equity_df()`` keeps compounding the run. On a two-phase program the
    two numbers are deliberately different: a verdict at 105,205 in the
    Verification can sit on a curve worth 116,569.

    Two more things worth knowing before reading a result:

      * a program that PASSES also stops trading, exactly like one that
        breaches. The equity curve is flat from the verdict on, so
        ``res.metrics`` is computed over a mostly frozen curve and can read as
        a success on an account the firm closed. Read ``res.account`` first.
      * ``RAN_OUT_OF_TIME`` only ever appears when a phase sets
        ``max_calendar_days``. Without one, an account that never reaches its
        target simply stays open and ``res.account`` is ``None``, which means
        "still alive", not "failed".
    """

    phases: List[Any] = field(default_factory=list)
    """At least one. Two for the mainstream 2-Step."""
    sessions: List[int] = field(default_factory=list)
    """UTC instants each session starts at: see :func:`account_sessions`."""
    consistency: Optional[float] = None
    """Largest share of the winning sessions' profit one session may hold, e.g.
    ``0.50``. It does NOT close the account: it blocks the payout, and the
    verdict reports it as ``consistency_ok``. ``None`` == no cap."""

    def to_json_dict(self) -> dict:
        if not self.phases:
            raise ValueError(
                "account rules need at least one phase: "
                "AccountRules(phases=[AccountPhase(...)], sessions=...)"
            )
        if self.consistency is not None and not math.isfinite(float(self.consistency)):
            raise ValueError(
                f"AccountRules.consistency must be a finite share in (0, 1], "
                f"got {self.consistency!r}."
            )
        return {
            "phases": [
                p.to_json_dict() if hasattr(p, "to_json_dict") else p
                for p in self.phases
            ],
            "day_boundary": {"starts": [int(t) for t in self.sessions]},
            "consistency": (
                None if self.consistency is None
                else {"max_best_day_share": float(self.consistency)}
            ),
        }


@dataclass
class BacktestConfig:
    universe: List[int] = field(default_factory=lambda: [1])
    time_range_start: int = 0
    time_range_end: int = 4_000_000_000
    initial_capital: float = 1000.0
    currency: str = "USD"
    bar_interval: Any = None
    """Bar step the simulation runs on, e.g. ``Interval.hours(1)``.
    ``None`` means one minute. Steps below one minute are a Pro feature;
    Community simulates at one minute at the finest, whatever the store holds."""
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    fees: FeeConfig = field(default_factory=FeeConfig)
    slippage: Any = None
    data_version: Optional[str] = None
    rng_seed: Optional[int] = None
    trading_days_per_year: float = 365.25
    """Annualisation factor: 365.25 for crypto/futures, 252 for equities."""
    risk_free_rate: float = 0.0
    """Annual risk-free rate used in Sharpe/Sortino (excess return). Default 0.0
    = raw Sharpe (consistent with raptorbt/vectorbt and most reporting). Set a
    non-zero rate (e.g. 0.025) for an excess-return Sharpe."""
    output_resolution: Any = None
    """Downsample output timeseries (equity, positions).
    None = auto (uses resample_to if set, else bar_interval), so the output
    keeps the step the simulation ran at.
    Use ``{"Seconds": 1}``, ``{"Hours": 1}``, ``{"Days": 1}``, etc. for explicit
    control. Pro returns any step down to one second; Community output is
    coarsened to daily."""
    resample_to: Any = None
    """Resample raw bars to this interval before simulation.
    E.g. ``{"Minutes": 60}`` aggregates 1-min data into 60-min OHLCV bars.
    ``None`` (default) = use data as-is from the store."""
    feature_sets: List[str] = field(default_factory=list)
    """Feature sets to preload. Feature columns from
    ``features/{set_name}/{symbol_id}/`` are injected into signal env."""
    symbol_names: Dict[str, int] = field(default_factory=dict)
    """Mapping of human-readable symbol names to SymbolId integers.
    Required when using ``bt.asset()`` or ``symbol_ref()`` cross-asset references.
    Example: ``{"BTCUSDT": 1, "ETHUSDT": 2}``"""
    warmup_bars: int = 0
    """Number of bars to skip before acting on signals.
    Allows indicators (EMA, SMA, etc.) to stabilise. During warmup,
    equity tracking runs but no trades are generated.
    Set to at least the longest indicator window (e.g. 25 for EMA(25))."""
    accuracy: bool = False
    """When True, simulation runs on 1-minute bars regardless of bar_interval.
    Signals are still evaluated at bar_interval resolution (hybrid mode).
    Use for precise SL/TP fills and intraday drawdown tracking. Slower."""
    extra_timeframes: Dict[str, Any] = field(default_factory=dict)
    """Additional timeframes for multi-timeframe strategies.
    Maps labels to Interval dicts. The engine resamples native bars
    and injects prefixed columns (e.g. "1h.close", "4h.high").
    Example: ``{"1h": Interval.hours(1), "4h": Interval.hours(4)}``"""
    exo_data: List[str] = field(default_factory=list)
    """Exogenous data series names to inject into signal evaluation.
    Each name corresponds to an ``exo/{name}/`` directory in the data store
    (written via ``bt.register_exo()``). Columns are ASOF-joined onto
    bar timestamps and accessible as ``col("exo.{name}.{column}")``.
    Example: ``["hashrate", "fear_greed"]``"""
    signal_source: Any = None
    """Signal data source. Dict mapping provider → list of normalized symbols.
    Example: ``{"binance": ["BTC-USDT:perp", "ETH-USDT:perp"]}``
    Also accepts a string (single provider for all symbols) for backward compat."""
    execution_source: Any = None
    """Execution data source. Same format as ``signal_source``.
    Fill prices come from this source. When absent, same as ``signal_source``.
    Example: ``{"dydx": ["BTC-USD:perp", "ETH-USD:perp"]}``"""
    pair_map: Dict[str, str] = field(default_factory=dict)
    """Explicit mapping from signal symbol to execution symbol.
    Required when signal and execution have different tickers.
    Example: ``{"BTC-USDT:perp": "BTC-USD:perp"}``"""
    option_underlyings: Dict[int, int] = field(default_factory=dict)
    """Maps an option symbol id to the symbol whose price settles it.

    Required for every option in the universe. Deribit settles against its own
    index, whose ticker matches no series you can ingest, so the substitute is
    yours to name (``BTC-PERPETUAL`` in practice). The engine refuses to run an
    option without one rather than settle it against its own last traded
    premium, which on an illiquid strike is days stale.
    Example: ``{2: 1}`` to settle option id 2 against symbol id 1."""
    option_margin_model: str = "none"
    """Margin formula short option positions pay: ``"none"`` or ``"deribit"``.

    ``"none"`` charges nothing, which is only honest when the strategy never
    sells an option. ``"deribit"`` applies the venue's published per-contract
    formula, refuses a short that does not fit initial margin, and force-closes
    the book when maintenance margin passes equity."""
    account_rules: Any = None
    """Prop-firm limits enforced *during* the run (see :class:`AccountRules`).
    ``None`` == no limits.

    A breached account is CLOSED: it liquidates, stops trading, and its equity
    is flat from there on. That is the whole point of the option, and it is why
    ``metrics`` afterwards describes a curve the account never traded. Read
    ``result.account`` first; the run says so in ``result.warnings``.

    The full run and the lite sweep walk the same rule, mark by mark, so a pass
    rate over a parameter grid means something. The fast kernels and the GPU
    sweep refuse the configuration by name instead of ignoring it."""
    option_contracts: Dict = field(default_factory=dict)
    """Contract terms per option symbol id. Filled automatically from the data
    store at run time; set it by hand only to override what was ingested.

    Positions on an option are counted in **units of the underlying**, not in
    exchange contracts. On Deribit the two are the same thing (contract size 1).
    On a listed equity option, one contract is 100 units: to hold one SPY
    contract quoted at 4.70, target 100, which costs the 470 a contract costs
    and settles for what a contract settles for. ``contract_size`` in the terms
    is what converts a position back into contracts."""
    book_levels: Optional[int] = None
    """Levels per side of the stored order book the run reads: every level
    stored when ``None`` (the default); a number reads the smaller of it and
    the depth stored.

    Every reader of the book sees that depth and no more: the queue ahead of
    a resting order (``fill_model={"queue": {"depth_source": "book"}}``), the
    book columns (``bt.book.bid_price(k)`` and the rest), the trade clock's
    quotes, the fill marks, and the book a quoting strategy reads at each
    wake-up (``bt.book.bid_price_at(k)``, ``bt.book.bid_levels()``). Past it
    the run knows nothing, as past the depth stored: an order resting further
    out joins a queue the book cannot measure (``assumed_queue`` then, or a
    refusal by name), and a book column naming a deeper level is refused
    before anything is read. A bound is for speed: on a BTCUSDT day stored
    200 levels deep as whole ladders, a maker quoting three levels a side
    every 200 ms runs in 0.75 to 1.3 s reading every level and about 0.55 s
    bounded to 25; stored as changes (the default shape), about 0.6 s either
    way."""
    # Deprecated — kept for backward compat
    provider: Optional[str] = None
    exo_sources: Dict = field(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        # A plain dataclass takes any attribute, so `cfg.tick_size = 0.1` (a
        # field of ExecutionConfig, not of this class) was stored and never
        # read: the run went ahead as if it had not been written. Only the
        # fields serialised by to_json_dict reach the engine, so any other
        # public name is refused where it is written.
        if name.startswith("_") or name in type(self).__dataclass_fields__:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(_unknown_config_attribute(type(self).__name__, name))

    def to_json_dict(self) -> dict:
        d: dict = {
            "universe": self.universe,
            "time_range": {
                "start": self.time_range_start,
                "end": self.time_range_end,
            },
            "bar_interval": self.bar_interval or {"Minutes": 1},
            "initial_capital": self.initial_capital,
            "currency": self.currency,
            "execution": self.execution.to_json_dict(),
            "fees": self.fees.to_json_dict(),
            "slippage": self.slippage or {"FixedBps": {"bps": 0.0}},
            "data_version": self.data_version,
            "rng_seed": self.rng_seed,
            "trading_days_per_year": self.trading_days_per_year,
            "risk_free_rate": self.risk_free_rate,
        }
        if self.output_resolution is not None:
            d["output_resolution"] = self.output_resolution
        if self.resample_to is not None:
            d["resample_to"] = self.resample_to
        if self.feature_sets:
            d["feature_sets"] = self.feature_sets
        if self.symbol_names:
            d["symbol_names"] = self.symbol_names
        if self.warmup_bars > 0:
            d["warmup_bars"] = self.warmup_bars
        if self.accuracy:
            d["precise"] = True
        if self.extra_timeframes:
            d["extra_timeframes"] = self.extra_timeframes
        if self.exo_data:
            d["exo_data"] = self.exo_data
        if self.signal_source:
            d["signal_source"] = self.signal_source
        if self.execution_source:
            d["execution_source"] = self.execution_source
        if self.option_contracts:
            d["option_contracts"] = {
                str(sid): spec for sid, spec in self.option_contracts.items()
            }
        if self.option_margin_model and self.option_margin_model != "none":
            d["option_margin_model"] = self.option_margin_model
        if self.account_rules is not None:
            rules = self.account_rules
            d["account_rules"] = (
                rules.to_json_dict() if hasattr(rules, "to_json_dict") else rules
            )
        if self.book_levels is not None:
            d["book_levels"] = int(self.book_levels)
        # Deprecated fields (backward compat)
        if self.provider:
            d["provider"] = self.provider
        if self.exo_sources:
            d["exo_sources"] = {
                str(sid): list(src) for sid, src in self.exo_sources.items()
            }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict())


def _unknown_config_attribute(owner: str, name: str) -> str:
    """The refusal for ``BacktestConfig.<name> = ...`` on a name it lacks."""
    msg = f"{owner} has no setting {name!r}: assigning it would have no effect."
    for holder, cls, path in (
        ("ExecutionConfig", ExecutionConfig, "execution"),
        ("FeeConfig", FeeConfig, "fees"),
    ):
        if name in cls.__dataclass_fields__:
            msg += (
                f" It is a setting of {holder}: set cfg.{path}.{name}, or pass "
                f"{holder}({name}=...) as {path}=."
            )
            break
    else:
        close = difflib.get_close_matches(name, BacktestConfig.__dataclass_fields__, n=3)
        if close:
            msg += " Did you mean " + " or ".join(repr(c) for c in close) + "?"
    return msg


def resolve_universe(
    universe: List[Union[int, str]],
    store: Any,
    symbol_names: Optional[Dict[str, int]] = None,
) -> List[int]:
    """Resolve a mixed list of symbol IDs and ticker names to integer IDs.

    Args:
        universe: List of integer IDs or string ticker names.
        store: A ``DataStore`` instance (must have ``resolve_symbol()``).
        symbol_names: Optional name-to-ID mapping (checked before store).

    Returns:
        List of integer symbol IDs.

    Raises:
        ValueError: If a ticker name cannot be resolved.
        TypeError: If store is None and string tickers are present.
    """
    result = []
    for item in universe:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, str):
            if symbol_names and item in symbol_names:
                result.append(symbol_names[item])
            elif store is None:
                raise TypeError(
                    f"DataStore required to resolve symbol name {item!r}. "
                    f"Pass integer IDs or provide a store."
                )
            else:
                result.append(store.resolve_symbol(item))
        else:
            result.append(int(item))
    return result
