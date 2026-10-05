"""Strategy definition that serializes to Rust ``StrategyDef`` JSON.

Supports both direct construction and fluent builder pattern::

    # Direct (existing API)
    strategy = Strategy(name="ema", signals={...}, position_sizing=expr)

    # Fluent builder (new)
    strategy = (
        Strategy.create("ema")
        .signal("fast", ema(close, 10))
        .signal("slow", ema(close, 25))
        .signal("trend", col("fast") > col("slow"))
        .size(when(col("trend"), lit(0.5), lit(0.0)))
        .stop_loss(pct=2.0)
    )
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from manifoldbt._serde import scalar_value_to_json
from manifoldbt.expr import Expr, MultiExpr, _ParamRef, lit, param as _param


def _as_expr(value: Any, where: str) -> "Expr":
    """The expression ``where`` needs, or an error naming what arrived instead.

    Everything downstream -- parameter collection, JSON serialisation -- assumes
    an ``Expr`` and says so in the language of its own internals when it is not
    one: ``'tuple' object has no attribute '_param_meta'`` for a multi-output
    indicator handed over whole, the same on ``str`` for an expression written
    as text. Both are unactionable, so both are caught at the door instead.
    """
    if isinstance(value, Expr):
        return value
    if isinstance(value, MultiExpr):
        raise value._reject(f"{where} takes one of them.")
    if isinstance(value, (list, tuple)) and any(isinstance(v, Expr) for v in value):
        raise TypeError(
            f"{where} takes one expression, got a {type(value).__name__} of "
            f"{len(value)}. Unpack it and pass the one you meant."
        )
    if isinstance(value, str):
        raise TypeError(
            f"{where} takes an expression, got the string {value!r}. Expressions "
            f"are built, not parsed from text: col(\"close\"), sma(col(\"close\"), 20), "
            f"col(\"fast\") > col(\"slow\")."
        )
    raise TypeError(
        f"{where} takes an expression, got {type(value).__name__}. Wrap a constant "
        f"in lit(), a column in col()."
    )


_QUOTE_TIFS = ("GTX", "GTC", "IOC", "FOK")

_SIZE_WITH_QUOTES = (
    "a strategy that quotes sizes each quote itself (quote(size=...)), in units, and "
    "has no target position for .size() to size: drop .size(), or drop the quotes"
)


def _quote_expr(value: Any, where: str) -> "Expr":
    """A quote's price or size: an expression, or a number made one."""
    if isinstance(value, bool):
        raise TypeError(f"{where} takes a number or an expression, got a bool")
    if isinstance(value, (int, float)):
        return lit(float(value))
    return _as_expr(value, where)


def _quote_cooldown(value: Any, where: str) -> int:
    """A quote's cooldown in nanoseconds: an ``Interval`` or a number of
    nanoseconds, as ``execution.latency`` takes them."""
    from manifoldbt.expr import _interval_to_nanos

    if isinstance(value, str) and value == "Trades":
        raise TypeError(
            f"{where}: Interval.trades() is a simulation clock, not a duration. "
            "Pass Interval.millis(n) or Interval.seconds(n)."
        )
    nanos = _interval_to_nanos(value)
    if nanos is None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{where} takes a duration: Interval.millis(n), Interval.seconds(n), "
                f"or a number of nanoseconds, got {value!r}"
            )
        nanos = value
    if nanos < 0:
        raise ValueError(f"{where} must be a duration of zero or more, got {nanos} ns")
    return int(nanos)


def _same_value(a: Any, b: Any) -> bool:
    """Whether two declared defaults are the same value: ``20`` and ``20.0``
    are, ``1`` and ``True`` are not, and a NaN is the same as a NaN."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or (a != a and b != b)
    return type(a) is type(b) and a == b


class _Declarations:
    """The ``param()`` declarations found in ONE strategy's expressions.

    Every metadata comes from the expression tree itself (a ``Parameter`` or
    ``Choice`` node, or the name an indicator period or span kept, see
    ``_ParamRef``), never from state shared with another strategy. Several
    declarations of one name merge: one without a default takes the default
    another gives, and the first range and the first description given are
    kept. Two DIFFERENT defaults are a conflict, recorded here and refused by
    ``Strategy.to_json_dict`` unless the strategy settles it with ``.param()``.
    """

    def __init__(self) -> None:
        self.metas: Dict[str, Dict[str, Any]] = {}
        self.conflicts: Dict[str, List[Any]] = {}

    def add(self, meta: Dict[str, Any]) -> None:
        name = meta["name"]
        seen = self.metas.get(name)
        if seen is None:
            self.metas[name] = meta
            return
        if seen is meta:
            return
        merged = dict(seen)
        new = meta.get("default")
        old = seen.get("default")
        if old is None:
            merged["default"] = new
        elif new is not None and not _same_value(old, new):
            values = self.conflicts.setdefault(name, [old])
            if not any(_same_value(v, new) for v in values):
                values.append(new)
        for key in ("range", "description"):
            if not merged.get(key) and meta.get(key):
                merged[key] = meta[key]
        self.metas[name] = merged


def _collect_params(expr: "Expr", out: _Declarations) -> None:
    """Walk an Expr tree and collect the param() declarations it holds."""
    if expr._param_meta is not None:
        out.add(expr._param_meta)
    for arg in expr._args:
        _collect_params_arg(arg, out)


def _collect_params_arg(arg: Any, out: _Declarations) -> None:
    if isinstance(arg, Expr):
        _collect_params(arg, out)
    elif isinstance(arg, (list, tuple)):
        # Scan nodes carry their init/update expressions as LISTS of Exprs;
        # skipping them silently dropped every param() used inside a scan
        # ("strategy uses undefined parameters: q" on a swept Kalman).
        for item in arg:
            _collect_params_arg(item, out)
    elif isinstance(arg, _ParamRef):
        # A param() used as an indicator period or span: the name the node
        # kept carries the declaration. A plain string (a column, a signal, a
        # branch of a choice) declares nothing, whatever its name.
        out.add(arg._param_meta)


def _conflict_message(strategy: str, name: str, values: List[Any]) -> str:
    shown = ", ".join(repr(v) for v in values)
    return (
        f"strategy '{strategy}': parameter '{name}' is declared with "
        f"different defaults ({shown}). One run gives it one value, so the "
        f"strategy has to say which: give the default once "
        f"(param(\"{name}\", default=...)) and write param(\"{name}\") "
        f"elsewhere, or declare it on the strategy with "
        f".param(\"{name}\", default=...), which every use then reads."
    )


class Strategy:
    """A backtester strategy definition.

    Serializes to the JSON strategy definition the engine reads.
    Supports both direct construction and fluent builder.
    """

    def __init__(
        self,
        name: str,
        signals: Optional[Dict[str, Expr]] = None,
        position_sizing: Optional[Expr] = None,
        parameters: Optional[Dict[str, Expr]] = None,
        constraints: Optional[List[Any]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.name = name
        self.signals = signals if signals is not None else {}
        for key, value in self.signals.items():
            _as_expr(value, f"signal {key!r}")
        self.position_sizing = (
            lit(1.0)
            if position_sizing is None
            else _as_expr(position_sizing, "position sizing")
        )
        self._parameters = parameters or {}
        self._constraints = constraints or []
        self._description = description
        self._orders: Optional[Dict[str, Any]] = None
        self._quotes: List[Dict[str, Any]] = []
        # Set by .size() and by a position_sizing given here: a strategy that
        # quotes sizes each quote itself and refuses both.
        self._size_set = position_sizing is not None
        # Memoised to_json() (invalidated by every builder mutation below).
        self._json_cache: Optional[str] = None

    # ------------------------------------------------------------------
    # Fluent builder API
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str) -> "Strategy":
        """Create an empty strategy for fluent construction.

        Example::

            strategy = Strategy.create("my_strat").signal("x", expr).size(expr)
        """
        return cls(name=name)

    def signal(self, name: str, expr: Expr) -> "Strategy":
        """Add a named signal expression (returns self for chaining)."""
        self.signals[name] = _as_expr(expr, f"signal {name!r}")
        self._json_cache = None
        return self

    def size(self, expr: Expr) -> "Strategy":
        """Set the position sizing expression (returns self for chaining)."""
        if self._quotes:
            raise ValueError(_SIZE_WITH_QUOTES)
        self.position_sizing = _as_expr(expr, "position sizing")
        self._size_set = True
        self._json_cache = None
        return self

    def quote(
        self,
        side: str,
        price: Any,
        size: Any,
        tif: str = "GTX",
        enabled: Any = None,
        cooldown: Any = None,
    ) -> "Strategy":
        """Keep a quote at the venue (returns self for chaining).

        A strategy with quotes runs on the order-level simulation: ``mbt.run``
        wakes it every ``bar_interval`` (``Interval.millis(100)``, say), and at
        each wake-up every quote is evaluated, then acts, in the order the
        quotes were declared:

        1. ``enabled`` null, or ``price`` or ``size`` not a number (NaN, a
           null): the quote holds whatever it has;
        2. ``enabled`` false, or ``size`` zero or less: its live order is
           cancelled and the quote stands for nothing;
        3. a live order already at ``price`` (the same price on the book's
           grid): kept, with its place in the queue;
        4. otherwise the live order is cancelled and a new one is posted at
           ``price`` for ``size``, at the back of the queue.

        A quote with a ``cooldown`` posts nothing for that long after it
        last sent a cancellation, by rule 2 or by rule 4: a requote then
        cancels at once and posts at the first wake-up past the pause, at the
        price wanted then.

        ``price`` is put on the symbol's tick grid (the store's
        ``tick_size``) before rule 3 compares it and before it is posted: a
        price a few ulps off a tick is that tick (``ask + 2 * 0.1``), one
        between two ticks goes down for a buy and up for a sell and is
        counted in ``order_activity["levels_snapped"]``.

        An order that has filled, been cancelled or been refused frees its
        quote without a cancellation. The expressions may read the batch
        (the book columns ``bt.book.*``, the strategy's own signals, params)
        and the state at the wake-up: ``position()``, ``cash()``,
        ``live_qty(side)``, ``order_age(side)``, ``order_price(side)``,
        ``last_fill_px(side)``, ``queue_ahead(side)``, the ages
        ``position_age()``, ``last_fill_age(side)`` and
        ``last_cancel_age(side)`` (seconds, NaN before the event),
        ``bt.book.bid_price_at(level)`` and
        ``bt.book.bid_levels()`` and their ask twins, combined with
        arithmetic, comparisons, ``when``, ``round``, ``clip``, ``min_val`` /
        ``max_val``, ``abs_val`` and ``round_to`` / ``floor_to`` /
        ``ceil_to``. The state is what the strategy knows: fills and answers
        after ``execution.latency["response"]``, the book as it stood
        ``execution.latency["feed"]`` earlier.

        A strategy that quotes takes no ``.size()``, no bracket and no entry
        order: each quote carries its own size, in units.

        ``mbt.run_sweep``, ``run_sweep_lite``, ``run_batch`` and
        ``run_batch_lite`` take it as they take any strategy: each combination
        is the run ``mbt.run`` makes alone with those parameter values and
        that execution config. See "Quoting in the DSL" in the strategy
        authoring guide.

        Args:
            side: ``"buy"`` or ``"sell"``.
            price: The limit, an expression or a number.
            size: The quantity, in units, an expression or a number.
            tif: ``"GTX"`` (post-only, the default: refused on arrival if it
                would cross), ``"GTC"``, ``"IOC"`` or ``"FOK"``.
            enabled: A condition; the quote stands while it holds. Always
                when left out.
            cooldown: How long the quote posts nothing after it sent a
                cancellation (a withdrawal, or the cancel of a requote),
                counted from the sending on the strategy's clock:
                ``Interval.millis(500)``, or a number of nanoseconds. A fill,
                an expired IOC or a refused GTX starts no pause, and a quote
                stands before its first cancellation. The pause is the
                quote's own, not its side's. ``None`` or zero: no pause.

        Example::

            clips = mbt.round(mbt.position() / 0.01)
            strategy = (
                mbt.Strategy.create("maker")
                .quote("buy", bt.book.bid_price_at(mbt.clip(clips, 0, 9) + 1), 0.01,
                       enabled=mbt.position() < 0.05)
            )
        """
        if self._size_set:
            raise ValueError(_SIZE_WITH_QUOTES)
        if not isinstance(side, str) or side.lower() not in ("buy", "sell"):
            raise ValueError(f'quote: side must be "buy" or "sell", got {side!r}')
        if not isinstance(tif, str) or tif.upper() not in _QUOTE_TIFS:
            raise ValueError(
                'quote: tif must be "GTX" (post-only), "GTC", "IOC" or "FOK", '
                f"got {tif!r}"
            )
        n = len(self._quotes) + 1
        spec: Dict[str, Any] = {
            "side": side.lower(),
            "price": _quote_expr(price, f"the price of quote {n}"),
            "size": _quote_expr(size, f"the size of quote {n}"),
            "tif": tif.upper(),
        }
        if enabled is not None:
            if isinstance(enabled, bool):
                spec["enabled"] = lit(enabled)
            else:
                spec["enabled"] = _as_expr(enabled, f"the condition of quote {n}")
        if cooldown is not None:
            nanos = _quote_cooldown(cooldown, f"the cooldown of quote {n}")
            # Zero is no pause: left out, so the JSON is the one without it.
            if nanos > 0:
                spec["cooldown_ns"] = nanos
        self._quotes.append(spec)
        self._json_cache = None
        return self

    @property
    def quotes(self) -> List[Dict[str, Any]]:
        """The quotes declared with :meth:`quote`, in order."""
        return list(self._quotes)

    def param(
        self,
        name: str,
        default: Any = None,
        range: Optional[Tuple[Any, Any]] = None,
        description: str = "",
    ) -> "Strategy":
        """Register a sweep parameter (returns self for chaining).

        A ``param()`` used in the strategy's expressions is declared by that
        use; this call is for a parameter no expression of the strategy holds
        (a custom fill rule's), or to settle its default: the value given here
        wins over every default written in the expressions.

        Args:
            name: Parameter name (must match ``param("name")`` in expressions).
            default: Default value.
            range: Optional ``(min, max)`` bounds for sweeps.
            description: Human-readable description.
        """
        self._parameters[name] = _param(name, default=default, range=range, description=description)
        self._json_cache = None
        return self

    def _exit_order(self, key: str, spec: Dict[str, Any], side: str) -> "Strategy":
        from .config import exit_order

        if self._orders is None:
            self._orders = {}
        self._orders[key] = exit_order(spec, side, key)
        self._json_cache = None
        return self

    def stop_loss(
        self,
        pct: Optional[float] = None,
        side: str = "both",
        *,
        signal: Optional[str] = None,
    ) -> "Strategy":
        """Convenience: attach a stop-loss order (returns self for chaining).

        Pass exactly one of ``pct`` (one distance for the whole run) or
        ``signal`` (the name of a signal holding the distance, in percent, so
        it can widen with volatility). A ``signal`` distance is read on the bar
        whose signal opened the trade and frozen for the life of that trade::

            stop_dist = mbt.lit(2.0) * atr(14) / close * mbt.lit(100.0)
            strategy.signal("stop_dist", stop_dist).stop_loss(signal="stop_dist")

        Args:
            pct: Distance from entry as percentage (e.g. ``2.0`` = 2%).
            side: ``"both"`` (default), ``"long"`` or ``"short"``: which
                positions the stop arms on. A stop on the shorts only leaves
                the longs without one.
            signal: Name of a signal holding the distance in percent.
        """
        from .config import exit_distance

        return self._exit_order(
            "stop_loss",
            {"stop_pct": exit_distance(pct, signal, "stop_loss")},
            side,
        )

    def take_profit(
        self,
        pct: Optional[float] = None,
        side: str = "both",
        *,
        signal: Optional[str] = None,
    ) -> "Strategy":
        """Convenience: attach a take-profit order (returns self for chaining).

        Pass exactly one of ``pct`` or ``signal``; see :meth:`stop_loss` for
        what a signal distance means.

        Args:
            pct: Distance from entry as percentage (e.g. ``5.0`` = 5%).
            side: ``"both"`` (default), ``"long"`` or ``"short"``.
            signal: Name of a signal holding the distance in percent.
        """
        from .config import exit_distance

        return self._exit_order(
            "take_profit",
            {"profit_pct": exit_distance(pct, signal, "take_profit")},
            side,
        )

    def trailing_stop(
        self,
        pct: Optional[float] = None,
        use_high: bool = True,
        side: str = "both",
        *,
        signal: Optional[str] = None,
    ) -> "Strategy":
        """Convenience: attach a trailing stop (returns self for chaining).

        Pass exactly one of ``pct`` or ``signal``. A signal distance is frozen
        at the entry like the other two; the ratchet that follows is unchanged.

        Args:
            pct: Trail distance as percentage (e.g. ``3.0`` = 3%).
            use_high: Track bar high/low (True) or close (False).
            side: ``"both"`` (default), ``"long"`` or ``"short"``.
            signal: Name of a signal holding the trail distance in percent.
        """
        from .config import exit_distance

        return self._exit_order(
            "trailing_stop",
            {
                "trail_pct": exit_distance(pct, signal, "trailing_stop"),
                "use_high": use_high,
            },
            side,
        )

    def _entry(
        self,
        trigger: str,
        offset_bps: Optional[float],
        price: Optional[float],
        signal: Optional[str],
        time_in_force: Any,
        size_at_fill_price: bool,
        limit_price: Optional[Dict[str, Any]] = None,
    ) -> "Strategy":
        from .config import entry_price
        from .config import time_in_force as normalise_tif

        if self._orders is None:
            self._orders = {}
        entry: Dict[str, Any] = {
            "price": entry_price(offset_bps=offset_bps, price=price, signal=signal),
            "trigger": trigger,
            "time_in_force": normalise_tif(time_in_force),
            "size_at_fill_price": size_at_fill_price,
        }
        if limit_price is not None:
            entry["limit_price"] = limit_price
        self._orders["limit_entry"] = entry
        self._json_cache = None
        return self

    def limit_entry(
        self,
        *,
        offset_bps: Optional[float] = None,
        price: Optional[float] = None,
        signal: Optional[str] = None,
        time_in_force: Any = "GTC",
        size_at_fill_price: bool = False,
    ) -> "Strategy":
        """Rest the entry passively at a level instead of taking a market fill.

        Pass exactly one of ``offset_bps`` (distance from the signal-bar close),
        ``price`` (a fixed level), or ``signal`` (the name of a signal this
        strategy defines, so the level can be any series the DSL computes).

        A passive fill pays maker fees, takes no slippage, and lands on the
        level exactly. It can also never fill: check ``result.warnings``.

        An order that expires unfilled is posted again on the next bar, at that
        bar's level, for as long as the target holds and still differs from the
        position held, so a time-limited order is how a strategy requotes.

        Args:
            offset_bps: Distance from the signal close in bps (positive = more passive).
            price: A fixed price level.
            signal: Name of a signal to read the level from.
            time_in_force: ``"GTC"`` (default, rests until filled or until the
                target moves), ``{"GTB": 5}`` (expires after 5 EVENTS, then is
                posted again while the target holds), ``"IOC"`` (one event), or
                a DURATION -- ``Interval.seconds(1)``, ``Interval.millis(500)``
                -- which cancels the order on the first event stamped at or
                after ``posted_at + duration``, before that event's prints are
                offered to it. Requoting IS cancelling and posting again: the
                new order goes to the back of the queue with its own timestamp,
                even when its level has not moved.
            size_at_fill_price: Size off the order's level instead of the close.

        See "Time in force" and "Requoting is cancelling and posting again" in
        the strategy authoring guide.
        """
        return self._entry(
            "Limit", offset_bps, price, signal, time_in_force, size_at_fill_price
        )

    def stop_entry(
        self,
        *,
        offset_bps: Optional[float] = None,
        price: Optional[float] = None,
        signal: Optional[str] = None,
        time_in_force: Any = "GTC",
        size_at_fill_price: bool = False,
    ) -> "Strategy":
        """Enter on a breakout: fill once price trades **through** the level.

        The mirror of :meth:`limit_entry`. It crosses the book, so it pays taker
        fees and slippage, and a bar that gaps through the level fills at the
        open rather than at the level.
        """
        return self._entry(
            "Stop", offset_bps, price, signal, time_in_force, size_at_fill_price
        )

    def market_if_touched(
        self,
        *,
        offset_bps: Optional[float] = None,
        price: Optional[float] = None,
        signal: Optional[str] = None,
        time_in_force: Any = "GTC",
        size_at_fill_price: bool = False,
    ) -> "Strategy":
        """Wait for price to come to the level, then take a market fill.

        Same trigger as :meth:`limit_entry`, but the fill crosses the book:
        taker fees and slippage apply.
        """
        return self._entry(
            "MarketIfTouched",
            offset_bps,
            price,
            signal,
            time_in_force,
            size_at_fill_price,
        )

    def stop_limit_entry(
        self,
        *,
        stop: Optional[float] = None,
        stop_signal: Optional[str] = None,
        limit: Optional[float] = None,
        limit_signal: Optional[str] = None,
        time_in_force: Any = "GTC",
        size_at_fill_price: bool = False,
    ) -> "Strategy":
        """Breakout that arms a resting limit.

        The ``stop`` level arms the order; it then rests at ``limit`` and fills
        there with maker fees. Pass each level either as a number or as the name
        of a signal.
        """
        from .config import entry_price

        return self._entry(
            "StopLimit",
            None,
            stop,
            stop_signal,
            time_in_force,
            size_at_fill_price,
            limit_price=entry_price(price=limit, signal=limit_signal),
        )

    def describe(self, text: str) -> "Strategy":
        """Set strategy description (returns self for chaining)."""
        self._description = text
        self._json_cache = None
        return self

    @property
    def orders(self) -> Optional[Dict[str, Any]]:
        """Order config dict (stop-loss, take-profit, trailing), or None."""
        return self._orders

    def to_json_dict(self) -> dict:
        """Serialize to a dict matching Rust ``StrategyDef`` serde format."""
        # `signals` is a plain public dict, so a caller can drop anything into it
        # after construction. Re-check here rather than let the walk below trip
        # over it.
        for key, value in self.signals.items():
            _as_expr(value, f"signal {key!r}")
        _as_expr(self.position_sizing, "position sizing")

        # Auto-collect params from this strategy's own expressions: the
        # default of a param() is the one written in the expression that uses
        # it, never one another strategy of the process declared.
        declared = _Declarations()
        for expr in self.signals.values():
            _collect_params(expr, declared)
        _collect_params(self.position_sizing, declared)
        for q in self._quotes:
            for key in ("price", "size", "enabled"):
                if key in q:
                    _collect_params(q[key], declared)

        # Merge: explicit .param() calls override auto-collected. One without
        # a default keeps the default the expressions give (like a param()
        # without one does): it settles the range or the description, never
        # erases the value.
        all_metas: Dict[str, Any] = dict(declared.metas)
        settled = set()
        for name, param_expr in self._parameters.items():
            meta = getattr(param_expr, "_param_meta", None)
            if meta is None:
                continue
            base = all_metas.get(name)
            if base is not None and meta.get("default") is None:
                merged = dict(base)
                for key in ("range", "description"):
                    if meta.get(key):
                        merged[key] = meta[key]
                all_metas[name] = merged
            else:
                all_metas[name] = meta
                settled.add(name)
        # Two different defaults for one name, and no .param() to settle it:
        # whichever won would be an accident of the walk order, so refuse.
        for name, values in declared.conflicts.items():
            if name not in settled:
                raise ValueError(_conflict_message(self.name, name, values))

        # Build ParamSpec dicts
        params: Dict[str, Any] = {}
        for param_name, meta in all_metas.items():
            spec: Dict[str, Any] = {
                "name": meta["name"],
                "default": scalar_value_to_json(meta.get("default")),
                "description": meta.get("description", ""),
            }
            if meta.get("range") is not None:
                lo, hi = meta["range"]
                spec["range"] = [
                    scalar_value_to_json(lo),
                    scalar_value_to_json(hi),
                ]
            else:
                spec["range"] = None
            params[param_name] = spec

        out = {
            "name": self.name,
            "signals": {
                name: expr.to_json() for name, expr in self.signals.items()
            },
            "position_sizing": self.position_sizing.to_json(),
            "parameters": params,
            "constraints": list(self._constraints),
            "metadata": {
                "description": self._description,
            },
        }
        # Per-strategy SL/TP/trailing orders travel with the strategy so the
        # engine applies them per-strategy in a single batch/sweep call (the
        # Rust StrategyDef.orders field; omitted when unset for a clean JSON).
        if self._orders:
            out["orders"] = self._orders
        # Same rule for the quotes: absent unless declared, so a strategy that
        # does not quote serialises byte for byte as it always did.
        if self._quotes:
            out["quotes"] = [
                {k: (v.to_json() if isinstance(v, Expr) else v) for k, v in q.items()}
                for q in self._quotes
            ]
        return out

    def to_json(self) -> str:
        """Serialize to a JSON string matching Rust ``StrategyDef``.

        Memoised: builder mutations reset the cache, so repeated runs of the
        same strategy skip the (O(expression tree)) re-serialisation.
        """
        if self._json_cache is None:
            self._json_cache = json.dumps(self.to_json_dict())
        return self._json_cache
