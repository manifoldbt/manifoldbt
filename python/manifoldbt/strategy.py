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
from manifoldbt.expr import Expr, MultiExpr, lit, param as _param


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


def _collect_params(expr: "Expr", out: Dict[str, Any]) -> None:
    """Walk an Expr tree and collect all param() metadata."""
    if expr._param_meta is not None:
        name = expr._param_meta["name"]
        if name not in out:
            out[name] = expr._param_meta
    for arg in expr._args:
        _collect_params_arg(arg, out)


def _collect_params_arg(arg: Any, out: Dict[str, Any]) -> None:
    if isinstance(arg, Expr):
        _collect_params(arg, out)
    elif isinstance(arg, (list, tuple)):
        # Scan nodes carry their init/update expressions as LISTS of Exprs;
        # skipping them silently dropped every param() used inside a scan
        # ("strategy uses undefined parameters: q" on a swept Kalman).
        for item in arg:
            _collect_params_arg(item, out)
    elif isinstance(arg, str):
        # DynPeriod/DynFloat param name — check global registry
        from manifoldbt.expr import _param_registry
        if arg in _param_registry and arg not in out:
            out[arg] = _param_registry[arg]


class Strategy:
    """A backtester strategy definition.

    Serializes to JSON matching the Rust ``bt_strategy::StrategyDef``
    serde format. Supports both direct construction and fluent builder.
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
        self.position_sizing = _as_expr(expr, "position sizing")
        self._json_cache = None
        return self

    def param(
        self,
        name: str,
        default: Any = None,
        range: Optional[Tuple[Any, Any]] = None,
        description: str = "",
    ) -> "Strategy":
        """Register a sweep parameter (returns self for chaining).

        Args:
            name: Parameter name (must match ``param("name")`` in expressions).
            default: Default value.
            range: Optional ``(min, max)`` bounds for sweeps.
            description: Human-readable description.
        """
        self._parameters[name] = _param(name, default=default, range=range, description=description)
        self._json_cache = None
        return self

    def stop_loss(self, pct: float) -> "Strategy":
        """Convenience: attach a stop-loss order (returns self for chaining).

        Args:
            pct: Distance from entry as percentage (e.g. ``2.0`` = 2%).
        """
        if self._orders is None:
            self._orders = {}
        self._orders["stop_loss"] = {"stop_pct": pct}
        self._json_cache = None
        return self

    def take_profit(self, pct: float) -> "Strategy":
        """Convenience: attach a take-profit order (returns self for chaining).

        Args:
            pct: Distance from entry as percentage (e.g. ``5.0`` = 5%).
        """
        if self._orders is None:
            self._orders = {}
        self._orders["take_profit"] = {"profit_pct": pct}
        self._json_cache = None
        return self

    def trailing_stop(self, pct: float, use_high: bool = True) -> "Strategy":
        """Convenience: attach a trailing stop (returns self for chaining).

        Args:
            pct: Trail distance as percentage (e.g. ``3.0`` = 3%).
            use_high: Track bar high/low (True) or close (False).
        """
        if self._orders is None:
            self._orders = {}
        self._orders["trailing_stop"] = {"trail_pct": pct, "use_high": use_high}
        self._json_cache = None
        return self

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

        if self._orders is None:
            self._orders = {}
        entry: Dict[str, Any] = {
            "price": entry_price(offset_bps=offset_bps, price=price, signal=signal),
            "trigger": trigger,
            "time_in_force": time_in_force,
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

        Args:
            offset_bps: Distance from the signal close in bps (positive = more passive).
            price: A fixed price level.
            signal: Name of a signal to read the level from.
            time_in_force: ``"GTC"`` (default), ``{"GTB": 5}``, or ``"IOC"``.
            size_at_fill_price: Size off the order's level instead of the close.
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

        # Auto-collect params from expressions (bt.param() in indicators)
        auto_params: Dict[str, Any] = {}
        for expr in self.signals.values():
            _collect_params(expr, auto_params)
        _collect_params(self.position_sizing, auto_params)

        # Merge: explicit .param() calls override auto-collected
        all_metas: Dict[str, Any] = {}
        for name, meta in auto_params.items():
            all_metas[name] = meta
        for name, param_expr in self._parameters.items():
            meta = getattr(param_expr, "_param_meta", None)
            if meta is not None:
                all_metas[name] = meta

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
        return out

    def to_json(self) -> str:
        """Serialize to a JSON string matching Rust ``StrategyDef``.

        Memoised: builder mutations reset the cache, so repeated runs of the
        same strategy skip the (O(expression tree)) re-serialisation.
        """
        if self._json_cache is None:
            self._json_cache = json.dumps(self.to_json_dict())
        return self._json_cache
