"""Order-level simulation against a stored book and tape.

``mbt.sim.run`` hands a strategy function a simulated venue. The function moves
the clock, reads the book and the tape as they reach it, and posts, cancels and
replaces its own orders, which reach the venue after their latency. Each order
resting in the book is served by the same queue model as ``mbt.run``'s
``fill_model={"queue": ...}``: a strategy that rests one order at a time gets
the same fills through either door.

Example::

    import manifoldbt as mbt

    def maker(sim):
        while sim.elapse(mbt.Interval.millis(100)):
            book = sim.book("BTCUSDT")
            if book.best_bid is None:
                continue
            if not sim.orders("BTCUSDT", side="buy", status="live"):
                sim.post("BTCUSDT", "buy", book.best_bid, 0.01, tif="GTX")

    res = mbt.sim.run(maker, mbt.sim.Config(latency={"entry": mbt.Interval.millis(5)}),
                     store, symbols=["BTCUSDT"], start="2025-08-13", end="2025-08-14")
    res.fills_df()

What the venue does:

* an order that would execute on arrival takes the displayed opposite ladder
  within its limit, level by level, and the rest of it expires: the simulated
  fills take nothing out of the stored book, so a rest left behind would be
  crossed at once by the very liquidity it just took;
* an order that would not execute rests at the back of its level's queue,
  behind the size the book shows there and behind the strategy's own earlier
  orders at that price (one of them that leaves or shrinks moves the orders
  behind it up by what it held);
* a print at the price of a resting order burns its queue only from the side
  that consumes it (a selling aggressor for a bid, a buying one for an ask);
  a print strictly through the price serves the rest, whichever side crossed;
* at one venue instant the market comes first: the prints, then the book state
  that shows them, then the actions that arrive at that instant. A print
  stamped at a GTD order's ``expire_at`` still serves it;
* a cancellation that reaches the venue before its order (a cancel latency
  shorter than the entry latency) is refused, and the order then rests and
  can be served;
* a replacement at the same price keeps the order's place when its size is
  smaller or unchanged; a total size that does not exceed what filled cancels
  the rest (status ``cancelled``, ``qty`` unchanged);
* a GTD order whose ``expire_at`` is at or before its arrival expires without
  executing; a FOK that cannot fill whole ends ``expired``;
* with ``Config(prints_update_book="aggressed_side")`` or ``"both_sides"``,
  each print updates the book until the next book state, for whatever executes
  against the book: a market order sent on a print fills at that print;
* a stop order (``sim.post(..., trigger=p)``) waits at the venue, out of the
  book, until a print at or through ``p`` (``trigger_on="last"``) or a stored
  best ask (a buy) or best bid (a sell) at or through it (``"bid_ask"``); it
  then executes at the end of that instant, once every market event stamped
  there is in and before the actions arriving then, as a market order
  (``price=None``) or a limit order at ``price``;
* a bracket (``sim.post_bracket``) is a parent order and the exits the venue
  arms once it has filled, for what it filled: a stop, a target, an exit at a
  time. The first one set off takes the others off at the same instant, and
  what the book does not fill of it executes again at the venue's next event.

An order plan (:class:`Plan`) writes down what a strategy sends -- when it
decides, the order, the side, the size, the exits the venue holds -- and the
engine runs it without calling back into Python: ``mbt.sim.run(plan)``, or
``mbt.sim.sweep(plan, grid)`` over a grid of its ``mbt.param``, the variants in
parallel, each the run ``mbt.sim.run`` makes of it.

A market stored order by order (``mbt.ingest_mbo``) serves a resting order in
its exact place in the queue instead: behind the orders that reached its level
before it, served when the market executes an order that arrived after it, or
goes through its level, or brings an order of the other side to its price.
Its book by price is read as any stored book.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from manifoldbt._native import (
    Book,
    Market,
    Order,
    Sim,
    Wakeup,
    sim_run as _sim_run,
    sim_run_plan as _sim_run_plan,
    sim_sweep as _sim_sweep,
)

__all__ = [
    "Book",
    "Config",
    "Market",
    "Order",
    "Plan",
    "PlanSweep",
    "Sim",
    "SimResult",
    "Wakeup",
    "alternate",
    "bracket",
    "closes",
    "every",
    "limit",
    "market",
    "notional",
    "run",
    "sessions",
    "setups",
    "stop",
    "stop_limit",
    "sweep",
]

_LATENCY_KEYS = ("entry", "cancel", "response", "feed")
_PRINT_RULES = ("aggressed_side", "both_sides")
_STOP_IN_MARKET = ("trigger", "reject")


def _duration_ns(what: str, raw: Any) -> int:
    """Nanoseconds behind an ``Interval``, a ``timedelta`` or an int."""
    from manifoldbt.expr import _interval_to_nanos

    if raw == "Trades":
        raise TypeError(
            f"{what}: Interval.trades() is a simulation clock, not a duration; "
            "pass Interval.millis(n), Interval.seconds(n) or nanoseconds"
        )
    try:
        nanos = _interval_to_nanos(raw)
    except (ValueError, OverflowError) as err:
        raise ValueError(f"{what} must be a finite duration, got {raw!r} ({err})") from None
    if nanos is None and hasattr(raw, "total_seconds") and hasattr(raw, "microseconds"):
        nanos = (raw.days * 86_400 + raw.seconds) * 1_000_000_000 + raw.microseconds * 1_000
    if nanos is None:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(
                f"{what} takes a duration, got {raw!r}: Interval.millis(n), "
                "Interval.seconds(n), a datetime.timedelta, or a number of nanoseconds"
            )
        nanos = raw
    if nanos < 0:
        raise ValueError(f"{what} must be a duration of zero or more, got {nanos} ns")
    return int(nanos)


def _finite(what: str, raw: Any) -> float:
    """``raw`` as a float, refused by name when it is NaN or infinite."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise TypeError(f"{what} takes a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {raw!r}")
    return value


def _instant_ns(what: str, raw: Any) -> int:
    """Nanoseconds since the epoch, UTC, behind a date, an instant or an int."""
    from manifoldbt.helpers import date_to_ns

    if isinstance(raw, bool):
        raise TypeError(f"{what} takes a date, not a bool")
    if isinstance(raw, int):
        return raw
    if hasattr(raw, "value") and hasattr(raw, "tz_localize"):  # pandas.Timestamp
        ts = raw if raw.tzinfo is not None else raw.tz_localize("UTC")
        return int(ts.value)
    if isinstance(raw, datetime):
        dt = raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000) * 1_000
    if isinstance(raw, str):
        return date_to_ns(raw)
    raise TypeError(
        f"{what} takes a date (\"2025-08-13\"), a datetime, or nanoseconds since "
        f"the epoch; got {type(raw).__name__}"
    )


# The columns `Market.from_arrays` reads from each table; any other is left
# alone, and never converted.
_ARRAY_COLUMNS = {
    "quotes": ("ts_ns", "timestamp", "bid_px", "bid_qty", "ask_px", "ask_qty"),
    "trades": ("ts_ns", "timestamp", "price", "qty", "aggressor"),
}


def _arrow_array(what: str, name: str, raw: Any) -> Any:
    """One column as a contiguous ``pyarrow.Array``, without a copy when its
    buffer can be shared (a contiguous numpy array, an Arrow array)."""
    import pyarrow as pa

    if isinstance(raw, pa.ChunkedArray):
        return raw.combine_chunks()
    if isinstance(raw, pa.Array):
        return raw
    lib = type(raw).__module__.split(".")[0]
    try:
        if lib == "polars":
            return _arrow_array(what, name, raw.to_arrow())
        if lib == "pandas":
            return pa.Array.from_pandas(raw)
        return pa.array(raw)
    except (TypeError, ValueError, pa.ArrowException) as err:
        raise TypeError(
            f"{what}: column {name!r} cannot be read as an array "
            f"({type(raw).__name__}): {err}"
        ) from None


def _arrow_columns(obj: Any, what: str) -> Any:
    """The columns of ``obj`` that ``Market.from_arrays`` reads, as one
    ``pyarrow.RecordBatch``: from a pandas or polars DataFrame, a pyarrow
    Table or RecordBatch, a dict of arrays, or anything exporting an Arrow
    stream."""
    import pyarrow as pa

    wanted = _ARRAY_COLUMNS[what]
    if isinstance(obj, pa.RecordBatch):
        return obj
    lib = type(obj).__module__.split(".")[0]
    if isinstance(obj, pa.Table):
        cols = [(n, obj.column(n)) for n in obj.column_names if n in wanted]
    elif isinstance(obj, dict):
        cols = [(str(k), v) for k, v in obj.items() if str(k) in wanted]
    elif lib == "polars" and hasattr(obj, "to_arrow"):
        present = [n for n in obj.columns if n in wanted]
        table = obj.select(present).to_arrow()
        cols = [(n, table.column(n)) for n in table.column_names]
    elif lib == "pandas" and hasattr(obj, "items"):
        cols = [(str(k), v) for k, v in obj.items() if str(k) in wanted]
        index = getattr(obj.index, "name", None)
        has_ts = any(n in ("ts_ns", "timestamp") for n, _ in cols)
        if index in ("ts_ns", "timestamp") and not has_ts:
            cols.insert(0, (index, obj.index))
    elif hasattr(obj, "__arrow_c_stream__"):
        table = pa.table(obj)
        cols = [(n, table.column(n)) for n in table.column_names if n in wanted]
    else:
        raise TypeError(
            f"{what} takes a pandas or polars DataFrame, a pyarrow Table or "
            f"RecordBatch, or a dict of arrays; got {type(obj).__name__}"
        )
    arrays = [(n, _arrow_array(what, n, c)) for n, c in cols]
    lengths = {n: len(a) for n, a in arrays}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"{what}: columns differ in length: {lengths}")
    return pa.RecordBatch.from_arrays([a for _, a in arrays], names=[n for n, _ in arrays])


@dataclass
class Config:
    """How the simulated venue behaves.

    Attributes:
        latency: The four delays between the strategy and its venue, each a
            duration (``Interval.millis(n)``, a ``datetime.timedelta`` or
            nanoseconds): ``"entry"`` for a post or a replacement to reach the
            venue, ``"cancel"`` for a cancellation (``"entry"`` when left
            out), ``"response"`` for an acknowledgement or a fill to reach the
            strategy, ``"feed"`` for a market event to. Zero when left out.
        queue: The queue model, as ``fill_model={"queue": ...}`` takes it:
            ``{"model": "risk_adverse" | "power" | "log", "n": ...,
            "depth_source": "book" | "assumed", "assumed_queue": ...,
            "cancel_ahead_rate": ...}``, or ``{"model": "fifo"}``, the exact
            queue of a market stored order by order (refused on a book
            stored by price). Left out: the exact queue on a market stored
            order by order, ``risk_adverse`` reading the stored book on the
            others. Named, it is the model of every market of the run.
            ``depth_source="assumed"`` makes the queue ``assumed_queue``
            times the order, whatever the book shows: it is not brought down
            to the displayed size, which is the rule of ``"book"``. ``"power"``
            and ``"log"`` read the depth at the order's level, so they need
            ``depth_source="book"``: with ``"assumed"`` the run is refused by
            name before it starts.
        maker_fee_bps: Fee on a fill of a resting order, in basis points of the
            notional. Negative is a rebate: the fee is negative and credits the
            cash.
        taker_fee_bps: Fee on a fill at arrival; negative is a rebate too.
        initial_cash: Cash at the start.
        seed: Accepted and passed to the venue, but no queue model draws from
            it: ``risk_adverse``, ``power``, ``log`` and the exact queue are
            deterministic, so two seeds give the same run to the bit.
        prints_update_book: Let each print update the book that an order
            executing on arrival meets, until the next stored book state, so
            that a market order sent on a print (zero latency, and no book
            state at the print's instant) fills at that print's price. Two
            rules:

            * ``"aggressed_side"``: a print updates the side it traded
              against. One where the seller crossed leaves no bid above its
              price, at its price the print's size or the size the stored book
              shows there when it is larger, and the stored bids below it
              behind; one where the buyer crossed does the same to the asks.
              The other side stays as stored.
            * ``"both_sides"``: a print sets both sides, whoever crossed, as a
              top-of-book venue that executes on trades has them: each side is
              the print alone, its price for its size, no stored level behind
              it. A market order (``GTC`` or ``GTD``) larger than the print
              fills its rest one tick worse than the print; an ``IOC`` lets
              the rest expire, a ``FOK`` takes nothing, and a limit order that
              executes lets its rest expire.

            Each side keeps the last print that set it. Everything that
            executes against the book reads it: a market order, a limit order
            that would execute, a ``FOK``, the refusal of a ``GTX`` and of a
            replacement whose new price would execute. Under either rule,
            resting orders are served as without it, by the queue model, off
            the prints and the stored states; what the strategy reads
            (``sim.book()``, ``sim.trades()``) and the equity's mid do not
            change. The side a print traded against is the tape's aggressor: a
            print that came without one took the tick rule's side when it was
            ingested, or when ``Market.from_arrays`` built the market.
            ``None`` (the default): the stored book alone.
        stop_in_market: What the venue does with a stop order whose trigger
            the market has already reached when the order arrives (the last
            print, for ``trigger_on="last"``; the stored state standing then,
            for ``"bid_ask"``): ``"trigger"`` (the default) sets it off at
            once, and it executes at its arrival; ``"reject"`` refuses it, and
            it ends ``rejected`` with nothing filled.
    """

    latency: Optional[Dict[str, Any]] = None
    queue: Optional[Dict[str, Any]] = None
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    initial_cash: float = 0.0
    seed: int = 0
    prints_update_book: Optional[str] = None
    stop_in_market: str = "trigger"

    def _to_json(self) -> str:
        latency = dict(self.latency or {})
        unknown = set(latency) - set(_LATENCY_KEYS)
        if unknown:
            raise TypeError(
                f"unknown latency key(s) {sorted(unknown)}: mbt.sim.Config.latency takes "
                '"entry", "cancel", "response" and "feed"'
            )
        if self.seed < 0:
            raise ValueError(f"mbt.sim.Config.seed must be zero or more, got {self.seed!r}")
        rule = self.prints_update_book
        if rule is not None and not (isinstance(rule, str) and rule in _PRINT_RULES):
            raise ValueError(
                'mbt.sim.Config.prints_update_book takes "aggressed_side" or "both_sides" '
                f"(or None, the stored book alone), got {rule!r}"
            )
        stops = self.stop_in_market
        if not (isinstance(stops, str) and stops in _STOP_IN_MARKET):
            raise ValueError(
                'mbt.sim.Config.stop_in_market takes "trigger" (a stop already in the market '
                'when it arrives is set off at once) or "reject" (it is refused), '
                f"got {stops!r}"
            )
        body = {
            "latency": {
                k: _duration_ns(f"latency.{k}", latency[k]) for k in _LATENCY_KEYS if k in latency
            },
            "maker_fee_bps": _finite("mbt.sim.Config.maker_fee_bps", self.maker_fee_bps),
            "taker_fee_bps": _finite("mbt.sim.Config.taker_fee_bps", self.taker_fee_bps),
            "initial_cash": _finite("mbt.sim.Config.initial_cash", self.initial_cash),
            "seed": int(self.seed),
        }
        if self.queue is not None:
            queue = dict(self.queue)
            for key, value in queue.items():
                if isinstance(value, float):
                    _finite(f"mbt.sim.Config.queue[{key!r}]", value)
            body["queue"] = queue
        if rule is not None:
            body["prints_update_book"] = rule
        if stops != "trigger":
            body["stop_in_market"] = stops
        return json.dumps(body)


def run(
    strategy: Union[Callable[[Sim], Any], "Plan"],
    config: Optional[Config] = None,
    store: Any = None,
    symbols: Union[str, int, Sequence[Union[str, int]], None] = None,
    start: Any = None,
    end: Any = None,
    *,
    markets: Optional[Sequence[Market]] = None,
    levels: Optional[int] = None,
    by_day: Any = "auto",
    device: str = "cpu",
) -> "SimResult":
    """Run ``strategy(sim)`` against a simulated venue.

    Args:
        strategy: A function of one argument, the :class:`Sim`. The run ends
            when it returns; what it returns is ignored. Or a :class:`Plan`,
            run by the engine with its parameters at their defaults, without
            calling back into Python, on one market: the run each variant of
            :func:`sweep` is.
        config: A :class:`Config`; zero latency, no fees and the book-reading
            ``risk_adverse`` queue when left out.
        store: The store holding the book and the tape of each symbol.
        symbols: Tickers or symbol ids, one market each.
        start: First instant simulated: a date (``"2025-08-13"``), a datetime,
            or nanoseconds since the epoch.
        end: The instant the data stops, excluded.
        markets: Instead of ``store``/``symbols``/``start``/``end``, markets
            already built, from :meth:`Market.from_store` or written out by
            hand with :meth:`Market.from_ladders`.
        levels: Levels per side of the stored book to load: every stored
            level when left out (1000 for a market stored order by order); at
            most that many otherwise, all of them when the book is stored
            shallower. A level past it reads as absent, as past the stored
            depth, and an order resting there joins a queue the venue cannot
            measure (``assumed_queue``). Without ``assumed_queue``, that order
            is refused on its own and the run goes on: a new order ends
            ``rejected``, a replacement is answered ``replace_rejected`` and
            the order stays as it was. The first order refused on each market
            is said in a ``UserWarning`` (the price, the side, the levels
            read). On a market stored order by order, a side that shows fewer
            levels than are read shows every level it has: an order past its
            last level joins an empty queue.
        by_day: Hold each market a UTC day at a time (``True``), whole
            (``False``), or day by day when ``[start, end)`` spans more than
            one day (``"auto"``, the default). Held day by day, the engine
            holds the day the strategy is in, the one before it and the next
            ones, prepared ahead, whatever the length of the range; the run is
            the run of the markets held whole, bit for bit. ``sim.trades()``
            then keeps the prints of the day the strategy is in and of the day
            before: read it at least once a day.
        device: Where the days' book is prepared when held day by day:
            ``"cpu"`` (the default) or ``"cuda"``, with the same result. A
            build or a machine without CUDA prepares them on the CPU and says
            so in a warning.

    Returns:
        A :class:`SimResult`: every order, fill and order event as the venue
        recorded them, the equity at each wake-up, and the markouts after each
        fill.
    """
    is_plan = isinstance(strategy, Plan)
    if not is_plan and not callable(strategy):
        raise TypeError(
            "mbt.sim.run: strategy is a function of one argument, the simulation, "
            "or a mbt.sim.Plan"
        )
    config = config if config is not None else Config()
    if not isinstance(config, Config):
        raise TypeError("mbt.sim.run: config is a mbt.sim.Config")
    if markets is not None:
        if store is not None or symbols is not None or start is not None or end is not None:
            raise TypeError(
                "mbt.sim.run: give either markets=, or store, symbols, start and end"
            )
        markets = list(markets)
    else:
        if store is None or symbols is None or start is None or end is None:
            raise TypeError(
                "mbt.sim.run needs a store, symbols, start and end (or markets=)"
            )
        if isinstance(symbols, (str, int)):
            symbols = [symbols]
        start_ns, end_ns = _instant_ns("start", start), _instant_ns("end", end)
        if isinstance(by_day, str) and by_day == "auto":
            by_day = end_ns - start_ns > 86_400_000_000_000
        elif not isinstance(by_day, bool):
            raise ValueError(f'mbt.sim.run: by_day takes "auto", True or False, got {by_day!r}')
        if device not in ("cpu", "cuda", "gpu"):
            raise ValueError(f'mbt.sim.run: device takes "cpu" or "cuda", got {device!r}')
        if device != "cpu":
            if not by_day:
                raise ValueError(
                    'mbt.sim.run: device="cuda" prepares the days of markets held day by day; '
                    "leave by_day to \"auto\" or pass by_day=True"
                )
            from manifoldbt import _require_grant_for_gpu

            _require_grant_for_gpu(device, "GPU day preparation")
        markets = [
            Market.from_store(store, s, start_ns, end_ns, levels, by_day=by_day, device=device)
            for s in symbols
        ]
    if not markets:
        raise ValueError("mbt.sim.run: at least one market")
    if is_plan:
        raw = _sim_run_plan(strategy._to_json(markets), config._to_json(), markets)
    else:
        raw = _sim_run(strategy, markets, config._to_json())
    return SimResult(raw, config)


def _to_frame(cols: Dict[str, Any], times: Sequence[str], backend: str) -> Any:
    from manifoldbt.dataframe import _resolve_backend

    backend = _resolve_backend(backend)
    if backend == "pandas":
        import pandas as pd

        df = pd.DataFrame(cols)
        for c in times:
            df[c] = pd.to_datetime(df[c], unit="ns", utc=True)
        return df
    if backend == "polars":
        import polars as pl

        df = pl.DataFrame(cols)
        return df.with_columns(pl.col(c).cast(pl.Datetime("ns", "UTC")) for c in times)
    return cols


def _frame(
    arrow: Callable[[], Any],
    columns: Callable[[], Dict[str, Any]],
    times: Sequence[str],
    backend: str,
) -> Any:
    """A table of the record as a DataFrame, from its Arrow columns.

    ``arrow`` hands the table over from the engine without a copy; ``columns``
    is the same table as one list per column, kept for what Arrow cannot give
    the same way: the plain dict of the ``"arrow"`` backend, and an empty
    table, whose list columns pandas and polars type on their own.
    """
    from manifoldbt.dataframe import _resolve_backend, record_to_frame

    backend = _resolve_backend(backend)
    if backend in ("pandas", "polars"):
        batch = arrow()
        if batch.num_rows:
            return record_to_frame(batch, times, backend)
    return _to_frame(columns(), times, backend)


def _orders_columns(raw: Any) -> Dict[str, Any]:
    cols = raw.orders_columns()
    cols["sent_at"] = cols.pop("sent_ns")
    return cols


class SimResult:
    """What a simulation recorded, as the venue saw it."""

    __slots__ = ("_raw", "config", "_totals")

    def __init__(self, raw: Any, config: Config) -> None:
        self._raw = raw
        self.config = config
        self._totals = None

    @property
    def raw(self) -> Any:
        return self._raw

    @property
    def symbols(self) -> List[str]:
        return self._raw.symbols

    @property
    def position(self) -> Dict[str, float]:
        """Final position per symbol."""
        return self._raw.position

    @property
    def cash(self) -> float:
        return self._raw.cash

    @property
    def fees(self) -> float:
        return self._raw.fees

    @property
    def fill_marks(self) -> Dict[str, Any]:
        """What the market did after each fill, aggregated per horizon (100 ms,
        1 s, 10 s), the horizons counted from the fill itself
        (``anchor == "fill"``). Same fields as ``mbt.run``'s
        ``result.fill_marks``."""
        return self._raw.fill_marks

    @property
    def fill_fragility(self) -> Dict[str, Any]:
        """What the queue decided for the orders that rested, in the shape of
        ``mbt.run``'s ``result.fill_fragility`` under a queue: ``maker_fills``,
        ``queue`` (``queue_decided_fills``, ``traverse_fills``,
        ``fills_from_book_cross``, ``partial_fills``, ``book_unknown_at_post``:
        the orders posted behind ``assumed_queue``), ``would_fill_touch`` and
        ``would_fill_traverse`` (per life of an order at one level, whether a
        print reached it or went through it). ``touch_only_fills`` is a bar
        convention and stays 0.

        Units differ. ``queue_decided_fills``, ``traverse_fills`` (the
        ``traverse`` and the ``book_cross`` fills) and ``fills_from_book_cross``
        count fills, and the first two add up to ``maker_fills``;
        ``partial_fills`` and ``book_unknown_at_post`` count orders;
        ``would_fill_touch`` and ``would_fill_traverse`` count lives. An order
        served in several pieces is several fills but one life, so fills cannot
        be set against lives. On a market stored by price, with
        ``orders_served`` the orders that have a maker fill::

            would_fill_traverse <= orders_served
                               <= would_fill_touch + fills_from_book_cross

        and ``orders_served <= queue_decided_fills + traverse_fills``. A market
        stored order by order is served from its order events, not from the
        prints these two counters read: no such envelope holds there."""
        return self._raw.fill_fragility

    @property
    def levels_snapped(self) -> int:
        """Orders whose price was between two ticks of the symbol and moved to
        the passive side (a bid down, an ask up) when they were sent. A price
        a few ulps off a tick is that tick and is not counted."""
        return self._raw.levels_snapped

    def orders_df(self, backend: str = "auto") -> Any:
        """Every order as it ended: ``order_id``, ``symbol``, ``side``,
        ``price`` (``None`` for a market order), ``qty``, ``filled``,
        ``avg_price``, ``status``, ``tif``, ``trigger_price`` and
        ``trigger_on`` (``None`` for an order that is not a stop or a leg on a
        trigger), ``parent`` and ``role`` (``"stop"``, ``"target"``,
        ``"time"``: the legs of a bracket, ``None`` for any other order),
        ``sent_at`` (local time; for a leg, the venue instant it was armed).

        Built from Arrow columns the engine hands over without a copy; each
        call builds a new frame, so changing one leaves the next untouched."""
        return _frame(
            self._raw.orders_arrow, lambda: _orders_columns(self._raw), ["sent_at"], backend
        )

    def fills_df(self, backend: str = "auto") -> Any:
        """Every fill at the venue's instant: ``timestamp``, ``order_id``,
        ``symbol``, ``side``, ``price``, ``qty``, ``fee``, ``channel``
        (``"queue"``, ``"traverse"``, ``"book_cross"`` or ``"taker"``) and
        ``maker``.

        The columns are the venue's, as in :meth:`orders_df` and
        :meth:`events_df` (``qty``, ``fee``, ``side`` as ``"buy"`` or
        ``"sell"``). The trade log of ``mbt.run``, ``trades_df``, names them
        in its own words: ``quantity``, ``fill_price``, ``fees``,
        ``execution_timestamp``, ``side`` as 1 or 2."""
        return _frame(self._raw.fills_arrow, self._raw.fills_columns, ["timestamp"], backend)

    def events_df(self, backend: str = "auto") -> Any:
        """The order journal: what the strategy sent, at its local time, and
        what the venue did, at the venue's. ``queue_ahead`` is the volume ahead
        of an order when the venue put it in the book. A stop order is
        ``armed`` when it reaches the venue and waits, and ``triggered`` (at
        its trigger's price) when the market sets it off. A leg of a bracket
        is ``armed`` when the venue arms it, ``triggered`` when the market or
        its time sets it off (no price for an exit at a time), ``carried``
        (with the size left) when the book could not fill all of it, and
        ``cancelled`` when another leg of its bracket went first."""
        return _frame(self._raw.events_arrow, self._raw.events_columns, ["timestamp"], backend)

    def equity_df(self, backend: str = "auto") -> Any:
        """Cash plus each position marked at its venue mid, at each wake-up of
        the strategy."""

        def columns() -> Dict[str, Any]:
            ts, equity = self._raw.equity_columns()
            return {"timestamp": ts, "equity": equity}

        return _frame(self._raw.equity_arrow, columns, ["timestamp"], backend)

    def fill_marks_df(self, backend: str = "auto") -> Any:
        """The per-fill detail behind :attr:`fill_marks`; ``symbol_id`` is the
        market's index in :attr:`symbols`."""
        return _frame(
            self._raw.fill_marks_arrow,
            self._raw.fill_marks_detail,
            ["timestamp", "anchor"],
            backend,
        )

    @property
    def metrics(self) -> Dict[str, Any]:
        """Counts and totals of the run: orders, fills (maker and taker),
        volume and notional traded, fees, final equity, P&L against the
        initial cash, and the deepest drawdown of the equity samples.

        ``orders_filled`` counts the orders served at least in part, not the
        orders whose status is ``filled``: one filled in part and then
        cancelled or expired is counted.

        Counted in the engine, once: a new dict at each read, from totals kept
        after the first."""
        totals = self._totals
        if totals is None:
            # The builtin sum() compensates its rounding from Python 3.12 on:
            # the engine sums as the interpreter would, to the last bit.
            totals = self._raw.metrics_totals(sys.version_info >= (3, 12))
            self._totals = totals
        orders, served, fills, maker, taker, volume, notional, drawdown, last = totals
        final = last if last is not None else self.config.initial_cash
        return {
            "orders": orders,
            "orders_filled": served,
            "fills": fills,
            "maker_fills": maker,
            "taker_fills": taker,
            "volume": volume,
            "notional": notional,
            "fees": self.fees,
            "final_equity": final,
            "pnl": final - self.config.initial_cash,
            "max_drawdown": drawdown,
            "position": self.position,
        }

    def summary(self) -> str:
        """One line: orders, the orders served at least in part ('N filled'),
        fills, volume, P&L and fees."""
        m = self.metrics
        return (
            f"{m['orders']} orders, {m['orders_filled']} filled; {m['fills']} fills "
            f"({m['maker_fills']} maker, {m['taker_fills']} taker), volume {m['volume']:g}; "
            f"P&L {m['pnl']:.2f} after {m['fees']:.2f} of fees"
        )

    def __repr__(self) -> str:
        return f"SimResult(symbols={self.symbols}, {self.summary()})"


# ---------------------------------------------------------------------------
# Order plans
# ---------------------------------------------------------------------------

_UNITS = ("ticks", "bps", "pct", "price")
_TRIGGER_ON = ("last", "bid_ask")
_NS_PER_DAY = 86_400_000_000_000


def _plain(raw: Any) -> Any:
    """A numpy scalar as the Python number it holds; anything else as is."""
    if (
        hasattr(raw, "item")
        and hasattr(raw, "dtype")
        and not isinstance(raw, (bool, int, float, str, dict))
    ):
        return raw.item()
    return raw


def _is_param(value: Any) -> bool:
    from manifoldbt.expr import Expr

    return isinstance(value, Expr) and value._variant == "Parameter"


def _scalar(what: str, raw: Any, kind: str) -> Any:
    """``raw`` as the JSON a plan field holds: nanoseconds for a duration
    (``kind="ns"``), a number (``"num"``), a whole number (``"int"``) or a
    side (``"side"``)."""
    raw = _plain(raw)
    if kind == "ns":
        return _duration_ns(what, raw)
    if kind == "side":
        if isinstance(raw, str) and raw.lower() in ("buy", "sell"):
            return raw.lower()
        raise ValueError(f'{what} takes "buy" or "sell", got {raw!r}')
    if isinstance(raw, bool):
        raise TypeError(f"{what} takes a number, not a bool")
    if kind == "int":
        if isinstance(raw, float) and raw.is_integer():
            raw = int(raw)
        if not isinstance(raw, int):
            raise TypeError(f"{what} takes a whole number, got {raw!r}")
        return raw
    if isinstance(raw, int):
        return raw
    return _finite(what, raw)


def _field(what: str, raw: Any, kind: str, params: Dict[str, Any]) -> Any:
    """A plan field: ``{"param": name}`` for ``mbt.param(name, default=...)``
    (its default noted in ``params``), the value converted otherwise."""
    from manifoldbt.expr import Expr

    if _is_param(raw):
        name = raw._args[0]
        meta = raw._param_meta or {}
        default = meta.get("default")
        if default is not None:
            value = _scalar(f"{what} (the default of mbt.param({name!r}))", default, kind)
            if name in params and params[name] != value:
                raise ValueError(
                    f"mbt.param({name!r}) is given two defaults in one plan: "
                    f"{params[name]!r} and {value!r}"
                )
            params[name] = value
        return {"param": name}
    if isinstance(raw, Expr):
        raise TypeError(
            f"{what}: a plan field takes a number or mbt.param(...), not an expression"
        )
    return _scalar(what, raw, kind)


def _choice(what: str, raw: Any, allowed: Sequence[str]) -> str:
    if not (isinstance(raw, str) and raw in allowed):
        names = ", ".join(f'"{a}"' for a in allowed)
        raise ValueError(f"{what} takes {names}, got {raw!r}")
    return raw


class _Every:
    __slots__ = ("period", "on", "anchor")

    def __init__(self, period: Any, on: str, anchor: str) -> None:
        self.period = period
        self.on = _choice("every(on=)", on, ("clock", "print"))
        self.anchor = _choice("every(anchor=)", anchor, ("start", "session"))

    def _spec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "kind": "every",
            "period_ns": _field("every(period)", self.period, "ns", params),
            "on": self.on,
            "anchor": self.anchor,
        }

    def __repr__(self) -> str:
        return f"every({self.period!r}, on={self.on!r}, anchor={self.anchor!r})"


def every(period: Any, *, on: str = "clock", anchor: str = "start") -> _Every:
    """Decide every ``period`` (an ``Interval``, a ``timedelta``,
    nanoseconds, or ``mbt.param``), at ``anchor + k * period`` for ``k`` from
    1: ``anchor="start"`` counts from the first instant the market covers,
    ``"session"`` from each session's open (``Plan(sessions=...)``).

    ``on="clock"`` decides at the instant itself; ``on="print"`` at the first
    print the strategy sees at or after it, when that print comes inside the
    same entry window. The next instant is the first one strictly after the
    decision."""
    return _Every(period, on, anchor)


class _Setups:
    __slots__ = ("columns",)

    def __init__(self, columns: Dict[str, Any]) -> None:
        self.columns = columns

    def _spec(self, _params: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self.columns, kind="setups")

    def __repr__(self) -> str:
        return f"setups({len(self.columns['ts_ns'])} rows)"


def _column(table: Any, name: str) -> Any:
    """Column ``name`` of a DataFrame, a pyarrow table, or a dict of arrays;
    ``None`` when it has none."""
    import pyarrow as pa

    if isinstance(table, (pa.Table, pa.RecordBatch)):
        return table.column(name) if name in table.schema.names else None
    if isinstance(table, dict):
        return table.get(name)
    columns = getattr(table, "columns", None)
    if columns is not None and name in list(columns):
        return table[name]
    return None


def _as_list(what: str, name: str, raw: Any) -> List[Any]:
    import pyarrow as pa

    arr = _arrow_array(what, name, raw)
    if pa.types.is_floating(arr.type):
        return [None if (x is None or math.isnan(x)) else float(x) for x in arr.to_pylist()]
    return arr.to_pylist()


def setups(
    table: Any,
    *,
    ts: str = "ts_ns",
    side: Optional[str] = "side",
    level: Optional[str] = "level",
    size: Optional[str] = "size",
) -> _Setups:
    """Decide once per row of ``table`` (a pandas or polars DataFrame, a
    pyarrow table, or a dict of arrays), at its instant.

    ``ts`` is the strategy's local instant, nanoseconds since the epoch or a
    timestamp column, the rows in time order. The other columns are read when
    the table has them: ``side`` (``"buy"`` / ``"sell"``) for
    ``Plan(side="from_setups")``, ``level`` the reference price the entry's
    levels are set from (a missing value: the last print seen), ``size`` for
    ``Plan(size="from_setups")``. A row before the market's first instant, or
    outside every entry window of ``Plan(sessions=...)``, is not a
    decision."""
    import pyarrow as pa

    raw = _column(table, ts)
    if raw is None:
        raise ValueError(f"setups: the table has no {ts!r} column")
    arr = _arrow_array("setups", ts, raw)
    if pa.types.is_timestamp(arr.type):
        arr = arr.cast(pa.timestamp("ns", tz=arr.type.tz)).cast(pa.int64())
    elif pa.types.is_integer(arr.type):
        arr = arr.cast(pa.int64())
    else:
        raise TypeError(
            f"setups: column {ts!r} holds instants: integer nanoseconds or timestamps, "
            f"got {arr.type}"
        )
    if arr.null_count:
        raise ValueError(f"setups: column {ts!r} has a missing instant")
    cols: Dict[str, Any] = {"ts_ns": arr.to_pylist()}
    for key, name in (("side", side), ("level", level), ("size", size)):
        if name is None:
            continue
        col = _column(table, name)
        if col is None:
            continue
        values = _as_list("setups", name, col)
        if key == "side":
            values = [_setup_side(name, v) for v in values]
        cols[key] = values
    return _Setups(cols)


def _setup_side(name: str, v: Any) -> str:
    if isinstance(v, str) and v.lower() in ("buy", "sell"):
        return v.lower()
    if not isinstance(v, bool) and v in (1, -1):
        return "buy" if v == 1 else "sell"
    raise ValueError(f'setups: column {name!r} takes "buy" / "sell" (or 1 / -1), got {v!r}')


class _Entry:
    __slots__ = ("kind", "args")

    def __init__(self, kind: str, **args: Any) -> None:
        self.kind = kind
        self.args = args

    def _spec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        a = self.args
        out: Dict[str, Any] = {"kind": self.kind}
        if self.kind == "market":
            return out
        out["offset"] = _field(f"{self.kind}(offset)", a["offset"], "num", params)
        out["unit"] = a["unit"]
        if self.kind == "limit":
            out["tif"] = a["tif"]
            return out
        if self.kind == "stop_limit":
            out["limit_offset"] = _field(
                "stop_limit(limit_offset)", a["limit_offset"], "num", params
            )
            out["after_trigger"] = a["after_trigger"]
        out["on"] = a["on"]
        if a["expire_after"] is not None:
            out["expire_after_ns"] = _field(
                f"{self.kind}(expire_after)", a["expire_after"], "ns", params
            )
        return out

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.kind}({args})"


def market() -> _Entry:
    """Each decision sends a market order (``GTC``): it takes what the
    displayed opposite ladder offers at its arrival."""
    return _Entry("market")


def limit(offset: Any, *, unit: str = "ticks", tif: str = "IOC") -> _Entry:
    """Each decision sends a limit order through the market: a buy at the
    reference plus ``offset``, a sell at the reference minus it, on the tick
    grid. It executes at its arrival (``tif="IOC"``, or ``"FOK"``: whole or
    nothing) and the rest expires. The reference is the setup's ``level``, or
    the last print the strategy has seen."""
    return _Entry(
        "limit",
        offset=offset,
        unit=_choice("limit(unit=)", unit, _UNITS),
        tif=_choice("limit(tif=)", str(tif).upper(), ("IOC", "FOK")),
    )


def stop(
    offset: Any, *, unit: str = "ticks", on: str = "last", expire_after: Any = None
) -> _Entry:
    """Each decision sends a stop-market order: a buy set off at the reference
    plus ``offset``, a sell at the reference minus it, the trigger on the
    tick grid away from the market, set off by a print (``on="last"``) or by
    the stored best bid or ask (``"bid_ask"``). ``expire_after`` (a duration
    from its arrival at the venue) makes it ``GTD``. A stop the market has
    already reached when it arrives follows ``Config(stop_in_market=...)``."""
    return _Entry(
        "stop",
        offset=offset,
        unit=_choice("stop(unit=)", unit, _UNITS),
        on=_choice("stop(on=)", on, _TRIGGER_ON),
        expire_after=expire_after,
    )


def stop_limit(
    offset: Any,
    limit_offset: Any,
    *,
    unit: str = "ticks",
    on: str = "last",
    expire_after: Any = None,
    after_trigger: str = "rest",
) -> _Entry:
    """Each decision sends a stop-limit order: the trigger of :func:`stop`,
    and a limit ``limit_offset`` past it (above a buy's trigger, below a
    sell's), on the tick grid. Once set off it executes as a limit order, and
    what does not execute at once rests in the book (``after_trigger="rest"``)
    or expires (``"ioc"``)."""
    return _Entry(
        "stop_limit",
        offset=offset,
        limit_offset=limit_offset,
        unit=_choice("stop_limit(unit=)", unit, _UNITS),
        on=_choice("stop_limit(on=)", on, _TRIGGER_ON),
        expire_after=expire_after,
        after_trigger=_choice("stop_limit(after_trigger=)", after_trigger, ("rest", "ioc")),
    )


class _Alternate:
    __slots__ = ("first",)

    def __init__(self, first: Any) -> None:
        self.first = first

    def _spec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"kind": "alternate", "first": _field("alternate(first)", self.first, "side", params)}

    def __repr__(self) -> str:
        return f"alternate({self.first!r})"


def alternate(first: Any = "buy") -> _Alternate:
    """The side of each entry, ``first`` (``"buy"``, ``"sell"`` or
    ``mbt.param``) then the other, entry after entry: a decision skipped does
    not count."""
    return _Alternate(first)


class _Notional:
    __slots__ = ("amount", "lot")

    def __init__(self, amount: Any, lot: float) -> None:
        self.amount = amount
        self.lot = _finite("notional(lot=)", lot)

    def _spec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "kind": "notional",
            "amount": _field("notional(amount)", self.amount, "num", params),
            "lot": self.lot,
        }

    def __repr__(self) -> str:
        return f"notional({self.amount!r}, lot={self.lot!r})"


def notional(amount: Any, *, lot: float = 1) -> _Notional:
    """The size of each entry, ``floor(amount / price / lot) * lot``, the
    price the entry's own level (a stop's trigger, a limit's price) or, for a
    market entry, the mid the strategy sees. A size of zero sends nothing."""
    return _Notional(amount, lot)


class _Closes:
    __slots__ = ("n",)

    def __init__(self, n: Any) -> None:
        self.n = n

    def __repr__(self) -> str:
        return f"closes({self.n!r})"


def closes(n: Any) -> _Closes:
    """Hold a position until the ``n``-th session close strictly after its
    entry filled (``Plan(sessions=...)``)."""
    return _Closes(n)


class _Bracket:
    __slots__ = ("stop", "target", "unit", "on", "hold")

    def __init__(self, stop: Any, target: Any, unit: str, on: str, hold: Any) -> None:
        self.stop = stop
        self.target = target
        self.unit = _choice("bracket(unit=)", unit, _UNITS)
        self.on = _choice("bracket(on=)", on, _TRIGGER_ON)
        self.hold = hold

    def _spec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"on": self.on}
        for key in ("stop", "target"):
            value = getattr(self, key)
            if value is not None:
                out[key] = {
                    "value": _field(f"bracket({key})", value, "num", params),
                    "unit": self.unit,
                }
        if isinstance(self.hold, _Closes):
            out["hold"] = {"kind": "closes", "n": _field("closes(n)", self.hold.n, "int", params)}
        elif self.hold is not None:
            out["hold"] = {"kind": "ns", "ns": _field("bracket(hold)", self.hold, "ns", params)}
        return out

    def __repr__(self) -> str:
        return (
            f"bracket(stop={self.stop!r}, target={self.target!r}, unit={self.unit!r}, "
            f"on={self.on!r}, hold={self.hold!r})"
        )


def bracket(
    stop: Any = None,
    target: Any = None,
    *,
    unit: str = "ticks",
    on: str = "last",
    hold: Any = None,
) -> _Bracket:
    """The exits the venue holds for each position: a stop ``stop`` away from
    the price the entry filled at (below it for a position bought, above it
    for one sold), a target ``target`` away on the other side, in ``unit``
    (``"ticks"``, ``"bps"``, ``"pct"`` or ``"price"``), set off by ``on``, and
    a hold: a duration from the fill, or :func:`closes`. The first exit set
    off takes the others off. With an exit, a decision is skipped while the
    strategy knows of a live order or an open position."""
    return _Bracket(stop, target, unit, on, hold)


def _clock(what: str, raw: Any) -> Any:
    from datetime import time

    if isinstance(raw, time):
        return raw
    if isinstance(raw, str):
        try:
            return time.fromisoformat(raw)
        except ValueError:
            pass
    raise ValueError(f'{what} takes a local time of day, "09:30" or a datetime.time; got {raw!r}')


def _ns_between(a: Any, b: Any) -> int:
    """Nanoseconds from the time of day ``a`` to ``b`` on one day."""
    from datetime import date, datetime, timedelta

    d = date(2000, 1, 3)
    return (datetime.combine(d, b) - datetime.combine(d, a)) // timedelta(microseconds=1) * 1_000


class _Sessions:
    __slots__ = ("open", "close", "tz", "entries")

    def __init__(self, open: Any, close: Any, tz: Any, entries: Any) -> None:
        self.open = _clock("sessions(open=)", open)
        self.close = _clock("sessions(close=)", close)
        self.tz = tz
        if entries is not None:
            if not (isinstance(entries, (tuple, list)) and len(entries) == 2):
                raise ValueError(
                    'sessions(entries=) takes (from, until), two local times such as '
                    '("09:45", "15:30")'
                )
            entries = (
                _clock("sessions(entries=)", entries[0]),
                _clock("sessions(entries=)", entries[1]),
            )
        self.entries = entries

    def _zone(self) -> Any:
        from datetime import timezone, tzinfo

        if isinstance(self.tz, tzinfo):
            return self.tz
        if self.tz in ("UTC", "utc"):
            return timezone.utc
        try:
            import zoneinfo

            return zoneinfo.ZoneInfo(self.tz)
        except Exception as err:  # noqa: BLE001 - said by name below
            raise ValueError(
                f"sessions(tz=): the time zone {self.tz!r} is not known to this Python "
                f"({err}); on Windows, `pip install tzdata` brings the time zones"
            ) from None

    def _pairs(self, span: Sequence[int]) -> List[List[int]]:
        from datetime import datetime, timedelta, timezone

        zone = self._zone()
        day = timedelta(days=1)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

        def at(d: Any, t: Any) -> int:
            local = datetime.combine(d, t).replace(tzinfo=zone)
            return (local.astimezone(timezone.utc) - epoch) // timedelta(microseconds=1) * 1_000

        overnight = self.close <= self.open
        first = datetime.fromtimestamp(span[0] / 1e9, tz=timezone.utc).astimezone(zone).date()
        last = datetime.fromtimestamp(span[1] / 1e9, tz=timezone.utc).astimezone(zone).date()
        pairs: List[List[int]] = []
        d = first - day
        while d <= last + day:
            if d.weekday() < 5:
                close_day = d + day if overnight else d
                pairs.append([at(d, self.open), at(close_day, self.close)])
            d += day
        return pairs

    def _spec(self, span: Optional[Sequence[int]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"open_close": self._pairs(span) if span is not None else []}
        if self.entries is not None:
            start, until = self.entries
            length = _ns_between(self.open, self.close)
            if length <= 0:
                length += _NS_PER_DAY
            from_ns = _ns_between(self.open, start) % _NS_PER_DAY
            until_ns = _ns_between(until, self.close) % _NS_PER_DAY
            if from_ns + until_ns > length:
                raise ValueError(
                    "sessions(entries=): the window must lie within the session, its start "
                    "before its end"
                )
            out["entries_from_ns"] = from_ns
            out["entries_until_ns"] = until_ns
        return out

    def __repr__(self) -> str:
        return (
            f"sessions(open={self.open.isoformat()!r}, close={self.close.isoformat()!r}, "
            f"tz={self.tz!r}, entries={self.entries!r})"
        )


def sessions(open: Any, close: Any, tz: Any = "UTC", *, entries: Any = None) -> _Sessions:
    """The sessions of the market: each weekday from ``open`` to ``close``
    (local times of day, ``"09:30"``), in time zone ``tz`` (a name such as
    ``"America/New_York"``, or a ``tzinfo``), a close at or before the open
    being the next day's. They are set against the market's span when the
    plan runs, and a session that holds no event of the market (a holiday) is
    dropped. ``entries=(from, until)`` bounds, within each session, the
    window in which entries are decided."""
    return _Sessions(open, close, tz, entries)


class Plan:
    """An order plan: what a strategy sends, written down, which the engine
    runs without calling back into Python, alone (:func:`run`) or swept over
    a grid of its parameters (:func:`sweep`).

    Args:
        when: When it decides: :func:`every` or :func:`setups`.
        entry: What each decision sends: :func:`market`, :func:`limit`,
            :func:`stop` or :func:`stop_limit`.
        side: ``"buy"``, ``"sell"``, :func:`alternate`, or ``"from_setups"``.
        size: A size (a number or ``mbt.param``), :func:`notional`, or
            ``"from_setups"``.
        exit: The exits the venue holds for each position, :func:`bracket`;
            without it, every decision sends its entry.
        sessions: The market's sessions, :func:`sessions`: their entry windows
            bound the decisions.
        sample_every: The step of the equity samples the metrics read, a
            duration; a minute when left out.

    Every number and duration of the plan takes a value or
    ``mbt.param("name", default=...)``, which :func:`sweep` sweeps.
    """

    __slots__ = ("when", "entry", "side", "size", "exit", "sessions", "sample_every")

    def __init__(
        self,
        when: Any,
        entry: Any,
        side: Any,
        size: Any,
        exit: Any = None,
        sessions: Any = None,
        sample_every: Any = None,
    ) -> None:
        if not isinstance(when, (_Every, _Setups)):
            raise TypeError("Plan(when=) takes mbt.sim.every(...) or mbt.sim.setups(...)")
        if not isinstance(entry, _Entry):
            raise TypeError(
                "Plan(entry=) takes mbt.sim.market(), limit(...), stop(...) or stop_limit(...)"
            )
        if not (isinstance(side, _Alternate) or side in ("buy", "sell", "from_setups")):
            raise ValueError(
                'Plan(side=) takes "buy", "sell", mbt.sim.alternate(...) or "from_setups", '
                f"got {side!r}"
            )
        if exit is not None and not isinstance(exit, _Bracket):
            raise TypeError("Plan(exit=) takes mbt.sim.bracket(...)")
        if sessions is not None and not isinstance(sessions, _Sessions):
            raise TypeError("Plan(sessions=) takes mbt.sim.sessions(...)")
        self.when = when
        self.entry = entry
        self.side = side
        self.size = size
        self.exit = exit
        self.sessions = sessions
        self.sample_every = sample_every

    def _spec(self, markets: Sequence[Market]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if isinstance(self.side, _Alternate):
            side: Dict[str, Any] = self.side._spec(params)
        else:
            side = {"kind": self.side}
        if isinstance(self.size, _Notional):
            size: Dict[str, Any] = self.size._spec(params)
        elif isinstance(self.size, str) and self.size == "from_setups":
            size = {"kind": "from_setups"}
        else:
            size = {"kind": "qty", "qty": _field("Plan(size=)", self.size, "num", params)}
        body: Dict[str, Any] = {
            "when": self.when._spec(params),
            "entry": self.entry._spec(params),
            "side": side,
            "size": size,
        }
        if self.exit is not None:
            body["exit"] = self.exit._spec(params)
        if self.sessions is not None:
            span = markets[0].span if markets else None
            body["sessions"] = self.sessions._spec(span)
        if self.sample_every is not None:
            body["sample_every_ns"] = _duration_ns("Plan(sample_every=)", self.sample_every)
        body["params"] = params
        return body

    def _to_json(self, markets: Sequence[Market]) -> str:
        return json.dumps(self._spec(markets))

    def __repr__(self) -> str:
        parts = [
            f"when={self.when!r}",
            f"entry={self.entry!r}",
            f"side={self.side!r}",
            f"size={self.size!r}",
        ]
        if self.exit is not None:
            parts.append(f"exit={self.exit!r}")
        if self.sessions is not None:
            parts.append(f"sessions={self.sessions!r}")
        return f"Plan({', '.join(parts)})"


def _grid_value(what: str, raw: Any) -> Any:
    """A value of a parameter axis, as the JSON the engine reads: a duration
    in nanoseconds, a number, or a word."""
    from manifoldbt.expr import _interval_to_nanos

    raw = _plain(raw)
    if isinstance(raw, bool):
        raise TypeError(f"{what}: a value of the grid is a number or a word, not a bool")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict) or hasattr(raw, "total_seconds"):
        nanos = _interval_to_nanos(raw)
        return nanos if nanos is not None else _duration_ns(what, raw)
    if isinstance(raw, int):
        return raw
    return _finite(what, raw)


def _exec_value(path: str, raw: Any) -> Any:
    """A value of an execution axis: a latency in nanoseconds, a rule name or
    ``None``, a number."""
    what = f"execution_grid[{path!r}]"
    raw = _plain(raw)
    if path.startswith("latency."):
        return _duration_ns(what, raw)
    if raw is None or isinstance(raw, str):
        return raw
    if isinstance(raw, bool):
        raise TypeError(f"{what} takes a number or a name, not a bool")
    if isinstance(raw, int):
        return raw
    return _finite(what, raw)


def _axes(what: str, grid: Any, value: Callable[[str, Any], Any]) -> Dict[str, List[Any]]:
    if grid is None:
        return {}
    if not isinstance(grid, dict):
        raise TypeError(f"{what} takes a dict of {{name: [values]}}")
    out: Dict[str, List[Any]] = {}
    for name, values in grid.items():
        if not isinstance(name, str):
            raise TypeError(f"{what}: an axis is named by a string, got {name!r}")
        if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
            raise TypeError(f"{what}[{name!r}] takes a list of values, got {values!r}")
        out[name] = [value(name, v) for v in values]
    return out


def _sweep_market(
    markets: Optional[Sequence[Market]], store: Any, symbol: Any, start: Any, end: Any,
    by_day: Any,
) -> List[Market]:
    if markets is not None:
        if store is not None or symbol is not None or start is not None or end is not None:
            raise TypeError(
                "mbt.sim.sweep: give either markets=, or store, symbol, start and end"
            )
        markets = list(markets)
    else:
        if store is None or symbol is None or start is None or end is None:
            raise TypeError("mbt.sim.sweep needs markets=, or a store, a symbol, start and end")
        start_ns, end_ns = _instant_ns("start", start), _instant_ns("end", end)
        if isinstance(by_day, str) and by_day == "auto":
            by_day = end_ns - start_ns > _NS_PER_DAY
        elif not isinstance(by_day, bool):
            raise ValueError(f'mbt.sim.sweep: by_day takes "auto", True or False, got {by_day!r}')
        markets = [Market.from_store(store, symbol, start_ns, end_ns, None, by_day=by_day)]
    if len(markets) != 1:
        raise ValueError(
            f"mbt.sim.sweep: a plan runs on one market, got {len(markets)}: one sweep per symbol"
        )
    return markets


def sweep(
    plan: Plan,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    config: Optional[Config] = None,
    *,
    markets: Optional[Sequence[Market]] = None,
    store: Any = None,
    symbol: Union[str, int, None] = None,
    start: Any = None,
    end: Any = None,
    by_day: Any = "auto",
    execution_grid: Optional[Dict[str, Sequence[Any]]] = None,
    engine: str = "auto",
    max_parallelism: Optional[int] = None,
    keep: str = "metrics",
) -> "PlanSweep":
    """Every variant of ``plan`` over one market, in the engine, in parallel.

    Each variant is the run :func:`run` makes of the same plan with the same
    parameters and the same config, to the bit: its result does not depend on
    the other variants, on their order, or on the number of threads.

    Args:
        plan: A :class:`Plan`.
        param_grid: ``{name: [values]}`` for the ``mbt.param`` the plan names
            (durations as ``Interval``, ``timedelta`` or nanoseconds); a
            parameter left out takes its default. An axis no field reads is
            refused by name.
        config: A :class:`Config`.
        markets: The one market, built already; or ``store``, ``symbol``,
            ``start`` and ``end`` (and ``by_day``) to read it as
            :meth:`Market.from_store` does.
        by_day: Hold the market read from ``store`` a UTC day at a time
            (``True``), whole (``False``), or day by day over a range
            longer than a day (``"auto"``, the default). Held day by day,
            the variants go through the days together: each day is read
            once for all of them and let go once every one is past it, and
            what a variant holds overnight (a position, a stop armed, an
            exit at a later close) goes on to the next day with it. The
            results are those of the market held whole, to the bit.
        execution_grid: ``{path: [values]}`` over the keys of :class:`Config`,
            dotted (``"latency.entry"``, ``"latency.feed"``,
            ``"prints_update_book"``, ``"taker_fee_bps"``).
        engine: How the variants run, to the same results:
            ``"direct"`` reads the market only at the instants a variant
            acts, the book each order meets computed once for every variant;
            ``"sim"`` runs each variant on the order-level simulation, which
            plays every event of the market; ``"auto"`` (the default) takes
            ``"direct"`` for a plan it computes and ``"sim"`` otherwise, and
            :attr:`PlanSweep.engine_reason` says why. ``"direct"`` computes
            the plans in which no order rests in the book: entries at the
            market, limits that execute at their arrival (``tif="IOC"`` or
            ``"FOK"``), stops, stop-limits with ``after_trigger="ioc"``, with
            or without an exit, over one market, held whole or day by day,
            whose book holds whole ladders; an order the venue sets off is
            handed the first market event that reaches it. It refuses any
            other plan by name.
        max_parallelism: Threads; every core when left out.
        keep: ``"metrics"`` (the default) keeps one row of metrics per
            variant; ``"fills"`` also keeps every variant's fills.

    Returns:
        A :class:`PlanSweep`. The parameters vary in name order, the last the
        fastest, and the execution axes are slower than all of them: variant
        ``i`` runs parameter combination ``i % P`` under execution
        combination ``i // P``.
    """
    if not isinstance(plan, Plan):
        raise TypeError("mbt.sim.sweep: plan is a mbt.sim.Plan")
    config = config if config is not None else Config()
    if not isinstance(config, Config):
        raise TypeError("mbt.sim.sweep: config is a mbt.sim.Config")
    if engine not in ("auto", "sim", "direct"):
        raise ValueError(
            'mbt.sim.sweep: engine takes "auto" (the default), "sim" or "direct", '
            f"got {engine!r}"
        )
    if keep not in ("metrics", "fills"):
        raise ValueError(f'mbt.sim.sweep: keep takes "metrics" or "fills", got {keep!r}')
    markets = _sweep_market(markets, store, symbol, start, end, by_day)
    grid = _axes("param_grid", param_grid, lambda n, v: _grid_value(f"param_grid[{n!r}]", v))
    exec_grid = _axes("execution_grid", execution_grid, _exec_value)
    threads = 0 if max_parallelism is None else int(max_parallelism)
    if max_parallelism is not None and threads < 1:
        raise ValueError(
            f"mbt.sim.sweep: max_parallelism must be at least 1, got {max_parallelism!r}"
        )
    raw = _sim_sweep(
        plan._to_json(markets),
        json.dumps(grid),
        config._to_json(),
        markets,
        threads,
        json.dumps(exec_grid),
        keep,
        engine,
    )
    return PlanSweep(raw, plan, markets, config, keep)


class PlanSweep:
    """What :func:`sweep` gives: one row of metrics per variant, and any
    variant run again on demand.

    The columns of :meth:`to_df`: ``variant``, ``param_<name>`` per
    parameter axis, ``exec_<path>`` per execution axis, then the metrics:
    ``orders`` (entries and the legs of their brackets), ``entries``,
    ``entries_filled`` (whole or in part), ``entries_expired`` and
    ``entries_rejected`` (with nothing filled), ``entries_skipped_busy``,
    ``entries_skipped_no_reference``, ``entries_skipped_level`` (a level at
    or below zero), ``entries_skipped_size`` (a size of zero),
    ``exits_stop``, ``exits_target``, ``exits_time`` (exits that filled),
    ``exits_carried`` (exits whose rest the book could not fill at once and
    that executed again at a later event, each counted once), ``fills``,
    ``volume``, ``notional``, ``fees``, ``final_equity``, ``pnl``,
    ``max_drawdown`` (in cash, zero or less), ``round_trips`` (spans of the
    position from flat back to flat), ``win_rate``, ``avg_round_trip_pnl``,
    ``exposure`` (the share of equity samples with a position open) and
    ``position`` (at the end). The equity is read at the plan's samples
    (``Plan(sample_every=)``) and at the end.
    """

    __slots__ = ("_raw", "plan", "markets", "config", "keep")

    def __init__(
        self, raw: Any, plan: Plan, markets: List[Market], config: Config, keep: str
    ) -> None:
        self._raw = raw
        self.plan = plan
        self.markets = markets
        self.config = config
        self.keep = keep

    def __len__(self) -> int:
        return len(self._raw)

    @property
    def engine(self) -> str:
        """The way the variants ran: ``"direct"``, the market read at the
        instants each variant acts, or ``"sim"``, the order-level
        simulation. The results are the same either way."""
        return self._raw.engine

    @property
    def engine_reason(self) -> str:
        """Why they ran that way: the engine asked for, or what took
        ``engine="auto"`` to the simulation."""
        return self._raw.engine_reason

    def to_arrow(self) -> Any:
        """The table, as a ``pyarrow.RecordBatch`` handed over without a
        copy."""
        return self._raw.metrics_arrow()

    def to_df(self, backend: str = "auto") -> Any:
        """The table, one row per variant, as a DataFrame (``backend``:
        ``"pandas"``, ``"polars"``, ``"arrow"`` for a dict of lists, or
        ``"auto"``)."""

        def columns() -> Dict[str, Any]:
            batch = self.to_arrow()
            return {n: batch.column(i).to_pylist() for i, n in enumerate(batch.schema.names)}

        return _frame(self.to_arrow, columns, [], backend)

    def params(self, i: int) -> Dict[str, Any]:
        """The parameters variant ``i`` ran with (durations in nanoseconds)."""
        return json.loads(self._raw.params_json(i))

    def config_of(self, i: int) -> Config:
        """The :class:`Config` variant ``i`` ran under."""
        return Config(**json.loads(self._raw.config_json(i)))

    def run(self, i: int) -> SimResult:
        """Variant ``i`` run again, alone: the whole record of :func:`run`
        (orders, fills, journal, equity, markouts)."""
        return SimResult(self._raw.run(i, True), self.config_of(i))

    def fills_df(self, i: int, backend: str = "auto") -> Any:
        """The fills of variant ``i``: those the sweep kept
        (``keep="fills"``), or those of the variant run again."""
        kept = self._raw.fills(i)
        raw = kept if kept is not None else self._raw.run(i, False)
        return SimResult(raw, self.config_of(i)).fills_df(backend)

    def __repr__(self) -> str:
        return f"PlanSweep(variants={len(self)}, engine={self.engine!r})"
