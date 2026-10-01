"""Order-level simulation against a stored book and tape.

``bt.sim.run`` hands a strategy function a simulated venue. The function moves
the clock, reads the book and the tape as they reach it, and posts, cancels and
replaces its own orders, which reach the venue after their latency. Each order
resting in the book is served by the same queue model as ``bt.run``'s
``fill_model={"queue": ...}``: a strategy that rests one order at a time gets
the same fills through either door.

Example::

    import manifoldbt as bt

    def maker(sim):
        while sim.elapse(bt.Interval.millis(100)):
            book = sim.book("BTCUSDT")
            if book.best_bid is None:
                continue
            if not sim.orders("BTCUSDT", side="buy", status="live"):
                sim.post("BTCUSDT", "buy", book.best_bid, 0.01, tif="GTX")

    res = bt.sim.run(maker, bt.sim.Config(latency={"entry": bt.Interval.millis(5)}),
                     store, symbols=["BTCUSDT"], start="2025-08-13", end="2025-08-14")
    res.fills_df()

What the venue does:

* an order that would execute on arrival takes the displayed opposite ladder
  within its limit, level by level, and the rest of it expires: the simulated
  fills take nothing out of the stored book, so a rest left behind would be
  crossed at once by the very liquidity it just took;
* an order that would not execute rests at the back of its level's queue,
  behind the size the book shows there and behind the strategy's own earlier
  orders at that price;
* at one venue instant the market comes first: the prints, then the book state
  that shows them, then the actions that arrive at that instant.

A market stored order by order (``bt.ingest_mbo``) serves a resting order in
its exact place in the queue instead: behind the orders that reached its level
before it, served when the market executes an order that arrived after it, or
goes through its level, or brings an order of the other side to its price.
Its book by price is read as any stored book.
"""
from __future__ import annotations

import json
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
)

__all__ = ["Book", "Config", "Market", "Order", "Sim", "SimResult", "Wakeup", "run"]

_LATENCY_KEYS = ("entry", "cancel", "response", "feed")


def _duration_ns(what: str, raw: Any) -> int:
    """Nanoseconds behind an ``Interval``, a ``timedelta`` or an int."""
    from manifoldbt.expr import _interval_to_nanos

    if raw == "Trades":
        raise TypeError(
            f"{what}: Interval.trades() is a simulation clock, not a duration; "
            "pass Interval.millis(n), Interval.seconds(n) or nanoseconds"
        )
    nanos = _interval_to_nanos(raw)
    if nanos is None and hasattr(raw, "total_seconds") and hasattr(raw, "microseconds"):
        nanos = (raw.days * 86_400 + raw.seconds) * 1_000_000_000 + raw.microseconds * 1_000
    if nanos is None:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(
                f"{what} takes a duration: Interval.millis(n), Interval.seconds(n), "
                "a datetime.timedelta, or a number of nanoseconds"
            )
        nanos = raw
    if nanos < 0:
        raise ValueError(f"{what} must be a duration of zero or more, got {nanos} ns")
    return int(nanos)


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
        maker_fee_bps: Fee on a fill of a resting order, in basis points of the
            notional.
        taker_fee_bps: Fee on a fill at arrival.
        initial_cash: Cash at the start.
        seed: Seed of the draws a probabilistic queue model makes.
    """

    latency: Optional[Dict[str, Any]] = None
    queue: Optional[Dict[str, Any]] = None
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    initial_cash: float = 0.0
    seed: int = 0

    def _to_json(self) -> str:
        latency = dict(self.latency or {})
        unknown = set(latency) - set(_LATENCY_KEYS)
        if unknown:
            raise TypeError(
                f"unknown latency key(s) {sorted(unknown)}: bt.sim.Config.latency takes "
                '"entry", "cancel", "response" and "feed"'
            )
        body = {
            "latency": {
                k: _duration_ns(f"latency.{k}", latency[k]) for k in _LATENCY_KEYS if k in latency
            },
            "maker_fee_bps": float(self.maker_fee_bps),
            "taker_fee_bps": float(self.taker_fee_bps),
            "initial_cash": float(self.initial_cash),
            "seed": int(self.seed),
        }
        if self.queue is not None:
            body["queue"] = dict(self.queue)
        return json.dumps(body)


def run(
    strategy: Callable[[Sim], Any],
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
            when it returns; what it returns is ignored.
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
            level when left out; at most that many otherwise, all of them when
            the book is stored shallower. A level past it reads as absent,
            as past the stored depth, and an order resting there joins a queue
            the venue cannot measure (``assumed_queue``).
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
    if not callable(strategy):
        raise TypeError("bt.sim.run: strategy is a function of one argument, the simulation")
    config = config if config is not None else Config()
    if not isinstance(config, Config):
        raise TypeError("bt.sim.run: config is a bt.sim.Config")
    if markets is not None:
        if store is not None or symbols is not None or start is not None or end is not None:
            raise TypeError(
                "bt.sim.run: give either markets=, or store, symbols, start and end"
            )
        markets = list(markets)
    else:
        if store is None or symbols is None or start is None or end is None:
            raise TypeError(
                "bt.sim.run needs a store, symbols, start and end (or markets=)"
            )
        if isinstance(symbols, (str, int)):
            symbols = [symbols]
        start_ns, end_ns = _instant_ns("start", start), _instant_ns("end", end)
        if isinstance(by_day, str) and by_day == "auto":
            by_day = end_ns - start_ns > 86_400_000_000_000
        elif not isinstance(by_day, bool):
            raise ValueError(f'bt.sim.run: by_day takes "auto", True or False, got {by_day!r}')
        if device not in ("cpu", "cuda", "gpu"):
            raise ValueError(f'bt.sim.run: device takes "cpu" or "cuda", got {device!r}')
        if device != "cpu":
            if not by_day:
                raise ValueError(
                    'bt.sim.run: device="cuda" prepares the days of markets held day by day; '
                    "leave by_day to \"auto\" or pass by_day=True"
                )
            from manifoldbt import _require_grant_for_gpu

            _require_grant_for_gpu(device, "GPU day preparation")
        markets = [
            Market.from_store(store, s, start_ns, end_ns, levels, by_day=by_day, device=device)
            for s in symbols
        ]
    if not markets:
        raise ValueError("bt.sim.run: at least one market")
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
        (``anchor == "fill"``). Same fields as ``bt.run``'s
        ``result.fill_marks``."""
        return self._raw.fill_marks

    @property
    def fill_fragility(self) -> Dict[str, Any]:
        """What the queue decided for the orders that rested, in the shape of
        ``bt.run``'s ``result.fill_fragility`` under a queue: ``maker_fills``,
        ``queue`` (``queue_decided_fills``, ``traverse_fills``,
        ``fills_from_book_cross``, ``partial_fills``, ``book_unknown_at_post``:
        the orders posted behind ``assumed_queue``), ``would_fill_touch`` and
        ``would_fill_traverse`` (per life of an order at one level, whether a
        print reached it or went through it). ``touch_only_fills`` is a bar
        convention and stays 0."""
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
        ``avg_price``, ``status``, ``tif``, ``sent_at`` (local time).

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
        ``"sell"``). The trade log of ``bt.run``, ``trades_df``, names them
        in its own words: ``quantity``, ``fill_price``, ``fees``,
        ``execution_timestamp``, ``side`` as 1 or 2."""
        return _frame(self._raw.fills_arrow, self._raw.fills_columns, ["timestamp"], backend)

    def events_df(self, backend: str = "auto") -> Any:
        """The order journal: what the strategy sent, at its local time, and
        what the venue did, at the venue's. ``queue_ahead`` is the volume ahead
        of an order when the venue put it in the book."""
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
        m = self.metrics
        return (
            f"{m['orders']} orders, {m['orders_filled']} filled; {m['fills']} fills "
            f"({m['maker_fills']} maker, {m['taker_fills']} taker), volume {m['volume']:g}; "
            f"P&L {m['pnl']:.2f} after {m['fees']:.2f} of fees"
        )

    def __repr__(self) -> str:
        return f"SimResult(symbols={self.symbols}, {self.summary()})"
