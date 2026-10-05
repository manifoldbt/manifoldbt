"""``bt.reconcile``: a broker's fills against the backtest of the same days.

A queue depth, a latency, a cancellation rate: each of them makes the P&L of a
maker, and none of them is knowable from a backtest alone. The one thing that
can settle them is a journal of fills the market actually granted over the same
days, marked the same way and counted the same way. That is what this module
builds, and what :class:`Reconciliation` reads.

The measurement lives in the engine, which marks both sides
through the very same code the run's own ``fill_marks`` went through. Here we
only turn a DataFrame of fills into columns, and the answer back into tables.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_NS_PER_SECOND = 1_000_000_000

# What a journal may call each column, lower-cased. The first name found wins,
# so a broker export drops in without being renamed first.
_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "time", "ts", "datetime", "execution_timestamp",
                  "filled_at", "transacttime"],
    "side": ["side", "direction", "buy_sell", "action"],
    "price": ["price", "fill_price", "avg_price", "execution_price", "px"],
    "qty": ["qty", "quantity", "size", "amount", "filled_qty", "executed_qty"],
    "symbol_id": ["symbol_id", "symbol", "instrument_id"],
}

# How a journal spells a side. Anything else is refused by name rather than
# read as a sell, which would flip the sign of every mark it touches.
_BUY = {"buy", "b", "bid", "long", "1", "buy_long", "open_long"}
_SELL = {"sell", "s", "ask", "short", "2", "-1", "sell_short", "close_long"}


class Reconciliation:
    """What a journal of real fills and a backtest of the same days did.

    Every number is a measurement, and :attr:`verdict` reads them one sentence
    at a time. Nothing here names a setting to change: the point of the
    exercise is to put the two execution assumptions side by side, and
    ``execution_grid`` (see ``run_sweep``) is how the choice between them gets
    published as a surface rather than made in private.

    Tables (each a DataFrame, or a dict of columns without pandas):

    ``by_day_df()`` / ``by_hour_df()``
        fills served on each side, per UTC day and per hour of the UTC day,
        with ``matched``, ``live_only``, ``sim_only`` and ``served_ratio``.
    ``horizons_df()``
        the markout of each side at +100 ms, +1 s and +10 s, each with its
        bootstrap standard error, plus the difference taken PAIRWISE over the
        fills that matched -- the only form of the difference with a small
        error bar, since the two samples share their fills and their minutes.
    ``pairs_df()``
        one row per matched pair: the two timestamps, the two prices, the price
        edge in basis points, the time gap, and each side's three marks.

    Aggregates (``dict``): :attr:`service`, :attr:`half_spread`,
    :attr:`price_gap_bps`, :attr:`time_gap_ns`, plus the counts on the object
    itself.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw = raw

    # -- counts ---------------------------------------------------------

    @property
    def raw(self) -> Dict[str, Any]:
        """Everything the engine returned, as plain dicts and lists."""
        return self._raw

    @property
    def live_fills(self) -> int:
        """Fills in the journal."""
        return self._raw["live_fills"]

    @property
    def sim_fills(self) -> int:
        """Fills in the backtest's trade log."""
        return self._raw["sim_fills"]

    @property
    def matched(self) -> int:
        """Pairs: one real fill and the backtest fill it is the same fill as."""
        return self._raw["matched"]

    @property
    def live_only(self) -> int:
        """Real fills with no backtest equivalent inside the tolerance."""
        return self._raw["live_only"]

    @property
    def sim_only(self) -> int:
        """Backtest fills the market never granted."""
        return self._raw["sim_only"]

    @property
    def matched_share_live(self) -> float:
        """``matched / live_fills``."""
        return self._raw["matched_share_live"]

    @property
    def matched_share_sim(self) -> float:
        """``matched / sim_fills``."""
        return self._raw["matched_share_sim"]

    # -- aggregates -----------------------------------------------------

    @property
    def service(self) -> Dict[str, Any]:
        """How much of what was posted got served, on each side.

        ``live_rate`` and ``sim_rate`` share one denominator, the quotes the
        BACKTEST posted (``result.order_activity["orders_posted"]``): the two
        sides ran the same decisions, so the same posting count prices both.
        ``NaN`` when the run tracked no order activity; ``served_ratio``
        (live over backtest fills) reads without a denominator.
        """
        return self._raw["service"]

    @property
    def horizons(self) -> List[Dict[str, Any]]:
        """One entry per horizon: ``100ms``, ``1s``, ``10s``."""
        return self._raw["horizons"]

    @property
    def half_spread(self) -> Dict[str, Any]:
        """What the quote earned at the instant it was served, on each side.

        ``NaN`` without a stored book: the marks then read the last print,
        which is the mid give or take a half-spread -- enough for the drift,
        not enough to say what the quote captured.
        """
        return self._raw["half_spread"]

    @property
    def price_gap_bps(self) -> Dict[str, float]:
        """The price edge over the paired fills, in bp of the backtest price.

        **Negative = the real fill was worse**: it paid more on a buy, received
        less on a sell. Same sign convention as a markout.
        """
        return self._raw["price_gap_bps"]

    @property
    def time_gap_ns(self) -> Dict[str, float]:
        """``live - sim`` over the paired fills, in nanoseconds."""
        return self._raw["time_gap_ns"]

    @property
    def verdict(self) -> List[str]:
        """One factual sentence per quantity. No recommendation."""
        return self._raw["verdict"]

    @property
    def tolerance_ns(self) -> int:
        """The matching window that produced :attr:`matched`."""
        return self._raw["tolerance_ns"]

    # -- tables ---------------------------------------------------------

    def by_day_df(self, backend: str = "auto") -> Any:
        """Fills served per UTC day, on both sides. ``day`` is a timestamp."""
        return self._bucket_df("by_day", "day", True, backend)

    def by_hour_df(self, backend: str = "auto") -> Any:
        """Fills served per hour of the UTC day, on both sides.

        Hour of DAY, ``0`` to ``23``, not absolute hour: over several days it
        is the intraday profile, which is what a maker reads.
        """
        return self._bucket_df("by_hour", "hour", False, backend)

    def _bucket_df(self, key: str, name: str, as_time: bool, backend: str) -> Any:
        cols = dict(self._raw[key])
        cols[name] = cols.pop("key")
        ordered = {name: cols[name], **{k: v for k, v in cols.items() if k != name}}
        backend = _backend(backend)
        if backend == "pandas":
            import pandas as pd
            df = pd.DataFrame(ordered)
            if as_time:
                df[name] = pd.to_datetime(df[name], unit="ns", utc=True)
            return df
        if backend == "polars":
            import polars as pl
            df = pl.DataFrame(ordered)
            if as_time:
                df = df.with_columns(
                    pl.from_epoch(pl.col(name), time_unit="ns").alias(name)
                )
            return df
        return ordered

    def horizons_df(self, backend: str = "auto") -> Any:
        """The markouts of both sides at the three horizons, and the difference."""
        rows = self._raw["horizons"]
        backend = _backend(backend)
        if backend == "pandas":
            import pandas as pd
            return pd.DataFrame(rows)
        if backend == "polars":
            import polars as pl
            return pl.DataFrame(rows)
        return rows

    def pairs_df(self, backend: str = "auto") -> Any:
        """One row per matched pair, with each side's marks beside it."""
        cols = self._raw["pairs"]
        backend = _backend(backend)
        if backend == "pandas":
            import pandas as pd
            df = pd.DataFrame(cols)
            for c in ("live_timestamp", "sim_timestamp"):
                df[c] = pd.to_datetime(df[c], unit="ns", utc=True)
            return df
        if backend == "polars":
            import polars as pl
            df = pl.DataFrame(cols)
            return df.with_columns(
                [
                    pl.from_epoch(pl.col(c), time_unit="ns").alias(c)
                    for c in ("live_timestamp", "sim_timestamp")
                ]
            )
        return cols

    def summary(self) -> str:
        """The verdict, one sentence per line."""
        return "\n".join(self._raw["verdict"])

    def plot(self, **kwargs: Any) -> Any:
        """Shortcut for :func:`manifoldbt.plot.reconcile`."""
        from manifoldbt import plot

        return plot.reconcile(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Reconciliation(live={self.live_fills}, backtest={self.sim_fills}, "
            f"matched={self.matched})"
        )

    def _repr_html_(self) -> str:
        rows = "".join(f"<li>{line}</li>" for line in self._raw["verdict"])
        return f"<b>{self!r}</b><ul>{rows}</ul>"


def _backend(backend: str) -> str:
    from manifoldbt.dataframe import _resolve_backend

    return _resolve_backend(backend)


# ---------------------------------------------------------------------------
# Reading a journal
# ---------------------------------------------------------------------------


def _column(frame: Any, wanted: str) -> Any:
    """The column of *frame* a journal may have spelled several ways."""
    names = {str(c).strip().lower(): c for c in _columns_of(frame)}
    for alias in _ALIASES[wanted]:
        if alias in names:
            return _values_of(frame, names[alias])
    raise ValueError(
        f"live_fills has no {wanted!r} column. A journal needs a UTC timestamp, "
        f"a side, a price and a quantity; any of {_ALIASES[wanted]} is read as "
        f"{wanted!r}. Columns seen: {sorted(names.values(), key=str)}"
    )


def _columns_of(frame: Any) -> Any:
    if hasattr(frame, "columns"):
        return list(frame.columns)
    if isinstance(frame, dict):
        return list(frame)
    raise TypeError(
        "live_fills takes a pandas or polars DataFrame, or a dict of columns"
    )


def _values_of(frame: Any, name: Any) -> Any:
    import numpy as np

    if isinstance(frame, dict):
        return np.asarray(frame[name])
    col = frame[name]
    if hasattr(col, "to_numpy"):
        return col.to_numpy()
    return np.asarray(col)


def _timestamps_ns(raw: Any) -> Any:
    """A journal's timestamps as UTC nanoseconds.

    A naive datetime column is read as UTC and said so, rather than shifted by
    a local offset nobody wrote down: an hour of drift would put every fill in
    the wrong bucket and match none of them.
    """
    import numpy as np

    arr = np.asarray(raw)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]").astype("int64")
    if arr.dtype.kind in "iu":
        return arr.astype("int64")
    if arr.dtype.kind == "f":
        return arr.astype("int64")
    # Strings, or a column of Python datetimes.
    import pandas as pd

    parsed = pd.to_datetime(pd.Series(arr), utc=True)
    return parsed.values.astype("datetime64[ns]").astype("int64")


def _sides(raw: Any) -> Any:
    """A journal's sides as the trade log's own encoding: 1 buy, 2 sell."""
    import numpy as np

    arr = np.asarray(raw)
    if arr.dtype.kind in "iuf":
        out = np.where(np.asarray(arr, dtype="float64") > 0, 1, 2).astype("uint8")
        # An explicit 2 already means sell; only -1/0 and 1 need the mapping
        # above, so a log that uses the engine's own codes round-trips.
        both = np.asarray(arr, dtype="float64")
        out = np.where(both == 2, 2, out).astype("uint8")
        return out
    out = np.empty(len(arr), dtype="uint8")
    for i, v in enumerate(arr):
        key = str(v).strip().lower()
        if key in _BUY:
            out[i] = 1
        elif key in _SELL:
            out[i] = 2
        else:
            raise ValueError(
                f"live_fills row {i}: side {v!r} is neither a buy nor a sell. "
                f"Buy is any of {sorted(_BUY)}, sell any of {sorted(_SELL)}; "
                "an unreadable side would flip the sign of every mark it touches"
            )
    return out


def fills_to_columns(live_fills: Any, symbol_id: Optional[int]) -> Dict[str, Any]:
    """A journal DataFrame as the five columns the engine reads.

    Sorted by timestamp, because the marks accumulate a position fill by fill
    and the matching walks both sides forward. A stable sort, so two fills of
    one instant keep the order the journal wrote them in.
    """
    import numpy as np

    ts = _timestamps_ns(_column(live_fills, "timestamp"))
    side = _sides(_column(live_fills, "side"))
    price = np.asarray(_column(live_fills, "price"), dtype="float64")
    qty = np.asarray(_column(live_fills, "qty"), dtype="float64")

    n = len(ts)
    for name, col in (("side", side), ("price", price), ("qty", qty)):
        if len(col) != n:
            raise ValueError(
                f"live_fills: the {name!r} column has {len(col)} rows against "
                f"{n} timestamps"
            )

    try:
        symbols = np.asarray(_column(live_fills, "symbol_id"), dtype="uint32")
    except (ValueError, TypeError):
        if symbol_id is None:
            raise ValueError(
                "live_fills has no numeric symbol_id column, so reconcile cannot "
                "tell which symbol its fills belong to: pass symbol_id=<the id "
                "the run used>"
            ) from None
        symbols = np.full(n, symbol_id, dtype="uint32")

    order = np.argsort(ts, kind="stable")
    return {
        "ts": ts[order].tolist(),
        "side": side[order].tolist(),
        "price": price[order].tolist(),
        "qty": qty[order].tolist(),
        "symbol": symbols[order].tolist(),
    }
