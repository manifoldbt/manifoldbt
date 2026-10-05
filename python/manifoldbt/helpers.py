"""Convenience helpers for configuration.

Simplifies creating ``BacktestConfig`` by accepting human-readable dates,
named slippage models, and bar intervals.

Usage::

    from manifoldbt.helpers import date_to_ns, time_range, Slippage, Interval

    start, end = time_range("2022-01-01", "2024-01-01")
    config = bt.BacktestConfig(
        time_range_start=start,
        time_range_end=end,
        slippage=Slippage.fixed_bps(1.0),
        bar_interval=Interval.minutes(1),
    )
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

NANOS_PER_SECOND = 1_000_000_000


def date_to_ns(date_str: str) -> int:
    """Convert a date string to nanoseconds since Unix epoch (UTC).

    Accepted formats:
        - ``"2021-01-15"``
        - ``"2021-01-15 09:30:00"``
        - ``"2021-01-15T09:30:00"``
    """
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp()) * NANOS_PER_SECOND
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date '{date_str}'. "
        "Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
    )


def time_range(start: str, end: str) -> Tuple[int, int]:
    """Convert two date strings to a ``(start_ns, end_ns)`` tuple."""
    return date_to_ns(start), date_to_ns(end)


# ---------------------------------------------------------------------------
# Slippage model factories
# ---------------------------------------------------------------------------


class Slippage:
    """Factory for slippage configuration dicts."""

    @staticmethod
    def fixed_bps(bps: float) -> Dict[str, Any]:
        """Fixed basis-point slippage on every fill."""
        return {"FixedBps": {"bps": bps}}

    @staticmethod
    def volume_impact(impact_coeff: float, exponent: float = 1.5) -> Dict[str, Any]:
        """Volume-participation impact model.

        Cost = ``impact_coeff * participation_rate ^ exponent``.
        """
        return {"VolumeImpact": {"impact_coeff": impact_coeff, "exponent": exponent}}

    @staticmethod
    def spread_based(spread_fraction: float = 1.0) -> Dict[str, Any]:
        """Spread-based slippage (fraction of bid-ask spread)."""
        return {"SpreadBased": {"spread_fraction": spread_fraction}}

    @staticmethod
    def none() -> Dict[str, Any]:
        """No slippage."""
        return {"FixedBps": {"bps": 0.0}}


# ---------------------------------------------------------------------------
# Bar interval factories
# ---------------------------------------------------------------------------


class Interval:
    """Factory for bar interval configuration dicts."""

    @staticmethod
    def millis(n: int = 1) -> Dict[str, int]:
        """Milliseconds.

        Not a bar size: no store holds sub-second bars, and the bar engine
        refuses it as ``bar_interval`` by name. It is a DURATION, for the
        places that read one: a window in time
        (``rolling_sum(Interval.millis(500))``), the lifetime of an order
        (``time_in_force=Interval.millis(500)``), a latency
        (``execution.latency={"order": Interval.millis(5)}``), and the
        wake-up clock of a strategy that quotes (``Strategy.quote``), which
        takes it as ``bar_interval``: ``Interval.millis(100)`` wakes it ten
        times a second.
        """
        return {"Millis": n}

    @staticmethod
    def seconds(n: int = 1) -> Dict[str, int]:
        return {"Seconds": n}

    @staticmethod
    def minutes(n: int = 1) -> Dict[str, int]:
        return {"Minutes": n}

    @staticmethod
    def hours(n: int = 1) -> Dict[str, int]:
        return {"Hours": n}

    @staticmethod
    def days(n: int = 1) -> Dict[str, int]:
        return {"Days": n}

    @staticmethod
    def trades() -> str:
        """The trade clock: one simulation row per trade of the stored tape.

        Not a duration, so it is a bare name rather than a count. Passed as
        ``bar_interval``, the simulation grid becomes the tape itself, in the
        order the venue printed it: ``open``, ``high``, ``low``, ``close`` and
        ``vwap`` all equal the trade's price, ``volume`` is its quantity, and
        the flow columns ``side`` (+1 aggressive buy, -1 aggressive sell),
        ``buy_volume`` and ``sell_volume`` say who crossed. ``bid``, ``ask``,
        ``spread``, ``depth_at_best_bid`` and ``depth_at_best_ask`` carry the
        stored book as of each print, and are null when no book is stored.

        Every count the engine expresses in bars then counts EVENTS:
        ``signal_delay``, ``warmup_bars``, ``time_in_force={"GTB": n}`` and any
        window given as an integer. A window given as a duration still counts
        time, so ``rolling_sum(Interval.seconds(34))`` reads thirty-four
        seconds of prints and ``rolling_sum(34)`` reads thirty-four prints.

        Passed as ``output_resolution``, it asks for the equity and position
        curves per event instead of the one-second default this clock uses.
        See the "Trade Clock" section of the strategy authoring guide.
        """
        return "Trades"


# ---------------------------------------------------------------------------
# Execution price constants
# ---------------------------------------------------------------------------


class ExecutionPrice:
    """Constants matching Rust ``ExecutionPrice`` enum variants."""

    NEXT_BAR_OPEN = "NextBarOpen"
    NEXT_BAR_CLOSE = "NextBarClose"
    NEXT_BAR_VWAP = "NextBarVwap"
    AT_CLOSE = "AtClose"
    AT_OPEN = "AtOpen"
    AT_VWAP = "AtVwap"
    MID_PRICE = "MidPrice"

    @staticmethod
    def custom(name: str) -> Dict[str, str]:
        """Fill at a named bar column, or at a signal the strategy defines.

        The name resolves against the bar schema first (``vwap``, ``bid``, ...),
        then against the strategy's signals -- so a fill can land on any level
        the DSL computes (a band around an SMA, a prior swing, ...)::

            strat = strat.signal("exec_level", sma * 1.012)
            config.execution.execution_price = ExecutionPrice.custom("exec_level")

        The series is read at the order's SIGNAL row (no look-ahead beyond what
        the sizing already has; with the default ``signal_delay=0`` that is the
        execution bar). A row where the signal has no value falls back to the
        close with a warning, and a fill outside the bar's [low, high] range is
        warned about. A name that is neither a column nor a signal is rejected
        before the simulation starts.
        """
        return {"Custom": name}


# ---------------------------------------------------------------------------
# Fill model factories
# ---------------------------------------------------------------------------


class FillModel:
    """Factory for fill model configuration dicts."""

    @staticmethod
    def atomic() -> Dict[str, Any]:
        """Atomic fill — entire order at single price (default)."""
        return {"max_participation_rate": 0.0, "intra_bar_price": "SinglePoint"}

    @staticmethod
    def participation(
        rate: float, intra_bar_price: str = "SinglePoint"
    ) -> Dict[str, Any]:
        """Partial fill limited to a fraction of bar volume.

        Args:
            rate: Max fraction of bar volume per fill (e.g. 0.1 = 10%).
            intra_bar_price: "SinglePoint", "TypicalPrice", or "OhlcAverage".
        """
        return {"max_participation_rate": rate, "intra_bar_price": intra_bar_price}
