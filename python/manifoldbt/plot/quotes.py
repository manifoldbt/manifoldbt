"""A few minutes of quoting, one segment per order (plotly).

Every other chart in this family is an aggregate: a mean, a count, a share.
This one is the raw material they are made of, and it is the one that makes
them believable. Each quote is a horizontal segment at its own level, from the
instant it reached the book to the instant its life ended, and the market runs
behind it. A minute of a maker looks like a staircase of one-second segments
walking down behind a falling price, most of them grey because nothing ever
came, and every so often one ending in a dot.

The result does not carry the book, so the market behind the quotes is drawn
from what it does carry: the close of each output row, and the mid each fill
was marked against. There is no reconstructed bid and ask here, and the chart
does not pretend otherwise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from manifoldbt._convert import positions_arrays, run_currency
from manifoldbt.plot._theme import (
    ACCENT,
    ACCENT_ALT,
    DARK_GRAY,
    GRAY,
    GREEN,
    ORANGE,
    WHITE,
    theme_context,
)
from manifoldbt.plot._utils import finalize, is_quote_run, new_figure

#: How a quote's segment is drawn: its colour, its width, and how opaque.
#: `expired` is four quotes in five, so it is the background the three
#: interesting outcomes have to be legible against, not a competing colour.
_OUTCOMES = (
    ("expired", GRAY, 3.0, 0.85, "its time-in-force ran out, nothing served"),
    ("cancelled", DARK_GRAY, 3.0, 0.85, "withdrawn for another reason"),
    ("cancelled_requote", ACCENT, 3.0, 0.9, "replaced because the target moved"),
    ("partial", ACCENT_ALT, 3.4, 1.0, "served in part, then withdrawn"),
    ("stale_filled", ORANGE, 3.4, 1.0,
     "served after its cancellation was decided"),
    ("filled", GREEN, 3.4, 1.0, "served while it was live"),
)

#: The marker a fill gets, by the channel that served it. Open, always: a
#: filled quote lives about as long as a marker is wide, so a solid one covers
#: the segment it is supposed to be terminating.
_CHANNELS = (("queue", "circle-open"), ("traverse", "diamond-open"),
             ("book_cross", "x-thin-open"))

#: The window drawn when none is asked for. One minute rather than five: a
#: quote lives a second, and five minutes of them is a row of dots.
_DEFAULT_SPAN_NS = 60 * 1_000_000_000


def _segments(x0, x1, y):
    """One flat segment per order, joined by a gap so one trace draws them all.

    The gap is `NaT` and `NaN` in arrays of their own dtype, never `None` in an
    object array: numpy turns a `datetime64` into a plain integer on its way
    into an object array, and plotly then plots 1.7e18 on a date axis, which is
    off the side of every window and draws nothing at all.
    """
    n = len(y)
    xs = np.empty(n * 3, dtype="datetime64[ns]")
    xs[0::3], xs[1::3] = x0, x1
    xs[2::3] = np.datetime64("NaT", "ns")
    ys = np.empty(n * 3, dtype=np.float64)
    ys[0::3], ys[1::3] = y, y
    ys[2::3] = np.nan
    return xs, ys


def _as_ns(value) -> Optional[int]:
    """A user-supplied instant as int64 nanoseconds, or None.

    A naive instant is read as UTC and an aware one is converted to it. Both
    paths are needed: the obvious way to pick a window is off the journal's own
    column, and those stamps already carry a zone.
    """
    if value is None:
        return None
    import pandas as pd
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value)


def _busiest_window(filled_ns: np.ndarray, span_ns: int) -> int:
    """Start of the span holding the most fills, so the default shows life."""
    if len(filled_ns) == 0:
        return 0
    bins = filled_ns // span_ns
    values, counts = np.unique(bins, return_counts=True)
    return int(values[int(np.argmax(counts))] * span_ns)


def quotes(
    result,
    *,
    start=None,
    end=None,
    symbol_id: Optional[int] = None,
    title: str = "Quotes, one segment per order",
    figsize: Tuple[float, float] = (14, 6.5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Every quote alive over a window, drawn at its level for its lifetime.

    A segment runs from the instant the order reached the book (its decision
    instant plus ``execution.latency.order``) to the instant its life ended:
    the fill, or the cancellation taking effect. Its colour is the outcome, so
    a wall of grey is a maker nobody is trading with and a run of orange is a
    cancel latency being paid for. Fills sit at the end of their segment,
    shaped by the channel that granted them.

    Args:
        start: first instant of the window; anything ``pandas.Timestamp``
            accepts. Default: the start of the busiest minute, because a window
            picked at random on a maker's day is usually a minute of nothing.
        end: last instant. Default: one minute after ``start``. A window much
            wider than a few minutes turns second-long quotes into dots; that
            is a property of the quotes, not of the chart.
        symbol_id: which symbol to draw. Default: the only one, else the one
            that quoted most over the window.

    Raises:
        ValueError: when the run kept no journal of its resting orders, or no
            quote was alive over the window.
    """
    journal = getattr(result, "resting_orders_df", lambda: None)()
    if journal is None:
        if is_quote_run(result):
            raise ValueError(
                "this run comes from a strategy that quotes with .quote, which "
                "does not carry the journal of resting orders this chart draws: "
                "its orders are in result.orders_df() and its fills, with their "
                "channel, in result.fills_df()"
            )
        raise ValueError(
            "this run posted no resting order, so it has no quote to draw: "
            "the segments come from result.resting_orders_df()"
        )
    if not len(journal):
        raise ValueError("this run's journal of resting orders is empty")

    import pandas as pd

    posted = journal["posted_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    visible = journal["visible_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    filled = journal["filled_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    filled = np.where(journal["filled_at"].isna().to_numpy(), np.int64(-1), filled)
    cancel = journal["cancel_effective_at"].to_numpy(
        dtype="datetime64[ns]").astype("int64")
    cancel = np.where(journal["cancel_effective_at"].isna().to_numpy(),
                      np.int64(-1), cancel)

    start_ns = _as_ns(start)
    end_ns = _as_ns(end)
    if start_ns is None:
        served = filled[filled >= 0]
        start_ns = (_busiest_window(served, _DEFAULT_SPAN_NS)
                    if len(served) else int(posted.min()))
    if end_ns is None:
        end_ns = start_ns + _DEFAULT_SPAN_NS
    if end_ns <= start_ns:
        raise ValueError("end must be after start")

    # An order is on the picture when any part of its life overlaps the
    # window, not only when it was posted inside it: a quote standing across
    # the left edge is part of what the window shows.
    dies = np.where(filled >= 0, filled, np.where(cancel >= 0, cancel, end_ns))
    dies = np.maximum(dies, visible)
    live = (dies >= start_ns) & (visible <= end_ns)
    if symbol_id is not None:
        live &= journal["symbol_id"].to_numpy() == int(symbol_id)
    if not live.any():
        raise ValueError(
            "no quote was alive between "
            f"{pd.Timestamp(start_ns, tz='UTC')} and "
            f"{pd.Timestamp(end_ns, tz='UTC')}"
        )
    if symbol_id is None:
        ids, counts = np.unique(journal["symbol_id"].to_numpy()[live],
                                return_counts=True)
        sid = int(ids[int(np.argmax(counts))])
        live &= journal["symbol_id"].to_numpy() == sid
    else:
        sid = int(symbol_id)

    win = journal[live]
    x0 = visible[live].astype("datetime64[ns]")
    x1 = dies[live].astype("datetime64[ns]")
    level = np.asarray(win["level"], dtype=np.float64)
    outcome = np.asarray(win["outcome"])
    currency = run_currency(result)

    with theme_context():
        fig = new_figure(figsize, title)
        _add_market(fig, result, sid, start_ns, end_ns, currency)

        for name, colour, width, opacity, why in _OUTCOMES:
            m = outcome == name
            if not m.any():
                continue
            xs, ys = _segments(x0[m], x1[m], level[m])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name=f"{name.replace('_', ' ')} ({int(m.sum()):,})",
                legendgroup=name,
                line=dict(color=colour, width=width),
                opacity=opacity,
                hoverinfo="skip",
            ))
            _ = why

        _add_fills(fig, win, currency)

        fig.update_yaxes(title_text=f"Price ({currency})")
        fig.update_xaxes(
            range=[np.datetime64(start_ns, "ns"), np.datetime64(end_ns, "ns")],
            tickformat="%H:%M:%S",
        )
        span = (end_ns - start_ns) / 1e9
        served = int((win["filled_at"].notna()).sum())
        fig.update_layout(
            hovermode="closest", margin=dict(b=62),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=10)),
            annotations=list(fig.layout.annotations) + [dict(
                x=0, y=0, xref="paper", yref="paper",
                xanchor="left", yanchor="top", yshift=-42,
                showarrow=False, font=dict(color=GRAY, size=11),
                text=(
                    f"symbol {sid}, {span:,.0f} s from "
                    f"{pd.Timestamp(start_ns, tz='UTC'):%Y-%m-%d %H:%M:%S} UTC: "
                    f"{len(win):,} quotes, {served:,} of them served. A segment "
                    "runs from the instant the order reached the book to the "
                    "instant its life ended."
                ),
            )],
        )
        return finalize(fig, show=show, save=save)


def _add_market(fig, result, sid: int, start_ns: int, end_ns: int,
                currency: str) -> None:
    """What the run carries of the market: the close, and the marks' own mid."""
    pa = positions_arrays(result)
    ts = np.asarray(pa["timestamp"], dtype="datetime64[ns]")
    m = (np.asarray(pa["symbol_id"]) == sid)
    m &= (ts.astype("int64") >= start_ns) & (ts.astype("int64") <= end_ns)
    if m.any():
        # A backdrop and nothing more: the quotes rest AT the touch, so a close
        # drawn at full weight lands on top of every segment this chart exists
        # to show.
        fig.add_trace(go.Scatter(
            x=ts[m], y=np.asarray(pa["close"], dtype=np.float64)[m],
            mode="lines", name="Close", line=dict(color=GRAY, width=0.7),
            opacity=0.35,
            hovertemplate="%{x|%H:%M:%S}  %{y:,.2f}<extra>Close</extra>",
        ))


def _add_fills(fig, win, currency: str) -> None:
    """One marker per served quote, shaped by the channel that granted it."""
    served = win[win["filled_at"].notna()]
    if not len(served):
        return
    x = served["filled_at"].to_numpy(dtype="datetime64[ns]")
    y = np.asarray(served["level"], dtype=np.float64)
    channel = np.asarray(served["fill_channel"].fillna(""))
    qty = np.asarray(served["filled_qty"], dtype=np.float64)
    ahead = np.asarray(served["queue_ahead_at_post"], dtype=np.float64)

    for name, symbol in _CHANNELS:
        m = channel == name
        if not m.any():
            continue
        fig.add_trace(go.Scatter(
            x=x[m], y=y[m], mode="markers",
            name=f"{name.replace('_', ' ')} fill ({int(m.sum()):,})",
            marker=dict(symbol=symbol, size=9, color=WHITE, opacity=0.95,
                        line=dict(color=WHITE, width=1.4)),
            text=[f"{name.replace('_', ' ')} {q:,.6g} @ {p:,.2f} {currency}"
                  f"<br>{a:,.4g} units were ahead at post"
                  for q, p, a in zip(qty[m], y[m], ahead[m])],
            hoverinfo="text+x",
        ))
