"""What became of the quotes, and what served the ones that filled (plotly).

`order_activity` counts the quotes and `fill_fragility["queue"]` counts the
fills, and the two are read against each other or not at all: ten thousand
quotes for four hundred fills is a service rate, and a service rate is the
first number that says whether a queue model was worth turning on. They come
out of a run as two dicts of integers, which is a form nobody compares in their
head.

Two panels, because the two counts are not per the same thing. Left is per
QUOTE, and the base of every share is the number posted. Right is per FILL, and
the base is the number served.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
from manifoldbt.plot._utils import finalize, is_quote_run


def _rows(items: List[Tuple[str, float, str]], base: float):
    """Reverse the rows so the first named is the top bar, and label them."""
    items = list(reversed(items))
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    notes = [i[2] for i in items]
    text = [
        f"  {v:,.0f}" + (f"  ({v / base:.1%})" if base > 0 else "")
        for v in values
    ]
    return labels, values, text, notes


def _bar(fig, labels, values, text, notes, colour, row, col):
    fig.add_trace(go.Bar(
        y=labels, x=values, orientation="h", showlegend=False,
        marker=dict(color=colour, line=dict(color=DARK_GRAY, width=0.4)),
        text=text, textposition="outside",
        textfont=dict(color=WHITE, size=10),
        hovertext=notes, hoverinfo="text",
    ), row=row, col=col)


def order_life(
    result,
    *,
    title: str = "The life of a quote",
    figsize: Tuple[float, float] = (13, 5.5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Every counter a run keeps about its resting orders, on one picture.

    Left, **per quote posted**: how many replaced a standing one, how many
    expired with nothing served, how many were posted behind a queue the
    stored book could not see, how many had their level moved onto the price
    grid, and how many stood beside a quote that was still being cancelled.

    Right, **per fill served**: the channel that granted it -- the queue coming
    down to the order, a print through the level, or the book crossing it --
    plus the fills that ended a partially served order and the ones a quote
    took after its own cancellation had been decided.

    When the run kept a journal (``result.resting_orders_df()``), two more
    panels come below, and they are the ones a counter cannot give: **how long
    a quote stood** before it was served and before it expired, and **how much
    volume was in front of it** when it arrived, in multiples of its own size.
    The first says whether the time-in-force is doing anything; the second says
    whether the queue was ever going to reach the order at all.

    A strategy that quotes with ``.quote`` does not keep that journal: its run
    draws the two counter panels only, and a line under them points to
    ``result.orders_df()`` and ``result.fills_df()``, which hold the orders and
    the fills of such a run.

    Raises:
        ValueError: when the run tracked no resting orders at all.
    """
    activity = getattr(result, "order_activity", None) or {}
    frag = getattr(result, "fill_fragility", None) or {}
    queue = frag.get("queue") or {}
    posted = float(activity.get("orders_posted", 0) or 0)
    if posted <= 0:
        raise ValueError(
            "this run posted no resting order, so it has no quote to follow: "
            "the counters come from a strategy with a level entry order "
            "(.limit_entry(...)) run on the tape"
        )

    served = float(queue.get("queue_decided_fills", 0)
                   + queue.get("traverse_fills", 0)) or float(
        frag.get("maker_fills", 0) or 0)

    # No bar for `orders_posted` itself: it is the base of every share here,
    # it is written in the panel's own title, and drawn it would be the longest
    # bar on the picture for the one number the reader already has.
    quotes = [
        ("requoted", float(activity.get("requotes", 0)),
         "replaced a quote already standing, or one cancelled on the event "
         "before: a requote is a cancel and a new order at the back of the "
         "queue, never a move"),
        ("expired unfilled", float(activity.get("expired_unfilled", 0)),
         "the time-in-force ran out with nothing served"),
    ]
    if "book_unknown_at_post" in queue:
        quotes.append((
            "book unknown at post",
            float(queue["book_unknown_at_post"]),
            "the stored ladder had no row at this level and instant, so the "
            "order joined an assumed_queue instead of an observed one",
        ))
    if "levels_snapped" in activity:
        quotes.append((
            "level snapped to the grid",
            float(activity["levels_snapped"]),
            "tick_size moved the level the strategy asked for onto the "
            "venue's own grid",
        ))
    if "overlapping_live_orders" in activity:
        quotes.append((
            "live beside a cancelled one",
            float(activity["overlapping_live_orders"]),
            "posted while an earlier quote was still being withdrawn: the "
            "exposure a cancel latency buys",
        ))

    fills: List[Tuple[str, float, str]] = []
    if queue:
        fills.append((
            "queue-decided", float(queue.get("queue_decided_fills", 0)),
            "the volume ahead of the order was consumed and the trade reached "
            "it: what modelling the queue bought",
        ))
        fills.append((
            "through the level", float(queue.get("traverse_fills", 0)),
            "a print went through the level, or the book crossed it: what the "
            "traverse convention already gave for free",
        ))
        fills.append((
            "  of those, a book cross",
            float(queue.get("fills_from_book_cross", 0)),
            "decided by the best opposite price crossing the level, with no "
            "print through it",
        ))
        fills.append((
            "ended a partial", float(queue.get("partial_fills", 0)),
            "orders whose life ended having served part of their size",
        ))
    else:
        fills.append((
            "maker fills", float(frag.get("maker_fills", 0) or 0),
            "fills taken by an order resting in the book",
        ))
        fills.append((
            "touch only", float(frag.get("touch_only_fills", 0) or 0),
            "granted on a print AT the level: they would not exist under the "
            "traverse convention",
        ))
    if "stale_fills" in activity:
        fills.append((
            "taken by a stale quote", float(activity["stale_fills"]),
            "booked after the cancellation was decided and before it took "
            "effect: the fills a zero-latency backtest does not have",
        ))
    if "sniped_fills" in activity:
        fills.append((
            "  of those, sniped", float(activity["sniped_fills"]),
            "taken inside the extra window the snipe model added",
        ))

    journal = getattr(result, "resting_orders_df", lambda: None)()
    has_journal = journal is not None and len(journal) > 0
    # Two panels are missing from a `.quote` run, and a chart that simply
    # leaves them out reads as a chart with nothing to say about them.
    note = ""
    if journal is None and is_quote_run(result):
        note = (
            "<br>The lifetime and queue panels read the journal of resting "
            "orders, which a strategy that quotes with .quote does not carry: "
            "see result.orders_df() and result.fills_df()."
        )
    titles =[f"Per quote, of {posted:,.0f} posted",
              f"Per fill, of {served:,.0f} served"]
    if has_journal:
        titles += [
            f"How long a quote stood, in seconds: "
            f"<span style='color:{GREEN}'>to a fill</span> / "
            f"<span style='color:{GRAY}'>to an expiry</span>",
            "Volume ahead at post, x the order's own size",
        ]
        if figsize[1] < 9:
            figsize = (figsize[0], 10.0)

    with theme_context():
        fig = make_subplots(
            rows=2 if has_journal else 1, cols=2, horizontal_spacing=0.20,
            vertical_spacing=0.10, subplot_titles=titles,
        )
        _bar(fig, *_rows(quotes, posted), ACCENT, 1, 1)
        _bar(fig, *_rows(fills, served), ACCENT_ALT, 1, 2)
        clipped = _add_distributions(fig, journal) if has_journal else ""

        rate = served / posted if posted else float("nan")
        # Room on the right of each panel for the outside labels, which
        # otherwise run off the plotting area and are simply not drawn.
        for c, rows in ((1, quotes), (2, fills)):
            widest = max([v for _, v, _ in rows] + [1.0])
            fig.update_xaxes(title_text=None, showticklabels=False,
                             showgrid=False, range=[0, widest * 1.34],
                             row=1, col=c)
            fig.update_yaxes(automargin=True, row=1, col=c)
        if has_journal:
            fig.update_yaxes(title_text="quotes", row=2, col=1)
            fig.update_yaxes(title_text="quotes", row=2, col=2)
        fig.update_layout(
            title_text=title, showlegend=False,
            width=int(figsize[0] * 80), height=int(figsize[1] * 80),
            bargap=0.35, barmode="overlay", hovermode="closest",
            margin=dict(b=82 if note else 64, r=40),
            annotations=list(fig.layout.annotations) + [dict(
                x=0, y=0, xref="paper", yref="paper",
                xanchor="left", yanchor="top", yshift=-44,
                showarrow=False, font=dict(color=GRAY, size=11),
                text=(
                    (f"Service rate {rate:.2%}: one quote in "
                     f"{1 / rate:,.0f} was served."
                     if rate > 0 else
                     "No quote was served over these days.") + clipped + note
                ),
            )],
        )
        return finalize(fig, show=show, save=save)


def _hist(fig, values, colour, name, row, col, hi):
    """One histogram, sharing its panel, clipped to a readable right edge."""
    fig.add_trace(go.Histogram(
        x=values, name=name,
        xbins=dict(start=0.0, end=hi, size=hi / 48.0),
        marker=dict(color=colour, line=dict(width=0)),
        opacity=0.65,
        hovertemplate="%{x}<br>%{y:,} quotes<extra>" + name + "</extra>",
    ), row=row, col=col)


def _add_distributions(fig, journal) -> str:
    """The two panels only the journal can draw. Returns a note about clipping.

    A quote's life and the queue in front of it both have long right tails --
    one quote in a hundred stands for many seconds, and a level can carry a
    hundred times the order's own size. Drawn to the last point, every bar of
    interest lands in the first pixel column, so both panels are clipped at a
    high quantile and the note says where.
    """
    seconds = 1e-9
    posted = journal["posted_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    filled = journal["filled_at"]
    cancel = journal["cancel_decided_at"]

    served = filled.notna().to_numpy()
    to_fill = (filled.to_numpy(dtype="datetime64[ns]").astype("int64")[served]
               - posted[served]) * seconds
    gone = cancel.notna().to_numpy() & ~served
    to_expiry = (cancel.to_numpy(dtype="datetime64[ns]").astype("int64")[gone]
                 - posted[gone]) * seconds

    lives = np.concatenate([to_fill, to_expiry])
    hi_life = max(float(np.quantile(lives, 0.99)) if len(lives) else 1.0, 1e-3)
    if len(to_fill):
        _hist(fig, to_fill[to_fill <= hi_life], GREEN, "to a fill", 2, 1, hi_life)
    if len(to_expiry):
        _hist(fig, to_expiry[to_expiry <= hi_life], GRAY, "to an expiry",
              2, 1, hi_life)

    ahead = np.asarray(journal["queue_ahead_at_post"], dtype=np.float64)
    size = np.asarray(journal["size"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        clips = np.where(size > 0, ahead / size, np.nan)
    clips = clips[np.isfinite(clips)]
    if not len(clips):
        return ""

    hi_q = max(float(np.quantile(clips, 0.95)), 1e-6)
    _hist(fig, clips[clips <= hi_q], ORANGE, "queue at post", 2, 2, hi_q)
    known = journal["book_known_at_post"]
    assumed = float((known == False).mean()) if known.notna().any() else 0.0  # noqa: E712
    fig.add_annotation(
        x=1, y=1, xref="x domain", yref="y domain", xanchor="right",
        yanchor="top", showarrow=False, font=dict(color=GRAY, size=10),
        align="right", row=2, col=2,
        text=(f"median {np.median(clips):,.1f}x the order's own size"
              f"<br>{assumed:.1%} of quotes joined an assumed queue"),
    )
    return (f" Both lower panels are clipped: lives at the 99th percentile "
            f"({hi_life:,.1f} s), the queue at the 95th ({hi_q:,.0f}x).")

