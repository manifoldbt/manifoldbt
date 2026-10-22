"""Adverse selection, split by what served the fill (plotly).

`fill_marks()` averages every fill together, and that average hides the finding:
the three ways a resting order can be reached are three different trades. A
queue-decided fill means the flow that consumed the level stopped there. A
traversal means the price went THROUGH the level, so the order was served
precisely because the market was leaving. A book cross means the opposite best
price came over with no print at all.

The channel comes from ``result.resting_orders_df()``, joined to the marks on
the instant the fill was booked. No heuristic reconstructs it: an engine that
did not keep the journal has no channel to give, and this chart says so rather
than guessing one from the price.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from manifoldbt.plot._theme import (
    ACCENT,
    ACCENT_ALT,
    DARK_GRAY,
    GRAY,
    ORANGE,
    theme_context,
)
from manifoldbt.plot._utils import finalize, is_quote_run, new_figure

#: The columns drawn, left to right. The first is the quote's own edge -- the
#: mark at horizon zero -- and the three others are where it went afterwards.
_COLUMNS = (("at the fill", None), ("+100ms", "100ms"),
            ("+1s", "1s"), ("+10s", "10s"))

#: One colour per channel, and the order they are drawn in.
_CHANNELS = (
    ("queue", ACCENT, "the volume ahead of the order was consumed"),
    ("traverse", ORANGE, "a print went through the level"),
    ("book_cross", ACCENT_ALT, "the best opposite price crossed it"),
)

_LABELS = {"queue": "queue-decided", "traverse": "through the level",
           "book_cross": "book cross"}


def fills_by_channel(result):
    """The marked fills of one run or of several, each with its channel.

    A pandas DataFrame: every column of ``fill_marks_df()`` plus
    ``fill_channel`` and the two things only the journal knows about the order
    behind the fill, ``queue_ahead_at_post`` and ``book_known_at_post``.

    The join is on the instant the fill was booked, the symbol and the side --
    ``resting_orders_df()["filled_at"]`` is the same stamp
    ``trades_df()["execution_timestamp"]`` carries, so it is exact rather than
    nearest. A handful of events book two fills on the same side (a stale quote
    and the one replacing it): the first journal row of such an event wins,
    which keeps the row count equal to the number of marked entry fills.

    A list or tuple of results is pooled, which is how ten days of a tape are
    read: a day is a run of its own, and stitching non-consecutive days into
    one long run marks the first fills of each day against the last mid of the
    previous one.

    Raises:
        ValueError: when the run took no marks, or kept no journal.
    """
    import pandas as pd

    from manifoldbt.plot.hourly import as_results

    runs = as_results(result)
    if len(runs) > 1:
        return pd.concat([fills_by_channel(r) for r in runs], ignore_index=True)
    result = runs[0]
    marks = getattr(result, "fill_marks_df", lambda: None)()
    if marks is None:
        raise ValueError(
            "this run carries no fill marks: set execution.fill_marks=True "
            "(it needs the tape, so fill_resolution=\"ticks\" or the trade "
            "clock) and run again"
        )
    journal = getattr(result, "resting_orders_df", lambda: None)()
    if journal is None:
        if is_quote_run(result):
            raise ValueError(
                "this run comes from a strategy that quotes with .quote, which "
                "does not carry the journal of resting orders this chart reads: "
                "the channel of each fill is in result.fills_df()[\"channel\"], "
                "and what became of each order in result.orders_df()"
            )
        raise ValueError(
            "this run posted no resting order, so no fill has a channel: the "
            "channel is what served a quote, and a market order is not a quote"
        )

    keep = ["filled_at", "symbol_id", "side", "fill_channel",
            "queue_ahead_at_post", "book_known_at_post", "outcome"]
    served = (journal.dropna(subset=["filled_at"])[keep]
              .drop_duplicates(subset=["filled_at", "symbol_id", "side"],
                               keep="first"))
    return pd.merge(
        marks, served,
        left_on=["timestamp", "symbol_id", "side"],
        right_on=["filled_at", "symbol_id", "side"],
        how="inner",
    )


def _stats(df, channel: str):
    """Mean and standard error, in bp, for one channel and every column."""
    sel = df[df["fill_channel"] == channel]
    price = np.asarray(sel["price"], dtype=np.float64)
    side = np.asarray(sel["side"], dtype=np.float64)
    means, errs, counts = [], [], []
    for _, horizon in _COLUMNS:
        col = "ref_at_fill" if horizon is None else f"mark_{horizon}"
        ref = np.asarray(sel[col], dtype=np.float64)
        bps = 1e4 * side * (ref - price) / price
        bps = bps[np.isfinite(bps)]
        n = len(bps)
        counts.append(n)
        means.append(float(bps.mean()) if n else float("nan"))
        errs.append(float(bps.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
    return means, errs, counts


def marks_by_channel(
    result,
    *,
    title: str = "Adverse selection, by what served the fill",
    figsize: Tuple[float, float] = (12, 5.5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """The markouts of `fill_marks()`, one group of bars per service channel.

    Four columns. The first is the mark at horizon zero -- the distance between
    the fill price and the mid it was filled against, which is the half-spread
    the quote captured, positive when the fill landed on the good side. The
    three others are where that mid went 100 ms, 1 s and 10 s later, signed
    with the position, so negative is the market leaving in the direction that
    hurts.

    Whiskers are the standard error of each mean over its own fills, so a
    channel whose bar is shorter than its whisker is a channel with too few
    fills to draw a conclusion from -- which is usually the queue, and is worth
    seeing rather than being told.

    Raises:
        ValueError: when the run took no marks or kept no journal of its
            resting orders.
    """
    df = fills_by_channel(result)
    present = [c for c in _CHANNELS if (df["fill_channel"] == c[0]).any()]
    if not present:
        raise ValueError(
            "no fill in this run was served by a resting order, so there is "
            "no channel to split by"
        )

    with theme_context():
        fig = new_figure(figsize, title)
        totals = []
        for channel, colour, _why in present:
            means, errs, counts = _stats(df, channel)
            totals.append((channel, counts[0]))
            fig.add_trace(go.Bar(
                x=[label for label, _ in _COLUMNS], y=means,
                name=f"{_LABELS[channel]} ({counts[0]:,})",
                marker=dict(color=colour, line=dict(color=DARK_GRAY, width=0.5)),
                error_y=dict(type="data", array=errs, color=GRAY,
                             thickness=1.3, width=6),
                hovertext=[f"{_LABELS[channel]}<br>{m:+.3f} bp<br>{n:,} fills"
                           for m, n in zip(means, counts)],
                hoverinfo="text+x",
            ))

        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.8)
        fig.update_yaxes(title_text="Basis points of the fill price")
        fig.update_xaxes(title_text=None, type="category")
        why = "<br>".join(f"{_LABELS[c]}: {w}." for c, _col, w in present)
        fig.update_layout(
            barmode="group", bargap=0.28, bargroupgap=0.06,
            hovermode="closest", margin=dict(b=120),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            annotations=list(fig.layout.annotations) + [dict(
                x=0, y=0, xref="paper", yref="paper",
                xanchor="left", yanchor="top", yshift=-40,
                showarrow=False, font=dict(color=GRAY, size=11),
                align="left",
                text=(
                    f"{len(df):,} fills, each carrying the channel that served "
                    f"the FIRST unit of its order.<br>{why}"
                ),
            )],
        )
        return finalize(fig, show=show, save=save)
