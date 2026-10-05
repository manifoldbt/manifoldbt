"""Adverse selection after a fill, horizon by horizon (plotly).

Reads ``result.fill_marks``, which a run only fills in when it was asked for
(``execution.fill_marks=True``). The chart puts the two halves of a maker's
P&L side by side at the same scale, which is the whole point: on a liquid
spot book the captured half-spread is a hundredth of a basis point and the
drift against the position at one second is well over one, and a bar chart
of the drift alone hides that the edge it has to pay for is invisible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import plotly.graph_objects as go

from manifoldbt.plot._theme import ACCENT, DARK_GRAY, GRAY, GREEN, RED, theme_context
from manifoldbt.plot._utils import finalize, new_figure


def _line(fig, y, color, dash, label, position):
    """A horizontal reference line, labelled at one end.

    The two lines it draws sit within a hundredth of a basis point of each
    other on a liquid book -- that is the finding, not a rendering problem --
    so their labels go to opposite ends rather than on top of one another.
    """
    if y is None or y != y:  # None or NaN
        return
    fig.add_hline(
        y=y,
        line_color=color,
        line_width=1.2,
        line_dash=dash,
        annotation_text=f"{label} {y:+.4f} bp",
        annotation_position=position,
        annotation_font_color=color,
        annotation_font_size=11,
    )


def fill_marks(
    result,
    *,
    title: str = "Adverse selection after a fill",
    figsize: Tuple[float, float] = (10, 5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Mean markout per horizon, with its bootstrap error bar.

    One bar per horizon (+100 ms, +1 s, +10 s): the average of
    ``side * (reference - fill price) / fill price`` in basis points, signed
    in the direction the fill takes the position. **Negative is adverse
    selection** -- the market left in the direction that hurts -- so those
    bars are red and the rare positive ones green. The whiskers are the
    bootstrap standard error of the mean over the fills, so a bar shorter
    than its whisker is not a measurement.

    Two horizontal lines put the drift in proportion: the captured
    half-spread (the mark at horizon zero, when a book was stored) and, when
    the book gives it, the market's own half-spread at the same instants --
    the most a quote resting at the touch could ever have earned.

    Raises:
        ValueError: when the run did not ask for the marks.
    """
    marks = getattr(result, "fill_marks", None)
    if not marks:
        raise ValueError(
            "this run carries no fill marks: set "
            "execution.fill_marks=True (it needs the tape, so "
            'fill_resolution="ticks" or the trade clock) and run again'
        )

    horizons = marks.get("horizons", [])
    labels = [h["horizon"] for h in horizons]
    means = [h["mean_bps"] for h in horizons]
    errs = [h["stderr_bps"] for h in horizons]
    medians = [h["median_bps"] for h in horizons]
    counts = [h["marked_fills"] for h in horizons]

    with theme_context():
        fig = new_figure(figsize, title)
        fig.add_trace(go.Bar(
            x=labels,
            y=means,
            name="Mean markout",
            marker=dict(
                color=[RED if (m == m and m < 0) else GREEN for m in means],
                line=dict(color=DARK_GRAY, width=0.6),
            ),
            width=0.55,
            error_y=dict(type="data", array=errs, color=GRAY, thickness=1.4, width=8),
            # hovertext, not text: a number printed inside every bar competes
            # with the axis and says nothing the hover does not.
            hovertext=[
                f"{m:+.3f} bp<br>median {md:+.3f} bp<br>{n:,} fills marked"
                for m, md, n in zip(means, medians, counts)
            ],
            hoverinfo="text+x",
        ))

        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.8)
        _line(fig, marks.get("half_spread_captured_bps"), ACCENT, "solid",
              "half-spread captured", "top right")
        _line(fig, marks.get("book_half_spread_bps"), GRAY, "dot",
              "book half-spread", "top left")

        fig.update_yaxes(title_text="Basis points of the fill price")
        # No x title: the three categories name themselves, and the room goes
        # to the line that says how the marks were taken.
        fig.update_xaxes(title_text=None)
        # The anchor is part of the number, not a footnote: a reader comparing
        # this with another engine needs it in the picture.
        fig.update_layout(
            showlegend=False,
            margin=dict(b=56),
            annotations=list(fig.layout.annotations) + [dict(
                x=0, y=0, xref="paper", yref="paper",
                xanchor="left", yanchor="top", yshift=-40,
                showarrow=False, font=dict(color=GRAY, size=11),
                text=(
                    f"{marks.get('fills_total', 0):,} fills, marked from the "
                    f"{str(marks.get('anchor', '')).replace('_', ' ')}; "
                    f"error bars are {marks.get('bootstrap_draws', 0):,} "
                    "bootstrap resamples"
                ),
            )],
        )
        return finalize(fig, show=show, save=save)
