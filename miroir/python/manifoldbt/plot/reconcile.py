"""A journal of real fills against the backtest of the same days (plotly).

Two panels, because a reconciliation has exactly two questions and they are
not on the same scale.

Left, **service by hour of the UTC day**: how many fills each side booked. It
is the number a queue model exists to get right, it is the one it gets wrong
first, and an hour-of-day profile is where the two curves part company --
typically the quiet hours, where a backtest's touch convention hands out fills
a real queue never reaches.

Right, **adverse selection by horizon**: the markout of each side at +100 ms,
+1 s and +10 s, with the bootstrap standard error as a whisker. The paired
difference is annotated on each pair rather than drawn as a third bar: it is
taken over the fills that MATCHED, so its error bar is small where the
difference of two independent means would be large, and a bar drawn beside two
population means invites reading it as their subtraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from manifoldbt.plot._theme import ACCENT, ACCENT_ALT, DARK_GRAY, GRAY, WHITE, theme_context
from manifoldbt.plot._utils import finalize


def reconcile(
    reconciliation,
    *,
    title: str = "Real fills against the backtest",
    figsize: Tuple[float, float] = (13, 5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Service by hour and adverse selection by horizon, both sides at once.

    Args:
        reconciliation: what :func:`manifoldbt.reconcile` returned.
        title: figure title.
        figsize: matplotlib-style size, converted to pixels.
        show: ``None`` shows unless ``save`` was given.
        save: path; ``.html`` writes an interactive page, image extensions go
            through kaleido.

    Raises:
        ValueError: when nothing was reconciled, so there is nothing to draw.
    """
    raw = getattr(reconciliation, "raw", reconciliation)
    if not isinstance(raw, dict) or "horizons" not in raw:
        raise ValueError(
            "plot.reconcile takes what bt.reconcile returned, not a backtest "
            "result: rec = bt.reconcile(result, live_fills, store=store)"
        )
    if raw["live_fills"] == 0 and raw["sim_fills"] == 0:
        raise ValueError(
            "neither side booked a fill over these days, so there is nothing "
            "to reconcile"
        )

    hours = raw["by_hour"]
    horizons = raw["horizons"]

    with theme_context():
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.58, 0.42],
            horizontal_spacing=0.09,
            subplot_titles=(
                "Fills served, by hour of the UTC day",
                "Markout after a fill",
            ),
        )

        x = [f"{h:02d}" for h in hours["key"]]
        fig.add_trace(
            go.Bar(
                x=x,
                y=hours["sim_fills"],
                name="backtest",
                marker=dict(color=ACCENT, line=dict(color=DARK_GRAY, width=0.4)),
                hovertemplate="%{x}:00 UTC<br>backtest %{y:,} fills<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=hours["live_fills"],
                name="journal",
                marker=dict(color=ACCENT_ALT, line=dict(color=DARK_GRAY, width=0.4)),
                hovertemplate="%{x}:00 UTC<br>journal %{y:,} fills<extra></extra>",
            ),
            row=1,
            col=1,
        )

        labels = [h["horizon"] for h in horizons]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[h["sim_mean_bps"] for h in horizons],
                name="backtest",
                showlegend=False,
                marker=dict(color=ACCENT, line=dict(color=DARK_GRAY, width=0.4)),
                error_y=dict(
                    type="data",
                    array=[h["sim_stderr_bps"] for h in horizons],
                    color=GRAY,
                    thickness=1.3,
                    width=7,
                ),
                hovertext=[
                    f"backtest {h['sim_mean_bps']:+.3f} bp"
                    f"<br>{h['sim_marked_fills']:,} fills marked"
                    for h in horizons
                ],
                hoverinfo="text",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[h["live_mean_bps"] for h in horizons],
                name="journal",
                showlegend=False,
                marker=dict(color=ACCENT_ALT, line=dict(color=DARK_GRAY, width=0.4)),
                error_y=dict(
                    type="data",
                    array=[h["live_stderr_bps"] for h in horizons],
                    color=GRAY,
                    thickness=1.3,
                    width=7,
                ),
                hovertext=[
                    f"journal {h['live_mean_bps']:+.3f} bp"
                    f"<br>{h['live_marked_fills']:,} fills marked"
                    for h in horizons
                ],
                hoverinfo="text",
            ),
            row=1,
            col=2,
        )

        # The paired difference, written where the two bars of a horizon meet.
        # It is the number that survives composition, so it belongs on the
        # picture rather than in a caption.
        for h in horizons:
            if h["paired_fills"] == 0:
                continue
            lo = min(h["live_mean_bps"], h["sim_mean_bps"], 0.0)
            fig.add_annotation(
                x=h["horizon"],
                y=lo,
                yshift=-14,
                text=f"{h['paired_mean_bps']:+.3f} ± {h['paired_stderr_bps']:.3f}",
                showarrow=False,
                font=dict(color=WHITE, size=10),
                row=1,
                col=2,
            )

        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.8, row=1, col=2)
        fig.update_xaxes(title_text="hour (UTC)", type="category", row=1, col=1)
        fig.update_yaxes(title_text="fills", row=1, col=1)
        fig.update_xaxes(title_text=None, type="category", row=1, col=2)
        fig.update_yaxes(title_text="basis points, signed with the position",
                         row=1, col=2)

        service = raw["service"]
        if service.get("orders_posted"):
            served = (
                f"service {100 * service['sim_rate']:.1f}% backtest against "
                f"{100 * service['live_rate']:.1f}% journal, over "
                f"{service['orders_posted']:,} quotes posted"
            )
        else:
            served = (
                f"{raw['sim_fills']:,} backtest fills against "
                f"{raw['live_fills']:,} in the journal"
            )
        fig.update_layout(
            title_text=title,
            width=int(figsize[0] * 80),
            height=int(figsize[1] * 80),
            barmode="group",
            bargap=0.25,
            hovermode="closest",
            margin=dict(b=64),
            legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"),
            annotations=list(fig.layout.annotations)
            + [
                dict(
                    x=0,
                    y=0,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                    yshift=-46,
                    showarrow=False,
                    font=dict(color=GRAY, size=11),
                    text=(
                        f"{served}; {raw['matched']:,} pairs matched, "
                        f"{raw['live_only']:,} real fills with no backtest twin, "
                        f"{raw['sim_only']:,} backtest fills never granted. "
                        "The number under each horizon is the PAIRED difference."
                    ),
                )
            ],
        )
        return finalize(fig, show=show, save=save)
