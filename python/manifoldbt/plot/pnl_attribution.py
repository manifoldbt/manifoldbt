"""Where a maker's P&L came from, fill by fill (plotly).

`equity()` says how much was lost. This says of what. A resting order earns
exactly one thing -- the distance between its level and the mid it was filled
against -- and pays three: the drift of that mid in the seconds after the fill,
the fees, and whatever the inventory it was left holding did before it came
back to flat.

Three of the four are measured per fill and the fourth is what is left, which
is stated on the chart rather than hidden: an attribution that closes only
because one term absorbs the error is worth reading, and one that pretends
otherwise is not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from manifoldbt._convert import equity_with_dates, run_currency, trades_arrays
from manifoldbt.plot._theme import (
    DARK_GRAY,
    GRAY,
    GREEN,
    RED,
    WHITE,
    theme_context,
)
from manifoldbt.plot._utils import finalize, new_figure

#: The horizons `fill_marks_df()` carries, in the order the marks are taken.
HORIZONS = ("100ms", "1s", "10s")


def attribution(result, horizon: str = "1s") -> Dict[str, float]:
    """The four terms and the total, in the account currency.

    Split out of the chart so the numbers can be asserted, joined to something
    else, or printed without drawing anything.

    - ``half_spread``: ``sum(side * (ref_at_fill - price) * qty)``. What the
      quote earned against the mid at the anchor of its own fill. Positive when
      the fill landed on the good side of the mid, which is the whole point of
      resting rather than crossing -- and routinely negative for a quote served
      by a print going through its level.
    - ``adverse``: ``sum(side * (mark_h - ref_at_fill) * qty)``. Where that
      same mid went between the anchor and the horizon, signed with the
      position. Negative is the market leaving in the direction that hurts.
    - ``fees``: minus the fees of every fill in the trade log, brackets
      included: the exits of a maker's inventory are part of what being quoted
      cost.
    - ``inventory``: the remainder, ``total - half_spread - adverse - fees``.
      It is not measured. It holds everything the first three do not: what the
      position did after the horizon ran out, the path between a fill and the
      exit that closed it, and the mark-to-market of whatever was still open
      when the run ended.
    - ``total``: last equity minus first, which is the run's own P&L.

    Args:
        horizon: which markout column to charge the adverse-selection term to.
            One of ``"100ms"``, ``"1s"``, ``"10s"``.

    Raises:
        ValueError: when the run took no marks, or the horizon is not one of
            the three.
    """
    if horizon not in HORIZONS:
        raise ValueError(
            f"horizon must be one of {', '.join(HORIZONS)}, not {horizon!r}"
        )
    df = getattr(result, "fill_marks_df", lambda: None)()
    if df is None:
        raise ValueError(
            "this run carries no fill marks, so its P&L cannot be attributed: "
            "set execution.fill_marks=True (it needs the tape, so "
            'fill_resolution="ticks" or the trade clock) and run again'
        )

    dates, equity = equity_with_dates(result)
    total = float(equity[-1] - equity[0]) if len(equity) > 1 else 0.0
    _ = dates

    side = np.asarray(df["side"], dtype=np.float64)
    qty = np.asarray(df["qty"], dtype=np.float64)
    price = np.asarray(df["price"], dtype=np.float64)
    ref = np.asarray(df["ref_at_fill"], dtype=np.float64)
    mark = np.asarray(df[f"mark_{horizon}"], dtype=np.float64)

    half_spread = float(np.nansum(side * (ref - price) * qty))
    adverse = float(np.nansum(side * (mark - ref) * qty))
    fees = -float(np.nansum(np.asarray(trades_arrays(result)["fees"],
                                       dtype=np.float64)))
    return {
        "half_spread": half_spread,
        "adverse": adverse,
        "fees": fees,
        "inventory": total - half_spread - adverse - fees,
        "total": total,
        "fills": int(len(df)),
        "horizon": horizon,
    }


def pnl_attribution(
    result,
    *,
    horizon: str = "1s",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11, 5.5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """The four terms of a maker's P&L, as a waterfall down to the run's own.

    Bars are money, in the account currency, not basis points: a term that is
    tiny per fill and paid ten thousand times is the one that decides the day,
    and per-fill units hide exactly that. The total is the last bar and it is
    the run's equity change, so what the chart says and what `equity()` shows
    are the same number by construction.

    Args:
        horizon: which markout to charge the adverse-selection term to
            (``"100ms"``, ``"1s"`` or ``"10s"``). The choice is a claim about
            how long a fill's information lasts, so it is written on the bar.

    Raises:
        ValueError: when the run took no marks.
    """
    parts = attribution(result, horizon)
    currency = run_currency(result)
    if title is None:
        title = f"Where the P&L came from ({currency})"

    labels = [
        "Half-spread<br>captured",
        f"Adverse selection<br>at +{horizon}",
        "Fees",
        "Inventory<br>(the remainder)",
        "Total",
    ]
    values = [parts["half_spread"], parts["adverse"], parts["fees"],
              parts["inventory"], parts["total"]]
    measures = ["relative", "relative", "relative", "relative", "total"]

    with theme_context():
        fig = new_figure(figsize, title)
        fig.add_trace(go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=[f"{v:+,.0f}" for v in values],
            textposition="outside",
            textfont=dict(color=WHITE, size=11),
            connector=dict(line=dict(color=DARK_GRAY, width=1)),
            increasing=dict(marker=dict(color=GREEN)),
            decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=GRAY)),
            hovertemplate="%{x}<br>%{y:+,.2f} " + currency + "<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.8)
        fig.update_yaxes(title_text=currency)
        fig.update_xaxes(title_text=None)
        per_fill = parts["total"] / parts["fills"] if parts["fills"] else float("nan")
        fig.update_layout(
            showlegend=False,
            # Room above the tallest bar for its label, and below for the note.
            margin=dict(t=64, b=72),
            annotations=list(fig.layout.annotations) + [dict(
                x=0, y=0, xref="paper", yref="paper",
                xanchor="left", yanchor="top", yshift=-52,
                showarrow=False, font=dict(color=GRAY, size=11),
                text=(
                    f"{parts['fills']:,} fills, {per_fill:+,.4f} {currency} each. "
                    "The first three terms are measured per fill against the "
                    f"mid at its anchor; the fourth is the remainder, not a "
                    "measurement."
                ),
            )],
        )
        return finalize(fig, show=show, save=save)
