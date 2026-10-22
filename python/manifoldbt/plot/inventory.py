"""The inventory a maker carried, against the market it carried it in (plotly).

A market maker's P&L is not made of trades, it is made of the position it was
left holding between them. `trades()` draws round trips and answers a
directional strategy's question; this one answers a maker's, which is the
opposite: when did the book hand me an inventory, at what price, and how long
did I sit on it before it came back to flat.

The two things share an axis of time and nothing else, so they get a panel
each. An inventory that grows while the price falls is a maker buying a move
all the way down, and that is read by looking straight down from one panel to
the other.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from manifoldbt._convert import (
    date_tickformat,
    positions_arrays,
    run_currency,
    trades_arrays,
)
from manifoldbt.plot._decimate import maybe_decimate
from manifoldbt.plot._theme import (
    ACCENT,
    DARK_GRAY,
    GRAY,
    ORANGE,
    WHITE,
    theme_context,
)
from manifoldbt.plot._utils import finalize
from manifoldbt.plot.backtest import _rgba


def _fill_marker(n_fills: int) -> Tuple[float, float]:
    """Marker size and opacity that leave the price readable at any density.

    A maker on one liquid day books thousands of fills, so the size that made
    `trades()` legible at 246 round trips paints a solid band here. Both shrink
    with the count: past a couple of thousand the markers are meant to read as
    a haze around the price saying where the maker was busy, and the individual
    fill is reached by hovering.
    """
    if n_fills <= 200:
        return 8.0, 0.9
    if n_fills <= 2_000:
        return 5.0, 0.7
    return 3.0, 0.45


def _pick_symbol(pa: dict, symbol_id: Optional[int]) -> int:
    """The symbol to draw: the one asked for, the only one, else the busiest."""
    if symbol_id is not None:
        return int(symbol_id)
    ids = np.asarray(pa["symbol_id"])
    uniq = np.unique(ids)
    if len(uniq) <= 1:
        return int(uniq[0]) if len(uniq) else 0
    pos = np.abs(np.asarray(pa["position"], dtype=np.float64))
    held = [float(pos[ids == s].sum()) for s in uniq]
    return int(uniq[int(np.argmax(held))])


def inventory(
    result,
    *,
    symbol_id: Optional[int] = None,
    target: Optional[float] = None,
    title: str = "Inventory",
    figsize: Tuple[float, float] = (14, 7),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """The market on top, the position underneath, every fill on both.

    Above, the close of one symbol with a marker per fill -- a triangle up for
    a buy, down for a sell -- so the level the inventory was bought at is read
    off the price itself. Below, the position through the run: under the trade
    clock it is sampled one row per second by default, so a clip held for a
    tenth of a second is a spike and one held for a minute is a plateau.

    Args:
        symbol_id: Which symbol to draw. Default: the only one in the universe,
            otherwise the one that carried the most inventory.
        target: The inventory the strategy was aiming for, in units. Drawn as a
            dotted line on each side of zero. The engine cannot infer it -- a
            size is an expression, not a number -- so it is an argument rather
            than something read off the result. It is worth passing: a peak
            above it is the overshoot a cancel latency buys, and there is
            nothing to read that against without the line.

    Raises:
        ValueError: when the result carries no position trace to draw.
    """
    pa = positions_arrays(result)
    if not len(pa.get("timestamp", ())):
        raise ValueError("this result carries no positions to draw")

    sid = _pick_symbol(pa, symbol_id)
    mask = np.asarray(pa["symbol_id"]) == sid
    if not mask.any():
        raise ValueError(f"symbol {sid} is not in this run's universe")

    ts = pa["timestamp"][mask].astype("datetime64[ns]")
    position = np.asarray(pa["position"], dtype=np.float64)[mask]
    close = np.asarray(pa["close"], dtype=np.float64)[mask]
    currency = run_currency(result)

    with theme_context():
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.62, 0.38], vertical_spacing=0.05,
        )

        # The fills first, so the close draws over the haze they make rather
        # than under it: at a few thousand markers the price is the line that
        # has to survive.
        n_fills = _add_fills(fig, result, sid, currency)
        d_ts, d_close = maybe_decimate(ts, close)
        fig.add_trace(go.Scatter(
            x=d_ts, y=d_close, mode="lines", name="Close",
            line=dict(color=WHITE, width=1.0), opacity=0.9,
            hovertemplate="%{x|%d %b %H:%M:%S}  %{y:,.2f}<extra>Close</extra>",
        ), row=1, col=1)

        # A step, because a position does not interpolate between two rows: it
        # is what it is until a fill changes it.
        p_ts, p_pos = maybe_decimate(ts, position)
        fig.add_trace(go.Scatter(
            x=p_ts, y=p_pos, mode="lines", name="Position", showlegend=False,
            line=dict(color=ACCENT, width=0.6, shape="hv"),
            fill="tozeroy", fillcolor=_rgba(ACCENT, 0.45),
            hovertemplate="%{x|%d %b %H:%M:%S}  %{y:,.6f}<extra>Position</extra>",
        ), row=2, col=1)

        peak = float(np.nanmax(np.abs(position))) if len(position) else 0.0
        if target is not None and target > 0:
            levels = [float(target)]
            if float(np.nanmin(position)) < -1e-12:
                levels.append(-float(target))
            for y in levels:
                fig.add_hline(y=y, line_color=ORANGE, line_width=1.0,
                              line_dash="dot", row=2, col=1)

        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.8, row=2, col=1)
        fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
        fig.update_yaxes(title_text="Position (units)", row=2, col=1)
        fig.update_xaxes(tickformat=date_tickformat(ts), row=2, col=1)

        flat = float((np.abs(position) < 1e-12).mean()) if len(position) else 0.0
        note = (
            f"symbol {sid}: flat on {flat:.1%} of the rows, peak inventory "
            f"{peak:,.6g}"
        )
        if target is not None and target > 0:
            note += (f" against the dotted target of {float(target):,.6g}"
                     f" ({peak / float(target):,.1f}x)")
        if n_fills:
            note += f"; {n_fills:,} fills"
        fig.update_layout(
            title_text=title,
            width=int(figsize[0] * 80), height=int(figsize[1] * 80),
            hovermode="closest", margin=dict(b=62),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            annotations=list(fig.layout.annotations) + [dict(
                x=0, y=0, xref="paper", yref="paper",
                xanchor="left", yanchor="top", yshift=-42,
                showarrow=False, font=dict(color=GRAY, size=11), text=note,
            )],
        )
        return finalize(fig, show=show, save=save)


def _add_fills(fig, result, sid: int, currency: str) -> int:
    """Every fill of one symbol, on the price panel. Returns how many."""
    try:
        trades = result.trades
    except Exception:
        return 0
    if trades is None or trades.num_rows == 0:
        return 0

    ta = trades_arrays(result)
    m = np.asarray(ta["symbol_id"]) == sid
    if not m.any():
        return 0
    x = ta["execution_timestamp"][m]
    y = np.asarray(ta["fill_price"], dtype=np.float64)[m]
    side = np.asarray(ta["side"])[m]
    qty = np.asarray(ta["quantity"], dtype=np.float64)[m]
    size, opacity = _fill_marker(int(m.sum()))

    for name, colour, symbol, sel in (
        ("Buy", ACCENT, "triangle-up", side == 1),
        ("Sell", ORANGE, "triangle-down", side != 1),
    ):
        if not sel.any():
            continue
        fig.add_trace(go.Scatter(
            x=x[sel], y=y[sel], mode="markers", name=name,
            marker=dict(symbol=symbol, size=size, color=colour,
                        opacity=opacity, line=dict(width=0)),
            text=[f"{name.lower()} {q:,.6g} @ {p:,.2f} {currency}"
                  for q, p in zip(qty[sel], y[sel])],
            hoverinfo="text+x",
        ), row=1, col=1)
    return int(m.sum())
