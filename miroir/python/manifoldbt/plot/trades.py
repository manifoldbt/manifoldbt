"""Trade-level charts: round trips on the price, PnL per round trip (plotly).

Both read the fill log through :func:`manifoldbt._trades.round_trips`, the
Python mirror of the engine's own pairing, so what they draw is what
``trade_stats`` counted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from manifoldbt._convert import (
    date_tickformat,
    percent_tickformat,
    positions_arrays,
    run_currency,
)
from manifoldbt._trades import EXIT_REASON_LABELS, SIDE_LONG, round_trips
from manifoldbt.plot._decimate import maybe_decimate
from manifoldbt.plot._theme import (
    ACCENT,
    DARK_GRAY,
    GRAY,
    GREEN,
    ORANGE,
    RED,
    theme_context,
)
from manifoldbt.plot._utils import finalize, new_figure
from manifoldbt.plot.backtest import _rgba

# Above this many round trips the shaded entry-to-exit zones stop reading as
# zones and turn into vertical stripes; the connecting segments stay.
_ZONES_AUTO_MAX = 120


def _marker_style(n_round_trips: int) -> Tuple[int, dict]:
    """Marker size and outline that leave the price visible at any density.

    Measured on two years of hourly BTC with 246 round trips: size-10
    markers with a 1px white outline covered the close entirely. Size and
    outline shrink with the count so a dense chart still shows the path.
    """
    if n_round_trips <= 60:
        return 10, dict(color="white", width=1)
    if n_round_trips <= 200:
        return 8, dict(color="white", width=0.8)
    return 6, dict(color="white", width=0.5)


def _reason_label(code: int) -> str:
    return EXIT_REASON_LABELS.get(int(code), "open")


def _pick_symbol(rt: dict, result) -> int:
    """Symbol to chart when none is asked for: the only one, else the busiest."""
    ids = np.asarray(rt["symbol_id"])
    if len(ids) == 0:
        pos_ids = np.unique(np.asarray(positions_arrays(result)["symbol_id"]))
        return int(pos_ids[0]) if len(pos_ids) else 0
    uniq, counts = np.unique(ids, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _segments(x0, y0, x1, y1):
    """One polyline per pair, joined with None so a single trace draws them all."""
    n = len(x0)
    if n == 0:
        return np.zeros(0, dtype="datetime64[ns]"), np.zeros(0)
    xs = np.empty(n * 3, dtype=object)
    ys = np.empty(n * 3, dtype=object)
    xs[0::3], xs[1::3], xs[2::3] = x0, x1, None
    ys[0::3], ys[1::3], ys[2::3] = y0, y1, None
    return xs, ys


# ── Trades on price ──────────────────────────────────────────────────────────


def trades(
    result,
    *,
    symbol_id: Optional[int] = None,
    zones: Optional[bool] = None,
    title: str = "Trades",
    figsize: Tuple[float, float] = (14, 6),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Round trips drawn on the close: entries, exits, and the path between.

    The close of one symbol as a backdrop; an entry marker per round trip;
    an exit marker coloured by the outcome (green profit, red loss, orange
    when the position is still open at the last bar and marked to it); a
    segment from entry to exit in the same colour; and, when there are few
    enough trades to read them, a shaded zone over the holding period.

    Args:
        symbol_id: Which symbol to chart. Default: the only one in the
            universe, otherwise the one with the most round trips.
        zones: Shade the entry-to-exit period. ``None`` (default) shades
            when the symbol carries at most 120 round trips; beyond that
            the zones read as stripes, so only the markers and the
            entry-to-exit segments are drawn. Markers also shrink with the
            count so the close stays visible on a dense chart.
    """
    with theme_context():
        fig = new_figure(figsize, title)
        rt = round_trips(result)
        sid = _pick_symbol(rt, result) if symbol_id is None else int(symbol_id)
        currency = run_currency(result)

        # Backdrop: the symbol's close.
        pa = positions_arrays(result)
        mask = np.asarray(pa["symbol_id"]) == sid
        ts = pa["timestamp"][mask].astype("datetime64[ns]")
        close = np.asarray(pa["close"], dtype=np.float64)[mask]
        d_ts, d_close = maybe_decimate(ts, close)
        fig.add_trace(go.Scatter(
            x=d_ts, y=d_close, mode="lines", name="Close",
            line=dict(color=GRAY, width=1.0), opacity=0.9,
            hovertemplate="%{x|%d %b %Y %H:%M}  %{y:,.4f}<extra>Close</extra>",
        ))

        sel = np.asarray(rt["symbol_id"]) == sid
        if sel.any():
            e_ts = rt["entry_timestamp"][sel]
            x_ts = rt["exit_timestamp"][sel]
            e_px = rt["entry_price"][sel]
            x_px = rt["exit_price"][sel]
            side = rt["side"][sel]
            q = rt["quantity"][sel]
            pnl = rt["pnl"][sel]
            ret = rt["return_pct"][sel]
            is_open = rt["is_open"][sel]
            reason = rt["exit_reason"][sel]

            win = (pnl > 0) & ~is_open
            loss = (pnl <= 0) & ~is_open
            n_rt = int(sel.sum())
            size, outline = _marker_style(n_rt)

            if zones is None:
                zones = n_rt <= _ZONES_AUTO_MAX
            if zones:
                for color, m in ((GREEN, win), (RED, loss), (ORANGE, is_open)):
                    for a, b in zip(e_ts[m], x_ts[m]):
                        fig.add_vrect(
                            x0=a, x1=b, fillcolor=_rgba(color, 0.07),
                            line_width=0, layer="below",
                        )

            # Entry-to-exit segments, one trace per outcome. They share the
            # legend entry of the matching exit marker (same legendgroup) so
            # the legend stays at five items and toggling an outcome hides
            # both its markers and its paths.
            for group, color, m in (("profit", GREEN, win), ("loss", RED, loss),
                                    ("open", ORANGE, is_open)):
                if not m.any():
                    continue
                xs, ys = _segments(e_ts[m], e_px[m], x_ts[m], x_px[m])
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", name=f"Path ({group})",
                    legendgroup=group, showlegend=False,
                    line=dict(color=color, width=1.0), opacity=0.7,
                    hoverinfo="skip",
                ))

            # Entry markers: one style, the direction is in the hover.
            fig.add_trace(go.Scatter(
                x=e_ts, y=e_px, mode="markers", name="Entry",
                marker=dict(symbol="triangle-up", size=size, color=ACCENT,
                            line=outline),
                text=[f"{'Long' if s == SIDE_LONG else 'Short'} {qq:.6f} @ {p:,.4f}"
                      for s, qq, p in zip(side, q, e_px)],
                hoverinfo="text+x",
            ))

            # Exit markers coloured by outcome.
            for name, group, color, symbol, m in (
                ("Exit (profit)", "profit", GREEN, "triangle-down", win),
                ("Exit (loss)", "loss", RED, "triangle-down", loss),
                ("Open (marked)", "open", ORANGE, "circle", is_open),
            ):
                if not m.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=x_ts[m], y=x_px[m], mode="markers", name=name,
                    legendgroup=group,
                    marker=dict(symbol=symbol, size=size, color=color,
                                line=outline),
                    text=[f"{p:+,.2f} {currency} ({r:+.2%})  {_reason_label(c)}"
                          for p, r, c in zip(pnl[m], ret[m], reason[m])],
                    hoverinfo="text+x",
                ))

        fig.update_yaxes(title_text="Price")
        fig.update_xaxes(tickformat=date_tickformat(ts) if len(ts) else None)
        # Horizontal legend above the plot, right-aligned like summary(): the
        # title sits at the left edge of the margin, so a left-anchored legend
        # ran under it as soon as the title outgrew the y-axis margin.
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return finalize(fig, show=show, save=save)


# ── PnL per round trip ───────────────────────────────────────────────────────


def trade_pnl(
    result,
    *,
    pct_scale: bool = False,
    marker_size_range: Tuple[float, float] = (6.0, 14.0),
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 5),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """PnL of every round trip at the time it closed.

    One marker per round trip, placed at its exit bar: green above the zero
    line for a profit, red below for a loss, orange for a position still
    open at the last bar (marked to it). Marker size follows the magnitude,
    so the trades that made or cost the most stand out from the noise.

    Args:
        pct_scale: Plot the return on the entry notional instead of the PnL
            in the account currency.
        marker_size_range: Smallest and largest marker, in pixels.
    """
    with theme_context():
        currency = run_currency(result)
        if title is None:
            title = "Trade Returns" if pct_scale else f"Trade PnL ({currency})"
        fig = new_figure(figsize, title)
        rt = round_trips(result)

        x = rt["exit_timestamp"]
        y = rt["return_pct"] if pct_scale else rt["pnl"]
        other = rt["pnl"] if pct_scale else rt["return_pct"]
        is_open = rt["is_open"]
        win = (y > 0) & ~is_open
        loss = (y <= 0) & ~is_open

        lo, hi = marker_size_range
        mag = np.abs(y)
        peak = float(mag.max()) if len(mag) else 0.0
        sizes = np.full(len(y), (lo + hi) / 2.0) if peak <= 0 else lo + (hi - lo) * mag / peak

        def _text(m):
            return [
                (f"{v:+.2%}  ({o:+,.2f} {currency})" if pct_scale
                 else f"{v:+,.2f} {currency}  ({o:+.2%})")
                + f"  {_reason_label(c)}  sym {s}"
                for v, o, c, s in zip(y[m], other[m], rt["exit_reason"][m], rt["symbol_id"][m])
            ]

        for name, color, m in (("Profit", GREEN, win), ("Loss", RED, loss),
                               ("Open (marked)", ORANGE, is_open)):
            if not m.any():
                continue
            fig.add_trace(go.Scatter(
                x=x[m], y=y[m], mode="markers", name=name,
                marker=dict(size=sizes[m], color=color, opacity=0.85,
                            line=dict(color="white", width=0.6)),
                text=_text(m), hoverinfo="text+x",
            ))

        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.6)
        if pct_scale:
            fig.update_yaxes(title_text="Return",
                             tickformat=percent_tickformat(peak if peak > 0 else 0.01))
        else:
            # No explicit tick format: the axis keeps the theme's SI ticks
            # (400, 10.5k), the exact amount lives in the hover.
            fig.update_yaxes(title_text=f"PnL ({currency})")
        if len(x):
            fig.update_xaxes(tickformat=date_tickformat(x))
        # Same placement as trades(): right-aligned, clear of the title.
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return finalize(fig, show=show, save=save)
