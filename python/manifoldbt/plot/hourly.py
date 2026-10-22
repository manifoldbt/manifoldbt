"""A maker's day, hour of the UTC day by hour of the UTC day (plotly).

Nothing about a maker is stationary across a session. The flow that serves it
arrives when the venue is awake, and so does the flow that runs it over; a
quote that is fairly priced at four in the morning is a gift at two in the
afternoon. Four panels on one hour axis say whether the fills, the exposure,
the adverse selection and the fees move together -- which is the difference
between "my edge is too thin" and "my edge is fine except for three hours".

On more than one day the hours are pooled: hour 14 is every 14:00 UTC the runs
cover, and the note says over how many days. A day of a liquid tape is millions
of events, so ten days are ten runs rather than one long one; pass the list and
the panels pool it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from manifoldbt._convert import positions_arrays, run_currency, trades_arrays
from manifoldbt.plot._theme import (
    ACCENT,
    DARK_GRAY,
    GRAY,
    GREEN,
    ORANGE,
    RED,
    theme_context,
)
from manifoldbt.plot._utils import finalize

_HOURS = [f"{h:02d}" for h in range(24)]


def as_results(result) -> list:
    """One result, or a sequence of them, as a list.

    The tape layer is run a day at a time on purpose, so every chart that
    pools across days has to take the days as it got them. A tuple or a list
    is a sequence of runs; anything else is one run.
    """
    if isinstance(result, (list, tuple)):
        runs = list(result)
        if not runs:
            raise ValueError("no result to draw: the sequence is empty")
        return runs
    return [result]


def _hour_of(ts: np.ndarray) -> np.ndarray:
    """Hour of the UTC day of a datetime64[ns] array, as int."""
    if len(ts) == 0:
        return np.zeros(0, dtype=np.int64)
    # A tz-aware pandas column goes through `to_numpy`, which converts to UTC
    # and drops the zone; handing it to numpy directly does the same thing and
    # warns about it. Either way the values are UTC nanoseconds, which is what
    # an hour of the UTC day is counted from.
    arr = (ts.to_numpy(dtype="datetime64[ns]") if hasattr(ts, "to_numpy")
           else np.asarray(ts, dtype="datetime64[ns]"))
    ns = arr.astype("int64")
    return (ns // 3_600_000_000_000) % 24


def _by_hour(hours: np.ndarray, values: Optional[np.ndarray] = None) -> np.ndarray:
    """Sum of *values* (or a count) per hour of the day, always 24 long."""
    return np.bincount(hours, weights=values, minlength=24)[:24]


def hourly(
    result,
    *,
    horizon: str = "1s",
    title: str = "The day, hour by hour",
    figsize: Tuple[float, float] = (13, 10),
    show: "bool | str | None" = None,
    save: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Fills, exposure, adverse selection and fees, per hour of the UTC day.

    Top to bottom, on one shared hour axis:

    1. **fills served**, stacked by side. A maker whose two sides do not
       arrive in the same hours is not making a market, it is taking a view.
    2. **exposure**: the share of the run's rows in that hour where the
       position was not flat. Under the trade clock those rows are one second
       apart by default, so it reads as the share of the hour spent holding
       something.
    3. **markout at the chosen horizon**, in basis points signed with the
       position, with its standard error as a whisker. Negative is adverse.
       An hour whose bar is shorter than its whisker is an hour with too few
       fills to say anything about, and the whisker says so.
    4. **fees**, in the account currency.

    Args:
        result: one run, or a list of runs to pool. A day of a liquid tape is
            a run of its own, so ten days arrive as ten results and the hours
            are summed across them. Pooling them by stitching the days into
            one long run instead would be worse than inconvenient: on days
            that are not consecutive, the first fills of each day are marked
            against the last mid of the previous one.
        horizon: which markout to average, ``"100ms"``, ``"1s"`` or ``"10s"``.

    Raises:
        ValueError: when no run booked a fill, so there is no day to draw.
    """
    runs = as_results(result)
    buys = np.zeros(24)
    sells = np.zeros(24)
    fees_by_hour = np.zeros(24)
    held_rows = np.zeros(24)
    rows = np.zeros(24)
    bps: list = [[] for _ in range(24)]
    days: set = set()
    n_fills = 0

    for run in runs:
        ta = trades_arrays(run)
        exec_ts = ta.get("execution_timestamp",
                         np.zeros(0, dtype="datetime64[ns]"))
        if len(exec_ts):
            t_hours = _hour_of(exec_ts)
            side = np.asarray(ta["side"])
            buys += _by_hour(t_hours[side == 1])
            sells += _by_hour(t_hours[side != 1])
            fees_by_hour += _by_hour(
                t_hours, np.asarray(ta["fees"], dtype=np.float64))
            n_fills += len(exec_ts)

        pa = positions_arrays(run)
        p_ts = pa.get("timestamp", np.zeros(0, dtype="datetime64[ns]"))
        if len(p_ts):
            p_hours = _hour_of(p_ts)
            held = (np.abs(np.asarray(pa["position"], dtype=np.float64))
                    > 1e-12).astype(float)
            held_rows += _by_hour(p_hours, held)
            rows += _by_hour(p_hours)
        days |= _distinct_days(p_ts if len(p_ts) else exec_ts)
        _collect_markouts(run, horizon, bps)

    if n_fills == 0:
        raise ValueError("no run booked a fill, so there is no hourly profile")

    currency = run_currency(runs[0])
    exposure = np.divide(held_rows, rows, out=np.full(24, np.nan),
                         where=rows > 0)
    means, errs, marked = _summarise(bps)
    days = len(days)

    with theme_context():
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
            subplot_titles=(
                "Fills served",
                "Exposure: share of rows holding a position",
                f"Markout at +{horizon} (bp, negative is adverse)",
                f"Fees ({currency})",
            ),
        )

        for name, colour, y in (("Buy", ACCENT, buys), ("Sell", ORANGE, sells)):
            fig.add_trace(go.Bar(
                x=_HOURS, y=y, name=name,
                marker=dict(color=colour, line=dict(color=DARK_GRAY, width=0.4)),
                hovertemplate="%{x}:00 UTC<br>" + name.lower()
                              + " %{y:,} fills<extra></extra>",
            ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=_HOURS, y=exposure, name="Exposure", showlegend=False,
            marker=dict(color=GRAY, line=dict(color=DARK_GRAY, width=0.4)),
            hovertemplate="%{x}:00 UTC<br>%{y:.1%} of rows<extra></extra>",
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=_HOURS, y=means, name="Markout", showlegend=False,
            marker=dict(
                color=[RED if (m == m and m < 0) else GREEN for m in means],
                line=dict(color=DARK_GRAY, width=0.4),
            ),
            error_y=dict(type="data", array=errs, color=GRAY,
                         thickness=1.2, width=5),
            hovertext=[f"{m:+.3f} bp<br>{int(n):,} fills marked"
                       for m, n in zip(means, marked)],
            hoverinfo="text+x",
        ), row=3, col=1)

        fig.add_trace(go.Bar(
            x=_HOURS, y=fees_by_hour, name="Fees", showlegend=False,
            marker=dict(color=ORANGE, line=dict(color=DARK_GRAY, width=0.4)),
            hovertemplate="%{x}:00 UTC<br>%{y:,.2f} " + currency + "<extra></extra>",
        ), row=4, col=1)

        fig.add_hline(y=0, line_color=DARK_GRAY, line_width=0.8, row=3, col=1)
        fig.update_yaxes(title_text="fills", row=1, col=1)
        fig.update_yaxes(title_text=None, tickformat=".0%", row=2, col=1)
        fig.update_yaxes(title_text="bp", row=3, col=1)
        fig.update_yaxes(title_text=currency, row=4, col=1)
        for r in (1, 2, 3, 4):
            fig.update_xaxes(type="category", row=r, col=1)
        fig.update_xaxes(title_text="hour of the UTC day", row=4, col=1)

        day_note = ("one UTC day" if days == 1
                    else f"{days:,} UTC days, pooled by hour of the day")
        fig.update_layout(
            title_text=title,
            width=int(figsize[0] * 80), height=int(figsize[1] * 80),
            barmode="stack", bargap=0.2, hovermode="closest",
            # The hour axis has a title of its own, so the note goes below it
            # rather than through it.
            margin=dict(b=96),
            legend=dict(orientation="h", yanchor="bottom", y=1.03,
                        xanchor="right", x=1),
            annotations=list(fig.layout.annotations) + [dict(
                x=0, y=0, xref="paper", yref="paper",
                xanchor="left", yanchor="top", yshift=-74,
                showarrow=False, font=dict(color=GRAY, size=11),
                text=(
                    f"{n_fills:,} fills over {day_note}. Whiskers are the "
                    "standard error of the hour's mean markout."
                ),
            )],
        )
        return finalize(fig, show=show, save=save)


def _collect_markouts(result, horizon: str, into: list) -> None:
    """Append one run's markouts, in bp, to the per-hour buckets."""
    df = getattr(result, "fill_marks_df", lambda: None)()
    if df is None or f"mark_{horizon}" not in getattr(df, "columns", ()):
        return
    hours = _hour_of(df["timestamp"])
    side = np.asarray(df["side"], dtype=np.float64)
    price = np.asarray(df["price"], dtype=np.float64)
    mark = np.asarray(df[f"mark_{horizon}"], dtype=np.float64)
    bps = 1e4 * side * (mark - price) / price
    ok = np.isfinite(bps)
    for h in range(24):
        sel = ok & (hours == h)
        if sel.any():
            into[h].append(bps[sel])


def _summarise(buckets: list):
    """Mean markout, its standard error and the count, per hour."""
    means = np.full(24, np.nan)
    errs = np.zeros(24)
    counts = np.zeros(24)
    for h, parts in enumerate(buckets):
        if not parts:
            continue
        values = np.concatenate(parts)
        n = len(values)
        counts[h] = n
        means[h] = float(values.mean())
        # Standard error of the mean, not the bootstrap the aggregates use:
        # an hour holds hundreds of fills, and a thousand resamples of
        # hundreds of fills says what the closed form already says.
        errs[h] = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return means, errs, counts


def _distinct_days(ts) -> set:
    """The UTC days a run's rows fall on, so pooled runs are not counted twice."""
    if not len(ts):
        return set()
    arr = (ts.to_numpy(dtype="datetime64[ns]") if hasattr(ts, "to_numpy")
           else np.asarray(ts, dtype="datetime64[ns]"))
    return set(np.unique(arr.astype("datetime64[D]")).tolist())
