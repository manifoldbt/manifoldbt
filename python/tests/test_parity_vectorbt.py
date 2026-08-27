"""Cross-engine parity: manifoldbt vs vectorbt on brackets, shorts, fees.

This suite pins manifoldbt's fill semantics against an independent engine
(vectorbt) on controlled synthetic bars, so a refactor that silently changes a
fill price, a stop level, or PnL booking is caught here rather than in the wild.

Coverage — only what vectorbt can legitimately model apples-to-apples:

* market entry + take-profit          (test_market_take_profit_parity)
* market entry + stop-loss            (test_market_stop_loss_parity)
* combined SL+TP bracket              (test_bracket_sl_tp_parity)
* short entry + take-profit           (test_short_take_profit_parity)
* trailing stop                       (test_trailing_stop_parity)
* fees over multiple round-trips      (test_fees_multi_trade_parity)

Out of scope for vectorbt (validated separately, NOT against vectorbt):

* determined-price / resting limit entry — vectorbt has no resting order, so
  ``test_limit_entry_matches_independent_reference`` pins it against a NumPy
  model instead.
* sizing under fees — with ``FractionOfEquity`` the engines size differently
  once fees exist (manifoldbt charges the fee on top of a full-equity notional;
  vectorbt reserves it out of cash). Both are legitimate; the fee test sizes in
  fixed units to compare the fee arithmetic without that policy difference.

What is compared, and why only this:

* Trade fills (entry price, exit price, exit reason) and final ``total_return``.
  These are computed at full internal resolution and are exact. The *equity
  curve* is deliberately NOT compared: on a Community build the output series is
  capped to daily resolution, so its shape is not apples-to-apples with
  vectorbt. The realised trades and the final equity are unaffected by that cap.

Convention alignment (measured against manifoldbt 0.14.1, not assumed):

* ``signal_delay=0`` + ``AtClose`` → a market entry fills at the *close* of the
  signal bar. vectorbt ``from_signals`` fills the entry bar at close by default,
  so entries line up with no shift.
* ``FractionOfEquity`` sizing is taken at the *signal-bar close*
  (``size_at_fill_price=False``). For a market entry that equals the fill price,
  so vectorbt ``size_type="percent"`` matches. For a resting limit entry the
  signal close and the fill price differ, so vectorbt is fed an explicit unit
  size to reproduce manifoldbt's "size at signal close" rule.
* Take-profit is a passive target: it fills at the level even if the bar gaps
  through it. Stop-loss fills at the level (or worse on a gap). vectorbt's
  ``stop_exit_price=StopMarket`` reproduces the level fill on these
  no-gap-at-open scenarios.

vectorbt has no resting entry order, so the limit-entry scenario also carries an
independent NumPy reference for *where* the order fills; vectorbt only checks the
downstream take-profit off that fill.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
vbt = pytest.importorskip("vectorbt")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, when  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402
from vectorbt.portfolio.enums import StopExitPrice, Direction  # noqa: E402

CAPITAL = 10_000.0
REL_TOL = 1e-6

# Exit-reason codes emitted in trades_df (measured):
REASON_NONE, REASON_SL, REASON_TP, REASON_TRAIL = 0, 1, 2, 3


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bars(o, h, l, c, start="2023-01-01"):
    ts = pd.date_range(start, periods=len(c), freq="1h", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": list(map(float, o)), "high": list(map(float, h)),
         "low": list(map(float, l)), "close": list(map(float, c)),
         "volume": [1000.0] * len(c)}
    )


def _mbt_run(df, strat, tmp_path, name, *, delay=0, allow_short=False,
             sizing="FractionOfEquity", fees=None):
    """Run manifoldbt on an in-memory OHLC frame; return the Result."""
    root = str(tmp_path / name)
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df, symbol="TEST", symbol_id=1, interval="1h",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    ts = df["timestamp"]
    cfg = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=int(ts.iloc[-1].value) + 30 * 86_400_000_000_000,
        bar_interval=Interval.hours(1),
        initial_capital=CAPITAL,
        execution=bt.ExecutionConfig(
            signal_delay=delay, execution_price="AtClose",
            max_position_pct=1.0, allow_short=allow_short,
            position_sizing_mode=sizing,
        ),
        fees=fees if fees is not None else bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )
    return bt.run(strat, cfg, store)


def _mbt_trades(res):
    """(entry_fill, exit_fill, exit_reason) from a two-row round-trip."""
    tr = res.trades_df()
    assert len(tr) == 2, f"expected one round-trip, got {len(tr)} rows:\n{tr}"
    entry = tr.iloc[0]
    exit_ = tr.iloc[1]
    return float(entry["fill_price"]), float(exit_["fill_price"]), int(exit_["exit_reason"])


def _vbt_from_signals(df, entries, *, exits=None, tp=None, sl=None,
                      sl_trail=False, size=1.0, size_type="percent",
                      direction=None, fees=0.0):
    idx = pd.DatetimeIndex(df["timestamp"])
    close = pd.Series(df["close"].values, index=idx, dtype=float)
    ent = pd.Series(entries, index=idx)
    ex = pd.Series(exits if exits is not None else False, index=idx)
    kwargs = dict(
        open=pd.Series(df["open"].values, index=idx, dtype=float),
        high=pd.Series(df["high"].values, index=idx, dtype=float),
        low=pd.Series(df["low"].values, index=idx, dtype=float),
        init_cash=CAPITAL, size=size, size_type=size_type,
        fees=fees, slippage=0.0, sl_stop=sl, tp_stop=tp, sl_trail=sl_trail,
        stop_exit_price=StopExitPrice.StopMarket,
        freq="1h", accumulate=False,
    )
    if direction is not None:
        kwargs["direction"] = direction
    return vbt.Portfolio.from_signals(close, ent, ex, **kwargs)


def _assert_close(a, b, msg):
    assert abs(a - b) <= REL_TOL * max(1.0, abs(b)), f"{msg}: {a} != {b}"


# --------------------------------------------------------------------------- #
# Scenario A — market entry + take-profit
# --------------------------------------------------------------------------- #
def test_market_take_profit_parity(tmp_path):
    # Enter long at bar 0 close (100). TP +10% (110) is crossed at bar 3
    # (open 108 < 110 < high 115): both engines fill the target at 110.
    df = _bars(
        o=[100, 100, 104, 108, 111, 113],
        h=[101, 102, 106, 115, 112, 114],
        l=[99, 99, 103, 107, 110, 112],
        c=[100, 100, 105, 112, 111, 113],
    )
    # Long only while close in (99.5, 106): true on bars 0-2, false after, so
    # the position is a single clean round-trip closed by the TP.
    entry = when((col("close") > lit(99.5)) & (col("close") < lit(106.0)),
                 lit(1.0), lit(0.0))
    strat = (bt.Strategy.create("mkt_tp")
             .signal("d", col("close")).size(entry).take_profit(pct=10.0))

    res = _mbt_run(df, strat, tmp_path, "A")
    m_entry, m_exit, reason = _mbt_trades(res)
    assert reason == REASON_TP
    _assert_close(m_entry, 100.0, "mbt entry")
    _assert_close(m_exit, 110.0, "mbt tp exit")

    pf = _vbt_from_signals(df, [True, False, False, False, False, False], tp=0.10)
    v_tr = pf.trades.records_readable.iloc[0]
    _assert_close(float(v_tr["Avg Entry Price"]), m_entry, "entry price")
    _assert_close(float(v_tr["Avg Exit Price"]), m_exit, "exit price")
    _assert_close(pf.total_return(), res.metrics["total_return"], "total_return")


# --------------------------------------------------------------------------- #
# Scenario B — market entry + stop-loss
# --------------------------------------------------------------------------- #
def test_market_stop_loss_parity(tmp_path):
    # Enter long at bar 0 close (100). SL -5% (95) is hit at bar 3
    # (open 97 > 95, low 94 <= 95): both engines fill the stop at 95.
    df = _bars(
        o=[100, 100, 99, 97, 96, 95],
        h=[101, 101, 100, 98, 97, 96],
        l=[99, 99, 96, 94, 95, 94],
        c=[100, 100, 98, 96, 96, 95],
    )
    entry = when(col("close") >= lit(97.0), lit(1.0), lit(0.0))
    strat = (bt.Strategy.create("mkt_sl")
             .signal("d", col("close")).size(entry).stop_loss(pct=5.0))

    res = _mbt_run(df, strat, tmp_path, "B")
    m_entry, m_exit, reason = _mbt_trades(res)
    assert reason == REASON_SL
    _assert_close(m_entry, 100.0, "mbt entry")
    _assert_close(m_exit, 95.0, "mbt sl exit")

    pf = _vbt_from_signals(df, [True, False, False, False, False, False], sl=0.05)
    v_tr = pf.trades.records_readable.iloc[0]
    _assert_close(float(v_tr["Avg Entry Price"]), m_entry, "entry price")
    _assert_close(float(v_tr["Avg Exit Price"]), m_exit, "exit price")
    _assert_close(pf.total_return(), res.metrics["total_return"], "total_return")


# --------------------------------------------------------------------------- #
# Scenario C — resting limit entry at a determined price + take-profit
# --------------------------------------------------------------------------- #
def _resting_limit_reference(df, signal_bar, offset_frac, tp_frac, capital,
                             size_bar=None):
    """Independent NumPy model of a resting buy-limit + take-profit.

    Mirrors the measured manifoldbt rule: the limit rests at
    ``signal_close * (1 - offset_frac)``, fills on the first bar AFTER the
    signal bar whose low touches it (fill AT the level), then a passive TP at
    ``fill * (1 + tp_frac)`` closes it on the first later bar whose high
    reaches it.

    Two different bars decide the level and the size. The LEVEL comes from the
    signal bar. The SIZE (``size_at_fill_price=False``) comes from the close of
    the bar the order is placed on, which is the signal bar shifted by
    ``signal_delay`` — the same bar at delay 0, the next one at delay 1.
    ``size_bar`` says which; it defaults to the signal bar.
    """
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    signal_close = close[signal_bar]
    limit = signal_close * (1.0 - offset_frac)
    qty = capital / close[signal_bar if size_bar is None else size_bar]

    fill_bar = next((i for i in range(signal_bar + 1, len(low)) if low[i] <= limit), None)
    assert fill_bar is not None, "limit never filled in reference"
    tp = limit * (1.0 + tp_frac)
    exit_bar = next((i for i in range(fill_bar, len(high)) if high[i] >= tp), None)
    assert exit_bar is not None, "TP never reached in reference"
    total_return = qty * (tp - limit) / capital
    return dict(limit=limit, qty=qty, fill_bar=fill_bar, tp=tp,
                exit_bar=exit_bar, total_return=total_return)


def test_limit_entry_matches_independent_reference(tmp_path):
    """Determined-price (resting limit) entry — validated WITHOUT vectorbt.

    vectorbt has no resting entry order: it cannot wait across bars for price to
    trade down to a level, so a "vs vectorbt" check would not be apples-to-apples
    and is deliberately not attempted. This manifoldbt-only feature is pinned
    against an independent NumPy model of the resting fill instead. The vectorbt
    suite above covers what both engines share (market entry, SL, TP).

    Signal at bar 0 (close 100). Limit rests 2% below (98). Bar 1 low 97 <= 98
    fills at 98. TP +5% off the fill (102.9) is reached at bar 3 (open 102 < the
    target, so it fills the passive target at the level, not on a gap).

    Runs at ``delay=1``, unlike the market-entry scenarios above: a resting
    order is gated against the bar it was placed from, so delay 0 would price
    it off the same bar whose low decides the fill. The engine refuses that.
    The reference model already assumes this shape (level from the signal bar,
    fill from the next one), so the expected numbers are unchanged.
    """
    df = _bars(
        o=[100, 99, 101, 102, 104, 105],
        h=[100.5, 100, 102, 104, 105, 106],
        l=[99.5, 97, 100, 101.5, 103, 104],
        c=[100, 99, 101, 103, 104, 105],
        start="2023-01-02",
    )
    # The target holds until the take-profit bar, so the bracket decides the
    # exit. A target that dropped back to 0 first would plan a signal exit and
    # the take-profit would never be reached: at delay 1 that is what happens,
    # and at delay 0 it was only hidden because the target had already been
    # recorded as 0 by the time the entry filled.
    entry = when(col("close") <= lit(103.0), lit(1.0), lit(0.0))
    strat = (bt.Strategy.create("lim_tp")
             .signal("d", col("close"))
             .size(entry)
             .limit_entry(offset_bps=200, time_in_force="GTC")  # 200 bps = 2%
             .take_profit(pct=5.0))

    res = _mbt_run(df, strat, tmp_path, "C", delay=1)
    m_entry, m_exit, reason = _mbt_trades(res)

    # Level from bar 0 (the signal), size from bar 1 (where delay 1 places it).
    ref = _resting_limit_reference(df, signal_bar=0, offset_frac=0.02,
                                   tp_frac=0.05, capital=CAPITAL, size_bar=1)
    _assert_close(m_entry, ref["limit"], "limit fill price")   # 98.0
    _assert_close(m_exit, ref["tp"], "tp exit price")          # 102.9
    assert reason == REASON_TP
    _assert_close(res.metrics["total_return"], ref["total_return"], "total_return")


# --------------------------------------------------------------------------- #
# Scenario D — combined SL+TP bracket (both armed, the right one fires)
# --------------------------------------------------------------------------- #
def test_bracket_sl_tp_parity(tmp_path):
    # SL -5% (95) AND TP +10% (110) armed together. Price rises, so the TP fires
    # at bar 3 and the stop never triggers — the bracket must not misfire.
    df = _bars(
        o=[100, 100, 104, 108, 111, 113],
        h=[101, 102, 106, 115, 112, 114],
        l=[99, 99, 103, 107, 110, 112],
        c=[100, 100, 105, 112, 111, 113],
    )
    entry = when((col("close") > lit(99.5)) & (col("close") < lit(106.0)),
                 lit(1.0), lit(0.0))
    strat = (bt.Strategy.create("bracket")
             .signal("d", col("close")).size(entry)
             .stop_loss(pct=5.0).take_profit(pct=10.0))

    res = _mbt_run(df, strat, tmp_path, "D")
    m_entry, m_exit, reason = _mbt_trades(res)
    assert reason == REASON_TP
    _assert_close(m_exit, 110.0, "mbt tp exit")

    pf = _vbt_from_signals(df, [True, False, False, False, False, False],
                           sl=0.05, tp=0.10)
    v_tr = pf.trades.records_readable.iloc[0]
    _assert_close(float(v_tr["Avg Exit Price"]), m_exit, "exit price")
    _assert_close(pf.total_return(), res.metrics["total_return"], "total_return")


# --------------------------------------------------------------------------- #
# Scenario E — short entry + take-profit
# --------------------------------------------------------------------------- #
def test_short_take_profit_parity(tmp_path):
    # Short at bar 0 close (100). TP -5% (95, profit for a short) is hit at bar 3
    # (open 96 > 95, low 94 <= 95): both engines cover at 95 for a +5% return.
    # Signal is short on bars 0-2 and flat from bar 3, so the TP closes it with
    # no re-entry.
    df = _bars(
        o=[100, 99, 98, 96, 95, 94],
        h=[100.5, 100, 99, 97, 96, 95],
        l=[99.5, 98, 97, 94, 94, 93],
        c=[100, 98, 97, 95, 94, 93],
    )
    entry = when(col("close") >= lit(96.0), lit(-1.0), lit(0.0))
    strat = (bt.Strategy.create("short_tp")
             .signal("d", col("close")).size(entry).take_profit(pct=5.0))

    res = _mbt_run(df, strat, tmp_path, "E", allow_short=True)
    m_entry, m_exit, reason = _mbt_trades(res)
    assert reason == REASON_TP
    _assert_close(m_entry, 100.0, "short entry")
    _assert_close(m_exit, 95.0, "short cover")

    pf = _vbt_from_signals(df, [True, False, False, False, False, False],
                           tp=0.05, direction=Direction.ShortOnly)
    v_tr = pf.trades.records_readable.iloc[0]
    _assert_close(float(v_tr["Avg Entry Price"]), m_entry, "entry price")
    _assert_close(float(v_tr["Avg Exit Price"]), m_exit, "cover price")
    _assert_close(pf.total_return(), res.metrics["total_return"], "total_return")


# --------------------------------------------------------------------------- #
# Scenario F — trailing stop
# --------------------------------------------------------------------------- #
def test_trailing_stop_parity(tmp_path):
    # Always long. The high peaks at 112 (bar 3-4), so a 5% trailing stop rests
    # at 112 * 0.95 = 106.4. Bar 5 low (104) trades through it: both engines exit
    # at 106.4. (vectorbt's sl_trail also trails off the high when high is given.)
    df = _bars(
        o=[100, 101, 106, 110, 111, 108],
        h=[100, 102, 108, 112, 112, 109],
        l=[100, 100, 105, 109, 109, 104],
        c=[100, 102, 107, 111, 110, 105],
    )
    strat = (bt.Strategy.create("trail")
             .signal("d", col("close")).size(lit(1.0))
             .trailing_stop(pct=5.0, use_high=True))

    res = _mbt_run(df, strat, tmp_path, "F")
    tr = res.trades_df()
    # Always-long re-enters at the exit bar's close (a mark-flat no-op on the
    # last bar), so the round-trip is the first two rows; assert on those.
    assert float(tr.iloc[1]["fill_price"]) == pytest.approx(106.4)
    assert int(tr.iloc[1]["exit_reason"]) == REASON_TRAIL

    pf = _vbt_from_signals(df, [True, False, False, False, False, False],
                           sl=0.05, sl_trail=True)
    v_tr = pf.trades.records_readable.iloc[0]
    _assert_close(float(v_tr["Avg Exit Price"]), 106.4, "trailing exit")
    _assert_close(pf.total_return(), res.metrics["total_return"], "total_return")


# --------------------------------------------------------------------------- #
# Scenario G — fees over multiple round-trips (cumulative accounting)
# --------------------------------------------------------------------------- #
def test_fees_multi_trade_parity(tmp_path):
    # Two round-trips with a 20 bps taker fee, sized in FIXED UNITS. Fixed units
    # are deliberate: under FractionOfEquity the engines size differently once
    # fees exist (manifoldbt charges the fee on top of a full-equity notional,
    # vectorbt reserves the fee out of cash), which is a legitimate design choice
    # rather than a parity bug. Fixing the unit count isolates the thing both
    # engines must agree on — the fee arithmetic and its cumulative effect.
    units = 50.0
    close = [100, 101, 102, 99, 98, 103, 99]
    df = _bars(
        o=close, h=[c + 0.5 for c in close], l=[c - 0.5 for c in close], c=close,
        start="2023-06-01",
    )
    # Long while close > 100: enters bar 1, exits bar 3, re-enters bar 5, exits
    # bar 6 → two clean round-trips.
    entry = when(col("close") > lit(100.0), lit(units), lit(0.0))
    strat = bt.Strategy.create("fees").signal("d", col("close")).size(entry)
    fees = bt.FeeConfig(maker_fee_bps=10.0, taker_fee_bps=20.0)

    res = _mbt_run(df, strat, tmp_path, "G", sizing="Units", fees=fees)

    sig = pd.Series(close, dtype=float) > 100
    entries = sig & ~sig.shift(1, fill_value=False)
    exits = ~sig & sig.shift(1, fill_value=False)
    pf = _vbt_from_signals(df, entries.tolist(), exits=exits.tolist(),
                           size=units, size_type="amount", fees=0.002)
    _assert_close(pf.total_return(), res.metrics["total_return"], "total_return")
