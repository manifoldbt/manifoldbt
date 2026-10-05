"""``round_trips`` pairs an engine result's fill log in the engine.

The pairing walked the fills in the interpreter, one numpy scalar at a time:
tens of milliseconds for a run of a few thousand fills, seconds for a quoting
run. The engine walks them now (``round_trips_columns``), the same walk on the
same doubles; the Python walk stays for any other object. Held here against
the walk as it was, kept below verbatim: the same columns, of the same dtypes,
with the same values to the bit (a NaN to a NaN), on real runs that open,
add, reduce, close and flip, on one symbol and on two, with positions left
open at the end, with and without them; and on fill logs written by hand with
zero and NaN quantities and prices, which the Python walk still answers.
"""
import os

import numpy as np
import pyarrow as pa
import pytest

import manifoldbt as bt
from manifoldbt import _trades
from manifoldbt._convert import positions_arrays, trades_arrays

pd = pytest.importorskip("pandas")


def old_round_trips(result, *, include_open=True):
    """``round_trips`` as it was, walking the numpy arrays."""
    SIDE_LONG, SIDE_SHORT = _trades.SIDE_LONG, _trades.SIDE_SHORT
    ta = trades_arrays(result)
    n = len(ta.get("symbol_id", ()))

    sym = np.asarray(ta["symbol_id"], dtype=np.uint32) if n else np.zeros(0, np.uint32)
    exec_ts = ta["execution_timestamp"].astype("datetime64[ns]") if n else np.zeros(0, "datetime64[ns]")
    exec_ns = exec_ts.view(np.int64)
    side = np.asarray(ta["side"], dtype=np.int64) if n else np.zeros(0, np.int64)
    qty = np.asarray(ta["quantity"], dtype=np.float64) if n else np.zeros(0)
    price = np.asarray(ta["fill_price"], dtype=np.float64) if n else np.zeros(0)
    fees = np.asarray(ta["fees"], dtype=np.float64) if n else np.zeros(0)
    reason = np.asarray(ta["exit_reason"], dtype=np.int64) if n else np.zeros(0, np.int64)

    open_pos = {}
    out = []

    def _close(o, s, i, close_qty, fill_price, fill_fees):
        if o[1] == SIDE_LONG:
            pnl = (fill_price - o[2]) * close_qty - o[4] - fill_fees
        else:
            pnl = (o[2] - fill_price) * close_qty - o[4] - fill_fees
        out.append((
            s, o[0], exec_ns[i], o[1], o[2], fill_price, close_qty,
            o[4] + fill_fees, pnl, reason[i], False, o[5], i,
        ))

    with np.errstate(all="ignore"):
        for i in range(n):
            s = int(sym[i])
            signed = qty[i] if side[i] == SIDE_LONG else -qty[i]
            o = open_pos.get(s)
            if o is None:
                open_pos[s] = [exec_ns[i], int(side[i]), price[i], qty[i], fees[i], i]
                continue
            old_pos = o[3] if o[1] == SIDE_LONG else -o[3]
            new_pos = old_pos + signed
            if (old_pos > 0.0 and new_pos > old_pos) or (old_pos < 0.0 and new_pos < old_pos):
                total_cost = o[2] * o[3] + price[i] * qty[i]
                o[3] = abs(new_pos)
                o[2] = total_cost / o[3]
                o[4] += fees[i]
                continue
            close_qty = min(qty[i], o[3])
            _close(o, s, i, close_qty, price[i], fees[i])
            if abs(new_pos) < 1e-12:
                del open_pos[s]
            elif (old_pos > 0.0 > new_pos) or (old_pos < 0.0 < new_pos):
                new_side = SIDE_LONG if new_pos > 0.0 else SIDE_SHORT
                open_pos[s] = [exec_ns[i], new_side, price[i], abs(new_pos), 0.0, i]
            else:
                o[3] = abs(new_pos)
                o[4] = 0.0

        if include_open and open_pos:
            pa_ = positions_arrays(result)
            p_sym = np.asarray(pa_["symbol_id"])
            p_ts = pa_["timestamp"].astype("datetime64[ns]").view(np.int64)
            p_close = np.asarray(pa_["close"], dtype=np.float64)
            for s in sorted(open_pos):
                o = open_pos[s]
                rows = np.flatnonzero(p_sym == s)
                if len(rows) == 0:
                    continue
                last = rows[-1]
                mark, mark_ns = p_close[last], p_ts[last]
                if o[1] == SIDE_LONG:
                    pnl = (mark - o[2]) * o[3] - o[4]
                else:
                    pnl = (o[2] - mark) * o[3] - o[4]
                out.append((
                    s, o[0], mark_ns, o[1], o[2], mark, o[3], o[4], pnl,
                    _trades.EXIT_REASON_OPEN, True, o[5], -1,
                ))

        m = len(out)
        cols = list(zip(*out)) if m else [()] * 13
        entry_ns = np.asarray(cols[1], dtype=np.int64)
        exit_ns = np.asarray(cols[2], dtype=np.int64)
        entry_price = np.asarray(cols[4], dtype=np.float64)
        quantity = np.asarray(cols[6], dtype=np.float64)
        pnl = np.asarray(cols[8], dtype=np.float64)
        notional = entry_price * quantity
        return_pct = np.where(notional > 0.0, pnl / notional, 0.0)

    return {
        "symbol_id": np.asarray(cols[0], dtype=np.uint32),
        "entry_timestamp": entry_ns.view("datetime64[ns]"),
        "exit_timestamp": exit_ns.view("datetime64[ns]"),
        "side": np.asarray(cols[3], dtype=np.uint8),
        "entry_price": entry_price,
        "exit_price": np.asarray(cols[5], dtype=np.float64),
        "quantity": quantity,
        "fees": np.asarray(cols[7], dtype=np.float64),
        "pnl": pnl,
        "return_pct": return_pct,
        "exit_reason": np.asarray(cols[9], dtype=np.int16),
        "holding_seconds": (exit_ns - entry_ns) / 1_000_000_000,
        "is_open": np.asarray(cols[10], dtype=bool),
        "entry_row": np.asarray(cols[11], dtype=np.int64),
        "exit_row": np.asarray(cols[12], dtype=np.int64),
    }


def assert_same(new, old):
    assert list(new) == list(old)
    for k in old:
        a, b = new[k], old[k]
        assert a.dtype == b.dtype, k
        assert a.shape == b.shape, k
        if a.dtype.kind == "f":
            same = (a.view(np.int64) == b.view(np.int64)) | (np.isnan(a) & np.isnan(b))
            assert same.all(), (k, np.flatnonzero(~same)[:5], a[~same][:5], b[~same][:5])
        else:
            assert np.array_equal(a.view(np.int64) if a.dtype.kind == "M" else a,
                                  b.view(np.int64) if b.dtype.kind == "M" else b), k


# ---------------------------------------------------------------------------
# Real runs: the engine's walk
# ---------------------------------------------------------------------------


def _store(tmp_path, symbols):
    n = 4 * 1440
    rng = np.random.default_rng(3)
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    ts = pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC")
    store = None
    for sid in range(1, symbols + 1):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
        df = pd.DataFrame({"timestamp": ts, "open": close, "high": close * 1.001,
                           "low": close * 0.999, "close": close, "volume": np.full(n, 1_000.0)})
        store = bt.import_dataframe(
            df, symbol=f"S{sid}", symbol_id=sid, interval="1m",
            data_root=os.path.join(root, "data"), metadata_db=os.path.join(root, "meta.sqlite"),
        )
    return store, int(ts[-1].value) + 60_000_000_000


def _runs(tmp_path, symbols):
    store, end = _store(tmp_path, symbols)
    config = bt.BacktestConfig(
        universe=list(range(1, symbols + 1)), time_range_start=0, time_range_end=end,
        bar_interval=bt.Interval.minutes(1), initial_capital=10_000.0,
        execution=bt.ExecutionConfig(allow_short=True),
        fees=bt.FeeConfig(maker_fee_bps=2.0, taker_fee_bps=5.0),
    )
    c = bt.col("close")
    f = bt.indicators.ema(c, 5)
    s = bt.indicators.ema(c, 21)
    flips = (bt.Strategy.create("flips").signal("f", f).signal("s", s)
             .size(bt.when(f > s, 0.7, -0.4)))
    # A size that moves every bar: adds, partial closes and flips.
    drift = (bt.Strategy.create("drift").signal("f", f).signal("s", s)
             .size(bt.clip((f - s) / s * 400.0, -0.9, 0.9)))
    brackets = (bt.Strategy.create("brackets").signal("f", f).signal("s", s)
                .size(bt.when(f > s, 0.5, 0.0)).stop_loss(pct=0.3).take_profit(pct=0.5))
    return [bt.run(st, config, store) for st in (flips, drift, brackets)]


@pytest.mark.parametrize("symbols", [1, 2])
def test_the_engine_pairs_a_real_run_as_the_walk_did(tmp_path, monkeypatch, symbols):
    results = _runs(tmp_path, symbols)

    def no_walk(*_a, **_k):
        raise AssertionError("an engine result took the Python walk")

    monkeypatch.setattr(_trades, "_walk", no_walk)
    for res in results:
        assert res.trade_count > 50
        for include_open in (True, False):
            assert_same(_trades.round_trips(res.raw, include_open=include_open),
                        old_round_trips(res.raw, include_open=include_open))
            assert_same(_trades.round_trips(res, include_open=include_open),
                        old_round_trips(res.raw, include_open=include_open))
        df_new = res.round_trips_df(backend="pandas")
        df_old = pd.DataFrame(old_round_trips(res.raw))
        pd.testing.assert_frame_equal(df_new, df_old, check_exact=True)
    # The runs did what they are here for.
    rt = [old_round_trips(r.raw) for r in results]
    assert any(t["is_open"].any() for t in rt)
    assert any((t["exit_reason"] == 1).any() or (t["exit_reason"] == 2).any() for t in rt)


# ---------------------------------------------------------------------------
# Hand-written fill logs: the Python walk, as before
# ---------------------------------------------------------------------------


class FakeResult:
    """What ``round_trips`` reads of a result: the trades and the positions."""

    def __init__(self, trades, positions):
        self.trades = trades
        self.positions = positions


def fake_result(rng, n_fills, n_symbols, odd=False):
    ts = np.sort(rng.integers(1_600_000_000, 1_600_100_000, n_fills)).astype(np.int64) * 1_000_000_000
    sym = rng.integers(1, n_symbols + 1, n_fills).astype(np.uint32)
    side = rng.integers(1, 3, n_fills).astype(np.uint8)
    qty = np.round(rng.uniform(0.0, 3.0, n_fills), rng.integers(0, 4))
    price = rng.uniform(90.0, 110.0, n_fills)
    fees = rng.uniform(0.0, 0.2, n_fills)
    reason = rng.integers(0, 7, n_fills).astype(np.uint8)
    if odd:
        k = max(1, n_fills // 20)
        qty[rng.integers(0, n_fills, k)] = 0.0
        qty[rng.integers(0, n_fills, k)] = np.nan
        price[rng.integers(0, n_fills, k)] = np.nan
        price[rng.integers(0, n_fills, k)] = 0.0
        fees[rng.integers(0, n_fills, k)] = -0.0
        qty[rng.integers(0, n_fills, k)] = 1e-13
    trades = pa.RecordBatch.from_pydict({
        "symbol_id": pa.array(sym, pa.uint32()),
        "execution_timestamp": pa.array(ts, pa.timestamp("ns", tz="UTC")),
        "side": pa.array(side, pa.uint8()),
        "quantity": pa.array(qty, pa.float64()),
        "fill_price": pa.array(price, pa.float64()),
        "fees": pa.array(fees, pa.float64()),
        "exit_reason": pa.array(reason, pa.uint8()),
    })
    p_ts = np.repeat(np.arange(1_600_000_000, 1_600_100_001, 10_000) * 1_000_000_000, n_symbols)
    p_sym = np.tile(np.arange(1, n_symbols + 1, dtype=np.uint32), len(p_ts) // n_symbols)
    positions = pa.RecordBatch.from_pydict({
        "timestamp": pa.array(p_ts, pa.timestamp("ns", tz="UTC")),
        "symbol_id": pa.array(p_sym, pa.uint32()),
        "close": pa.array(rng.uniform(90.0, 110.0, len(p_ts)), pa.float64()),
    })
    return FakeResult(trades, positions)


def no_walk(*_a, **_k):
    raise AssertionError("a table in the engine's shape took the Python walk")


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("include_open", [True, False])
def test_a_hand_written_log_is_paired_as_before(monkeypatch, seed, include_open):
    """Tables in the engine's shape go to the engine too: zero, NaN and tiny
    quantities, NaN and zero prices, -0.0 fees, several symbols."""
    rng = np.random.default_rng(seed)
    res = fake_result(rng, n_fills=int(rng.integers(0, 600)), n_symbols=int(rng.integers(1, 4)),
                      odd=seed % 3 == 0)
    monkeypatch.setattr(_trades, "_walk", no_walk)
    with np.errstate(all="ignore"):
        new = _trades.round_trips(res, include_open=include_open)
    assert_same(new, old_round_trips(res, include_open=include_open))


def test_a_table_of_another_shape_takes_the_walk():
    rng = np.random.default_rng(99)
    res = fake_result(rng, n_fills=300, n_symbols=2, odd=True)
    # Sides as int64: not the engine's column type.
    t = res.trades
    res.trades = t.set_column(t.schema.get_field_index("side"), "side",
                              t.column("side").cast(pa.int64()))
    walked = []
    real_walk = _trades._walk

    def spy(*a, **k):
        walked.append(1)
        return real_walk(*a, **k)

    _trades._walk = spy
    try:
        with np.errstate(all="ignore"):
            new = _trades.round_trips(res)
    finally:
        _trades._walk = real_walk
    assert walked
    assert_same(new, old_round_trips(res))


def test_open_positions_without_a_positions_table_raise_as_before():
    rng = np.random.default_rng(5)
    res = fake_result(rng, n_fills=50, n_symbols=1)
    del res.positions
    with pytest.raises(AttributeError):
        old_round_trips(res)  # a position is left open, the walk needs the table
    with pytest.raises(AttributeError):
        _trades.round_trips(res)
    # Without the open ones, nothing needs it.
    assert_same(_trades.round_trips(res, include_open=False),
                old_round_trips(res, include_open=False))
