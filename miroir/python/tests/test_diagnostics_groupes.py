"""The risk and exposure diagnostics group a curve by timestamp in one sort.

``risk_check`` and ``check_exposure_stability`` built one boolean mask over
the whole curve per timestamp: quadratic, hours on a curve of a million bars.
They group with one stable sort now, and a curve with one row per timestamp
(a single symbol) is aggregated without a loop. Held here against the mask
loops, kept below verbatim: the same arrays to the bit (a NaN to a NaN, -0.0
included) and the same per-timestamp dicts, on one symbol and several, rows
in order and shuffled, empty and single-row curves, and zero, negative, NaN
and infinite values.
"""
import numpy as np
import pyarrow as pa
import pytest

from manifoldbt import diagnostics as D


def old_compute_per_timestamp(pos):
    ts_ns = pos["timestamp"].view(np.int64) if pos["timestamp"].dtype.kind == "M" else pos["timestamp"]
    position = pos["position"].astype(np.float64)
    close = pos["close"].astype(np.float64)
    equity = pos["equity"].astype(np.float64)
    market_value = np.abs(position) * close
    unique_ts, inverse = np.unique(ts_ns, return_inverse=True)
    n = len(unique_ts)
    agg_equity = np.empty(n, dtype=np.float64)
    agg_exposure = np.zeros(n, dtype=np.float64)
    agg_hhi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = inverse == i
        agg_equity[i] = equity[mask][0]
        mv = market_value[mask]
        total_mv = mv.sum()
        agg_exposure[i] = total_mv
        if total_mv > 1e-12:
            weights = mv / total_mv
            agg_hhi[i] = (weights ** 2).sum()
        else:
            agg_hhi[i] = 0.0
    safe_eq = np.where(np.abs(agg_equity) > 1e-12, agg_equity, 1e-12)
    utilization = agg_exposure / safe_eq
    return {
        "timestamps": unique_ts.view("datetime64[ns]"),
        "equity": agg_equity,
        "exposure": agg_exposure,
        "utilization": utilization,
        "free_margin_ratio": 1.0 - utilization,
        "concentration": agg_hhi,
    }


def old_exposure_for_result(result):
    from manifoldbt._convert import positions_arrays

    pos = positions_arrays(result)
    ts_ns = pos["timestamp"].view(np.int64) if pos["timestamp"].dtype.kind == "M" else pos["timestamp"]
    position = pos["position"].astype(np.float64)
    close = pos["close"].astype(np.float64)
    equity = pos["equity"].astype(np.float64)
    sym_ids = pos["symbol_id"]
    market_value = np.abs(position) * close
    data = {}
    for ts in np.unique(ts_ns):
        mask = ts_ns == ts
        mv = market_value[mask]
        eq = equity[mask][0]
        total_mv = mv.sum()
        util = total_mv / max(abs(eq), 1e-12)
        sym_pos = {}
        for sid, p in zip(sym_ids[mask], position[mask]):
            sym_pos[int(sid)] = float(p)
        data[int(ts)] = {"utilization": float(util), "exposure": float(total_mv),
                         "equity": float(eq), "positions": sym_pos}
    return data


def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a.hex() == b.hex() or (a != a and b != b)
    if isinstance(a, dict):
        return list(a) == list(b) and all(same_value(a[k], b[k]) for k in a)
    return a == b


def same_array(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    if a.dtype.kind == "f":
        ok = (a.view(np.int64) == b.view(np.int64)) | (np.isnan(a) & np.isnan(b))
        assert ok.all(), (np.flatnonzero(~ok)[:5], a[~ok][:5], b[~ok][:5])
    else:
        assert np.array_equal(a.view(np.int64) if a.dtype.kind == "M" else a,
                              b.view(np.int64) if b.dtype.kind == "M" else b)


def positions(rng, n_ts, n_sym, shuffle, odd):
    ts = np.sort(rng.choice(np.arange(10**6), n_ts, replace=False)).astype(np.int64) * 60 * 10**9
    ts = np.repeat(ts, n_sym)
    sym = np.tile(np.arange(1, n_sym + 1, dtype=np.uint32), n_ts)
    n = len(ts)
    position = rng.normal(0.0, 2.0, n)
    position[rng.random(n) < 0.3] = 0.0
    close = rng.uniform(10.0, 200.0, n)
    equity = np.repeat(rng.uniform(1e3, 1e5, n_ts), n_sym)
    if odd and n:
        k = max(1, n // 10)
        close[rng.integers(0, n, k)] = -rng.uniform(1.0, 5.0, k)
        close[rng.integers(0, n, k)] = np.nan
        position[rng.integers(0, n, k)] = np.inf
        position[rng.integers(0, n, k)] = -0.0
        close[rng.integers(0, n, k)] = 1e-15
        equity[rng.integers(0, n, k)] = 0.0
    if shuffle and n:
        p = rng.permutation(n)
        ts, sym, position, close, equity = ts[p], sym[p], position[p], close[p], equity[p]
    return {"timestamp": ts.view("datetime64[ns]"), "symbol_id": sym, "position": position,
            "close": close, "equity": equity, "capital": equity * 0.5}


class FakeResult:
    def __init__(self, pos):
        self.positions = pa.RecordBatch.from_pydict({
            "timestamp": pa.array(pos["timestamp"].view(np.int64), pa.timestamp("ns", tz="UTC")),
            "symbol_id": pa.array(pos["symbol_id"], pa.uint32()),
            "position": pa.array(pos["position"], pa.float64()),
            "close": pa.array(pos["close"], pa.float64()),
            "capital": pa.array(pos["capital"], pa.float64()),
            "equity": pa.array(pos["equity"], pa.float64()),
        })


CASES = [(n_ts, n_sym, shuffle, odd)
         for n_ts in (0, 1, 2, 57, 800)
         for n_sym in (1, 3)
         for shuffle in (False, True)
         for odd in (False, True)]


@pytest.mark.parametrize("n_ts,n_sym,shuffle,odd", CASES)
def test_the_grouped_aggregates_are_the_masked_ones(n_ts, n_sym, shuffle, odd):
    rng = np.random.default_rng(n_ts * 31 + n_sym * 7 + shuffle * 3 + odd)
    pos = positions(rng, n_ts, n_sym, shuffle, odd)
    with np.errstate(all="ignore"):
        new = D._compute_per_timestamp(pos)
        old = old_compute_per_timestamp(pos)
    assert list(new) == list(old)
    for k in old:
        same_array(new[k], old[k])
    res = FakeResult(pos)
    with np.errstate(all="ignore"):
        assert same_value(D._exposure_for_result(res), old_exposure_for_result(res))


def test_a_single_symbol_curve_takes_no_loop_and_keeps_its_zeros():
    """-0.0 exposure: numpy's sum of one element is +0.0, and so is the new one."""
    pos = {"timestamp": (np.arange(4, dtype=np.int64) * 10**9).view("datetime64[ns]"),
           "symbol_id": np.ones(4, np.uint32),
           "position": np.array([-0.0, 1.0, 0.0, 2.0]),
           "close": np.array([5.0, -0.0, -3.0, np.nan]),
           "equity": np.array([1.0, 2.0, 3.0, 4.0]), "capital": np.ones(4)}
    new, old = D._compute_per_timestamp(pos), old_compute_per_timestamp(pos)
    for k in old:
        same_array(new[k], old[k])
    assert new["exposure"][1].hex() == "0x0.0p+0"
    res = FakeResult(pos)
    with np.errstate(all="ignore"):
        new_e, old_e = D._exposure_for_result(res), old_exposure_for_result(res)
    assert same_value(new_e, old_e)
    assert new_e[10**9]["exposure"].hex() == "0x0.0p+0"
