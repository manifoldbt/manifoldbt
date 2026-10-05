"""equity_with_dates rend, sans deduplication quand il n'y a rien a dedupliquer,
exactement ce que la deduplication rendait."""
import numpy as np
import pyarrow as pa

from manifoldbt._convert import _ts_to_int64, arrow_to_numpy, equity_with_dates


class _Resultat:
    def __init__(self, ts, equity):
        self.positions = pa.table({
            "timestamp": pa.array(np.asarray(ts, dtype=np.int64), pa.timestamp("ns", tz="UTC")),
            "equity": pa.array(np.asarray(equity, dtype=np.float64)),
        })


def _par_deduplication(result):
    ts_ns = _ts_to_int64(result.positions.column("timestamp"))
    eq_raw = arrow_to_numpy(result.positions.column("equity"))
    _, idx = np.unique(ts_ns, return_index=True)
    idx.sort()
    return ts_ns[idx].view("datetime64[ns]"), eq_raw[idx].astype(np.float64)


def _memes(a, b):
    assert a[0].dtype == b[0].dtype and a[1].dtype == b[1].dtype
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_un_symbole_rend_ce_que_la_deduplication_rendait():
    ts = np.arange(1_000, dtype=np.int64) * 60_000_000_000
    eq = 100.0 + np.cumsum(np.random.default_rng(1).normal(size=1_000))
    r = _Resultat(ts, eq)
    got = equity_with_dates(r)
    _memes(got, _par_deduplication(r))
    # Des copies qu'on peut modifier, comme avant.
    got[1][0] = -1.0
    assert r.positions.column("equity")[0].as_py() != -1.0


def test_plusieurs_symboles_sont_toujours_dedupliques():
    ts = np.repeat(np.arange(200, dtype=np.int64), 3) * 1_000_000_000
    eq = np.repeat(np.linspace(100.0, 120.0, 200), 3)
    r = _Resultat(ts, eq)
    got = equity_with_dates(r)
    assert len(got[0]) == 200
    _memes(got, _par_deduplication(r))


def test_les_bords():
    for ts, eq in [([], []), ([5], [1.0]), ([5, 5], [1.0, 1.0]), ([7, 3], [2.0, 1.0])]:
        r = _Resultat(ts, eq)
        _memes(equity_with_dates(r), _par_deduplication(r))
