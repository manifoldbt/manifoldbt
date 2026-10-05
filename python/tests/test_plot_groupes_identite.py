"""The charts group a curve by timestamp, day, month and year without sorting it.

``summary``, ``monthly_returns`` and ``annual_returns`` grouped with np.unique
(a full sort of the curve) and one boolean mask per group, most of their time
on a long curve. They read the runs of an array already in time order now,
and sort only an array that is not (several symbols per timestamp, a NaT).
Two things are held here: each helper gives what np.unique and the masks gave
(values, indices, dtypes) on arrays of every kind, and each chart draws the
same figure, to the byte of its JSON, as with the sorting helpers put back.
"""
import os

import numpy as np
import pytest

pytest.importorskip("plotly")
pd = pytest.importorskip("pandas")

import manifoldbt as bt  # noqa: E402
from manifoldbt.plot import backtest as B  # noqa: E402


# ---------------------------------------------------------------------------
# The sorting helpers, as the charts had them
# ---------------------------------------------------------------------------


def ref_unique_with_first_rows(a):
    values, first = np.unique(a, return_index=True)
    first.sort()
    return values, first


def ref_unique_values(a):
    return np.unique(a)


def ref_groups(keys):
    values = np.unique(keys)
    first = np.full(values.size, -1, dtype=np.intp)
    last = np.full(values.size, -1, dtype=np.intp)
    counts = np.zeros(values.size, dtype=np.intp)
    for i, k in enumerate(values):
        idx = np.nonzero(keys == k)[0]
        counts[i] = len(idx)
        if len(idx):
            first[i], last[i] = idx[0], idx[-1]
    return values, first, last, counts


def ref_calendar_groups(dates, unit, key=None):
    periods = dates.astype(f"datetime64[{unit}]")
    return ref_groups(key(periods) if key is not None else periods)


def year_int(y):
    return y.astype(int) + 1970


def same_array(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype.kind in "mM":
        assert np.array_equal(a.view(np.int64), b.view(np.int64))
    else:
        assert np.array_equal(a, b)


def arrays(rng):
    """Arrays in order, out of order, with repeats, with NaT, empty, single."""
    out = []
    for n in (0, 1, 2, 7, 500):
        base = np.sort(rng.integers(0, max(1, n // 3) + 1, n)).astype(np.int64)
        out += [
            base,
            np.unique(base),
            rng.permutation(base) if n else base,
            base[::-1].copy(),
        ]
    ns = np.sort(rng.integers(1_600_000_000, 1_700_000_000, 400)).astype(np.int64) * 10**9
    dt = ns.view("datetime64[ns]")
    out += [dt, dt.astype("datetime64[D]"), dt.astype("datetime64[M]"), dt.astype("datetime64[Y]")]
    out.append(np.repeat(dt, 2))
    with_nat = dt.copy()
    with_nat[[0, 57, 399]] = np.datetime64("NaT")
    out += [with_nat, with_nat.astype("datetime64[D]"), with_nat.astype("datetime64[M]")]
    lead_nat = dt.copy()
    lead_nat[0] = np.datetime64("NaT")
    out.append(lead_nat)
    out.append(dt.astype("datetime64[Y]").astype(int) + 1970)
    return out


@pytest.mark.parametrize("seed", range(5))
def test_each_helper_gives_what_the_sort_gave(seed):
    for a in arrays(np.random.default_rng(seed)):
        v, f = B._unique_with_first_rows(a)
        rv, rf = ref_unique_with_first_rows(a)
        same_array(v, rv)
        same_array(f, rf)
        same_array(B._unique_values(a), ref_unique_values(a))
        for x, y in zip(B._groups(a), ref_groups(a)):
            same_array(x, y)
        if a.dtype.kind == "M":
            for unit, key in (("M", None), ("Y", None), ("Y", year_int), ("D", None)):
                for x, y in zip(B._calendar_groups(a, unit, key),
                                ref_calendar_groups(a, unit, key)):
                    same_array(x, y)


# ---------------------------------------------------------------------------
# The charts, whole
# ---------------------------------------------------------------------------


def _store(tmp_path, symbols, n, freq):
    rng = np.random.default_rng(11)
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = None
    ts = pd.date_range("2021-11-20", periods=n, freq=freq, tz="UTC")
    for sid in range(1, symbols + 1):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n)))
        df = pd.DataFrame({"timestamp": ts, "open": close, "high": close * 1.002,
                           "low": close * 0.998, "close": close, "volume": np.full(n, 10.0)})
        store = bt.import_dataframe(
            df, symbol=f"S{sid}", symbol_id=sid, interval="1h",
            data_root=os.path.join(root, "data"), metadata_db=os.path.join(root, "meta.sqlite"),
        )
    return store, int(ts[-1].value) + 3_600_000_000_000


def _result(tmp_path, symbols):
    n = 20_000  # a little over two years and a quarter of hours
    store, end = _store(tmp_path, symbols, n, "1h")
    cfg = bt.BacktestConfig(
        universe=list(range(1, symbols + 1)), time_range_start=0, time_range_end=end,
        bar_interval=bt.Interval.hours(1), initial_capital=10_000.0,
        execution=bt.ExecutionConfig(allow_short=True),
    )
    f = bt.indicators.ema(bt.col("close"), 12)
    s = bt.indicators.ema(bt.col("close"), 60)
    strat = bt.Strategy.create("groupes").signal("f", f).signal("s", s).size(
        bt.when(f > s, 0.5, -0.5))
    return bt.run(strat, cfg, store).raw


def _figures(raw):
    return {
        name: getattr(B, name)(raw, show=False).to_json()
        for name in ("summary", "monthly_returns", "annual_returns", "drawdown", "equity")
    }


@pytest.mark.parametrize("symbols", [1, 2])
def test_each_chart_draws_the_figure_the_sort_drew(tmp_path, monkeypatch, symbols):
    raw = _result(tmp_path, symbols)
    if symbols == 2:
        # Two rows per timestamp: the positions are not in strict order, and
        # the summary takes the sorting road for them.
        ts = np.asarray(raw.positions.column("timestamp").cast("int64"))
        assert not np.all(ts[1:] > ts[:-1])
    new = _figures(raw)
    monkeypatch.setattr(B, "_unique_with_first_rows", ref_unique_with_first_rows)
    monkeypatch.setattr(B, "_unique_values", ref_unique_values)
    monkeypatch.setattr(B, "_groups", ref_groups)
    monkeypatch.setattr(B, "_calendar_groups", ref_calendar_groups)
    old = _figures(raw)
    for name in new:
        assert new[name] == old[name], name
    # The years and the months are really there to be grouped.
    assert '"2022"' in new["annual_returns"] and '"2023"' in new["annual_returns"]
