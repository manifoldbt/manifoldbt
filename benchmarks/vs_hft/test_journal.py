"""Checks on the pieces of the harness that need no engine and no data.

    python -m pytest benchmarks/vs_hft/test_journal.py -q

They need numpy and pandas, and nothing else: manifoldbt is replaced by a stub where
a module imports it.
"""
import io
import os
import re
import sys
import types
from contextlib import redirect_stdout

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import engine_mbt  # noqa: E402


class FakeResult:
    """The two tables of a run that `journal_of` reads."""

    def __init__(self, orders, fills):
        self._orders, self._fills = orders, fills

    def orders_df(self, backend="pandas"):
        return self._orders

    def fills_df(self, backend="pandas"):
        return self._fills


def _tables(sent, first_fill):
    """Orders sent at `sent` (ns since the epoch); order i has its first fill at first_fill[i], or none."""
    n = len(sent)
    orders = pd.DataFrame({
        "order_id": np.arange(100, 100 + n, dtype=np.int64),
        "sent_at": pd.to_datetime(np.asarray(sent, dtype=np.int64), unit="ns", utc=True),
        "side": ["buy" if i % 2 == 0 else "sell" for i in range(n)],
        "price": np.full(n, 100.0),
        "filled": [0.0 if t is None else 1.0 for t in first_fill],
    })
    rows = [(100 + i, t) for i, t in enumerate(first_fill) if t is not None]
    # two fills for the first filled order, so that the minimum is what has to be kept
    if rows:
        rows.append((rows[0][0], rows[0][1] + 5_000_000))
    fills = pd.DataFrame({
        "order_id": np.array([r[0] for r in rows], dtype=np.int64),
        "timestamp": pd.to_datetime(np.array([r[1] for r in rows], dtype=np.int64), unit="ns", utc=True),
    })
    return orders, fills


def test_first_fill_instant_is_exact_to_the_nanosecond():
    # a nanosecond epoch of today is ~1.76e18: a float64 holds it to the nearest 256 ns,
    # so any step that passes through float64 changes these instants
    base = 1_764_320_400_386_640_393
    first_fill = [base + 1, None, base + 255, base + 12_345_679, None, base + 3]
    sent = [base - 10_000_000] * len(first_fill)
    orders, fills = _tables(sent, first_fill)
    j = engine_mbt.journal_of(FakeResult(orders, fills), anchor_ns=base - 10_000_000, period_ns=100_000_000)
    assert j["fill_ts"].dtype == np.int64
    assert j["fill_ts"].tolist() == [0 if t is None else t for t in first_fill]
    assert j["filled"].tolist() == [0.0 if t is None else 1.0 for t in first_fill]


def test_journal_without_fills_is_all_zero():
    orders, fills = _tables([1_764_320_400_000_000_000] * 3, [None, None, None])
    j = engine_mbt.journal_of(FakeResult(orders, fills), anchor_ns=1_764_320_400_000_000_000, period_ns=100_000_000)
    assert j["fill_ts"].tolist() == [0, 0, 0]


def _ci_activate(monkeypatch, key, activate, ingest):
    stub = types.ModuleType("manifoldbt")
    stub.activate = activate
    stub.ingest_trades = stub.ingest_book = stub.ingest_mbo = ingest
    monkeypatch.setitem(sys.modules, "manifoldbt", stub)
    sys.modules.pop("ci_activate", None)
    if key is None:
        monkeypatch.delenv("MANIFOLDBT_CI_LICENSE", raising=False)
    else:
        monkeypatch.setenv("MANIFOLDBT_CI_LICENSE", key)
    import ci_activate

    out = io.StringIO()
    with redirect_stdout(out):
        rc = ci_activate.main()
    return rc, out.getvalue()


def test_ci_activate_needs_the_key_and_never_prints_it(monkeypatch):
    secret = "KEY-0123456789-not-to-be-printed"
    rc, out = _ci_activate(monkeypatch, None, lambda k: None, lambda *a, **kw: None)
    assert rc == 1 and "MANIFOLDBT_CI_LICENSE" in out

    def activate(k):
        raise ValueError("bad key " + k)

    rc, out = _ci_activate(monkeypatch, secret, activate, lambda *a, **kw: None)
    assert rc == 1 and secret not in out and "ValueError" in out

    rc, out = _ci_activate(monkeypatch, secret, lambda k: "activated for someone@example.com", lambda *a, **kw: None)
    assert rc == 0 and secret not in out and "someone@example.com" not in out


def test_ci_activate_checks_by_calling_and_tells_a_refusal_from_an_ordinary_failure(monkeypatch):
    def refused(*a, **kw):
        raise PermissionError("refused")

    def ordinary(*a, **kw):
        raise FileNotFoundError("no such data")

    rc, out = _ci_activate(monkeypatch, "k", lambda k: None, ordinary)
    assert rc == 0, out                       # got past the check, failed afterwards: expected

    rc, out = _ci_activate(monkeypatch, "k", lambda k: None, refused)
    assert rc == 1
    assert "licence check failed at: " in out
    assert not re.search(r"\b(plan|tier|pro|researcher|community|upgrade)\b", out.lower())
