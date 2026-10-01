"""Deterministic synthetic OHLCV bars, identical for every engine.

One generator, one seed, one fingerprint. Both engines receive the *same*
DataFrame object; nothing about the data can differ between them.

Two properties are deliberate, not incidental:

* ``open == previous close`` (no overnight gap). A bar that gaps through a stop
  is the one place where two engines can legitimately disagree on the fill price
  while both being correct. Removing gaps removes that whole class of false
  parity failures, so a real semantic drift is the only thing left that can trip
  the gate.
* the intrabar range is wide enough that percentage stops and targets actually
  trigger, otherwise the bracket workload would measure an empty branch.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

DEFAULT_SEED = 20260816


def make_ohlcv(
    rows: int,
    *,
    seed: int = DEFAULT_SEED,
    freq: str = "1min",
    start: str = "2020-01-01",
    vol: float = 3e-4,
    drift: float = 2e-7,
) -> pd.DataFrame:
    """A gap-free random walk of ``rows`` bars, reproducible from ``seed``."""
    rng = np.random.default_rng(seed)

    log_ret = rng.normal(drift, vol, size=rows)
    close = 100.0 * np.exp(np.cumsum(log_ret))

    open_ = np.empty(rows, dtype=np.float64)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Intrabar excursion beyond the open/close body, as a fraction of price.
    wick = rng.uniform(0.2, 1.8, size=rows) * vol * close
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi + wick
    low = np.maximum(body_lo - wick, 1e-8)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=rows, freq=freq, tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100.0, 10_000.0, size=rows),
        }
    )


def make_universe(rows: int, count: int, *, seed: int = DEFAULT_SEED) -> dict:
    """`count` independent series, keyed by the symbol id each engine will use.

    Independent, not correlated: a portfolio of copies of one asset would let a
    position-sizing bug cancel itself out across the book, which is exactly the
    class of mistake a multi-asset workload exists to catch. The seeds are
    spaced far apart and derived from the same base, so the whole universe is
    reproducible from `seed` alone and the first symbol is bit-identical to the
    single-asset series of the same length.
    """
    return {i + 1: make_ohlcv(rows, seed=seed + 1000 * i) for i in range(count)}


def digest(df: pd.DataFrame) -> str:
    """Short content fingerprint of the bars, recorded in the result envelope.

    Anyone re-running the harness can compare this before comparing timings: a
    different digest means a different dataset, which makes the numbers
    incomparable no matter how clean the machine was.
    """
    h = hashlib.sha256()
    for column in ("open", "high", "low", "close", "volume"):
        h.update(np.ascontiguousarray(df[column].to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]
