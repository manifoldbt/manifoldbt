"""Tests for bt.import_dataframe — in-memory DataFrame → Arrow IPC store.

The contract under test: import_dataframe is the in-memory twin of
import_csv. Same data through either path must produce an identical store
(same backtest results), and the normalisation layer must give clear errors
for bad inputs instead of a Rust panic.
"""
import os

import pytest

import manifoldbt as bt

pd = pytest.importorskip("pandas")

N_BARS = 120
START_MS = 1_577_836_800_000  # 2020-01-01T00:00:00Z


def _bars_df(n=N_BARS, tz="UTC"):
    """Synthetic 1m bars as a pandas DataFrame."""
    ts = pd.date_range("2020-01-01", periods=n, freq="1min", tz=tz)
    close = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0] * n,
        }
    )


def _store_paths(tmp_path, name):
    root = tmp_path / name
    return str(root / "data"), str(root / "metadata.sqlite")


def _import_df(df, tmp_path, name="df", **kw):
    data_root, metadata_db = _store_paths(tmp_path, name)
    os.makedirs(os.path.dirname(metadata_db), exist_ok=True)
    return bt.import_dataframe(
        df, symbol="BTCUSDT", symbol_id=1,
        data_root=data_root, metadata_db=metadata_db, **kw
    )


def _run_buy_and_hold(store):
    strategy = bt.Strategy(
        name="bh",
        signals={"signal": bt.lit(1.0)},
        position_sizing=bt.col("signal"),
    )
    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=START_MS * 1_000_000 + N_BARS * 60_000_000_000,
        bar_interval={"Minutes": 1},
        initial_capital=1000.0,
        currency="USD",
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            allow_fractional=True,
            skip_gap_bars=False,
            position_sizing_mode="Units",
        ),
        fees=bt.FeeConfig(),
        slippage={"FixedBps": {"bps": 0.0}},
        rng_seed=7,
    )
    return bt.run(strategy, config, store)


def test_import_dataframe_roundtrip(tmp_path):
    """DataFrame → store → run produces a usable backtest."""
    store = _import_df(_bars_df(), tmp_path)
    assert store.resolve_symbol("BTCUSDT") == 1

    result = _run_buy_and_hold(store)
    equity = result.equity_curve.to_pylist()
    assert len(equity) > 0
    # Price rises monotonically → buy & hold ends above initial capital.
    assert equity[-1] > 1000.0


def test_import_dataframe_matches_import_csv(tmp_path):
    """Same bars through import_csv and import_dataframe → identical results."""
    df = _bars_df()

    # CSV path (standard format: epoch-ms timestamp).
    #
    # Built from START_MS rather than derived from the datetime column:
    # `.astype("int64")` returns the underlying integer in the COLUMN's
    # resolution, which pandas picks for itself. Locally that was ns (so
    # //1e6 gave ms), on CI it was us (so //1e6 gave seconds) and the import
    # rejected the row. The bars are 1 minute apart by construction here, so
    # spelling the epoch out keeps the CSV identical on every pandas.
    csv_df = df.copy()
    csv_df["timestamp"] = [START_MS + i * 60_000 for i in range(len(csv_df))]
    csv_path = tmp_path / "bars.csv"
    csv_df.to_csv(csv_path, index=False)
    csv_root, csv_meta = _store_paths(tmp_path, "csv")
    os.makedirs(os.path.dirname(csv_meta), exist_ok=True)
    store_csv = bt.import_csv(
        str(csv_path), symbol="BTCUSDT", symbol_id=1,
        data_root=csv_root, metadata_db=csv_meta,
    )

    store_df = _import_df(df, tmp_path)

    res_csv = _run_buy_and_hold(store_csv)
    res_df = _run_buy_and_hold(store_df)
    assert res_df.equity_curve.to_pylist() == res_csv.equity_curve.to_pylist()
    assert res_df.metrics == res_csv.metrics


def test_import_dataframe_naive_timestamps_assumed_utc(tmp_path):
    """tz-naive datetimes are accepted and treated as UTC."""
    naive = _bars_df(tz=None)
    aware = _bars_df(tz="UTC")
    store_naive = _import_df(naive, tmp_path, name="naive")
    store_aware = _import_df(aware, tmp_path, name="aware")
    assert _run_buy_and_hold(store_naive).equity_curve.to_pylist() == \
        _run_buy_and_hold(store_aware).equity_curve.to_pylist()


def test_import_dataframe_datetime_index_promoted(tmp_path):
    """A pandas DatetimeIndex is used as the timestamp column."""
    df = _bars_df().set_index("timestamp")
    assert "timestamp" not in df.columns
    store = _import_df(df, tmp_path)
    assert store.resolve_symbol("BTCUSDT") == 1


def test_import_dataframe_polars(tmp_path):
    """Polars DataFrames go through the zero-copy to_arrow path."""
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(_bars_df())
    store = _import_df(df, tmp_path)
    assert store.resolve_symbol("BTCUSDT") == 1


def test_import_dataframe_missing_column_raises(tmp_path):
    df = _bars_df().drop(columns=["volume"])
    with pytest.raises(bt.DataError, match="volume"):
        _import_df(df, tmp_path)


def test_import_dataframe_integer_timestamp_raises(tmp_path):
    """Epoch integers are ambiguous (ms? ns?) — require datetimes."""
    df = _bars_df()
    df["timestamp"] = df["timestamp"].astype("int64")
    with pytest.raises(bt.DataError, match="datetime"):
        _import_df(df, tmp_path)


def test_import_dataframe_empty_raises(tmp_path):
    with pytest.raises(bt.DataError, match="no data rows"):
        _import_df(_bars_df(0), tmp_path)


def test_import_dataframe_daily_interval_runs(tmp_path):
    """Daily bars import AND backtest.

    Regression: the resolution table listed only 1m/1h, so a daily store
    resolved to the (empty) 1m directory and the run died with "empty bar
    dataset for symbol". A ``1d`` entry in the table lets the daily provider
    layout be found. 1m/1h were unaffected, which is exactly why this slipped.
    """
    n = 30
    ts = pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC")
    close = [100.0 + i for i in range(n)]  # strictly rising → buy & hold profits
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0] * n,
        }
    )
    store = _import_df(df, tmp_path, name="daily", interval="1d")
    assert store.resolve_symbol("BTCUSDT") == 1

    strategy = bt.Strategy(
        name="bh",
        signals={"signal": bt.lit(1.0)},
        position_sizing=bt.col("signal"),
    )
    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=int(ts[-1].value) + 5 * 86_400_000_000_000,
        bar_interval={"Days": 1},
        initial_capital=1000.0,
        execution=bt.ExecutionConfig(
            signal_delay=1, execution_price="AtClose",
            position_sizing_mode="Units",
        ),
        fees=bt.FeeConfig(),
        slippage={"FixedBps": {"bps": 0.0}},
    )
    result = bt.run(strategy, config, store)
    equity = result.equity_curve.to_pylist()
    assert len(equity) > 0
    assert equity[-1] > 1000.0


def _read_bars_back(tmp_path, name="df"):
    import glob

    import pyarrow.ipc as ipc

    data_root, _ = _store_paths(tmp_path, name)
    f = glob.glob(os.path.join(data_root, "**", "*.arrow"), recursive=True)[0]
    return ipc.open_file(f).read_all().to_pandas()


def test_import_dataframe_keeps_provided_quotes(tmp_path):
    """bid/ask a caller provides survive the import; spread is derived.

    The store schema has always carried bid/ask/spread and the engine reads
    them (MidPrice, spread_based slippage), but the import path dropped them
    silently: a user reported 0 non-null values out of 400 provided.
    """
    import numpy as np

    df = _bars_df()
    df["bid"] = df["close"] - 0.05
    df["ask"] = df["close"] + 0.05
    df.loc[10, "bid"] = np.nan  # a missing quote stays missing, not 0 or NaN-poison

    _import_df(df, tmp_path)
    t = _read_bars_back(tmp_path)

    assert t["bid"].notna().sum() == N_BARS - 1
    assert t["ask"].notna().sum() == N_BARS
    assert t["spread"].notna().sum() == N_BARS - 1
    ok = t["bid"].notna()
    assert np.allclose(t.loc[ok, "spread"].to_numpy(), 0.1)
    assert np.allclose(t.loc[ok, "bid"].to_numpy(), df.loc[ok.to_numpy(), "bid"].to_numpy())
    # Columns nobody provided stay null, exactly as before.
    assert t["buy_volume"].isna().all() and t["sell_volume"].isna().all()


def test_import_dataframe_without_quotes_is_unchanged(tmp_path):
    _import_df(_bars_df(), tmp_path)
    t = _read_bars_back(tmp_path)
    for col in ("bid", "ask", "spread", "buy_volume", "sell_volume"):
        assert t[col].isna().all(), col
