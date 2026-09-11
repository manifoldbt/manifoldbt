"""``plot.chart`` reads the store the way the engine does.

Found by running example 06 on a store filled with ``bt.ingest``: the chart
only looked for the legacy ``bars_1m/{id}.arrow`` file, so every store built
through the shipped import path raised ``No bar data found``. These tests pin
the provider layout, a ticker with a slash, and a coarser stored resolution.
"""
import numpy as np
import pandas as pd
import pytest

import manifoldbt as bt

pytest.importorskip("pyarrow")

import importlib  # noqa: E402

# ``manifoldbt.plot.chart`` the module, not the ``chart`` function the package re-exports.
chart_mod = importlib.import_module("manifoldbt.plot.chart")


def _frame(n: int, step_s: int, start_s: int = 1_700_000_000) -> pd.DataFrame:
    ts = pd.to_datetime(start_s + np.arange(n) * step_s, unit="s", utc=True)
    close = 100.0 + np.sin(np.arange(n) / 7.0)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.full(n, 10.0),
    })


def _run(store, symbol_id: int, start_s: int, end_s: int, interval):
    strategy = bt.Strategy.create("chart_probe").signal("one", bt.lit(1.0)).size(bt.col("one"))
    config = bt.BacktestConfig(
        universe=[symbol_id],
        time_range_start=start_s * 1_000_000_000,
        time_range_end=end_s * 1_000_000_000,
        bar_interval=interval,
        initial_capital=1_000.0,
    )
    return bt.run(strategy, config, store)


def test_chart_reads_the_provider_layout_written_by_import(tmp_path):
    n, step = 400, 60
    store = bt.import_dataframe(
        _frame(n, step), symbol="BTCUSDT", symbol_id=1, interval="1m",
        exchange="binance",
        data_root=str(tmp_path / "data"), metadata_db=str(tmp_path / "meta.sqlite"),
    )
    start, end = 1_700_000_000, 1_700_000_000 + n * step
    result = _run(store, 1, start, end, {"Minutes": 1})

    found = chart_mod._find_bar_file(store, 1, 60)
    assert found is not None and found.parts[-3:] == ("binance", "1m", "BTCUSDT.arrow")

    bars = chart_mod._load_bars(store, 1, start * 10**9, end * 10**9, 60)
    assert len(bars["timestamp"]) == n
    bars, offset, bar_interval_s, trades = chart_mod._prepare_chart_data(result, store, 1, 120)
    assert bar_interval_s == 60 and offset == n - 120


def test_chart_reads_a_ticker_with_a_slash_and_a_coarser_resolution(tmp_path):
    n, step = 300, 3_600
    start = 1_700_006_400  # a multiple of 4 h, so the buckets below are exact
    store = bt.import_dataframe(
        _frame(n, step, start), symbol="EUR/USD", symbol_id=7, interval="1h",
        exchange="dukascopy",
        data_root=str(tmp_path / "data"), metadata_db=str(tmp_path / "meta.sqlite"),
    )
    end = start + n * step

    found = chart_mod._find_bar_file(store, 7, 3_600)
    assert found is not None and found.parts[-4:] == ("dukascopy", "1h", "EUR", "USD.arrow")

    # A 4 h chart resamples the stored hourly bars; a 1 m chart has nothing to read.
    bars = chart_mod._load_bars(store, 7, start * 10**9, end * 10**9, 4 * 3_600)
    assert len(bars["timestamp"]) == n // 4
    assert bars["high"][0] == pytest.approx(max(_frame(4, step, start)["high"]))
    assert chart_mod._find_bar_file(store, 7, 60) is None
    assert chart_mod._load_bars(store, 7, start * 10**9, end * 10**9, 60) == {}


def test_chart_still_reads_the_legacy_layout(tmp_path):
    import pyarrow as pa
    n, step = 200, 60
    start = 1_700_000_100  # a multiple of 5 min, so the resampled count is exact
    df = _frame(n, step, start)
    legacy = tmp_path / "data" / "mega" / "bars_1m"
    legacy.mkdir(parents=True)
    table = pa.table({
        # nanoseconds by construction: pandas may keep a second resolution here
        "timestamp": pa.array((start + np.arange(n) * step) * 10**9, type=pa.int64()),
        "open": df["open"].to_numpy(), "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(), "close": df["close"].to_numpy(),
        "volume": df["volume"].to_numpy(),
    })
    with pa.ipc.new_file(str(legacy / "3.arrow"), table.schema) as w:
        w.write_table(table)
    store = bt.DataStore(str(tmp_path / "data"), str(tmp_path / "meta.sqlite"))

    found = chart_mod._find_bar_file(store, 3, 60)
    assert found is not None and found.parts[-2:] == ("bars_1m", "3.arrow")
    end = start + n * step
    bars = chart_mod._load_bars(store, 3, start * 10**9, end * 10**9, 300)
    assert len(bars["timestamp"]) == n // 5
