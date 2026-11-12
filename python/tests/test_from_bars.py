"""Tests for mbt.from_bars: bars ready for a run, without a file.

The contract under test: same bars, same run as a store on disk, whichever way
they arrive (frame, Arrow, file), and nothing written anywhere. What a store in
memory cannot do is refused by name, never answered from the working directory.
"""
import os

import pytest

import manifoldbt as bt
from manifoldbt.helpers import Interval, time_range

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")
pytest.importorskip("pyarrow.feather")

N = 2_000
BIG = 100_000
# Traverser alloue quelques Ko de metadonnees, pas une colonne de valeurs.
ONE_COLUMN = BIG * 8


def _bars(n=N, start="2024-01-01", slope=0.01, freq="1min"):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    # Une sinusoide : les signaux changent, donc les fills et la courbe aussi.
    close = 100.0 + np.sin(np.arange(n) / 25.0) * 5.0 + np.arange(n) * slope
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.roll(close, 1),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 10.0),
        }
    )


def _config(universe, bar_interval="1m"):
    start, end = time_range("2024-01-01", "2024-02-01")
    minutes = {"1m": Interval.minutes(1), "5m": Interval.minutes(5)}[bar_interval]
    return bt.BacktestConfig(
        universe=list(universe),
        time_range_start=start,
        time_range_end=end,
        bar_interval=minutes,
        initial_capital=100_000.0,
    )


def _run(store, universe=("SYN",), bar_interval="1m"):
    fast, slow = bt.col("close").rolling_mean(5), bt.col("close").rolling_mean(20)
    strategy = bt.Strategy(
        name="croisement",
        signals={"signal": bt.when(fast > slow, bt.lit(1.0), bt.lit(0.0))},
        position_sizing=bt.col("signal"),
    )
    return bt.run(strategy, _config(universe, bar_interval), store)


def _same_run(a, b):
    ea, eb = a.equity_df(), b.equity_df()
    assert len(ea) == len(eb) and len(ea) > 0
    assert (ea["equity"].to_numpy() == eb["equity"].to_numpy()).all()
    ta, tb = a.trades_df(), b.trades_df()
    assert len(ta) > 0, "sans fill, la comparaison ne prouverait rien"
    pd.testing.assert_frame_equal(ta, tb, check_exact=True)


def _disk(df, tmp_path, name="disk", **kw):
    root = tmp_path / name
    root.mkdir()
    return bt.import_dataframe(
        df, symbol="SYN", symbol_id=1,
        data_root=str(root / "data"), metadata_db=str(root / "m.sqlite"), **kw
    )


def test_same_run_as_a_store_on_disk(tmp_path):
    df = _bars()
    _same_run(_run(bt.from_bars(df, "SYN")), _run(_disk(df, tmp_path)))


def test_same_run_as_a_store_on_disk_when_resampled(tmp_path):
    df = _bars()
    _same_run(
        _run(bt.from_bars(df, "SYN"), bar_interval="5m"),
        _run(_disk(df, tmp_path), bar_interval="5m"),
    )


def test_same_run_with_quotes_kept(tmp_path):
    df = _bars()
    df["bid"], df["ask"] = df["close"] - 0.1, df["close"] + 0.1
    _same_run(_run(bt.from_bars(df, "SYN")), _run(_disk(df, tmp_path)))


def test_every_source_gives_the_same_run(tmp_path):
    df = _bars()
    table = pa.Table.from_pandas(df, preserve_index=False)
    reference = _run(bt.from_bars(df, "SYN"))

    arrow_file = tmp_path / "bars.arrow"
    with pa.OSFile(str(arrow_file), "wb") as sink, pa.ipc.new_file(sink, table.schema) as w:
        w.write_table(table)
    stream_file = tmp_path / "bars.arrows"
    with pa.OSFile(str(stream_file), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    feather_file = tmp_path / "bars.feather"
    pa.feather.write_feather(table, str(feather_file), compression="lz4")
    parquet_file = tmp_path / "bars.parquet"
    pa.parquet.write_table(table, str(parquet_file))

    sources = [
        table,
        table.to_batches()[0],
        {c: df[c].to_numpy() for c in df.columns},
        str(arrow_file),
        arrow_file,
        str(stream_file),
        str(feather_file),
        str(parquet_file),
    ]
    polars = pytest.importorskip("polars")
    sources.append(polars.from_pandas(df))
    for source in sources:
        _same_run(_run(bt.from_bars(source, "SYN")), reference)


def test_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = bt.from_bars(_bars(), "SYN")
    _run(store)
    assert os.listdir(tmp_path) == []


def test_arrow_columns_are_read_in_place():
    close = pa.array(np.linspace(100.0, 200.0, BIG))
    df = _bars(BIG)
    table = pa.table(
        {
            "timestamp": pa.array(df["timestamp"], type=pa.timestamp("ns", tz="UTC")),
            "open": close, "high": close, "low": close, "close": close,
            "volume": close,
        }
    )
    before = pa.total_allocated_bytes()
    store = bt.from_bars(table, "SYN")
    assert pa.total_allocated_bytes() - before < ONE_COLUMN, "une colonne a ete copiee"
    assert store.list_symbols() == [(1, "SYN")]


def test_an_arrow_file_is_read_in_place(tmp_path):
    df = _bars(BIG)
    table = pa.Table.from_pandas(df, preserve_index=False).set_column(
        0, pa.field("timestamp", pa.timestamp("ns", tz="UTC")),
        pa.array(df["timestamp"], type=pa.timestamp("ns", tz="UTC")),
    )
    path = tmp_path / "bars.arrow"
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as w:
        w.write_table(table)
    del table
    before = pa.total_allocated_bytes()
    store = bt.from_bars(path, "SYN")
    assert pa.total_allocated_bytes() - before < ONE_COLUMN, "une colonne a ete copiee"
    assert store.list_symbols() == [(1, "SYN")]


def test_store_answers_like_a_store():
    store = bt.from_bars(_bars(), "syn", asset_class="equity")
    assert store.dataset() == "memory"
    assert store.resolve_symbol("SYN") == 1
    assert store.resolve_symbol("syn") == 1
    assert store.list_symbols() == [(1, "SYN")]
    assert store.asset_classes() == {1: "Equity"}
    assert store.option_contracts() == {}
    assert store.active_version("bars_1m") == bt.from_bars(
        _bars(), "syn", asset_class="equity"
    ).active_version("bars_1m")
    with pytest.raises(ValueError, match="symbol"):
        store.resolve_symbol("NOPE")


def test_two_symbols_and_ids_in_the_order_given(tmp_path):
    btc, eth = _bars(), _bars(slope=-0.01)
    store = bt.from_bars({"BTCUSDT": btc, "ETHUSDT": eth})
    assert store.list_symbols() == [(1, "BTCUSDT"), (2, "ETHUSDT")]

    root = tmp_path / "disk"
    root.mkdir()
    keys = {"data_root": str(root / "data"), "metadata_db": str(root / "m.sqlite")}
    bt.import_dataframe(btc, symbol="BTCUSDT", symbol_id=1, **keys)
    disk = bt.import_dataframe(eth, symbol="ETHUSDT", symbol_id=2, **keys)

    universe = ("BTCUSDT", "ETHUSDT")
    _same_run(_run(store, universe), _run(disk, universe))


def test_same_config_on_two_stores_is_not_mixed_up():
    """La configuration preparee est memorisee : deux stores ne la partagent pas."""
    up, down = _bars(slope=0.05), _bars(slope=-0.05)
    a = _run(bt.from_bars({"AAA": up, "SYN": down}), ("SYN",))
    b = _run(bt.from_bars({"SYN": down}), ("SYN",))
    # SYN porte l'id 2 dans le premier store et 1 dans le second : c'est la preuve
    # que chacun a sa configuration, et la seule colonne qui doit differer.
    assert set(a.trades_df()["symbol_id"]) == {2}
    assert set(b.trades_df()["symbol_id"]) == {1}
    pd.testing.assert_frame_equal(
        a.trades_df().drop(columns="symbol_id"),
        b.trades_df().drop(columns="symbol_id"),
        check_exact=True,
    )


def test_symbol_ref_reads_the_other_symbol(tmp_path):
    other = _bars(slope=0.05)
    df = _bars()
    strategy = bt.Strategy(
        name="ref",
        signals={
            "signal": bt.when(
                bt.symbol_ref("OTHER", "close") > bt.lit(120.0), bt.lit(1.0), bt.lit(0.0)
            )
        },
        position_sizing=bt.col("signal"),
    )
    config = _config(["SYN"])
    root = tmp_path / "disk"
    root.mkdir()
    keys = {"data_root": str(root / "data"), "metadata_db": str(root / "m.sqlite")}
    bt.import_dataframe(df, symbol="SYN", symbol_id=1, **keys)
    disk = bt.import_dataframe(other, symbol="OTHER", symbol_id=2, **keys)
    _same_run(
        bt.run(strategy, config, bt.from_bars({"SYN": df, "OTHER": other})),
        bt.run(strategy, config, disk),
    )


def test_a_sweep_runs_on_it(tmp_path):
    df = _bars()
    window = bt.param("window")
    strategy = bt.Strategy(
        name="balayage",
        signals={"signal": bt.when(bt.col("close") > bt.col("close").rolling_mean(window), bt.lit(1.0), bt.lit(0.0))},
        position_sizing=bt.col("signal"),
    )
    config = _config(["SYN"])
    grid = {"window": [5, 10, 20]}
    memory = bt.run_sweep(strategy, grid, config, bt.from_bars(df, "SYN"))
    disk = bt.run_sweep(strategy, grid, config, _disk(df, tmp_path))
    assert memory.to_df().equals(disk.to_df())


@pytest.mark.parametrize(
    "ts",
    [
        ["2024-01-01 00:02", "2024-01-01 00:01"],
        ["2024-01-01 00:01", "2024-01-01 00:01"],
    ],
)
def test_rows_out_of_order_are_refused_by_row(ts):
    n = len(ts)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, utc=True),
            "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
            "close": [1.0] * n, "volume": [1.0] * n,
        }
    )
    with pytest.raises(bt.DataError, match="row 1 is not after row 0"):
        bt.from_bars(df, "SYN")


def test_bad_inputs_are_refused_by_name(tmp_path):
    df = _bars()
    with pytest.raises(TypeError, match="needs a symbol"):
        bt.from_bars(df)
    with pytest.raises(TypeError, match="holds columns"):
        bt.from_bars({c: df[c].to_numpy() for c in df.columns})
    with pytest.raises(bt.DataError, match="no such file"):
        bt.from_bars(str(tmp_path / "absent.arrow"), "SYN")
    other = tmp_path / "bars.csv"
    other.write_text("x")
    with pytest.raises(bt.DataError, match="cannot read"):
        bt.from_bars(str(other), "SYN")
    with pytest.raises(bt.DataError, match="missing required"):
        bt.from_bars(df.drop(columns=["close"]), "SYN")
    with pytest.raises(bt.DataError, match="no data rows"):
        bt.from_bars(df.iloc[:0], "SYN")
    with pytest.raises(bt.DataError, match="twice"):
        bt.from_bars({"AAA": df, "aaa": df})
    with pytest.raises(Exception, match="asset class"):
        bt.from_bars(df, "SYN", asset_class="nope")
    with pytest.raises(Exception, match="(?i)interval"):
        bt.from_bars(df, "SYN", interval="nope")


def test_what_needs_a_file_says_so_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = bt.from_bars(_bars(), "SYN")
    for call in (store.data_root, store.metadata_db):
        with pytest.raises(ValueError, match="lives in memory"):
            call()
    with pytest.raises(ValueError, match="lives in memory"):
        bt.register_exo(
            "x", pd.DataFrame({"timestamp": _bars()["timestamp"], "v": 1.0}), store=store
        )
    with pytest.raises(ValueError, match="lives in memory"):
        bt.attach_quotes(store, "SYN", "2024-01-01", "2024-01-01")
    with pytest.raises(bt.DataError, match="lives in memory"):
        bt.bars_from_trades(store, "SYN", "2024-01-01", "2024-01-02", interval="1m")
    assert os.listdir(tmp_path) == []


def test_a_cloud_job_is_refused_rather_than_run_on_another_symbol():
    from manifoldbt import cloud

    strategy = bt.Strategy(
        name="s", signals={"signal": bt.lit(1.0)}, position_sizing=bt.col("signal")
    )
    with pytest.raises(ValueError, match="lives in memory"):
        cloud._payload(strategy, _config(["SYN"]), bt.from_bars(_bars(), "SYN"))


def test_import_dataframe_takes_arrow_too(tmp_path):
    df = _bars()
    table = pa.Table.from_pandas(df, preserve_index=False)
    _same_run(_run(_disk(table, tmp_path, "a")), _run(_disk(df, tmp_path, "b")))
