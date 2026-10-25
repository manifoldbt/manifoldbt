"""cfg.liquidation from Python: the config reaches the engine, symbol names
resolve, and a leveraged position is closed where the venue would close it."""
import pytest

import manifoldbt as bt
from manifoldbt.expr import lit
from manifoldbt.helpers import Interval, Slippage

pd = pytest.importorskip("pandas")

LIQUIDATION = 7


def _frame():
    # Flat at 100, then one bar wicks to 60 and closes at 95.
    n = 48
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    o = [100.0] * n
    h = [100.0] * n
    lo = [100.0] * n
    c = [100.0] * n
    o[30], h[30], lo[30], c[30] = 98.0, 99.0, 60.0, 95.0
    for i in range(31, n):
        o[i] = h[i] = lo[i] = c[i] = 95.0
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": lo,
                         "close": c, "volume": [1e9] * n})


@pytest.fixture
def store(tmp_path):
    return bt.import_dataframe(_frame(), symbol="AAA", symbol_id=1, interval="1h",
                               data_root=str(tmp_path / "data"),
                               metadata_db=str(tmp_path / "meta.sqlite"))


def _cfg(liquidation=None):
    cfg = bt.BacktestConfig(
        universe=["AAA"], time_range_start=0, time_range_end=4_102_444_800_000_000_000,
        bar_interval=Interval.hours(1), initial_capital=1000.0,
        execution=bt.ExecutionConfig(execution_price="AtClose", max_position_pct=3.0),
        slippage=Slippage.fixed_bps(0.0),
    )
    cfg.liquidation = liquidation
    return cfg


def _hold_3x():
    return bt.Strategy.create("hold3x").signal("w", lit(3.0)).size(lit(3.0))


def _reasons(result):
    return list(result.trades_df(backend="pandas")["exit_reason"])


def test_a_3x_long_is_liquidated_on_the_wick(store):
    result = bt.run(_hold_3x(), _cfg(bt.Liquidation(maintenance_margin_rate=0.005)), store)
    trades = result.trades_df(backend="pandas")
    liq = trades[trades["exit_reason"] == LIQUIDATION]
    assert len(liq) == 1
    entry = trades.iloc[0]
    # The engine's liquidation price for the position it actually opened.
    cash = 1000.0 - entry["quantity"] * entry["fill_price"] - entry["fees"]
    expected = -cash / (entry["quantity"] * (1.0 - 0.005))
    assert liq["fill_price"].iloc[0] == pytest.approx(expected, rel=1e-9)
    assert any("liquidated" in w for w in result.warnings)


def test_without_liquidation_the_wick_is_survived(store):
    result = bt.run(_hold_3x(), _cfg(None), store)
    assert LIQUIDATION not in _reasons(result)


def test_per_symbol_rates_accept_symbol_names(store):
    # 30 % maintenance margin on AAA alone: liquidated far above the 0.5 % price.
    liq = bt.Liquidation(maintenance_margin_rate=0.005, per_symbol={"AAA": 0.3})
    result = bt.run(_hold_3x(), _cfg(liq), store)
    trades = result.trades_df(backend="pandas")
    price = trades[trades["exit_reason"] == LIQUIDATION]["fill_price"].iloc[0]
    assert price > 90.0


def test_the_lite_sweep_agrees_with_the_run(store):
    cfg = _cfg(bt.Liquidation(maintenance_margin_rate=0.005, fee_bps=20))
    full = bt.run(_hold_3x(), cfg, store)
    strat = bt.Strategy.create("sw").signal("w", lit(3.0)).size(lit(3.0) * bt.param("k"))
    lite = bt.run_sweep_lite(strat, {"k": [1.0]}, cfg, store, device="cpu")
    assert lite[0].final_equity == pytest.approx(full.equity_df(backend="pandas").iloc[-1, -1], rel=1e-12)


def test_an_impossible_rate_is_refused(store):
    with pytest.raises(Exception, match="maintenance_margin_rate"):
        bt.run(_hold_3x(), _cfg(bt.Liquidation(maintenance_margin_rate=1.5)), store)


def test_liquidation_is_labelled_in_round_trips():
    from manifoldbt._trades import EXIT_REASON_LABELS
    assert EXIT_REASON_LABELS[LIQUIDATION] == "liquidation"
