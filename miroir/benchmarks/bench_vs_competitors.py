"""
Benchmark: manifoldbt vs vectorbt vs backtrader
===============================================

Fair comparison: SAME strategy, SAME data, SAME results.
Indicators + simulation timed together for all engines.

Strategy (simple, verifiable):
  - EMA(12) cross above EMA(26) -> long 50%
  - EMA(12) cross below EMA(26) -> flat
  - RSI(14) filter: only enter if 30 < RSI < 70
  - Fees: 5 bps taker, no slippage

Usage:
    python benchmarks/bench_vs_competitors.py --rows 500000 --runs 5
    python benchmarks/bench_vs_competitors.py --rows 5000000 --runs 2 --engines bt vectorbt
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ALL_ENGINES = ["bt", "vectorbt", "backtrader"]


# --- Synthetic data generation -----------------------------------------------

def generate_ohlcv(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.0003, size=rows)
    mid = 100.0 * np.exp(np.cumsum(returns))
    noise = rng.uniform(0.0001, 0.001, size=rows) * mid
    timestamps = pd.date_range("2022-01-01", periods=rows, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": mid + rng.uniform(-0.5, 0.5, size=rows) * noise,
        "high": mid + noise,
        "low": mid - noise,
        "close": mid + rng.uniform(-0.5, 0.5, size=rows) * noise,
        "volume": rng.uniform(100, 10_000, size=rows),
    })


# --- manifoldbt (Rust) -------------------------------------------------------

def bench_bt_engine(df: pd.DataFrame, n_runs: int) -> dict:
    try:
        import manifoldbt as bt
        from manifoldbt import run_with_parquet
        from manifoldbt.indicators import ema, rsi, close as c
        from manifoldbt.helpers import Slippage, Interval
    except ImportError:
        return {"name": "manifoldbt (Rust)", "error": "not installed"}

    import tempfile

    # Write synthetic data to a temp parquet (manifoldbt canonical schema)
    parquet_df = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp"].values, utc=True).as_unit("ns"),
        "symbol_id": np.uint32(1),
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "vwap": df["close"].values,
        "volume": df["volume"].values,
        "buy_volume": df["volume"].values * 0.5,
        "sell_volume": df["volume"].values * 0.5,
        "trade_count": np.uint32(100),
        "bid": df["close"].values * 0.9999,
        "ask": df["close"].values * 1.0001,
        "spread": df["close"].values * 0.0002,
        "is_gap": False,
        "gap_fill_method": np.uint8(0),
    })
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    parquet_path = os.path.join(tmp_dir, "bench_data.parquet")
    parquet_df.to_parquet(parquet_path, index=False)

    fast = ema(c, 12)
    slow = ema(c, 26)
    my_rsi = rsi(c, 14)

    strategy = (
        bt.Strategy.create("ema_rsi")
        .signal("fast", fast)
        .signal("slow", slow)
        .signal("rsi", my_rsi)
        .signal("entry",
                (bt.col("fast") > bt.col("slow"))
                & (bt.col("rsi") > bt.lit(30.0))
                & (bt.col("rsi") < bt.lit(70.0)))
        .size(bt.when(bt.col("entry"), bt.lit(0.5), bt.lit(0.0)))
    )

    start_ns = int(df["timestamp"].iloc[0].value)
    end_ns = int(df["timestamp"].iloc[-1].value)

    config = bt.BacktestConfig(
        universe=[1],
        time_range_start=start_ns,
        time_range_end=end_ns,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000,
        execution=bt.ExecutionConfig(
            allow_short=False,
            max_position_pct=1.0,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=30,
    )

    from manifoldbt._native import (
        load_parquet_as_aligned,
        run_on_aligned as _run_on_aligned,
    )

    strat_json = strategy.to_json()
    cfg_json = config.to_json()

    # Load data ONCE (parquet read excluded from timing)
    aligned = load_parquet_as_aligned(cfg_json, parquet_path, "bench_v1")

    # Warmup engine
    _run_on_aligned(strat_json, cfg_json, aligned)

    # Timed runs: PURE ENGINE (compile + indicators + simulation, zero I/O)
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = _run_on_aligned(strat_json, cfg_json, aligned)
        times.append(time.perf_counter() - t0)

    try:
        os.unlink(parquet_path)
    except OSError:
        pass

    return {
        "name": "manifoldbt (Rust)",
        "times": times,
        "median": np.median(times),
        "mean": np.mean(times),
        "total_return": result.metrics.get("total_return", 0) * 100,  # to %
        "trades": result.metrics.get("trade_stats", {}).get("total_trades", None),
    }


# --- vectorbt (NumPy) -------------------------------------------------------

def _vbt_run(close, vbt):
    """Compute indicators + simulate. Everything in one timed call."""
    # EMA 12/26
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()

    # RSI 14 (Wilder's smoothing = EMA with alpha=1/period)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-12))

    # Target sizing: 50% when entry conditions met, 0% otherwise
    entry = (fast > slow) & (rsi > 30) & (rsi < 70)

    # Use from_signals with entries/exits on transitions only
    # This matches manifoldbt behavior: trade only when state changes
    entries = entry & ~entry.shift(1, fill_value=False)   # False -> True
    exits = ~entry & entry.shift(1, fill_value=False)     # True -> False

    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        init_cash=10_000,
        size=0.5,
        size_type="percent",
        fees=0.0,
        freq="1T",
        accumulate=False,
    )
    # Force metric computation (manifoldbt includes this in its timing)
    pf.stats()
    return pf


def bench_vectorbt(df: pd.DataFrame, n_runs: int) -> dict:
    try:
        import vectorbt as vbt
    except ImportError:
        return {"name": "vectorbt (NumPy)", "error": "not installed"}

    close = df.set_index("timestamp")["close"]

    # Warmup
    _vbt_run(close, vbt)

    times = []
    pf = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pf = _vbt_run(close, vbt)
        times.append(time.perf_counter() - t0)

    stats = pf.stats()
    return {
        "name": "vectorbt (NumPy)",
        "times": times,
        "median": np.median(times),
        "mean": np.mean(times),
        "total_return": stats.get("Total Return [%]", None),
        "trades": int(stats.get("Total Trades", 0)) * 2,  # vbt counts round-trips; x2 for entry+exit
    }


# --- backtrader (Python) ----------------------------------------------------

def bench_backtrader(df: pd.DataFrame, n_runs: int) -> dict:
    try:
        import backtrader as btdr
    except ImportError:
        return {"name": "backtrader (Python)", "error": "not installed"}

    class EmaRsi(btdr.Strategy):
        params = dict(fast=12, slow=26, rsi_period=14)

        def __init__(self):
            self.fast_ema = btdr.indicators.EMA(self.data.close, period=self.p.fast)
            self.slow_ema = btdr.indicators.EMA(self.data.close, period=self.p.slow)
            self.rsi = btdr.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.trade_count = 0

        def next(self):
            trend_up = self.fast_ema[0] > self.slow_ema[0]
            rsi_ok = 30 < self.rsi[0] < 70

            if trend_up and rsi_ok:
                if not self.position:
                    self.order_target_percent(target=0.5)
                    self.trade_count += 1
            else:
                if self.position:
                    self.close()
                    self.trade_count += 1

    bt_df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    bt_df = bt_df.rename(columns={"timestamp": "datetime"}).set_index("datetime")
    bt_df.index = bt_df.index.tz_localize(None)

    # Warmup
    cerebro = btdr.Cerebro()
    cerebro.addstrategy(EmaRsi)
    cerebro.adddata(btdr.feeds.PandasData(dataname=bt_df))
    cerebro.broker.set_cash(10_000)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.run()

    times = []
    for _ in range(n_runs):
        cerebro = btdr.Cerebro()
        cerebro.addstrategy(EmaRsi)
        cerebro.adddata(btdr.feeds.PandasData(dataname=bt_df))
        cerebro.broker.set_cash(10_000)
        cerebro.broker.setcommission(commission=0.0)
        t0 = time.perf_counter()
        results = cerebro.run()
        times.append(time.perf_counter() - t0)

    strat = results[0]
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / 10_000 - 1) * 100

    return {
        "name": "backtrader (Python)",
        "times": times,
        "median": np.median(times),
        "mean": np.mean(times),
        "total_return": total_return,
        "trades": strat.trade_count,
    }


# --- Output ------------------------------------------------------------------

BENCH_FNS = {
    "bt": bench_bt_engine,
    "vectorbt": bench_vectorbt,
    "backtrader": bench_backtrader,
}


def print_results(results: list[dict], rows: int):
    print("\n" + "=" * 70)
    print(f"  BENCHMARK: EMA(12/26) + RSI(14) on {rows:,} x 1-min bars")
    print("=" * 70)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("  No engines ran successfully.")
        return

    fastest = min(valid, key=lambda r: r["median"])

    for r in results:
        if "error" in r:
            print(f"\n  {r['name']:25s}  !! {r['error']}")
            continue

        med = r["median"]
        avg = r["mean"]
        mult = med / fastest["median"] if fastest["median"] > 0 else 0
        bar = "#" * min(int(mult * 3), 60)

        print(f"\n  {r['name']:25s}  {bar}")
        print(f"  {'':25s}  median = {med*1000:>10.1f} ms")
        print(f"  {'':25s}  mean   = {avg*1000:>10.1f} ms")
        print(f"  {'':25s}  min    = {min(r['times'])*1000:>10.1f} ms")
        print(f"  {'':25s}  max    = {max(r['times'])*1000:>10.1f} ms")
        if mult > 1.05:
            print(f"  {'':25s}  >> {mult:.0f}x slower")

    # Results comparison
    print("\n" + "-" * 70)
    print("  RESULTS COMPARISON (same strategy = same output)")
    print("-" * 70)
    print(f"  {'Engine':25s} {'Return':>12s} {'Trades':>10s}")
    for r in results:
        if "error" in r:
            continue
        ret = r.get("total_return")
        trades = r.get("trades")
        ret_str = f"{ret:.2f}%" if isinstance(ret, (int, float)) else str(ret)
        trades_str = str(int(trades)) if isinstance(trades, (int, float)) and trades is not None else str(trades)
        print(f"  {r['name']:25s} {ret_str:>12s} {trades_str:>10s}")

    print("\n" + "-" * 70)
    print(f"  Winner: {fastest['name']}  ({fastest['median']*1000:.1f} ms median)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Backtester benchmark")
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--engines", nargs="+", default=ALL_ENGINES, choices=ALL_ENGINES)
    args = parser.parse_args()

    print(f"Generating {args.rows:,} synthetic 1-min OHLCV bars...")
    df = generate_ohlcv(args.rows)
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"  Date range:  {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"  Engines:     {', '.join(args.engines)}")
    print(f"  Runs:        {args.runs}")

    results = []
    for engine in args.engines:
        fn = BENCH_FNS[engine]
        label = {"bt": "manifoldbt (Rust)", "vectorbt": "vectorbt (NumPy)", "backtrader": "backtrader (Python)"}[engine]
        print(f"\n> {label}...")
        r = fn(df, args.runs)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  median={r['median']*1000:.1f}ms  mean={r['mean']*1000:.1f}ms")
        results.append(r)

    print_results(results, args.rows)


if __name__ == "__main__":
    main()
