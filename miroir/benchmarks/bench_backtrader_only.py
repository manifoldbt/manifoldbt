"""Standalone backtrader benchmark - EMA(12/26) crossover on synthetic 1m data."""
import argparse
import time
import numpy as np
import pandas as pd
import backtrader as btdr


def generate_ohlcv(rows, seed=42):
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


class EmaCross(btdr.Strategy):
    def __init__(self):
        self.fast = btdr.indicators.EMA(self.data.close, period=12)
        self.slow = btdr.indicators.EMA(self.data.close, period=26)
        self.crossover = btdr.indicators.CrossOver(self.fast, self.slow)

    def next(self):
        if self.crossover > 0:
            self.order_target_percent(target=0.5)
        elif self.crossover < 0:
            self.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"Generating {args.rows:,} synthetic 1-min bars...")
    df = generate_ohlcv(args.rows)

    bt_df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    bt_df = bt_df.rename(columns={"timestamp": "datetime"}).set_index("datetime")
    bt_df.index = bt_df.index.tz_localize(None)

    # Warmup
    print("Warmup run...")
    cerebro = btdr.Cerebro()
    cerebro.addstrategy(EmaCross)
    cerebro.adddata(btdr.feeds.PandasData(dataname=bt_df))
    cerebro.broker.set_cash(10_000)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.run()

    # Timed runs
    print(f"Running {args.runs}x timed...")
    times = []
    for i in range(args.runs):
        cerebro = btdr.Cerebro()
        cerebro.addstrategy(EmaCross)
        cerebro.adddata(btdr.feeds.PandasData(dataname=bt_df))
        cerebro.broker.set_cash(10_000)
        cerebro.broker.setcommission(commission=0.0005)

        t0 = time.perf_counter()
        cerebro.run()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  run {i+1}: {elapsed*1000:.1f} ms")

    med = np.median(times)
    avg = np.mean(times)
    print(f"\nbacktrader results ({args.rows:,} bars):")
    print(f"  median = {med*1000:.1f} ms")
    print(f"  mean   = {avg*1000:.1f} ms")
    print(f"  min    = {min(times)*1000:.1f} ms")
    print(f"  max    = {max(times)*1000:.1f} ms")


if __name__ == "__main__":
    main()
