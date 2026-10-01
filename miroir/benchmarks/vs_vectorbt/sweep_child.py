"""One sweep point, one fresh process, one JSON line on stdout.

A parameter sweep is where vectorbt is supposed to be at its strongest: it
broadcasts the whole grid into one vectorised simulation. So it is the workload
where a speed claim needs the tightest gate, and where engines are the easiest
to compare *wrongly*.

Its own process, for two reasons:

* memory. The peak of a large grid is the number that decides what a machine can
  run at all, and it cannot be read once another engine has already grown the
  allocator in the same process.
* isolation. A 100k-combination grid that runs out of memory takes its process
  down with it; one point dying should not cost the whole benchmark.

    python sweep_child.py --bars 100000 --combos 250

The grid alignment trap
-----------------------
Every engine is asked for the same set of parameter pairs, but nothing forces
them to return the results in the same ORDER. manifoldbt enumerates its grid in
alphabetical key order (``fast`` outer, ``slow`` inner); vectorbt returns one
column per combination in the order its parameter product was built. Zip two of
them together wrongly and every number still looks plausible: the arrays have the
same length, the same distribution, even the same best value. Only the mapping
is scrambled, and the comparison silently becomes meaningless.

So the pairing is not assumed. Each engine returns its results keyed by the
``(fast, slow)`` pair it actually ran, and the comparison joins on that key.

Not every engine has a grid to broadcast
----------------------------------------
raptorbt has no fan-out API for a parameter grid on one instrument, so its
column here is a Python loop over ``run_single_backtest``. That is not a handicap
imposed by the harness, it is the only spelling available, and it is the one a
raptorbt user would write. It gets the same courtesy vectorbt gets on its own
path: each distinct moving average is computed once and reused across every
combination it appears in, rather than recomputed per cell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines import BY_CODE, ENGINES, REFERENCE  # noqa: E402

CAPITAL_TOL = 1e-9  # same yardstick as parity.py: difference over capital


def _mem_mb() -> float:
    import psutil

    info = psutil.Process().memory_info()
    # RSS under-reports under memory pressure, which is exactly the regime a
    # large grid puts the machine in. Prefer the private working set where the
    # platform exposes it.
    return (getattr(info, "private", None) or info.rss) / (1024 * 1024)


class _Peak(threading.Thread):
    # `_halt`, not `_stop`: Thread already has a private `_stop()` method, and
    # shadowing it with an Event makes `join()` raise "'Event' object is not
    # callable" on CPython 3.12, where `_wait_for_tstate_lock` calls it. The
    # sampler is the last thing anyone suspects when a sweep dies in join().
    def __init__(self, interval: float = 0.002):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0.0
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.peak = max(self.peak, _mem_mb())
            time.sleep(self.interval)

    def stop(self) -> float:
        self._halt.set()
        self.join(timeout=2.0)
        self.peak = max(self.peak, _mem_mb())
        return self.peak


def grid(combos: int) -> tuple[list[int], list[int]]:
    """Split `combos` into a near-square fast x slow grid.

    Near-square rather than a long strip: a 1 x N grid would sweep a single fast
    period and measure indicator reuse rather than fan-out. The fast and slow
    ranges are kept disjoint so no cell has fast >= slow, which would never trade
    and would pad the grid with empty simulations.
    """
    n_fast = int(combos ** 0.5)
    while combos % n_fast:
        n_fast -= 1
    n_slow = combos // n_fast
    fast_vals = [5 + 2 * i for i in range(n_fast)]
    slow_vals = [max(fast_vals) + 10 + 5 * i for i in range(n_slow)]
    return fast_vals, slow_vals


# The Community limit on a single sweep, as published on the pricing page.
COMMUNITY_SWEEP_LIMIT = 256


def tier() -> dict:
    """The licence tier the engine reports, read before and after the sweep.

    A number measured across a tier change is not a measurement of anything.
    """
    import manifoldbt as mbt

    name, _ = mbt.license_info()
    return {"pro": name == "Pro"}


def run_mbt(df, fast_vals, slow_vals, workdir, metrics=False):
    import manifoldbt as mbt
    from manifoldbt.expr import col, lit, when
    from manifoldbt.helpers import Interval, Slippage
    from manifoldbt.indicators import close as close_px, sma

    from workloads import CAPITAL

    store = mbt.import_dataframe(
        df,
        symbol="BENCH",
        symbol_id=1,
        interval="1m",
        data_root=os.path.join(workdir, "data"),
        metadata_db=os.path.join(workdir, "metadata.sqlite"),
    )
    last_ns = int(df["timestamp"].iloc[-1].value)
    config = mbt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=last_ns + 86_400_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=CAPITAL,
        execution=mbt.ExecutionConfig(
            signal_delay=0,
            execution_price="AtClose",
            max_position_pct=1.0,
            allow_short=False,
            position_sizing_mode="FractionOfEquity",
        ),
        fees=mbt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )
    strategy = (
        mbt.Strategy.create("sma_sweep")
        .signal("fast", sma(close_px, mbt.param("fast")))
        .signal("slow", sma(close_px, mbt.param("slow")))
        .size(when(col("fast") > col("slow"), lit(1.0), lit(0.0)))
    )

    def call():
        batch = mbt.run_sweep_lite(
            strategy, {"fast": fast_vals, "slow": slow_vals}, config, store
        )
        # Alphabetical key order: "fast" is the outer loop, "slow" the inner one.
        # Keyed by the pair rather than by position, so a change to that order
        # breaks the join loudly instead of scrambling the comparison.
        out = {}
        i = 0
        for f in fast_vals:
            for s in slow_vals:
                m = batch[i].metrics
                out[(f, s)] = (
                    (float(m["total_return"]), float(m["max_drawdown"]))
                    if metrics else float(m["total_return"])
                )
                i += 1
        if i != len(batch):
            raise RuntimeError(f"grid decode consumed {i} of {len(batch)} results")
        return out

    return call


def run_vbt(df, fast_vals, slow_vals, metrics=False):
    """vectorbt's grid path, taking the faster of its two ways of building it.

    The obvious spelling is to hand `MA.run` one window per combination and let
    `ma_above` line the two up. It is also the wrong one to benchmark: it
    recomputes every moving average once per combination it appears in, so a
    10 x 25 grid computes 500 averages instead of 35. Measured on 20k bars and
    250 combinations, that spelling takes 8.6x longer to build the same signal
    matrix, bit for bit.

    So the averages are computed once each and the product is formed by indexing
    columns. vectorbt is credited with its quicker path, exactly as the fee
    workload credits it with the quicker of its two metric paths.
    """
    import itertools

    import numpy as np
    import pandas as pd
    import vectorbt as vbt

    from workloads import CAPITAL, FREQ

    index = pd.DatetimeIndex(df["timestamp"])
    close = pd.Series(df["close"].to_numpy(dtype=np.float64), index=index)
    pairs = list(itertools.product(fast_vals, slow_vals))
    columns = pd.MultiIndex.from_tuples(pairs, names=["fast_window", "slow_window"])
    fast_idx = np.repeat(np.arange(len(fast_vals)), len(slow_vals))
    slow_idx = np.tile(np.arange(len(slow_vals)), len(fast_vals))

    def call():
        fast_ma = vbt.MA.run(close, window=fast_vals, short_name="fast").ma.to_numpy()
        slow_ma = vbt.MA.run(close, window=slow_vals, short_name="slow").ma.to_numpy()
        entries = pd.DataFrame(
            fast_ma[:, fast_idx] > slow_ma[:, slow_idx], index=index, columns=columns
        )
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=~entries,
            init_cash=CAPITAL,
            size=1.0,
            size_type="percent",
            fees=0.0,
            slippage=0.0,
            direction="longonly",
            accumulate=False,
            freq=FREQ,
        )
        returns = portfolio.total_return()
        # Columns carry a MultiIndex of the two window levels. Read the pair off
        # the index rather than trusting positional order.
        names = list(returns.index.names)
        fi = names.index("fast_window")
        si = names.index("slow_window")
        if not metrics:
            return {
                (int(kv[fi]), int(kv[si])): float(v)
                for kv, v in zip(returns.index, returns.to_numpy())
            }
        # The drawdown is what makes this workload different: it cannot be read
        # off the trade records, it needs the equity curve of every column, and
        # `from_signals` defers building that until something asks. Written out
        # rather than through `portfolio.max_drawdown()` because the accessor is
        # measurably slower on the same data, and vectorbt is credited with the
        # quicker of its two paths here exactly as it is everywhere else.
        equity = portfolio.value().to_numpy()
        drawdown = (equity / np.maximum.accumulate(equity, axis=0) - 1.0).min(axis=0)
        return {
            (int(kv[fi]), int(kv[si])): (float(r), float(d))
            for kv, r, d in zip(returns.index, returns.to_numpy(), drawdown)
        }

    return call


def run_rbt(df, fast_vals, slow_vals, metrics=False):
    """raptorbt's grid path: one backtest per cell, in Python.

    There is no fan-out entry point to call. ``run_multi_backtest`` broadcasts
    over *instruments*, not over parameters, so a parameter grid on one symbol is
    a loop, and the loop is what a user would write.

    The moving averages are hoisted out of it. Computing them per cell would
    recompute the same average once per combination it appears in, which is the
    spelling explicitly rejected on the vectorbt side; rejecting it there and
    accepting it here would be scoring the two engines with different rulers.
    """
    import numpy as np
    import raptorbt as rbt

    from workloads import CAPITAL

    timestamps = df["timestamp"].astype("int64").to_numpy()
    open_ = np.ascontiguousarray(df["open"].to_numpy(dtype=np.float64))
    high = np.ascontiguousarray(df["high"].to_numpy(dtype=np.float64))
    low = np.ascontiguousarray(df["low"].to_numpy(dtype=np.float64))
    close = np.ascontiguousarray(df["close"].to_numpy(dtype=np.float64))
    volume = np.ascontiguousarray(df["volume"].to_numpy(dtype=np.float64))
    config = rbt.BacktestConfig(
        initial_capital=CAPITAL, fees=0.0, slippage=0.0, upon_bar_close=True
    )

    def call():
        periods = sorted(set(fast_vals) | set(slow_vals))
        averages = {p: np.asarray(rbt.sma(close, p), dtype=np.float64) for p in periods}
        finite = {p: ~np.isnan(a) for p, a in averages.items()}
        out = {}
        for f in fast_vals:
            fast, fast_ok = averages[f], finite[f]
            for s in slow_vals:
                level = (fast > averages[s]) & fast_ok & finite[s]
                result = rbt.run_single_backtest(
                    timestamps, open_, high, low, close, volume,
                    level, ~level, config=config,
                )
                m = result.metrics
                ret = float(m.total_return_pct) / 100.0
                # Already computed inside the run, like the reference: asking
                # for it costs raptorbt nothing, and that is a result, not a
                # concession.
                out[(f, s)] = ((ret, -float(m.max_drawdown_pct) / 100.0)
                               if metrics else ret)
        return out

    return call


def builder(name: str, df, fast_vals, slow_vals, workdir: str, metrics: bool = False):
    """The grid closure for one engine, ready to be timed."""
    if name == "manifoldbt":
        return run_mbt(df, fast_vals, slow_vals, workdir, metrics)
    if name == "vectorbt":
        return run_vbt(df, fast_vals, slow_vals, metrics)
    if name == "raptorbt":
        return run_rbt(df, fast_vals, slow_vals, metrics)
    raise KeyError("no sweep path for engine {!r}".format(name))


def compare(reference_out: dict, other_out: dict) -> dict:
    """Join on the parameter pair, then gate on the worst cell, not the average.

    An average would hide a single badly wrong combination in a grid of
    thousands, which is precisely the failure a sweep can have and a single
    backtest cannot.
    """
    from workloads import CAPITAL

    missing = sorted(set(reference_out) ^ set(other_out))
    if missing:
        return {
            "status": "failed",
            "reason": f"{len(missing)} parameter pairs present in one engine only",
            "examples": [list(p) for p in missing[:5]],
        }
    # Both sides report a total return, a fraction of the account. Comparing
    # those directly is the same yardstick parity.py uses for a single backtest
    # (a difference in final equity over capital), without the round trip
    # through money.
    _ = CAPITAL
    # A cell is either a total return, or a (return, drawdown) pair when the
    # sweep was asked for a performance summary. Both components are gated: a
    # drawdown that disagrees is the same class of finding as a return that
    # does, and letting it through would publish a timing for work that was not
    # the same work.
    worst_pair, worst, worst_component = None, 0.0, None
    for pair, mv in reference_out.items():
        ov = other_out[pair]
        components = zip(mv, ov) if isinstance(mv, tuple) else ((mv, ov),)
        for index, (a, b) in enumerate(components):
            d = abs(a - b)
            if d > worst:
                worst, worst_pair = d, pair
                worst_component = ("return", "max_drawdown")[index]
    return {
        "status": "exact" if worst <= CAPITAL_TOL else "failed",
        "combos_compared": len(reference_out),
        "worst_abs_return_delta": worst,
        "worst_component": worst_component,
        "worst_pair": list(worst_pair) if worst_pair else None,
    }


# vectorbt is the only engine here whose grid footprint grows with the number of
# combinations, so it is the only one a point can declare out of scope. The
# others are looped or streamed and cost the same memory at any grid size.
OOS_ENGINE = "vectorbt"
OOS_REASON = (
    "vectorbt materialises the simulation per combination (measured: 1.57 MB "
    "per combination at 20k bars), so this grid would need tens of gigabytes. "
    "Running it would measure the swap file."
)


def main() -> int:
    """Entry point. The scratch store is removed on the way out, always.

    One point leaves one mkdtemp behind, and a matrix run spawns a child per
    point per engine per memory probe. A day of measurements left 87 of them and
    14.7 GB on the volume TEMP points at, which on a full system drive is not
    housekeeping, it is the benchmark failing with "Espace insuffisant sur le
    disque" in the middle of a run.
    """
    try:
        return _main(ap_parse())
    finally:
        _cleanup()


_SCRATCH: list = []


def _cleanup() -> None:
    import shutil

    for path in _SCRATCH:
        shutil.rmtree(path, ignore_errors=True)


def _scratch(prefix: str) -> str:
    """A temporary directory that will be removed when this process exits."""
    path = tempfile.mkdtemp(prefix=prefix)
    _SCRATCH.append(path)
    return path


def ap_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, required=True)
    ap.add_argument("--combos", type=int, required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="ask every cell for a performance summary, not just a total "
             "return. This is what a user sweeping thousands of combinations "
             "actually reads, and it is the one axis where the engines differ "
             "in kind rather than in speed: manifoldbt and raptorbt compute the "
             "drawdown inside the run whether it is read or not, while vectorbt "
             "defers the equity curve until something asks for it and then has "
             "to build one per column.",
    )
    ap.add_argument("--engines", nargs="+", default=list(ENGINES),
                    choices=list(ENGINES),
                    help="engines to run at this point; the reference is always "
                         "included whether it is named or not")
    ap.add_argument(
        "--parity-only",
        action="store_true",
        help="check that the engines agree, and skip timing. The only mode that "
             "says anything useful without a licence.",
    )
    ap.add_argument(
        "--vectorbt",
        choices=("run", "oos"),
        default="run",
        help="'oos' declares vectorbt out of scope for this point and runs the "
             "others. For grids whose vectorbt side does not fit in memory on "
             "any ordinary machine: not running it is the honest reading, since "
             "the alternative is a number produced by swapping.",
    )
    ap.add_argument(
        "--parity-anchor",
        type=int,
        default=250,
        help="with --vectorbt oos, the grid size vectorbt DOES run, so the point "
             "still carries a cross-engine check of the same code path.",
    )
    ap.add_argument(
        "--memory-only",
        choices=sorted(BY_CODE),
        help="run ONE engine and report the memory its grid added. Memory has to "
             "be measured this way: once any engine has run, the allocator has "
             "grown and whatever runs second is measured against a heap it can "
             "reuse. Measured in the same process, the second engine reads about "
             "half its true peak.",
    )
    return ap.parse_args()


def _main(args) -> int:
    import data as data_mod

    fast_vals, slow_vals = grid(args.combos)
    actual = len(fast_vals) * len(slow_vals)
    df = data_mod.make_ohlcv(args.bars)
    workdir = _scratch("mbt-sweep-")

    tier_before = tier()

    if args.memory_only:
        # What is wanted here is the PEAK of running this grid once, because that
        # is what a machine has to survive: a box that cannot hold the grid does
        # not return a slow number, it returns no number. Running the grid twice
        # and measuring the second call would answer a different question (what a
        # repeat costs in a warm process: 4.9 MB against 1.4 GB, measured), and
        # that number tells a reader nothing about whether the job fits.
        #
        # But a cold first call would also charge vectorbt its numba compilation,
        # a one-off that has nothing to do with grid size. So the engine is warmed
        # on a 2x2 grid -- enough to compile, too small to pre-allocate the real
        # one -- and the baseline is taken after that.
        name = BY_CODE[args.memory_only].name
        warm = builder(name, df, fast_vals[:2], slow_vals[:2],
                       os.path.join(workdir, "warm"), args.metrics)
        call = builder(name, df, fast_vals, slow_vals, workdir, args.metrics)
        warm()
        base = _mem_mb()
        watch = _Peak()
        watch.start()
        call()
        peak = watch.stop()
        print(json.dumps({
            "bars": args.bars,
            "combos": actual,
            "engine": name,
            "baseline_mb": round(base, 1),
            "added_mb": round(peak - base, 1),
            "tier_before": tier_before,
            "tier_after": tier(),
        }))
        return 0

    # Sweep timings are published from a licensed run only, the configuration
    # this benchmark describes. Timing is therefore refused outright without a
    # licence, rather than gated on grid size.
    #
    # Checking tier_before against tier_after is not enough on its own: a run
    # that starts AND finishes unlicensed shows no change at all, and would sail
    # through that comparison carrying a meaningless ratio.
    if not tier_before["pro"] and not args.parity_only:
        print(json.dumps({
            "bars": args.bars,
            "combos": actual,
            "status": "skipped",
            "reason": (
                "sweep timing requires a licence. "
                "Re-run with --parity-only to check agreement without timing."
            ),
            "tier_before": tier_before,
        }))
        return 2

    # Parity-only still has to fit the Community limit on a sweep.
    if args.parity_only and not tier_before["pro"] and actual > COMMUNITY_SWEEP_LIMIT:
        print(json.dumps({
            "bars": args.bars,
            "combos": actual,
            "status": "skipped",
            "reason": (
                f"{actual} combinations exceeds the Community limit of "
                f"{COMMUNITY_SWEEP_LIMIT} per sweep"
            ),
            "tier_before": tier_before,
        }))
        return 2

    running = [REFERENCE] + [n for n in args.engines if n != REFERENCE]
    out_of_scope = {}
    if args.vectorbt == "oos" and OOS_ENGINE in running:
        running.remove(OOS_ENGINE)
        out_of_scope[OOS_ENGINE] = {"status": "out of scope", "reason": OOS_REASON}

    calls = {
        name: builder(name, df, fast_vals, slow_vals,
                      os.path.join(workdir, name), args.metrics)
        for name in running
    }

    result = {
        "bars": args.bars,
        "combos": actual,
        "requested_combos": args.combos,
        "grid": {"fast": len(fast_vals), "slow": len(slow_vals)},
        "data_digest": data_mod.digest(df),
        "engines": running,
        "metrics": args.metrics,
        "tier_before": tier_before,
    }
    if out_of_scope:
        result["out_of_scope"] = out_of_scope

    # Warmup, discarded: it is where vectorbt compiles its numba kernels, and
    # charging that to every repetition would inflate the result. The warm call
    # is what the gate reads, so nothing is computed twice for it.
    warm = {name: call() for name, call in calls.items()}
    verdicts = {
        name: compare(warm[REFERENCE], out)
        for name, out in warm.items() if name != REFERENCE
    }

    for name, detail in out_of_scope.items():
        # The engine is not run at this size, so the point cannot carry a
        # cross-engine check against it on its own grid. It carries one of the
        # same code path at a size that engine can hold: same data, same
        # strategy, same adapters, fewer cells. That is weaker than checking the
        # grid itself, and the report says so rather than implying otherwise.
        a_fast, a_slow = grid(args.parity_anchor)
        anchor = compare(
            builder(REFERENCE, df, a_fast, a_slow,
                    os.path.join(workdir, "anchor"), args.metrics)(),
            builder(name, df, a_fast, a_slow, workdir, args.metrics)(),
        )
        verdicts[name] = {
            **anchor,
            "checked_at_combos": len(a_fast) * len(a_slow),
            "scope": "anchor grid, not this grid",
        }
        detail["parity_scope"] = "anchor grid"

    result["parity"] = verdicts
    result["status"] = "failed" if any(
        v["status"] == "failed" for v in verdicts.values()) else "exact"

    if result["status"] == "failed":
        # No timing for a grid the engines disagree on. That is the whole point
        # of the gate, and a sweep is where it earns its keep.
        result["timings"] = None
        result["note"] = "timing withheld: unexplained disagreement between engines"
        result["tier_after"] = tier()
        print(json.dumps(result))
        return 1

    samples = {name: [] for name in running}
    peaks = {name: 0.0 for name in running}
    base_mb = _mem_mb()
    for _ in range(args.reps):
        # Interleaved, so a runner that slows down mid-point penalises everyone.
        # Sampled per call rather than across the whole loop: a single peak over
        # every engine is the max of them, attributed to none, and memory is the
        # number that decides which grid sizes a machine can run at all.
        for name, call in calls.items():
            watch = _Peak()
            watch.start()
            t0 = time.perf_counter()
            call()
            elapsed = time.perf_counter() - t0
            peaks[name] = max(peaks[name], watch.stop())
            samples[name].append(elapsed)

    tier_after = tier()

    def med(xs):
        return sorted(xs)[len(xs) // 2]

    seconds = {name: med(samples[name]) for name in running}
    reference_s = max(1e-12, seconds[REFERENCE])
    result["timings"] = {
        "seconds": {**seconds, **{name: None for name in out_of_scope}},
        "ratio": {name: seconds[name] / reference_s
                  for name in running if name != REFERENCE},
        "reps": args.reps,
        "per_combo_us": seconds[REFERENCE] / actual * 1e6,
    }
    # Added by the call, on top of an already-built store / already-prepared
    # Series: what running the grid costs, not what holding the data costs.
    result["memory"] = {name: round(peaks[name] - base_mb, 1) for name in running}
    result["memory_baseline_mb"] = round(base_mb, 1)
    result["tier_after"] = tier_after

    # A sweep measured across a tier change is not a measurement, and a change
    # can only be seen after the fact.
    if tier_before["pro"] != tier_after["pro"]:
        result["timings"] = None
        result["note"] = (
            f"timing withheld: tier changed mid-point "
            f"(pro={tier_before['pro']} -> {tier_after['pro']})"
        )
        print(json.dumps(result))
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
