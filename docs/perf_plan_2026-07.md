# Perf plan: 2x on the CPU sweep (and friends) — execution order

Source: `docs/perf_audit_2026-07-18.md` (measured baselines, ranked table, ceilings).
Scope: run_sweep_lite CPU is the 2x target. Single run gets ~1.5-1.7x. GPU 2x is a
non-goal (fp64 parity ceiling ~1.35x, documented). Lite contract stays as-is.

Ground rules for every phase:
- Bit-for-bit parity CPU==GPU==general, verified against bt-expr goldens, never a mirror.
- When work moves off a path, break the old path on purpose to prove tests still bite.
- Portability: no target-cpu in shipped config; runtime feature detection only.
- Every perf claim: median of >=3 runs, MC-10M-CPU sentinel ~6s clean on both sides.
- One branch per workstream, `perf:` commits, no cross-stream stacking unless noted.

## End-to-end result so far (2026-07-18)

Measured A-B-A at the build level (HEAD, then main's engine sources, then HEAD
again) so drift between builds is visible rather than assumed. "after" is the
mean of the two HEAD runs.

| | before (main) | after | gain |
|---|---|---|---|
| CPU sweep (10k combos x 26k bars) | 51,332 c/s | ~64,567 c/s | **+26%** |
| SL/TP sweep | 27,662 c/s | ~33,885 c/s | **+22%** |
| `SweepResult.to_df()` | 63.2 us/combo | 12.0 us | **5.3x** |
| `SweepResult.best()` | 25.4 us/combo | 2.4 us | **10.5x** |
| `.metrics` per result | 27.9 us | 3.0 us | **9.4x** |
| GPU sweep (untouched, drift canary) | 164,067 c/s | ~167,436 c/s | +2% |

The GPU line is the control: nothing in this work touches that path and it does
not move, so the harness is not systematically biased.

**Caveats, so these are not over-read.** The two identical HEAD builds differed
by 6.7% (62,474 vs 66,661 c/s) and the baseline was measured once, so read the
sweep numbers as +26% with roughly +/-7%, solid in direction and magnitude but
not to the point. The post-processing ratios are far above the noise (the two
HEAD runs agree to 0.3%) and are reliable.

**One unattributed slice.** The CPU sweep gained ~26% while the paired A/B
credits the CAPM hoist with 11.9%. The likely source of the remaining ~14% is
the daily-equity unification, which was intended as a pure refactor: the merged
form does one division and one comparison per bar where the old form did
`last() == Some(ts/nanos*nanos)` (division AND multiplication) and then possibly
a second `last()/nanos != day` test. Redundant per-bar arithmetic was removed
without aiming for it. This attribution is PLAUSIBLE BUT UNVERIFIED; it needs
its own paired A/B before being claimed.

## Phase 0 — lock the ruler (DONE 2026-07-18)

- [x] Harness checked in at `benchmarks/audit_harness.py` (subcommands: sentinel,
      profile_sweep, bracket_probe, full_vs_lite, single1m, par_scale, sweep_cpu/gpu, access).
- [x] 2026-07-18 baselines are the reference row (see the audit doc).
- [x] Bracket-sweep probe run to size finding #3b.

**Measured (10k combos, ema grid, sentinel-clean):**

| bars | plain | SL/TP bracket | ratio | sim us/combo (plain -> bracket) |
|---|---|---|---|---|
| 26k | 54,800 c/s | 26,440 c/s | 2.07x slower | 242 -> 547 |
| 100k | 11,863 c/s | 3,737 c/s | 3.17x slower | 1,082 -> 3,972 |

**Re-sizing that forces (finding #3b):** the bracket penalty is large but scales
SUPER-linearly with bars (2.07x -> 3.17x), so it is dominated by per-bar bracket
check work, NOT by the `ohl_nan` copy (which is linear in bars). The copy is a
minority slice of the +305us/combo (26k). The audit's "25-40% tax, ~1.2-1.4x from
hoisting" was too optimistic: expect ~1.1x on bracket sweeps from the copy hoist
alone. The real bracket win lives in Phase 3 (loop/macro work), not Phase 1.
Consequence: **the ohl_nan/funding_nan hoist is demoted out of Phase 1** and folded
into the Phase 3 bracket work; Phase 1b keeps only the CAPM hoist (universal ~7%).

## Phase 1 — quick wins, zero parity risk (~2-3 days total)

Branch `perf/full-metrics-pydict` (DONE 2026-07-18)
- [x] `PyBacktestResult.metrics` + `profile` (result.rs): build via the existing
      `metrics_to_pydict` / `profile_to_pydict` (backtest.rs), no JSON round-trip.
- [x] **Nested `trade_stats` hand-mirrored too** (`trade_stats_to_pydict`). This was
      the missing half: `metrics_to_pydict` still round-tripped the nested
      TradeStatistics through JSON. Harmless for LITE sweeps (trade_stats = None) but
      the full `run_sweep` populates it on EVERY combo, so it was the dominant residual
      cost. Fixing only the outer getter gave 1.7x; adding this gave 4-6x.
- [x] Serde-parity pinned: 5 unit tests green, incl. a new
      `metrics_dict_matches_serde_with_trade_stats_signal_quality` for the
      nested-nested `signal_quality`. Drift guard verified to BITE (sabotaging one
      field fails both trade_stats tests with "drifted from serde").
- [x] End-to-end oracle: full `run_sweep` vs dedicated `mbt.run()` per combo (both
      full path, so exact) -- 9 combos x (189 scalar + 162 trade_stats) fields,
      **0 mismatches**.
- [ ] (deferred, optional) Route `to_df()`/`best()` through the `sweep_columns`
      buffer path. Not needed to close Phase 1: the getter fix already removed the
      JSON round-trip; what remains is inherent Python dict/DataFrame building.

**Measured (900 combos):**

| op | before | after | gain |
|---|---|---|---|
| `SweepResult.to_df()` | 49 us/combo | 11.7 us/combo | **4.2x** |
| `SweepResult.best()` | 15 us/combo | 2.4 us/combo | **6.3x** |
| `.metrics` per result | ~18 us | 2.8 us | **~6.4x** |

Note: the audit's "~50x" projection was wrong -- it carried over the scale of the
ORIGINAL 1M-sweep lite bug rather than this path's measured baseline. Real: 4-6x.

Python suite: 63 passed, 2 failed. Both failures (`test_golden_buy_and_hold`,
`test_sweep::test_sweep_returns_one_result_per_combo`) are **pre-existing** --
proven by stashing the change, rebuilding, and reproducing the identical
`2 failed, 63 passed`. Both share one root cause: the golden fixture yields a flat/
truncated equity curve (`[1000.0, 1000.0]`), which makes total_return 0.0 for every
combo. Tracked separately (see the pending `fix/python-test-suite` branch).

Branch `perf/hoist-capm` (Phase 1b, DONE 2026-07-18 -- perf number provisional)
- [x] CAPM benchmark returns hoisted into `run_sweep_lite` and `run_batch_lite`
      via `hoist_capm_benchmark()`; `run_lite_on_aligned` takes
      `hoisted_benchmark: Option<&[f64]>` and falls back to computing per run when
      None. walk_forward passes None (unchanged behaviour).
- **Guard (verified in code):** hoisting is bit-identical ONLY when the run's
  `sim_bars` is `&aligned.symbol_bars`, i.e. `coarse_bars` is None
  (orchestrator.rs:2300 `if signal_ns > native_ns`). The helper re-detects
  `native_ns` on the post-pre-resample bars and returns None under hybrid
  resampling, where closes come from coarse bars the driver has no handle on.
  A blind driver-side hoist would have silently changed alpha/beta there.
- [x] **Proven equivalent, not merely untested.** A temporary probe recomputed the
      per-run benchmark alongside the hoisted one and asserted `to_bits()`
      equality: green. Then the probe was made to `panic!` on entry to prove the
      hoisted branch is actually REACHED -- it is, by 8 tests, and exactly the
      right ones: `lite_matches_full_with_{stop_loss,take_profit,trailing_stop,
      gap_through_stop,stop_loss_short,full_bracket_and_costs}_at_native_resolution`,
      `lite_and_full_agree_on_max_drawdown_sign`, and
      `per_strategy_orders_apply_and_batch_is_heterogeneous` (batch_lite).
      Probe removed; `cargo test -p bt-core` = 91 passed, 0 failed.
- Note: `golden_buy_and_hold` fails under `--release` but passes in debug with
  `BT_UNLOCKED=1`. That is the known licensing artifact (the dev bypass is
  `#[cfg(debug_assertions)]`, so release runs locked and hits the Pro output
  floor), not this change. It also does not exercise this change at all: it runs
  `run()`, i.e. the full kernel, whose CAPM block was left untouched.

**Perf: MEASURED, interleaved A/B (10k combos x 26k bars, ema_cross).**

A plain before/after was NOT usable here: the Phase 0 baseline was taken while
the paper dashboard was loading the box, so comparing it against a later quiet
run would have credited the hoist with someone else's CPU. Instead the hoist was
put behind a temporary env toggle so ONE binary could run both arms, interleaved
A,B,A,B, in one environment. Paired deltas cancel any drift. Toggle removed after.

| | hoist OFF | hoist ON | delta |
|---|---|---|---|
| sim / combo | 246.2 us | 206.1 us | **+16.5%** |
| wall | 179.9 ms | 159.0 ms | **+11.9%** |
| throughput | 55,593 c/s | 62,883 c/s | |

Four paired rounds, tightly clustered (sim +15.9/+16.6/+17.5/+16.3%), which is
how we know the pairing worked. **The plan's ~1.08x estimate was too
conservative: the real figure is 1.12x on wall.** The audit had priced CAPM at
~1ns/bar (~26us/combo); the measured removal is ~40us/combo, because the pass
also does a resample-to-daily, a step_returns and their allocations per combo,
not just a linear scan.

### The MC-10M sentinel is NOT a load detector (correction)

Recorded because the old heuristic ("~6s clean vs ~16s loaded") is misleading
and cost real time this session. With every competing process killed and the
sweeps posting their best numbers of the day (66,458 c/s), the sentinel still
read ~16s, and inside one 3-run batch it printed `[16.14, 16.36, 7.47]`. It is
bimodal for reasons unrelated to CPU contention (10M paths: allocation /
first-touch / page-cache state), so it flags "loaded" on a quiet machine.

Use instead: interleaved A/B with paired deltas, which is robust to drift by
construction and needs no external notion of "clean".

Gate to close Phase 1: parity suite green (Rust goldens + Python mirrors), benches
re-run per protocol, numbers recorded in the audit doc.

## Phase 2 — medium effort, contained risk (~1-2 weeks)

Branch `perf/single-run-parallel` -- **CANCELLED 2026-07-18, premise was false.**

- [x] ~~rayon-join independent signal expressions~~ **ALREADY IMPLEMENTED.**
      orchestrator.rs:889 evaluates each dependency level with `level.par_iter()`
      whenever `level.len() >= 2`. The audit's exploration agent reported "fast/slow
      EMA computed sequentially"; that was a misread, and it propagated into this
      plan. Measured at 1M bars, N independent EMAs in one strategy:

      | signals | signal_eval | us/signal | if it were sequential |
      |---|---|---|---|
      | 1 | 6,843 us | 6,843 | - |
      | 2 | 10,071 us | 5,036 | 13,686 |
      | 4 | 10,580 us | 2,645 | 27,372 |
      | 16 | 17,110 us | 1,069 | 109,488 |

      16 signals cost 2.5x one signal, not 16x. The parallelism is real and
      working. Nothing to win here. (Each individual EMA is a sequential scan, so
      the residual per-signal cost is irreducible without changing the recurrence.)

- [x] ~~Overlap output_build components~~ **NOT WORTH IT.** `metrics` borrows
      `trace_equities` while the gather MOVES it, and metrics must run on
      full-resolution equity (max_drawdown depends on it, orchestrator.rs comment
      at the metrics call). Overlapping them needs an 8MB clone at 1M bars, which
      eats most of the ~2.8ms theoretical saving. Best case was ~7% of a 27ms run.

**Consequence for the ceiling:** the audit put the single run at ~1.7x reachable.
That number assumed a sequential signal phase that does not exist. With signals
already parallel and output_build ownership-bound, the single-run path is close to
its practical ceiling; expect well under 1.2x, and it is NOT where the 2x lives.
The 2x target remains the CPU sweep, where per-combo signal work is genuine CPU
load (the sweep saturates all threads across combos, so intra-combo signal
parallelism is degenerate there and the transpiled-sweep item still stands).

Branch `perf/simd-dispatch` (target: ~1.1-1.15x sweep, more on signal-heavy runs)
- [ ] `is_x86_feature_detected!` runtime dispatch on bt-expr elementwise kernels only
      (compare, IfElse, arithmetic). Folds and scans (EWM, rolling, sums) stay scalar.
- [ ] Parity: elementwise same-op-per-lane is bit-identical; add a test asserting
      dispatch on/off equality on goldens. Baseline fallback keeps portability.

Optional branch `perf/full-sweep-traces`
- [ ] Optional trace retention on full run_sweep (orchestrator.rs:3480-3547), or at
      minimum docs steering sweep users to lite + sweep_columns.

## Phase 3 — structural, the 2x closers (~3-5 weeks, sequential)

Branch `refactor/loop-extraction` FIRST (enabler, no behavior change)
- [ ] Mechanically extract the shared fill/equity/daily blocks from the four loop
      transcriptions (full/lite x general/fast) and the two bracket macros.
      Pure extraction: does not change WHICH metrics lite computes (lite contract).
- [ ] Golden + parity suites must be bit-identical before/after.

**De-risked 2026-07-18: the 1.33x premise holds, and is probably conservative.**

Measured with the paired-A/B discipline (each variant in its own process,
interleaved, only paired deltas trusted), 2500 combos x 26k bars:

| paired delta | value | rounds | reading |
|---|---|---|---|
| +1 indicator (fixed span, declared, unreferenced) | **-0.5 us/combo** | -0.8, -0.3, +11.1, -0.7 | an indicator costs ~nothing per combo |
| +1 elementwise pass (compare+when+add) | **+180 us/combo** | 178, 157, 183, 241 | one pass ~= 1.34x the WHOLE signal phase |

base: signal_eval 134.3 us/combo, simulation 199.7 us/combo.

The ~0 indicator delta is NOT pruning: the compiler compiles every entry of
`def.signals` with no dead-signal elimination
(crates/bt-strategy/src/compiler.rs:74-82). It is AMORTIZATION. A fixed-span EMA
is computed once and every combo hits IndicatorCache; a swept EMA over a 50x50
grid has 50 distinct spans shared by 50 combos each. So indicator math is
effectively free per combo in a 2D sweep, and nearly all of signal_eval is
per-combo overhead (env build, param binding, the elementwise chain, output
allocation) -- precisely what fusing the target into the per-bar loop removes.

Ceiling: the naive arithmetic says 1.68x, but do not quote that. It leans on a
slightly negative indicator delta (so "101% removable", an artifact), the
elementwise delta has 46% spread, and a real transpiled sweep still pays
per-combo param binding and indicator lookup. **Defensible: >= 1.33x, plausibly
1.4-1.7x.** Enough to justify the work; re-measure against the real
implementation rather than trusting this number.

Branch `perf/cpu-transpiled-sweep` (target: ~1.33x, stacked on Phase 1 => ~1.45x)
- [ ] Reuse the GPU hoist plan (build_hoist_plan) on CPU: fill hoisted indicator series
      once per sweep, compute the target in-loop via `sim_fast_lite_core_single`
      (orchestrator.rs:5766), which is already the CUDA kernel's CPU reference.
- [ ] Same eligibility gates as the GPU transpiler; anything else falls back to the
      current vectorized-signal path, unchanged.
- [ ] Deliberately break the old vectorized path (temporarily) to prove the fallback
      is still covered by tests. Parity anchored to bt-expr.

Branch `perf/lite-loop-tightening` (target: 36 -> ~24 cycles/bar, ~1.15-1.3x sweep)
- [ ] Branch elimination in the single-asset core: hoist the sizing_mode match, the
      no-rebalance gates, null-path dispatch. Control flow and layout ONLY.
- [ ] Forbidden: FP reordering, FMA, fast-math of any kind. Op order is the parity
      anchor. Bit-verify against bt-expr after every commit.

Gate to close Phase 3: composite CPU sweep >= 1.9x vs Phase 0 baseline on the 3-strategy
500k bench (median of 3, sentinel-clean), GPU numbers unchanged, parity green.

## Phase 4 — coverage (optional, per-gate decisions)

- [ ] Funding on run()'s CPU fast path (orchestrator.rs:1358-1364): aligns run() with
      lite/GPU, big win for perp single runs.
- [ ] GPU metrics kernel fuse/overlap (gpu_sweep.rs:4395-4656): <=1.16x GPU.
- [ ] Multi-asset GPU: brackets support, occupancy (33% today).
Each: own golden work, own decision. None blocks the 2x goal.

## Non-goals (explicit)

- GPU 2x under fp64 bit-parity: not reachable (sim kernel 73%, SASS-audited ceiling).
  fp32 stays the documented opt-in for speed-over-bits users.
- Changing the lite contract (cagr/calmar/ulcer != run()).
- Machine-specific build flags in shipped wheels.
- Data-loading work (cold 57ms @1M bars, <1% at bench scale): revisit only if a
  many-fresh-process workflow becomes a product path.

## Compatibility invariant

No strategy loses support at any phase. Fast paths widen or stay put; everything not
eligible falls back to today's code, which keeps its own test coverage (proven by the
deliberate-break rule). Example strategies remain testable and sweepable throughout;
`examples/` runs green at every phase gate.
