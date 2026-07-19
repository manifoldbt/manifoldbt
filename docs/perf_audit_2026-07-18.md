# Performance audit: path to 2x on the hot paths (2026-07-18)

Audit read-only: no engine code was modified. Machine: i5-13600KF (6P+8E, 20 threads),
RTX 3090, build `maturin develop --release --features cuda` (LTO, cgu=1, debuginfo kept
via env overrides). Every number is the median of at least 3 runs; machine load was
verified with the MC-10M-CPU sentinel (6.3-6.5s clean on both sides of each suite; one
loaded suite at 8.6-9.3s was discarded and rerun).

**Conclusion: the CPU lite sweep can reach 2x (three workstreams). The GPU sweep cannot
under the fp64 bit-parity constraint (ceiling ~1.35x). The single run tops out around
1.7x. The full `run_sweep` API still carries the unfixed twin of the JSON-getter bug:
its Python post-processing costs 3.6x the entire lite compute.**

## Method

Sampling profilers were unavailable (samply/ETW needs Administrator; py-spy cannot see
rayon threads). Attribution comes from the engine's own instrumentation, which proved
sufficient: 98% of wall*threads is attributed.

- `ProfileData` (orchestrator.rs:400): per-phase us counters on every result, including
  each lite sweep combo. Summed across combos and compared to wall*threads.
- `BT_PHASE6_DEBUG=1`: sub-splits output_build (metrics / capm / trade_stats / gather).
- `MBT_GPU_PROBE=1`: GPU phases (nvrtc / hoist / h2d / sim / metrics / d2h).
- Ablation: bars scaling (26k vs 100k), buy_and_hold vs ema_cross, max_parallelism 1..20,
  full vs lite API.

Harness: copy of the session scratchpad `audit_harness.py` (subcommands: sentinel,
profile_sweep, full_vs_lite, single1m, par_scale, batch3, load, access).

## Measured baselines (synthetic BTCUSDT 1h store, `benchmarks/_sweep_common.py` config)

| Path | Result | Phase split |
|---|---|---|
| Single run, 26k bars | 0.85ms | signal 43%, sim 22%, output 32% |
| Single run, 1M bars | 27ms | signal 9.9ms (37%), sim 8.4ms (31%), output 8.3ms (31%: metrics 4.5, capm 1.0, trade_stats ~2, gather/positions ~2.3) |
| CPU lite sweep, 10k combos x 26k bars | 171ms = 58k c/s | per combo: signal_eval 96us (28%), sim phase 239us (70%) |
| inside the sim phase | | per-bar loop ~200us (~8ns/bar = 36 cycles/bar), CAPM ~25us (measured 1ns/bar), O(days) metrics ~10us |
| CPU sweep, 500k combos | rsi 9.1s / ema 9.9s / trix 12.4s | 40-55k c/s |
| GPU sweep, 500k combos | rsi 1.67s / ema 0.98s / trix 2.35s | 5.4-10.2x vs CPU |
| GPU, 100k combos x 26k bars | 240ms = 418k c/s | sim kernel 176ms (73%), metrics kernel 34ms (14%), h2d 6ms, d2h ~2ms |
| Data load, 1M bars | cold 57ms, warm re-align ~0.5ms | store cache holds across calls |
| Lite packaging/access | ~1us per `.metrics` | fixed path is healthy |
| Full `run_sweep`, 900 combos | 80ms vs 16ms lite (5x) | then `.best()` 15us/combo, `.to_df()` 49us/combo in Python |

Cross-checks: these numbers predict the historical 1M x 100k CPU sweep at ~107s, inside
the observed 84-150s band. Parallel scaling is 9.9x on 20 threads, near the realistic
ceiling (~11-12x) for 6P+8E: rayon granularity is not a lever. Combo enumeration is now
lazy mixed-radix (sweep.rs:98); the old ~500B/combo materialization survives only in
walk_forward/sweep_2d. Per-combo strategy recompile (dynamic periods): ~2%, cleared.

## Ranked opportunities

Gain = on the bench that path owns. Effort: L (<1 day), M (days), H (week+).
Parity = risk to CPU==GPU==general bit-identity (anchored to bt-expr, never a mirror).

| # | Opportunity | Evidence | Where | Approach | Gain | Effort | Parity risk |
|---|---|---|---|---|---|---|---|
| 1 | Full-result `.metrics` = unfixed JSON twin | `.to_df()`+`.best()` = 58ms vs 16ms lite compute @900 combos | bt-python/src/result.rs:24-31, python sweep.py:75-93, dataframe.py:146-166 | Reuse `metrics_to_pydict` (backtest.rs:331); better: route `SweepResult.to_df/best` through `sweep_columns` | ~50x on full-sweep post-processing | L | None |
| 2 | CPU transpiled sweep: kill per-combo signal materialization | signal_eval = 28% of sweep CPU | orchestrator.rs:2402-2709; reference core orchestrator.rs:5766 | Do on CPU what the GPU does: hoist plan + in-loop target eval; `sim_fast_lite_core_single` is the transpile-ready reference | ~1.33x CPU sweep | H | Low; re-anchor goldens, deliberately break the old path to prove coverage |
| 3 | Hoist combo-invariant work: (a) CAPM benchmark pass, (b) ohl_nan/funding_nan copies | (a) measured 1ns/bar = ~7% of sweep; (b) 3 full O(bars) copies per combo on SL/TP sweeps, est. 25-40% tax (code-confirmed, not yet measured) | (a) orchestrator.rs:3395-3411 and 2082-2108; (b) orchestrator.rs:2821-2849 | Compute once in the sweep/batch drivers, pass by reference | 1.08x plain sweeps; ~1.2-1.4x SL/TP sweeps | L-M | None (same values, computed once) |
| 4 | Per-bar loop tightening | loop ~58% of sweep CPU at 36 cycles/bar; dependency floor ~12-18 cycles | simulate_fast_lite_single / core (orchestrator.rs:5307/5766) | Branch elimination, layout. No FP reordering, no FMA: op order is the parity anchor | 1.15-1.3x sweep | H | High if careless; bit-verify vs bt-expr |
| 5 | Single-run: parallelize independent signals + output_build components | @1M: signal 37% (fast/slow EMA sequential), output 31% (independent components) | orchestrator.rs:833-1204, 2059-2230 | rayon-join independent signal exprs (bit-safe); overlap output components; never parallelize the metric reductions | ~1.4-1.7x single run | M | Low if reductions stay sequential |
| 6 | SIMD via runtime dispatch (ships SSE2 baseline: no target-cpu anywhere) | elementwise ops are a large slice of signal eval | bt-expr evaluator kernels | `is_x86_feature_detected!` dispatch on elementwise kernels only; folds/scans stay scalar | ~1.1-1.15x sweep | M | None for elementwise; forbidden for folds |
| 7 | GPU metrics kernel fuse/overlap | 34ms of 240ms (14%) | gpu_sweep.rs:4395-4656 | Fuse into sim epilogue or stream overlap, same op order | <=1.16x GPU | M-H | Low |
| 8 | Full `run_sweep` materializes full traces per combo | 5x lite; ~600KB/combo retained | orchestrator.rs:3480-3547 | Optional trace retention; steer to lite + `sweep_columns` | up to 5x for full-sweep users | M | None |
| 9 | Coverage gates (funding on `run()` fast path, multi-asset+brackets on GPU, multi-asset kernel 33% occupancy) | gate list in bt-core | orchestrator.rs:1358-1364, 3998-4002 | Extend fast/GPU coverage case by case | 2-5x for affected configs | M-H each | Per-gate golden work |
| 10 | Data loading (no mmap on Arrow IPC, exact-key cache only, N+1 sqlite symbol_info) | cold 1M load 57ms | bt-data arrow_ipc_store.rs:327/461, metadata.rs:28 | mmap like mega_store, superset-slice cache, batched lookups | <1% on benches | L-M | None |

Maintenance note: the per-bar loop exists in four near-identical transcriptions
(full/lite x general/fast) plus two ~130-LOC bracket macros; every loop optimization is
written and parity-tested four times. Mechanical extraction of the shared fill/equity
blocks is a safe enabler for #4. Any unification touching WHICH metrics lite computes
would hit the lite contract (cagr/calmar/ulcer != run()), which stays as-is.

## Top 3 to reach 2x (CPU lite sweep)

1. **#3 hoists** (CAPM + bracket/funding copies): low effort, zero parity risk, ~1.08x
   plain sweeps, biggest single win on SL/TP sweeps.
2. **#2 CPU transpiled sweep**: ~1.33x, structural; the architecture is proven on GPU
   and the CPU core is already the kernel's reference. With #3: ~1.45x.
3. **#4 loop tightening** (+ #6 SIMD on residual signal eval): 36 -> ~24 cycles/bar
   closes the gap. Composite: **~1.9-2.2x**.

## Theoretical ceilings

- CPU sweep: signal+CAPM removed, loop untouched -> max 1.44x. 2x requires the loop
  work; at the ~12-18 cycle dependency floor the composite ceiling is ~3x.
- GPU sweep: sim kernel 73%, already SASS-audited to the fp64 parity ceiling; everything
  else free -> 1.37x max. 2x is not reachable under parity; opt-in fp32 remains the out.
- Single run 1M bars: sim loop is serial; practical ceiling ~1.7x.
- Data loading and lite packaging: <1% at bench scale, nothing to win.

## Compatibility guarantee for every item above

None of the proposals removes or restricts any strategy feature. #2/#4/#6 follow the
same pattern as the GPU path: strategies that qualify take the faster path, everything
else falls back to today's code, and the fallback stays golden-tested (the "break the
old path on purpose" rule applies when work moves). #3/#5/#7/#8/#10 compute identical
values in fewer places. #9 strictly widens fast-path coverage. Bit-for-bit parity is
re-verified against bt-expr for every change.
