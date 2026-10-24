# GPU parameter sweeps

The sweeps job of the [CPU benchmark](../vs_vectorbt/README.md#parameter-sweeps)
runs on a GitHub-hosted runner, which has no GPU, so it can only ever measure the
CPU path of a sweep. This benchmark runs the same sweeps on a GPU, and on the CPU
of the same machine beside it.

Same sweeps means the same code, not a copy of it: the data generator, the seed,
the grid and the strategy are imported from `../vs_vectorbt` (`data.py`,
`sweep_child.py`), and the default points are the CPU job's three. Each point
records the digest of its data, so its GPU number and the CPU job's number for
the same grid can be checked to be about the same bars.

| bars | combinations | grid (fast x slow) |
|---:|---:|---|
| 20,000 | 5,000 | 50 x 100 |
| 20,000 | 20,000 | 125 x 160 |
| 200,000 | 10,000 | 100 x 100 |

## Where the card comes from

GitHub's hosted runners have no GPU, and its GPU runners are billed by the minute
on paid plans only. The workflow (`.github/workflows/bench-gpu.yml`) therefore
starts the run on [Modal](https://modal.com) from a standard runner
(`modal_app.py`), and the runner only resolves the version, waits for the result
and publishes it.

Inside the container nothing differs from a user's machine: the published wheel
from PyPI with its `[gpu]` extra, the libraries pinned in `requirements-lock.txt`,
and an NVIDIA L4. Modal bills by the second for the life of the container, card and
reserved cores included, so a run costs its wall time: two minutes from start to
finish on the first runs, image already built, most of it the CPU side of the
comparison. The function's 15-minute timeout is the ceiling on what a stuck run
can cost.

## What the harness refuses to publish

- **A GPU number that was not computed on the GPU.** `device="cuda"` falls back to
  the CPU with a `UserWarning` when the card, the build or the strategy cannot
  take the sweep, so that a user's results are never affected. Here, any
  `UserWarning` raised during a GPU sweep fails the point. A machine where no card
  is visible fails before the first point.
- **A timing for different work.** Every cell of the GPU grid is compared with the
  CPU grid, keyed by its parameter pair, through the gate the CPU job applies to
  vectorbt: total return and maximum drawdown, worst cell against 1e-9 of capital.
  The engine documents the two paths as identical in fp64, so the result also
  counts the cells that are identical to the bit. Every timed repetition must
  reproduce the first call, cell for cell.
- **A timing measured across a licence change**, as in the CPU job. GPU
  acceleration is a Researcher feature, and the run activates the same CI licence
  as the CPU sweeps job.

## How it is timed

1. `calibrate_gpu()`, untimed: a few seconds that bring the card to its steady
   clocks and measure the costs the sweep uses to choose its execution path. The
   measured constants are kept in the result.
2. Per point, one untimed call on each side. It compiles the kernels on the GPU,
   which is why it is reported on its own (`first_call_s`): it is what a user's
   first sweep costs, not what a sweep costs.
3. `--reps` timed repetitions per side (7 by default, as in the CPU benchmark),
   GPU and CPU interleaved, in an order that alternates, so a clock that drifts
   over the run drifts on both sides. Medians are published with min, max and
   IQR; a point whose IQR exceeds 15% of its median is flagged, as in the CPU
   benchmark. Seven rather than five because one slow repetition out of five is
   enough to cross that line: it did on the first Modal run.

The CPU side runs on 4 cores reserved for it as a hard limit, like the four vCPUs
of the GitHub runner, and is told so (`max_parallelism`). A container can see
more cores than it may use, and a sweep that spawned one thread per visible core
on a 4-core quota would publish the cost of the oversubscription. What the
container saw is recorded under `environment.cores`; on Modal it is the 4 cores
reserved. The CPU model is not readable from inside Modal's sandbox, so it is
recorded as unknown.

## Reading the numbers

The speedup is the GPU against the CPU of the **same machine**. It is not a ratio
against the CPU job, which runs on another machine: the two files are read side by
side, never divided into each other.

The ratio grows with the work per sweep. On the first Modal runs (manifoldbt
0.28.1, L4 against its 4 reserved cores), the three default points ran 9.1, 10.3
and 18.8 times faster on the card: 33 ms against 296 ms, 111 ms against 1.15 s,
and 233 ms against 4.39 s. The published file is the reference; these are what
the harness was validated on.

Larger grids widen the gap. On a shared workstation (RTX 3090, CPU held to 4
threads, flagged noisy, so indicative only), a million bars by 20,000
combinations ran about 20 times faster on the card. That point is not in the
default set because its CPU side alone costs 37 s a call there, so a run would be
minutes of reserved card mostly waiting for the CPU. It can be asked for on a
manual run (`sweep: 1000000:20000`).

The numbers depend on the card. The published file is always an L4; a manual run
can name another one, and is then reported without being published.

## Reproducing it

On any machine with an NVIDIA card and its driver:

```bash
pip install "manifoldbt[gpu]" -r requirements-lock.txt
python bench.py --out results-gpu.json        # needs a licence with GPU acceleration
python report.py results-gpu.json
```

On Modal, as the workflow does:

```bash
pip install -r requirements-ci.txt
modal token new
BENCH_GPU=L4 MANIFOLDBT_VERSION=0.28.1 MANIFOLDBT_CI_LICENSE=<key> \
    modal run modal_app.py --out results-gpu.json
```

In a fork, the workflow needs three repository secrets: `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` (a Modal token) and `MANIFOLDBT_CI_LICENSE`. Without the Modal
token it measures nothing and says so in a warning instead of failing.

## Files

- `bench.py` - the harness: calibration, the gates, the timings, one JSON
- `modal_app.py` - runs `bench.py` on a Modal GPU and hands back its JSON
- `report.py` - JSON to Markdown, and to the GitHub job summary
- `publish.py` - wraps a green run with its provenance into `results/latest.json`
- `test_bench.py` - guards for the gates, the report and the publisher; no GPU and
  no engine needed
- `requirements-lock.txt` - what the Modal image installs beside manifoldbt
- `requirements-ci.txt` - what the runner installs: the Modal client
