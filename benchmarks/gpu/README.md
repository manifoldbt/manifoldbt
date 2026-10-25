# GPU parameter sweeps

The sweeps job of the [CPU benchmark](../vs_vectorbt/README.md#parameter-sweeps)
runs on a GitHub-hosted runner, which has no GPU, so it can only ever measure the
CPU path of a sweep. This benchmark times the same sweeps on a GPU. The CPU number
they are read against stays the CPU job's, published from its runner.

Same sweeps means the same code, not a copy of it: the data generator, the seed,
the grid and the strategy are imported from `../vs_vectorbt` (`data.py`,
`sweep_child.py`), and the default points are the CPU job's three. Each point
records the digest of its data, and the report matches a GPU point with a CPU
point only when bars, combinations and digest all agree.

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
reserved cores included, so a run costs its wall time: about 30 seconds from
start to finish, image already built. The function's 15-minute timeout is the
ceiling on what a stuck run can cost.

## What the harness refuses to publish

- **A GPU number that was not computed on the GPU.** `device="cuda"` falls back to
  the CPU with a `UserWarning` when the card, the build or the strategy cannot
  take the sweep, so that a user's results are never affected. Here, any
  `UserWarning` raised during a GPU sweep fails the point. A machine where no card
  is visible fails before the first point.
- **A timing for different work.** Each point also runs once on the CPU of the GPU
  machine, untimed, and every cell of the GPU grid is compared with that CPU grid,
  keyed by its parameter pair, through the gate the CPU job applies to vectorbt:
  total return and maximum drawdown, worst cell against 1e-9 of capital. The
  engine documents the two paths as identical in fp64, so the result also counts
  the cells that are identical to the bit. Every timed repetition must reproduce
  the first call, cell for cell.
- **A timing measured across a licence change**, as in the CPU job. GPU
  acceleration is a Researcher feature, and the run activates the same CI licence
  as the CPU sweeps job.

## How it is timed

1. `calibrate_gpu()`, untimed: a few seconds that bring the card to its steady
   clocks and measure the costs the sweep uses to choose its execution path. The
   measured constants are kept in the result.
2. Per point, one untimed GPU call. It compiles the kernels, which is why it is
   reported on its own (`first_call_s`): it is what a user's first sweep costs,
   not what a sweep costs.
3. `--reps` timed GPU repetitions (7 by default, as in the CPU benchmark). Medians
   are published with min, max and IQR; a point whose IQR exceeds 15% of its
   median is flagged, as in the CPU benchmark. Seven rather than five because one
   slow repetition out of five is enough to cross that line: it did on the first
   Modal run.

The CPU parity run gets 4 reserved cores, as a hard limit, and is told so
(`max_parallelism`): the card is billed while it runs, so it should be short, and
a sweep that spawned one thread per visible core on a smaller quota would crawl.
What the container saw is recorded under `environment.cores`; on Modal it is the
4 cores reserved. The CPU model is not readable from inside Modal's sandbox, so it
is recorded as unknown. None of this is a CPU timing: there is none in the result.

## Reading the numbers

The report puts the GPU median next to the CPU job's published median for the same
grid, and divides one by the other only when both runs measured the same
manifoldbt version. The two come from different machines, an L4 on Modal and a
4-vCPU GitHub runner, and the ratio says exactly that: what the sweep costs on the
runner the CPU numbers are published from, against what it costs on the card. It
is not the speed of the card against the CPU next to it, which depends as much on
that CPU as on the card.

On the validation run (manifoldbt 0.28.1), the three default points took 32 ms,
107 ms and 223 ms on the L4, against 228 ms, 902 ms and 3.53 s in the CPU job's
published run of the same version: 7.1, 8.4 and 15.9 times faster on the card.
The published files are the reference; these are what the harness was validated
on.

Larger grids widen the gap, and a manual run can ask for one
(`sweep: 1000000:20000`). The CPU job has no number for it, so the report shows the
GPU alone, and the untimed CPU parity run then costs about a minute of the run.

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

Modal runs GPU functions only on an account with a payment method. In a fork, the
workflow needs three repository secrets: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
(a Modal token) and `MANIFOLDBT_CI_LICENSE`. Without the Modal token it measures
nothing and says so in a warning instead of failing.

## Files

- `bench.py` - the harness: calibration, the gates, the GPU timings, one JSON
- `modal_app.py` - runs `bench.py` on a Modal GPU and hands back its JSON
- `report.py` - JSON to Markdown, with the CPU job's numbers beside it, and to the
  GitHub job summary
- `publish.py` - wraps a green run with its provenance into `results/latest.json`
- `test_bench.py` - guards for the gates, the report and the publisher; no GPU and
  no engine needed
- `requirements-lock.txt` - what the Modal image installs beside manifoldbt
- `requirements-ci.txt` - what the runner installs: the Modal client
