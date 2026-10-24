"""Run bench.py on a Modal GPU, from the workflow or from a laptop.

    BENCH_GPU=L4 MANIFOLDBT_VERSION=0.28.1 MANIFOLDBT_CI_LICENSE=<key> \\
        modal run modal_app.py --out results-gpu.json

GitHub's hosted runners have no GPU, and its GPU runners are billed by the minute
on paid plans only, so the card comes from Modal. Modal bills by the second for
the life of the container, card and reserved cores included, and nothing while no
container runs, so a run costs its wall time: a few minutes of an L4, which is a
few cents. The function's timeout below is the ceiling on that cost.

The image is the published wheel from PyPI with its ``[gpu]`` extra (the NVIDIA
runtime compiler that compiles the kernels; the driver comes with the card), plus
the pinned libraries the harness reads. The version is always pinned, never
"latest": Modal caches an image by its definition, so an unpinned install would
keep serving whichever version it resolved the first time.

The licence travels as a Modal secret built from this process's environment for
the length of the run, so it is never written into the image. Inside the
container ``../vs_vectorbt/ci_activate.py`` activates it, exactly as the CPU
sweeps job does, before ``bench.py`` runs.
"""
from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
REMOTE = "/root/benchmarks"

# Cores reserved for the CPU side of the comparison, as both the request and the
# hard limit, so the CPU number is not a measure of how idle the host was. Four,
# like the GitHub runner the CPU sweeps job runs on.
CPU_CORES = 4
# Room for the 1M-bar point; the sweeps themselves stay well under it.
MEMORY_MIB = 8192
# The whole run, image pull included, measured at a few minutes. The ceiling on
# what a stuck run can cost.
TIMEOUT_S = 15 * 60
# The card is read when the function is declared, not passed at call time: Modal
# checks the declared card when the app starts, so a function declared on an L4
# and called `with_options(gpu="T4")` still fails wherever an L4 cannot be had.
# Passed into the container too, so both sides declare the same function.
GPU = os.environ.get("BENCH_GPU", "").strip() or "L4"

VERSION = os.environ.get("MANIFOLDBT_VERSION", "").strip()
if modal.is_local() and not VERSION:
    raise SystemExit(
        "MANIFOLDBT_VERSION is empty: name the manifoldbt version to benchmark "
        "(the image pins it, see the module docstring)."
    )

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "manifoldbt[gpu]==" + VERSION if VERSION else "manifoldbt[gpu]",
        requirements=[str(HERE / "requirements-lock.txt")],
    )
    # Mounted when the container starts rather than baked into the image, so an
    # edit to the harness does not rebuild it.
    .add_local_dir(BENCHMARKS / "vs_vectorbt", REMOTE + "/vs_vectorbt",
                   ignore=["results", "**/__pycache__", "*.json"])
    .add_local_dir(HERE, REMOTE + "/gpu",
                   ignore=["results", "**/__pycache__", "*.json"])
)

app = modal.App("manifoldbt-gpu-bench")


@app.function(
    image=image,
    gpu=GPU,
    env={"BENCH_GPU": GPU},
    cpu=(CPU_CORES, CPU_CORES),
    memory=MEMORY_MIB,
    timeout=TIMEOUT_S,
    # A benchmark that silently ran twice is not a measurement of one run.
    retries=0,
    secrets=[modal.Secret.from_local_environ(["MANIFOLDBT_CI_LICENSE"])],
)
def measure(argv: list) -> dict:
    """Activate the licence, run the harness, hand back its exit code and JSON."""
    import subprocess
    import sys

    gpu_dir = REMOTE + "/gpu"
    activation = subprocess.run(
        [sys.executable, REMOTE + "/vs_vectorbt/ci_activate.py"], cwd=gpu_dir)
    if activation.returncode != 0:
        return {"stage": "activation", "exit_code": activation.returncode, "result": None}

    out = "/tmp/results-gpu.json"
    run = subprocess.run([sys.executable, "bench.py", *argv, "--out", out], cwd=gpu_dir)
    result = pathlib.Path(out).read_text(encoding="utf-8") if os.path.exists(out) else None
    return {"stage": "bench", "exit_code": run.returncode, "result": result}


@app.local_entrypoint()
def main(reps: int = 7, sweep: str = "", out: str = "results-gpu.json") -> None:
    argv = [
        "--reps", str(reps),
        "--cpu-threads", str(CPU_CORES),
        "--platform", "Modal {}, CPU side on {} reserved cores".format(GPU, CPU_CORES),
    ]
    if sweep.strip():
        argv += ["--sweep", *sweep.split()]
    reply = measure.remote(argv)
    if reply["result"] is not None:
        pathlib.Path(out).write_text(reply["result"], encoding="utf-8")
        print("wrote " + out)
    if reply["exit_code"] != 0:
        raise SystemExit("the {} step failed in the container (exit {})".format(
            reply["stage"], reply["exit_code"]))
