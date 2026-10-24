"""Guards for the GPU harness, its report and its publisher.

They need no GPU and no engine installed (numpy and pandas only, for the data
generator the harness imports), so they run anywhere:

    python -m pytest benchmarks/gpu
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    # By path: ../vs_vectorbt has a bench.py and a report.py too, and importing
    # this bench.py puts that directory first on sys.path.
    spec = importlib.util.spec_from_file_location("gpu_" + name,
                                                  os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("bench")
report = _load("report")


def test_the_default_points_are_the_cpu_jobs():
    # The GPU numbers are read next to the CPU sweeps job's: same grids. If the
    # CPU workflow's points move, these must move with them.
    workflow = os.path.join(HERE, "..", "..", ".github", "workflows",
                            "bench-vs-vectorbt.yml")
    text = open(workflow, encoding="utf-8").read()
    for point in bench.DEFAULT_POINTS:
        assert point in text, point


@pytest.mark.parametrize("text, expected", [
    ("20000:5000", (20000, 5000)),
    ("20000:20000:oos", (20000, 20000)),
])
def test_parse_point(text, expected):
    assert bench.parse_point(text) == expected


@pytest.mark.parametrize("text", ["20000", "a:b", "0:10", "10:-1"])
def test_parse_point_refuses(text):
    with pytest.raises(Exception):
        bench.parse_point(text)


def test_cgroup_quota(tmp_path):
    path = tmp_path / "cpu.max"
    path.write_text("400000 100000\n")
    assert bench.cgroup_quota(str(path)) == 4.0
    path.write_text("max 100000\n")
    assert bench.cgroup_quota(str(path)) is None
    assert bench.cgroup_quota(str(tmp_path / "missing")) is None


def test_summarise_matches_the_cpu_benchmark():
    s = bench.summarise([1.0, 2.0, 3.0, 4.0])
    assert s["median_s"] == 2.5
    assert s["iqr_s"] == 2.0  # median of the upper half minus the lower half
    assert s["iqr_over_median"] == pytest.approx(0.8)
    assert bench.summarise([2.0])["iqr_s"] == 0.0


def test_identical_cells_counts_bits_not_closeness():
    ref = {(5, 20): (0.1, -0.2), (5, 25): (0.3, -0.1), (7, 20): (math.nan, 0.0)}
    other = {(5, 20): (0.1, -0.2), (5, 25): (0.3 + 1e-16, -0.1), (7, 20): (math.nan, 0.0)}
    assert bench.identical_cells(ref, other) == 2


def test_timed_keeps_user_warnings_only():
    import warnings

    def call():
        warnings.warn("ran on the CPU", UserWarning)
        warnings.warn("old API", DeprecationWarning)
        return {"x": 1}

    _, out, caught = bench.timed(call)
    assert out == {"x": 1}
    assert caught == [("UserWarning", "ran on the CPU")]


def _entry(**changes):
    entry = {
        "fallbacks": [],
        "tier_before": {"pro": True},
        "tier_after": {"pro": True},
        "parity": {"status": "exact"},
        "unstable_reps": {"gpu": 0, "cpu": 0},
    }
    entry.update(changes)
    return entry


@pytest.mark.parametrize("changes, reason", [
    ({"fallbacks": [{"category": "UserWarning", "message": "no CUDA device"}]},
     "fell back to the CPU"),
    ({"tier_after": {"pro": False}}, "tier changed"),
    ({"parity": {"status": "failed"}}, "differ"),
    ({"unstable_reps": {"gpu": 1, "cpu": 0}}, "did not reproduce"),
])
def test_verdict_fails(changes, reason):
    status, why = bench.verdict(_entry(**changes))
    assert status == "failed"
    assert reason in why


def test_verdict_passes():
    assert bench.verdict(_entry()) == ("exact", None)


SAMPLE = {
    "schema_version": 1,
    "status": "exact",
    "reps": 5,
    "environment": {
        "platform": "Modal L4, CPU side on 4 reserved cores",
        "cpu": "AMD EPYC", "cpu_threads": 4,
        "gpus": [{"name": "NVIDIA L4", "driver_version": "570.0",
                  "memory.total": "23034 MiB"}],
        "versions": {"manifoldbt": "0.28.1"},
    },
    "sweeps": [
        {"bars": 20000, "combos": 5000, "status": "exact", "reason": None,
         "parity": {"status": "exact", "identical_cells": 5000},
         "timings": {"seconds": {"gpu": 0.02, "cpu": 0.2}, "speedup": 10.0,
                     "per_combo_us": {"gpu": 4.0, "cpu": 40.0},
                     "first_call_s": {"gpu": 0.03, "cpu": 0.21}, "noisy": False}},
        {"bars": 200000, "requested_combos": 10000, "status": "failed",
         "reason": "RuntimeError: out of memory"},
    ],
}


def test_report_renders_both_kinds_of_row():
    text = report.render(SAMPLE)
    assert "NVIDIA L4" in text
    assert "| 20,000 | 5,000 | 20.0 ms | 0.200 s | x10.0 |" in text
    assert "5,000/5,000 bit-identical" in text
    assert "failed: RuntimeError: out of memory" in text


def test_publish_refuses_a_failed_result(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    failed = dict(SAMPLE, status="failed")
    (artifacts / "results-gpu.json").write_text(json.dumps(failed))
    out = tmp_path / "latest.json"
    run = subprocess.run([sys.executable, os.path.join(HERE, "publish.py"),
                          str(artifacts), "--out", str(out)], capture_output=True)
    assert run.returncode == 1
    assert not out.exists()


def test_publish_wraps_a_green_result(tmp_path):
    artifacts = tmp_path / "artifacts" / "bench-gpu"
    artifacts.mkdir(parents=True)
    (artifacts / "results-gpu.json").write_text(json.dumps(SAMPLE))
    out = tmp_path / "results" / "latest.json"
    env = dict(os.environ, GITHUB_RUN_ID="42", GITHUB_REPOSITORY="manifoldbt/manifoldbt")
    run = subprocess.run([sys.executable, os.path.join(HERE, "publish.py"),
                          str(tmp_path / "artifacts"), "--out", str(out)],
                         capture_output=True, env=env)
    assert run.returncode == 0, run.stderr
    published = json.loads(out.read_text())
    assert published["run_url"].endswith("/manifoldbt/manifoldbt/actions/runs/42")
    assert published["gpu"]["sweeps"][0]["combos"] == 5000
