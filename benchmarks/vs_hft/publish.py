"""Merge a run's result artifacts into the one file the website reads.

    python publish.py <artifact-dir> --out results/latest.json

An artifact is not a publication: it needs a token to download, expires after
ninety days, and nothing outside GitHub can link to it. This writes the same
measurements to a path in the repository instead, fetchable by anyone over plain
HTTPS. The three jobs run on three runners and are stored side by side rather
than merged into one table: a timing taken on one machine and a timing taken on
another are not rows of the same table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "results-books.json": "books",
    "results-orders.json": "orders",
    "results-orders-exact-replay.json": "orders_exact_replay",
    "results-sweeps-maker.json": "sweeps_maker",
    "results-sweeps-latency.json": "sweeps_latency",
    "results-sweeps-taker.json": "sweeps_taker",
    "results-sweeps-mbt-maker.json": "sweeps_mbt_maker",
    "results-sweeps-mbt-latency.json": "sweeps_mbt_latency",
    "results-sweeps-mbt-taker.json": "sweeps_mbt_taker",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "manifoldbt/manifoldbt")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    published = {
        "repo": repo,
        "run_id": run_id,
        "run_url": "{}/{}/actions/runs/{}".format(server, repo, run_id) if run_id else "",
        "commit": os.environ.get("GITHUB_SHA", ""),
        "run_started_at": os.environ.get("GITHUB_RUN_STARTED_AT", ""),
        "conclusion": "success",
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    root = Path(args.artifacts)
    for name, key in SOURCES.items():
        found = next(iter(sorted(root.rglob(name))), None)
        if found is None:
            print("! {}: no {} under {}".format(key, name, root), file=sys.stderr)
            continue
        payload = json.loads(found.read_text(encoding="utf-8"))
        if payload.get("schema_version", 1) < 3:
            raise SystemExit("{} is schema {}, expected 3 or later".format(found, payload.get("schema_version", 1)))
        published[key] = payload
    if not any(k in published for k in SOURCES.values()):
        raise SystemExit("no artifact was found: nothing to publish")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    print("wrote {} ({:.0f} KB)".format(out, out.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
