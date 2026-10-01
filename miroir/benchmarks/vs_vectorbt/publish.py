"""Merge a run's result artifacts into the one file the website reads.

    python publish.py <artifact-dir> --out results/latest.json

An artifact is not a publication. It needs a token to download, it expires
after ninety days, and nothing outside GitHub can link to it, so a number that
only exists as an artifact is a number nobody can check. This writes the same
measurements to a path in the repository instead: fetchable by anyone, over
plain HTTPS, for as long as the repository exists.

The two jobs produce two payloads with two environments, because they run on two
runners. They are stored side by side rather than merged into one table: a
timing measured on one machine and a timing measured on another are not rows of
the same table, and pretending otherwise is how a benchmark starts lying
quietly. Provenance goes on top, from the workflow environment, so the file
names the run it came from without anyone having to trust the filename.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# artifact file name -> key in the published file.
SOURCES = {
    "results-ubuntu-latest.json": "backtests",
    "results-sweeps.json": "sweeps",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", help="directory the artifacts were downloaded into")
    parser.add_argument("--out", required=True, help="file to write")
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
        # This job only runs when both measuring jobs came back green, so the
        # conclusion is not read from anywhere: it is the precondition.
        "conclusion": "success",
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }

    root = Path(args.artifacts)
    for name, key in SOURCES.items():
        # download-artifact with merge-multiple flattens the two artifacts into
        # one directory, but a plain download nests each under its own name.
        # Take whichever layout the caller produced.
        found = next(iter(sorted(root.rglob(name))), None)
        if found is None:
            print("! {}: no {} under {}".format(key, name, root), file=sys.stderr)
            continue
        payload = json.loads(found.read_text(encoding="utf-8"))
        if payload.get("schema_version", 1) < 2:
            raise SystemExit(
                "{} is schema {}, expected 2 or later".format(
                    found, payload.get("schema_version", 1))
            )
        published[key] = payload

    if "backtests" not in published and "sweeps" not in published:
        raise SystemExit("neither artifact was found: nothing to publish")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    print("wrote {} ({:.0f} KB)".format(out, out.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
