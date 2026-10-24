"""Wrap a GPU run's result with its provenance, as the file the website reads.

    python publish.py <artifact-dir> --out results/latest.json

Same contract as ../vs_vectorbt/publish.py: the numbers are committed to the
repository rather than left in an artifact, which needs a token to download and
expires after ninety days.

The GPU result is a file of its own rather than a third key in the CPU file. It
comes from another machine, in another job, and a card that could not be had
must not hold back the CPU numbers, nor the reverse.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "results-gpu.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", help="directory the artifact was downloaded into")
    parser.add_argument("--out", required=True, help="file to write")
    args = parser.parse_args()

    found = next(iter(sorted(Path(args.artifacts).rglob(SOURCE))), None)
    if found is None:
        print("no {} under {}".format(SOURCE, args.artifacts), file=sys.stderr)
        return 1
    payload = json.loads(found.read_text(encoding="utf-8"))
    # The publish job only runs after a green measuring job, and the harness
    # exits non-zero on any failed point. Checked again here because this file
    # is what the website renders.
    if payload.get("status") != "exact":
        print("refusing to publish a result whose status is {!r}".format(
            payload.get("status")), file=sys.stderr)
        return 1

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
        "gpu": payload,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    print("wrote {} ({:.0f} KB)".format(out, out.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
