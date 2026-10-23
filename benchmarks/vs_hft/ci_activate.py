"""Activate the benchmark licence from the environment, then check it by using it.

    MANIFOLDBT_CI_LICENSE=<signed key> python ci_activate.py

Used by every job of the HFT benchmark. Exits non-zero rather than let the
benchmark run without a working licence.

The key is read from the environment and nowhere else: it is not asked for, not
printed, and neither is the message `activate` returns (it carries the licensee's
address, and a public run log is not the place for it). `activate` writes the key
to disk, so the child processes the harness spawns afterwards pick it up without
ever seeing the secret.

What `license_info()` says is not evidence that a call will be accepted, so the
check is a call. The three entry points the benchmark depends on are invoked with
arguments that must fail anyway (a date long before any data exists, a file that is
not there). An answer of any other kind than a refusal means the call got past the
licence check and failed afterwards, which is what is expected here.
"""
from __future__ import annotations

import os
import sys
import tempfile

import manifoldbt as mbt


def main() -> int:
    key = (os.environ.get("MANIFOLDBT_CI_LICENSE") or "").strip()
    if not key:
        print("MANIFOLDBT_CI_LICENSE is empty or unset: refusing to run the benchmark without it.")
        return 1
    try:
        mbt.activate(key)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        print(f"activation failed: {type(exc).__name__}")
        return 1

    root = tempfile.mkdtemp(prefix="bench_check_")
    kw = dict(data_root=root, metadata_db=os.path.join(root, "m.sqlite"))
    checks = {
        "tape ingestion": lambda: mbt.ingest_trades("bybit", "BTCUSDT", symbol_id=1, start="1999-01-01",
                                                    end="1999-01-01", **kw),
        "book ingestion": lambda: mbt.ingest_book("bybit", "BTCUSDT", symbol_id=1, start="1999-01-01",
                                                  end="1999-01-01", levels=25, category="spot",
                                                  cache_dir=os.path.join(root, "c"), **kw),
        "order-by-order ingestion": lambda: mbt.ingest_mbo(os.path.join(root, "absent"), {"AAPL": 1},
                                                           "1999-01-01", **kw),
    }
    refused = []
    for name, call in checks.items():
        try:
            call()
        except PermissionError:
            refused.append(name)
        except Exception:                          # noqa: BLE001 - failed after the check, as intended
            pass
    if refused:
        print("licence check failed at: " + ", ".join(refused))
        return 1
    print("licence check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
