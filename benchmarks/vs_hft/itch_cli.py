"""Download a Nasdaq TotalView-ITCH 5.0 day file, check it against its pinned
fingerprint, keep the messages of a few symbols, delete the day file.

    python itch_cli.py --day 2025-11-28 --symbols AAPL --out <dir>

The day file is several gigabytes compressed. It is read once, as it inflates;
what is kept is the extract of each symbol (tens of megabytes) in ITCH framing,
the format `manifoldbt.ingest_mbo` takes and the one `itch.py` decodes for the
other engines.
"""
import argparse
import json
import os
import sys
import time
from urllib.request import Request, urlopen

import data
import itch
import spec

URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/S{mm}{dd}{yy}-v50.txt.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=spec.MBOJoin.DAY)
    ap.add_argument("--symbols", nargs="+", default=[spec.MBOJoin.SYMBOL])
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    y, m, d = args.day.split("-")
    url = URL.format(mm=m, dd=d, yy=y[2:])
    pin = spec.NASDAQ_DAYS.get(args.day)
    if pin is None:
        sys.exit(f"{args.day} is not a pinned Nasdaq day (spec.NASDAQ_DAYS)")
    raw_dir = os.path.join(args.out, "raw-" + args.day)
    os.makedirs(raw_dir, exist_ok=True)
    gz = os.path.join(raw_dir, url.rsplit("/", 1)[1])
    t0 = time.perf_counter()
    if not os.path.exists(gz):
        req = Request(url, headers={"User-Agent": "manifoldbt-bench/1"})
        tmp = gz + ".part"
        with urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, gz)
    t_dl = time.perf_counter() - t0
    t0 = time.perf_counter()
    sha = data.sha256_file(gz)
    t_sha = time.perf_counter() - t0
    if sha != pin["sha256"]:
        sys.exit(f"fingerprint mismatch for {gz}: got {sha}, pinned {pin['sha256']}")
    stats = itch.extract(gz, args.symbols, args.out)
    stats.update({"url": url, "download_s": round(t_dl, 1), "sha256": sha, "sha256_s": round(t_sha, 1)})
    if not args.keep_raw:
        os.remove(gz)
        os.rmdir(raw_dir)
    with open(os.path.join(args.out, "extract.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
