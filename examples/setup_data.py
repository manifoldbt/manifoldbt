"""Populate the shared data store the examples read. Not an example itself.

Twenty of the files here are marked `Data: shared store`: they read real
market data from `data/` + `metadata/`, which are not in the repository
because market data does not belong in a git repository. This script
downloads what those files name, once.

    python examples/setup_data.py            # everything, ~10 symbols
    python examples/setup_data.py BTC        # just the symbols an ID matches

Everything is ingested at 1h. The engine aggregates upwards, so one 1h store
serves every example here, including the ones running on 12h or daily bars.

The connectors used (binance, dydx) are free on all tiers and need no API key.
Re-running is safe: an interval already present is not downloaded twice.

Usage:
    python examples/setup_data.py
"""
import sys

import manifoldbt as mbt

from _bootstrap import DATA_ROOT, METADATA_DB

# Every symbol the `Data: shared store` examples name, with the id it gets in
# the store. The ids only have to be unique -- the examples address symbols by
# name (`"BTC-USDT:perp"`), not by id.
#
# `asset_class="crypto_perp"` is the part that is easy to miss: the examples
# ask for `BTC-USDT:perp`, and an ingest left on the default `crypto_spot`
# stores a symbol they will not find.
BINANCE_PERPS = [
    ("BTCUSDT", 1),
    ("ETHUSDT", 2),
    ("LTCUSDT", 3),
    ("BNBUSDT", 4),
    ("DOTUSDT", 5),
    ("XRPUSDT", 6),
    ("ADAUSDT", 7),
    ("LINKUSDT", 8),
    ("DOGEUSDT", 9),
    ("AVAXUSDT", 10),
]

# 15_cross_exchange executes on dydx and takes its signal from binance. dydx
# v4 history starts in 2024, which is why this one has its own start date.
DYDX_PERPS = [("BTC-USD", 11)]

START = "2021-01-01T00:00:00Z"
DYDX_START = "2024-02-01T00:00:00Z"
END = "2026-03-01T00:00:00Z"
INTERVAL = "1h"


def _ingest(provider, symbol, symbol_id, start):
    try:
        mbt.ingest(
            provider=provider,
            symbol=symbol,
            symbol_id=symbol_id,
            start=start,
            end=END,
            interval=INTERVAL,
            asset_class="crypto_perp",
            data_root=DATA_ROOT,
            metadata_db=METADATA_DB,
        )
        return True
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        print(f"  {provider}:{symbol} FAILED -- {exc}")
        return False


def main(argv):
    wanted = [a.upper() for a in argv]
    jobs = [("binance", s, i, START) for s, i in BINANCE_PERPS]
    jobs += [("dydx", s, i, DYDX_START) for s, i in DYDX_PERPS]
    if wanted:
        jobs = [j for j in jobs if any(w in j[1].upper() for w in wanted)]
        if not jobs:
            raise SystemExit(f"No symbol matches {', '.join(wanted)}")

    print(f"Ingesting {len(jobs)} symbols at {INTERVAL} into {DATA_ROOT}\n")
    ok = sum(_ingest(*job) for job in jobs)

    print(f"\n{ok}/{len(jobs)} symbols in the store.")
    if ok:
        print("The `Data: shared store` examples can run now, e.g.")
        print("    python examples/01_trend_following.py")
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
