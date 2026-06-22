"""manifoldbt CLI — data ingestion from the command line.

Usage:
    manifoldbt ingest --provider binance --symbol BTCUSDT --symbol-id 1 \
        --start 2025-01-01T00:00:00Z --end 2025-01-31T00:00:00Z

    manifoldbt ingest --provider databento --dataset GLBX.MDP3 --symbol ESH5 \
        --symbol-id 1 --start 2025-01-01T00:00:00Z --end 2025-01-31T00:00:00Z \
        --exchange CME --asset-class future
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="manifoldbt", description="ManifoldBT CLI")
    sub = parser.add_subparsers(dest="command")

    # ── ingest ────────────────────────────────────────────────────────────
    ing = sub.add_parser("ingest", help="Ingest bars from a data provider")
    ing.add_argument("--provider", required=True, help="binance | bybit | hyperliquid | databento")
    ing.add_argument("--symbol", required=True, help="e.g. BTCUSDT, ESH5")
    ing.add_argument("--symbol-id", required=True, type=int, help="Unique integer ID for this symbol")
    ing.add_argument("--start", required=True, help="RFC3339 start (e.g. 2025-01-01T00:00:00Z)")
    ing.add_argument("--end", required=True, help="RFC3339 end")
    ing.add_argument("--interval", default="1m", help="Bar interval: 1s, 1m, 5m, 15m, 1h, 1d (default: 1m)")
    ing.add_argument("--dataset", default=None, help="Databento dataset (e.g. GLBX.MDP3)")
    ing.add_argument("--data-root", default="data", help="Parquet store directory (default: data)")
    ing.add_argument("--metadata-db", default="metadata/metadata.sqlite", help="Metadata SQLite path")
    ing.add_argument("--exchange", default=None, help="Exchange name for metadata")
    ing.add_argument("--asset-class", default="crypto_spot",
                     help="crypto_spot | crypto_perp | equity | future | option | forex | index")
    ing.add_argument("--license-key", default=None, help="Pro license key (or set via bt.activate())")

    # ── activate ──────────────────────────────────────────────────────────
    act = sub.add_parser("activate", help="Activate a Pro license key")
    act.add_argument("key", nargs="?", default=None,
                     help="Pro license key. If omitted, reads from stdin.")

    # ── import-csv ────────────────────────────────────────────────────────
    csv = sub.add_parser("import-csv", help="Import bars from a CSV file (standard/MT4/MT5)")
    csv.add_argument("csv_file", help="Path to the CSV file")
    csv.add_argument("--symbol", required=True, help="e.g. EURUSD, BTCUSDT")
    csv.add_argument("--symbol-id", required=True, type=int, help="Unique integer ID for this symbol")
    csv.add_argument("--interval", default="1m", help="Bar interval of the rows (default: 1m)")
    csv.add_argument("--data-root", default="data", help="Store directory (default: data)")
    csv.add_argument("--metadata-db", default="metadata/metadata.sqlite", help="Metadata SQLite path")
    csv.add_argument("--exchange", default="CSV", help="Exchange label for metadata (default: CSV)")
    csv.add_argument("--asset-class", default="crypto_spot",
                     help="crypto_spot | crypto_perp | equity | future | option | forex | index")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "activate":
        _cmd_activate(args)
    elif args.command == "import-csv":
        _cmd_import_csv(args)


def _cmd_activate(args: argparse.Namespace) -> None:
    import manifoldbt as bt

    key = args.key
    if not key:
        key = sys.stdin.read().strip()
    if not key:
        print("Error: missing license key", file=sys.stderr)
        sys.exit(1)

    try:
        bt.activate(key)
    except Exception as e:
        print(f"Activation failed: {e}", file=sys.stderr)
        sys.exit(1)

    tier, email = bt.license_info()
    if tier == "Pro":
        print(f"Activated: {tier}" + (f" ({email})" if email else ""))
    else:
        print("Activation did not produce a Pro license.", file=sys.stderr)
        sys.exit(1)


def _cmd_ingest(args: argparse.Namespace) -> None:
    import manifoldbt as bt

    if args.license_key:
        try:
            bt.activate(args.license_key)
        except Exception as e:
            print(f"License activation failed: {e}", file=sys.stderr)
            sys.exit(1)

    from manifoldbt.exceptions import LicenseError

    try:
        store = bt.ingest(
            provider=args.provider,
            symbol=args.symbol,
            symbol_id=args.symbol_id,
            start=args.start,
            end=args.end,
            interval=args.interval,
            dataset=args.dataset,
            data_root=args.data_root,
            metadata_db=args.metadata_db,
            exchange=args.exchange,
            asset_class=args.asset_class,
        )
        print(f"Ingested to {store.data_root()} (symbols: {store.list_symbols()})")
    except LicenseError:
        # Message already handled by _warn_pro atexit summary
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_import_csv(args: argparse.Namespace) -> None:
    import manifoldbt as bt

    try:
        store = bt.import_csv(
            args.csv_file,
            symbol=args.symbol,
            symbol_id=args.symbol_id,
            interval=args.interval,
            data_root=args.data_root,
            metadata_db=args.metadata_db,
            exchange=args.exchange,
            asset_class=args.asset_class,
        )
        print(f"Imported {args.csv_file} to {store.data_root()} (symbols: {store.list_symbols()})")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
