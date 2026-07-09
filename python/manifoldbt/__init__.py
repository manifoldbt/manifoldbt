"""manifoldbt: Fast research backtesting with Rust core + Python DSL."""
import copy
import json
from typing import Any, Dict, List, Optional, Tuple

import importlib as _importlib

from manifoldbt._native import (
    BacktestResult,
    BatchResultLite,
    DataStore,
    activate,
    license_info as _license_info,
    compile_strategy_json,
    run as _run_native,
    run_batch as _run_batch_native,
    run_batch_lite as _run_batch_lite_native,
    run_json,
    run_sweep as _run_sweep_native,
    run_sweep_lite as _run_sweep_lite_native,
    run_with_parquet,
    py_run_walk_forward as _run_walk_forward_native,
    py_run_sweep_2d as _run_sweep_2d_native,
    py_run_stability as _run_stability_native,
    py_replay as _replay_native,
    py_run_monte_carlo,
    py_run_stochastic as _run_stochastic_native,
    run_portfolio as _run_portfolio_native,
    py_ingest as _ingest_native,
    py_import_csv as _import_csv_native,
)
from manifoldbt._serde import scalar_value_to_json
from manifoldbt.config import (
    BacktestConfig,
    ExecutionConfig,
    FeeConfig,
    OrderConfig,
    VenueFees,
    resolve_universe,
)
from manifoldbt.exceptions import (
    BacktesterError,
    ConfigError,
    DataError,
    LicenseError,
    StrategyError,
)
from manifoldbt.expr import AssetRef, Expr, TimeframeRef, asset, col, exo, hold, lit, param, s, scan, symbol_ref, tf, when
from manifoldbt.helpers import (
    ExecutionPrice,
    FillModel,
    Interval,
    Slippage,
    date_to_ns,
    time_range,
)
from manifoldbt.portfolio import Portfolio
from manifoldbt.result import Result
from manifoldbt.strategy import Strategy
from manifoldbt.sweep import SweepResult
from manifoldbt import indicators

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("manifoldbt")
except Exception:
    __version__ = "0.1.0"

# ---------------------------------------------------------------------------
# License banner
# ---------------------------------------------------------------------------
def _print_banner():
    try:
        tier, email = _license_info()
        if tier == "Pro" and email:
            print(f"manifoldbt v{__version__} | \033[38;5;214mPro\033[0m | {email}")
        else:
            print(f"manifoldbt v{__version__} | \033[36mCommunity\033[0m | upgrade: www.manifoldbt.com")
    except Exception:
        print(f"manifoldbt v{__version__} | \033[36mCommunity\033[0m | upgrade: www.manifoldbt.com")

_print_banner()
del _print_banner


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_pro_warnings: list = []


def _warn_pro(msg: str) -> None:
    """Collect a Pro feature warning (printed at exit)."""
    if msg not in _pro_warnings:
        _pro_warnings.append(msg)


def _print_pro_summary() -> None:
    """Print collected Pro warnings at exit."""
    if _pro_warnings:
        print()
        for w in _pro_warnings:
            print(f"\033[38;5;214m[!] {w} -- Pro feature\033[0m")
        print("\033[38;5;214m  -> upgrade at www.manifoldbt.com\033[0m")


import atexit
atexit.register(_print_pro_summary)


def license_info() -> tuple:
    """Get license info: (tier, email). tier is "Pro" or "Community", email is str or None."""
    return _license_info()


def _is_pro() -> bool:
    """Check if current license is Pro."""
    try:
        tier, _ = _license_info()
        return tier == "Pro"
    except Exception:
        return False


def _require_pro(feature: str) -> None:
    """Raise LicenseError if the current license is not Pro.

    This used to ``raise SystemExit(0)``, which reads as a clean exit in a
    ``.py`` script but, in Jupyter/IPython, aborts the current cell with a bare
    ``SystemExit: 0`` (plus a spurious "To exit, use ..." warning) and silently
    skips the rest of the cell. ``LicenseError`` is a normal, catchable
    exception: a single clean traceback in a notebook, a real error in scripts.
    """
    if _is_pro():
        return
    raise LicenseError(
        f"'{feature}' is a Pro feature. Upgrade to Pro at www.manifoldbt.com"
    )


def _require_pro_for_gpu(device, feature: str) -> None:
    """Gate GPU acceleration (``device="cuda"``/``"gpu"``) behind Pro.

    GPU paths are also enforced natively, but that surfaces a ``PermissionError``
    with a full traceback (GPU sweep) or a bare ``ValueError`` (stochastic). Gating
    in Python first gives every GPU entry point the same clean ``LicenseError`` as
    the other Pro features. No-op for CPU or for Pro users.
    """
    if isinstance(device, str) and device.lower() in ("cuda", "gpu"):
        _require_pro(feature)


# Community fan-out budget: sweeps and batches may run up to this many backtests
# per call for free; beyond it requires Pro. Single run() is never affected.
# Keep in sync with the native bt_license::COMMUNITY_MAX_SWEEP_COMBOS.
_COMMUNITY_MAX_COMBOS = 500


def _grid_combos(param_grid) -> int:
    """Number of Cartesian combinations produced by a sweep param grid."""
    n = 1
    for values in param_grid.values():
        n *= max(1, len(values))
    return n


def _require_pro_over_combos(n_combos: int, what: str) -> None:
    """Raise LicenseError if a fan-out exceeds the Community combination limit.

    No-op at or below the limit, or for Pro users. Mirrors the native
    ``require_combo_limit`` so Community and Pro see identical behaviour.
    """
    if n_combos <= _COMMUNITY_MAX_COMBOS or _is_pro():
        return
    raise LicenseError(
        f"{what} with {n_combos} runs exceeds the Community limit of "
        f"{_COMMUNITY_MAX_COMBOS}. Upgrade to Pro at www.manifoldbt.com"
    )


def _classify_error(exc: Exception) -> Exception:
    """Wrap a Rust ValueError/RuntimeError in a more specific exception."""
    msg = str(exc)
    if any(kw in msg for kw in ("data", "parquet", "partition", "store", "version", "symbol")):
        return DataError(msg)
    if any(kw in msg for kw in ("strategy", "signal", "compile", "expression", "type")):
        return StrategyError(msg)
    if any(kw in msg for kw in ("config", "interval", "universe", "time_range")):
        return ConfigError(msg)
    return BacktesterError(msg)


# ---------------------------------------------------------------------------
# Config preparation (symbol resolution + strategy orders merge)
# ---------------------------------------------------------------------------

_AC_SUFFIX_MAP = {
    "spot": "CryptoSpot", "perp": "CryptoPerpetual",
    "future": "Future", "equity": "Equity",
    "option": "EquityOption", "fx": "Forex",
    "index": "Index",
}

# Symbol-name resolution is a pure function of (metadata_db, provider, name):
# SymbolIds are static once registered, so the (name→id) mapping never changes
# for a given metadata DB within a process. Every run()/run_sweep() call used to
# re-resolve — opening a fresh sqlite3 connection per symbol (~0.27ms each, i.e.
# the dominant slice of the per-call Python floor, and ~Nx that for an N-symbol
# universe). Memoising it collapses that to a dict hit. The DB path is part of
# the key so two stores on different metadata DBs never collide.
_RESOLVE_CACHE: Dict[Tuple[Any, str, str], int] = {}


def _resolve_normalized(sym: str, provider: str, store) -> int:
    """Resolve a normalized symbol name like 'BTC-USDT:perp' on a provider to SymbolId.

    Tries: 1) normalized parse → metadata lookup by (base, quote, asset_class, provider)
           2) fallback to raw ticker match

    Result is memoised per (metadata_db, provider, name) — see ``_RESOLVE_CACHE``.
    """
    import sqlite3

    try:
        meta_db = store.metadata_db()
    except Exception:
        meta_db = None

    ckey = (meta_db, provider, sym) if meta_db is not None else None
    if ckey is not None:
        cached = _RESOLVE_CACHE.get(ckey)
        if cached is not None:
            return cached

    # Parse normalized name: "BTC-USDT:perp" → base=BTC, quote=USDT, ac=CryptoPerpetual
    if ":" in sym:
        pair, suffix = sym.rsplit(":", 1)
        ac_db = _AC_SUFFIX_MAP.get(suffix)
    else:
        pair, ac_db = sym, None

    if "-" in pair:
        base, quote = pair.split("-", 1)
    else:
        base, quote = pair, ""

    resolved = None
    if ac_db and meta_db is not None:
        # Try metadata lookup by (base, quote, asset_class, provider)
        conn = sqlite3.connect(meta_db)
        row = conn.execute(
            "SELECT id FROM symbols WHERE base_currency=? COLLATE NOCASE "
            "AND quote_currency=? COLLATE NOCASE AND asset_class=? "
            "AND exchange=? COLLATE NOCASE ORDER BY id DESC LIMIT 1",
            (base, quote, ac_db, provider.upper()),
        ).fetchone()
        conn.close()
        if row:
            resolved = row[0]

    if resolved is None:
        # Fallback: try raw ticker match
        try:
            resolved = store.resolve_symbol(sym)
        except Exception:
            raise ValueError(
                f"Symbol '{sym}' not found on provider '{provider}'. "
                f"Searched: base={base}, quote={quote}, class={ac_db}"
            )

    if ckey is not None:
        _RESOLVE_CACHE[ckey] = resolved
    return resolved


def _resolve_source_dict(source, store):
    """Resolve a signal/execution source dict → list of (provider, norm_sym, symbol_id, raw_ticker).

    Returns the raw ticker from metadata (what the files are named on disk).
    """
    if isinstance(source, dict):
        import sqlite3
        conn = sqlite3.connect(store.metadata_db())
        resolved = []
        for provider, symbols in source.items():
            for sym in symbols:
                sid = _resolve_normalized(sym, provider, store)
                # Get raw ticker from metadata
                row = conn.execute("SELECT ticker FROM symbols WHERE id=?", (sid,)).fetchone()
                raw_ticker = row[0] if row else sym
                resolved.append((provider, sym, sid, raw_ticker))
        conn.close()
        return resolved
    return None


# _prepare_config() deepcopies the user's config (so it is never mutated) and
# re-resolves every name on each call. Both are pure functions of the config
# CONTENT, the strategy's order overrides and the store's metadata DB, so the
# prepared JSON is memoised on that content fingerprint — same pattern as
# _RESOLVE_CACHE (content keys, never object identity/heap address). The
# deepcopy alone is ~75us per call, the dominant slice of the per-call Python
# floor on small backtests.
_PREPARED_CFG_CACHE: Dict[Tuple[str, str, Any], str] = {}
_PREPARED_CFG_CACHE_MAX = 256


def _prepared_config_json(config: BacktestConfig, strategy, store: DataStore) -> str:
    """Content-memoised equivalent of ``_prepare_config(...).to_json()``."""
    try:
        meta_db = store.metadata_db()
    except Exception:
        meta_db = None
    if meta_db is None:
        return _prepare_config(config, strategy, store).to_json()

    orders = getattr(strategy, "_orders", None) if strategy is not None else None
    try:
        orders_key = json.dumps(orders, sort_keys=True, default=str) if orders else ""
        key = (config.to_json(), orders_key, meta_db)
    except (TypeError, ValueError):
        # Unserialisable config content — skip memoisation, never fail.
        return _prepare_config(config, strategy, store).to_json()

    cached = _PREPARED_CFG_CACHE.get(key)
    if cached is None:
        cached = _prepare_config(config, strategy, store).to_json()
        if len(_PREPARED_CFG_CACHE) >= _PREPARED_CFG_CACHE_MAX:
            _PREPARED_CFG_CACHE.clear()
        _PREPARED_CFG_CACHE[key] = cached
    return cached


def _prepare_config(config: BacktestConfig, strategy, store: DataStore) -> BacktestConfig:
    """Prepare config for execution: resolve symbols, convert deprecated fields."""
    cfg = copy.deepcopy(config)

    # --- Dict universe: {"binance": ["BTC-USDT:perp"], "onchain": ["hashrate"]} ---
    if isinstance(cfg.universe, dict):
        # Cross-exchange (multiple providers) is a Pro feature.
        if len(cfg.universe) > 1:
            _require_pro("Cross-exchange backtesting")

        resolved_universe = []
        qualified_names = {}  # "binance:BTC-USDT:perp" → SymbolId

        for provider, symbols in cfg.universe.items():
            for sym in symbols:
                sid = _resolve_normalized(sym, provider, store)
                resolved_universe.append(sid)
                qualified = f"{provider}:{sym}"
                qualified_names[qualified] = sid

        cfg.universe = resolved_universe
        cfg.symbol_names = qualified_names

        # Clear deprecated fields
        cfg.signal_source = None
        cfg.execution_source = None
        cfg.pair_map = {}
        cfg.exo_sources = {}
        cfg.provider = None

    # --- Legacy list universe: [1, 2, 3] or ["BTC-USD", "ETH-USD"] ---
    elif cfg.universe:
        if any(isinstance(s, str) for s in cfg.universe):
            cfg.universe = resolve_universe(cfg.universe, store, cfg.symbol_names)

        # Legacy exo_sources resolution
        if cfg.exo_sources and any(isinstance(k, str) for k in cfg.exo_sources):
            resolved = {}
            for key, val in cfg.exo_sources.items():
                sid = store.resolve_symbol(key) if isinstance(key, str) else key
                resolved[sid] = val
            cfg.exo_sources = resolved

        if cfg.provider and not cfg.signal_source:
            cfg.signal_source = cfg.provider

    # --- Resolve per-venue fee mapping: symbol_venue keys may be symbol names ---
    # Users key symbol_venue by name (e.g. "dydx:BTC-USD:perp" or "BTC-USDT:perp")
    # for ergonomics; the engine needs integer SymbolIds. Resolve them here using
    # the same name→id mapping as the universe.
    fees = getattr(cfg, "fees", None)
    if fees is not None and getattr(fees, "symbol_venue", None):
        resolved_sv = {}
        for key, venue in fees.symbol_venue.items():
            if isinstance(key, int):
                resolved_sv[key] = venue
            elif cfg.symbol_names and key in cfg.symbol_names:
                resolved_sv[int(cfg.symbol_names[key])] = venue
            else:
                resolved_sv[int(store.resolve_symbol(key))] = venue
        fees.symbol_venue = resolved_sv

    # Merge orders from strategy into execution config
    if strategy and hasattr(strategy, '_orders') and strategy._orders:
        if cfg.execution.orders is None:
            cfg.execution.orders = OrderConfig()
        for key, val in strategy._orders.items():
            setattr(cfg.execution.orders, key, val)

    return cfg


def _is_sub_daily(res: Any) -> bool:
    """Return True if an Interval dict represents sub-daily resolution."""
    if not isinstance(res, dict):
        return False
    if "Seconds" in res or "Minutes" in res:
        return True
    if "Hours" in res and res["Hours"] < 24:
        return True
    return False


def _interval_to_seconds(interval: Any) -> int:
    """Convert an Interval dict to total seconds."""
    if not isinstance(interval, dict):
        return 0
    if "Seconds" in interval:
        return interval["Seconds"]
    if "Minutes" in interval:
        return interval["Minutes"] * 60
    if "Hours" in interval:
        return interval["Hours"] * 3600
    if "Days" in interval:
        return interval["Days"] * 86400
    return 0


def _dataset_for_interval(interval: Any) -> str:
    """Map a bar interval to the best matching dataset (<= interval).

    Available: bars_1m (60s), bars_15m (900s), bars_1h (3600s), bars_1d (86400s).
    """
    secs = _interval_to_seconds(interval) if interval else 0
    secs = min(secs, 86400)
    if secs >= 86400:
        return "bars_1d"
    if secs >= 3600:
        return "bars_1h"
    if secs >= 900:
        return "bars_15m"
    return "bars_1m"


# Exact matches: bar_interval → dataset (no hybrid mode)
_EXACT_DATASETS = {60: "bars_1m", 900: "bars_15m", 3600: "bars_1h", 86400: "bars_1d"}


def _dataset_for_interval_exact(interval: Any) -> str:
    """Pick a dataset that avoids hybrid mode overhead.

    If bar_interval exactly matches a dataset resolution, use it.
    Otherwise, pick the closest LARGER dataset so the engine doesn't
    activate hybrid mode (signal on coarse + sim on fine = slow).
    Capped at bars_1d.
    """
    secs = _interval_to_seconds(interval) if interval else 0
    # Exact match — best case, no resample needed
    if secs in _EXACT_DATASETS:
        return _EXACT_DATASETS[secs]
    # No exact match: pick the next larger dataset to avoid hybrid overhead
    # e.g. 4h (14400s) → bars_1d (86400s), not bars_1h (3600s) which triggers hybrid
    for threshold, dataset in sorted(_EXACT_DATASETS.items()):
        if threshold >= secs:
            return dataset
    return "bars_1d"


def _resolve_store(config: BacktestConfig, store: DataStore) -> DataStore:
    """Select the right dataset based on config.

    Two modes:
      - **Normal** (default): dataset matches ``bar_interval`` exactly.
        If no exact match, picks the closest smaller dataset and sets
        ``resample_to`` so the engine resamples to bar_interval (no hybrid overhead).
      - **Precise** (``precise=True`` on config): always loads ``bars_1m``.
        Signals on ``bar_interval``, simulation on 1-min bars.
        Required for precise SL/TP fills.

    Skips auto-resolve if the user explicitly set a non-default dataset.
    """
    try:
        current = store.dataset()
    except Exception:
        return store

    # ArrowIpcDataStore handles multi-resolution internally via bar_interval —
    # skip Python-side dataset swapping. Detected by dataset() returning "arrow_ipc".
    if current == "arrow_ipc":
        return store

    # If user explicitly chose a non-default dataset, respect it
    if current != "bars_1m":
        return store

    # Accuracy mode: keep bars_1m (hybrid: signals on bar_interval, sim on 1m)
    if getattr(config, "precise", False):
        return store

    # Normal mode: pick dataset <= bar_interval.
    # The lite sim path runs on resampled bars, so no hybrid overhead.
    target = _dataset_for_interval(config.bar_interval)

    if target == current:
        return store

    # Try the target dataset; if it doesn't exist (no active version),
    # fall back to bars_1m — the engine will resample automatically.
    try:
        candidate = DataStore(
            data_root=store.data_root(),
            metadata_db=store.metadata_db(),
            dataset=target,
        )
        # Verify the dataset actually has an active version
        if candidate.active_version(target) is None:
            return store
        return candidate
    except Exception:
        return store


def _cap_output_resolution(config: BacktestConfig) -> BacktestConfig:
    """Cap output_resolution to daily for Community users (Pro feature)."""
    if config.output_resolution is None:
        return config
    if not _is_sub_daily(config.output_resolution):
        return config
    if _is_pro():
        return config
    _warn_pro("output_resolution capped to daily")
    config = copy.deepcopy(config)
    config.output_resolution = None
    return config


# ---------------------------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------------------------

def ingest(
    provider: str,
    symbol: Optional[str] = None,
    symbol_id: Optional[int] = None,
    start: str = "",
    end: str = "",
    *,
    symbols: Optional[list] = None,
    interval: str = "1m",
    dataset: Optional[str] = None,
    data_root: str = "data",
    metadata_db: str = "metadata/metadata.sqlite",
    exchange: Optional[str] = None,
    asset_class: str = "crypto_spot",
    progress: bool = True,
) -> DataStore:
    """Ingest bars from a data provider into the Arrow IPC store.

    Providers (free): ``"binance"``, ``"bybit"``, ``"hyperliquid"``, ``"dydx"``,
    ``"bitstamp"``. Pro: ``"databento"``, ``"massive"``.

    Returns a :class:`DataStore` ready for :func:`run`.

    Example (single symbol)::

        store = bt.ingest(
            provider="binance",
            symbol="BTCUSDT",
            symbol_id=1,
            start="2020-01-01T00:00:00Z",
            end="2025-01-01T00:00:00Z",
        )

    Example (multiple symbols)::

        store = bt.ingest(
            provider="binance",
            symbols=[("XMRUSDT", 26), ("VETUSDT", 27), ("ZECUSDT", 28)],
            start="2020-06-01T00:00:00Z",
            end="2026-03-01T00:00:00Z",
        )
    """
    _PRO_PROVIDERS = {"databento", "massive"}
    if provider in _PRO_PROVIDERS:
        _require_pro(f"Data connector: {provider}")

    # Build list of (symbol, symbol_id) pairs.
    if symbols is not None:
        pairs = [(s, sid) for s, sid in symbols]
    elif symbol is not None and symbol_id is not None:
        pairs = [(symbol, symbol_id)]
    else:
        raise ValueError("provide either symbol+symbol_id or symbols=[(ticker, id), ...]")

    if len(pairs) == 1:
        return _ingest_single(
            provider=provider, symbol=pairs[0][0], symbol_id=pairs[0][1],
            start=start, end=end, interval=interval, dataset=dataset,
            data_root=data_root, metadata_db=metadata_db,
            exchange=exchange, asset_class=asset_class, progress=progress,
        )

    # Multi-symbol: show all symbols with pending ones in grey.
    display = None
    callbacks = {}
    if progress:
        from manifoldbt._progress import make_multi_progress
        display, callbacks = make_multi_progress(pairs, provider)

    store = None
    try:
        for sym, sid in pairs:
            cb = callbacks.get(sym) if callbacks else None
            store = _ingest_native(
                provider=provider, symbol=sym, symbol_id=sid,
                start=start, end=end, interval=interval, dataset=dataset,
                data_root=data_root, metadata_db=metadata_db,
                exchange=exchange, asset_class=asset_class,
                progress_cb=cb,
            )
    finally:
        if display is not None:
            display.stop()

    return store


def import_csv(
    path: str,
    symbol: str,
    symbol_id: int,
    *,
    interval: str = "1m",
    data_root: str = "data",
    metadata_db: str = "metadata/metadata.sqlite",
    exchange: str = "CSV",
    asset_class: str = "crypto_spot",
) -> DataStore:
    """Import bars from a CSV file into the Arrow IPC store. Free on all tiers.

    Auto-detects standard (``timestamp,open,high,low,close,volume``),
    MetaTrader 4, and MetaTrader 5 exports. Returns a :class:`DataStore` ready
    for :func:`run` — the same store ``bt.ingest`` writes to.

    Example::

        store = bt.import_csv(
            "EURUSD_1m.csv", symbol="EURUSD", symbol_id=1,
            interval="1m", asset_class="forex",
        )
        result = bt.run(strategy, config, store)

    Args:
        path: Path to the CSV file (standard / MT4 / MT5 format).
        symbol: Ticker name (e.g. ``"EURUSD"``, ``"BTCUSDT"``).
        symbol_id: Unique integer ID for this symbol in the store.
        interval: Bar interval of the rows (``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``, ...).
        data_root: Store directory (default ``"data"``).
        metadata_db: Metadata SQLite path.
        exchange: Exchange label for metadata (default ``"CSV"``).
        asset_class: ``crypto_spot``, ``crypto_perp``, ``equity``, ``future``,
            ``option``, ``forex``, or ``index``.
    """
    return _import_csv_native(
        csv_path=str(path),
        symbol=symbol,
        symbol_id=symbol_id,
        interval=interval,
        data_root=data_root,
        metadata_db=metadata_db,
        exchange=exchange,
        asset_class=asset_class,
    )


def _ingest_single(
    *, provider, symbol, symbol_id, start, end, interval, dataset,
    data_root, metadata_db, exchange, asset_class, progress,
) -> DataStore:
    cb = None
    display = None
    if progress:
        from manifoldbt._progress import make_progress_display
        display, cb = make_progress_display(symbol, provider)

    try:
        return _ingest_native(
            provider=provider,
            symbol=symbol,
            symbol_id=symbol_id,
            start=start,
            end=end,
            interval=interval,
            dataset=dataset,
            data_root=data_root,
            metadata_db=metadata_db,
            exchange=exchange,
            asset_class=asset_class,
            progress_cb=cb,
        )
    finally:
        if display is not None:
            display.stop()


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def run(
    strategy: Strategy,
    config: BacktestConfig,
    store: DataStore,
) -> Result:
    """Run a backtest and return a rich Result.

    Returns a :class:`Result` with DataFrame conversion, summaries,
    and plotting methods. Access the raw Rust object via ``result.raw``.
    """
    try:
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        cfg_json = _prepared_config_json(config, strategy, store)
        raw = _run_native(strategy.to_json(), cfg_json, store)
        return Result(raw)
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def run_sweep(
    strategy: Strategy,
    param_grid: Dict[str, List[Any]],
    config: BacktestConfig,
    store: DataStore,
    *,
    max_parallelism: int = 0,
) -> SweepResult:
    """Run a parameter sweep in parallel (rayon) and return a SweepResult.

    Args:
        strategy: Strategy definition.
        param_grid: Mapping of parameter names to lists of values.
            Example: ``{"fast": [10, 20, 30], "slow": [50, 60]}``
            produces 6 combinations (Cartesian product).
        config: Backtest configuration.
        store: Data store.
        max_parallelism: Maximum threads. 0 = all available cores.

    Returns:
        A :class:`SweepResult` with ``.to_df()``, ``.best()``, ``.plot_metric()``.
    """
    _require_pro_over_combos(_grid_combos(param_grid), "Parameter sweep")
    try:
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        cfg_json = _prepared_config_json(config, strategy, store)
        grid_json = json.dumps({
            name: [scalar_value_to_json(v) for v in values]
            for name, values in param_grid.items()
        })
        raw_results = _run_sweep_native(
            strategy.to_json(),
            grid_json,
            cfg_json,
            store,
            max_parallelism,
        )
        return SweepResult(raw_results, param_grid)
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def run_batch(
    strategies: List[Strategy],
    config: BacktestConfig,
    store: DataStore,
    *,
    max_parallelism: int = 0,
) -> List[Result]:
    """Run many strategies in parallel sharing a single data load.

    Loads bars once, aligns timestamps once, then evaluates each strategy
    on a separate rayon thread.  Much faster than calling ``run()`` in a loop.

    Args:
        strategies: List of Strategy definitions.
        config: Shared backtest configuration (same universe/time range).
        store: Data store.
        max_parallelism: Maximum threads. 0 = all available cores.

    Returns:
        One :class:`Result` per strategy, in input order.
    """
    _require_pro_over_combos(len(strategies), "Batch backtesting")
    try:
        config = _prepare_config(config, None, store)
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        strategy_jsons = [strat.to_json() for strat in strategies]
        raw_results = _run_batch_native(
            strategy_jsons,
            config.to_json(),
            store,
            max_parallelism,
        )
        return [Result(r) for r in raw_results]
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def run_batch_lite(
    strategies: List[Strategy],
    config: BacktestConfig,
    store: DataStore,
    *,
    max_parallelism: int = 0,
) -> List["BatchResultLite"]:
    """Run many strategies in parallel, returning only metrics (no Arrow output).

    Much faster and lighter than ``run_batch`` — skips trade logging,
    position traces, and Arrow output construction.  Ideal for parameter sweeps
    where you only need metrics to select the best variant.

    Args:
        strategies: List of Strategy definitions.
        config: Shared backtest configuration (same universe/time range).
        store: Data store.
        max_parallelism: Maximum threads. 0 = all available cores.

    Returns:
        One :class:`BatchResultLite` per strategy (name, metrics, equity, trade_count).
    """
    _require_pro_over_combos(len(strategies), "Batch backtesting")
    try:
        config = _prepare_config(config, None, store)
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        strategy_jsons = [strat.to_json() for strat in strategies]
        return _run_batch_lite_native(
            strategy_jsons,
            config.to_json(),
            store,
            max_parallelism,
        )
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def run_sweep_lite(
    strategy: Strategy,
    param_grid: Dict[str, List[Any]],
    config: BacktestConfig,
    store: DataStore,
    *,
    max_parallelism: int = 0,
    device: str = "cpu",
) -> List["BatchResultLite"]:
    """Run a parameter sweep returning only metrics (no Arrow output).

    Same as ``run_sweep`` but uses the lite path — much faster for large grids.
    Supports ``param()`` in indicator periods (auto re-compilation per combo).

    Args:
        strategy: Strategy definition (may use ``param()`` in indicator periods).
        param_grid: Mapping of parameter names to lists of values.
        config: Backtest configuration.
        store: Data store.
        max_parallelism: Maximum threads. 0 = all available cores.
        device: ``"cpu"`` (default) or ``"cuda"``/``"gpu"``. The GPU path
            accelerates single-asset, AtClose + FixedBps sweeps and produces
            results numerically identical to the CPU path. **Pro-only**: a
            Community license raises ``PermissionError`` for ``device="cuda"``
            (Community keeps the full-speed CPU sweep with no restriction).
            Requires a build with ``--features cuda`` and a CUDA device; for any
            unsupported strategy/config (or when no GPU is present at runtime) it
            silently falls back to the CPU sweep, so results are never affected.

    Returns:
        One :class:`BatchResultLite` per combo (Cartesian product order).
    """
    _require_pro_over_combos(_grid_combos(param_grid), "Parameter sweep")
    _require_pro_for_gpu(device, "GPU sweep")
    try:
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        cfg_json = _prepared_config_json(config, strategy, store)
        grid_json = json.dumps({
            name: [scalar_value_to_json(v) for v in values]
            for name, values in param_grid.items()
        })
        return _run_sweep_lite_native(
            strategy.to_json(),
            grid_json,
            cfg_json,
            store,
            max_parallelism,
            device,
        )
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


# ---------------------------------------------------------------------------
# Research API
# ---------------------------------------------------------------------------

def run_walk_forward(
    strategy: Strategy,
    wf_config: Dict[str, Any],
    config: BacktestConfig,
    store: "DataStore",
) -> Dict[str, Any]:
    """Run walk-forward analysis (Pro only).

    Args:
        strategy: Strategy definition.
        wf_config: Walk-forward config dict with keys:
            method (str): "Anchored" or "Rolling"
            n_splits (int): Number of folds.
            train_ratio (float): Fraction for training (0, 1).
            optimize_metric (str): e.g. "sharpe", "sortino".
            param_grid (dict): Parameter grid for optimization.
            max_parallelism (int): Max threads.
        config: Backtest configuration.
        store: Data store.

    Returns:
        Dict with ``folds`` and ``best_params_per_fold``.
    """
    # Pro gate (friendly message + clean exit). Real enforcement lives natively
    # in `py_run_walk_forward` (check_feature("walk_forward")), so this cannot be
    # bypassed by calling the native function directly.
    _require_pro("Walk-forward optimization")
    config = _prepare_config(config, strategy, store)
    wf_json = json.dumps(_convert_param_grid_in_config(wf_config))
    return _run_walk_forward_native(strategy.to_json(), wf_json, config.to_json(), store)


def run_sweep_2d(
    strategy: Strategy,
    sweep_config: Dict[str, Any],
    config: BacktestConfig,
    store: "DataStore",
) -> Dict[str, Any]:
    """Run a 2D parameter sweep (heatmap).

    Args:
        strategy: Strategy definition.
        sweep_config: Dict with keys:
            x_param (str): First parameter name.
            x_values (list): Values for x_param.
            y_param (str): Second parameter name.
            y_values (list): Values for y_param.
            metric (str): Metric to collect.
            max_parallelism (int): Max threads.
        config: Backtest configuration.
        store: Data store.

    Returns:
        Dict with ``metric_grid`` (2D list), ``x_values``, ``y_values``, etc.
    """
    _require_pro_over_combos(
        len(sweep_config.get("x_values", [])) * len(sweep_config.get("y_values", [])),
        "2D parameter sweep",
    )
    config = _prepare_config(config, strategy, store)
    sweep_json = json.dumps(_convert_scalar_values_in_sweep(sweep_config))
    return _run_sweep_2d_native(strategy.to_json(), sweep_json, config.to_json(), store)


def run_stability(
    strategy: Strategy,
    stability_config: Dict[str, Any],
    config: BacktestConfig,
    store: "DataStore",
) -> Dict[str, Any]:
    """Run parameter stability analysis.

    Args:
        strategy: Strategy definition.
        stability_config: Dict with keys:
            param_name (str): Parameter to vary.
            values (list): Values to test.
            metric (str): Metric to evaluate.
            max_parallelism (int): Max threads.
        config: Backtest configuration.
        store: Data store.

    Returns:
        Dict with ``stability_score``, ``metric_values``, ``mean_metric``, ``std_metric``.
    """
    _require_pro_over_combos(len(stability_config.get("values", [])), "Parameter stability analysis")
    config = _prepare_config(config, strategy, store)
    stab_json = json.dumps(_convert_scalar_values_in_stability(stability_config))
    return _run_stability_native(strategy.to_json(), stab_json, config.to_json(), store)


def replay(
    manifest: Dict[str, Any],
    strategy: Strategy,
    store: "DataStore",
) -> Result:
    """Replay a backtest from a saved manifest.

    Args:
        manifest: RunManifest dict (as returned by a previous run).
        strategy: Original strategy definition (needed to recompile).
        store: Data store.

    Returns:
        Result from the replayed run.
    """
    raw = _replay_native(json.dumps(manifest), strategy.to_json(), store)
    return Result(raw)


# ---------------------------------------------------------------------------
# Stochastic simulation API
# ---------------------------------------------------------------------------

from manifoldbt.stochastic import StochasticModel


def run_stochastic(
    model,
    *,
    s0: float = 100.0,
    n_paths: int = 1000,
    n_steps: int = 252,
    dt: float = 1.0 / 252.0,
    params: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    confidence_levels: Optional[List[float]] = None,
    store_paths: bool = False,
    device: str = "cpu",
    precision: str = "f64",
) -> Dict[str, Any]:
    """Run a stochastic simulation via SDE expression DSL.

    All expressions are compiled to native Rust and executed with Rayon
    parallelism — no Python callback overhead.

    Args:
        model: Either a preset name (``"gbm"``, ``"heston"``, ``"merton"``,
            ``"garch_jd"``) or a :class:`StochasticModel` instance.
        s0: Initial price.
        n_paths: Number of simulation paths.
        n_steps: Number of time steps per path.
        dt: Time step in years (``1/252`` = daily, ``1/252/390`` = minute).
        params: Parameter overrides (merged with model defaults).
        seed: RNG seed for reproducibility.
        confidence_levels: Quantile levels for reporting.
        store_paths: Whether to store full price paths.
        device: ``"cpu"`` (default, Rayon parallel) or ``"cuda"``/``"gpu"``
            (CUDA GPU, requires build with ``--features cuda``).
        precision: ``"f64"`` (default, double) or ``"f32"`` (float, ~10-20x
            faster on consumer GPUs, suitable for research/prototyping).

    Returns:
        Dict with ``final_price``, ``final_return``, ``max_drawdown``,
        ``annualized_return``, ``annualized_vol`` (each with percentiles,
        mean, std, min, max), and optionally ``paths`` (Arrow array) +
        ``paths_n_steps``.

    Example:
        >>> result = mbt.run_stochastic("gbm", s0=100, n_paths=10000,
        ...     n_steps=252, dt=1/252, params={"mu": 0.05, "sigma": 0.2})
        >>> result["final_price"]["mean"]
        105.12

        >>> model = mbt.StochasticModel(
        ...     drift="mu", diffusion="sqrt(h)",
        ...     state_vars={"h": 1e-4},
        ...     state_update={"h": "omega + alpha * (ret - mu)**2 + beta * h"},
        ...     params={"mu": 0.08, "omega": 1e-6, "alpha": 0.1, "beta": 0.85},
        ... )
        >>> result = mbt.run_stochastic(model, s0=100, n_paths=5000)
    """
    _require_pro_for_gpu(device, "GPU stochastic simulation")
    config: Dict[str, Any] = {
        "s0": s0,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "dt": dt,
        "store_paths": store_paths,
        "device": device,
        "precision": precision,
    }

    if seed is not None:
        config["rng_seed"] = seed

    if confidence_levels is not None:
        config["confidence_levels"] = confidence_levels

    if isinstance(model, str):
        # Preset name
        config["preset"] = model
        if params:
            config["params"] = params
    elif isinstance(model, StochasticModel):
        model_dict = model.to_dict()
        if params:
            model_dict["params"].update(params)
        config["model"] = model_dict
    else:
        raise TypeError(
            f"model must be a preset name (str) or StochasticModel, got {type(model).__name__}"
        )

    try:
        return _run_stochastic_native(json.dumps(config))
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


# ---------------------------------------------------------------------------
# Portfolio API
# ---------------------------------------------------------------------------

def run_portfolio(
    portfolio: Portfolio,
    config: BacktestConfig,
    store: DataStore,
) -> Result:
    """Run a multi-strategy portfolio backtest.

    Args:
        portfolio: Portfolio definition with strategies and allocations.
        config: Backtest configuration (shared across all strategies).
        store: Data store.

    Returns:
        A :class:`Result` with combined portfolio metrics. Access per-strategy
        breakdown via ``result.per_strategy``.
    """
    try:
        config = _prepare_config(config, None, store)
        raw_combined, per_strategy_info = _run_portfolio_native(
            portfolio.to_json(),
            config.to_json(),
            store,
        )
        result = Result(raw_combined)
        result._per_strategy = per_strategy_info
        return result
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


# ---------------------------------------------------------------------------
# Lazy submodule imports
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    if name == "plot":
        return _importlib.import_module("manifoldbt.plot")
    if name == "diagnostics":
        return _importlib.import_module("manifoldbt.diagnostics")
    raise AttributeError(f"module 'manifoldbt' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convert_param_grid_in_config(wf_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert param_grid values to Rust ScalarValue JSON format."""
    result = dict(wf_config)
    if "param_grid" in result:
        result["param_grid"] = {
            name: [scalar_value_to_json(v) for v in values]
            for name, values in result["param_grid"].items()
        }
    return result


def _convert_scalar_values_in_sweep(sweep_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert x_values/y_values to Rust ScalarValue JSON format."""
    result = dict(sweep_config)
    if "x_values" in result:
        result["x_values"] = [scalar_value_to_json(v) for v in result["x_values"]]
    if "y_values" in result:
        result["y_values"] = [scalar_value_to_json(v) for v in result["y_values"]]
    return result


def _convert_scalar_values_in_stability(stability_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert values to Rust ScalarValue JSON format."""
    result = dict(stability_config)
    if "values" in result:
        result["values"] = [scalar_value_to_json(v) for v in result["values"]]
    return result


# ---------------------------------------------------------------------------
# Exogenous data registration
# ---------------------------------------------------------------------------

def register_exo(
    name: str,
    data,
    store: Optional["DataStore"] = None,
    data_root: str = "data",
    provider: Optional[str] = None,
    timeframe: str = "1d",
):
    """Register an exogenous data series for use in strategies.

    Without ``provider``: writes to ``{root}/exo/{name}.arrow`` (legacy layout).
    With ``provider``: writes to ``{root}/{provider}/{timeframe}/{name}.arrow``
    (unified layout, used for cross-exchange data).

    Args:
        name: Series identifier (e.g. ``"hashrate"``, ``"BTCUSDT"``).
        data: A pandas/polars DataFrame or dict with a ``"timestamp"`` column
              and one or more float value columns.
        store: Optional DataStore to infer ``data_root`` from.
        data_root: Root data directory (default ``"data"``).
        provider: Provider name for unified layout (e.g. ``"binance"``).
        timeframe: Timeframe label (e.g. ``"1d"``, ``"1h"``). Default ``"1d"``.

    Example::

        # Legacy (non-symbol exo like hashrate)
        bt.register_exo("hashrate", df)

        # Unified layout (cross-exchange)
        bt.register_exo("BTCUSDT", df, provider="binance", timeframe="1h")
    """
    import pyarrow as pa
    from pathlib import Path

    # Resolve data root
    if store is not None:
        root = Path(store.data_root()) / "mega"
    else:
        root = Path(data_root) / "mega"

    if provider:
        # Unified layout: {root}/{provider}/{timeframe}/{name}.arrow
        target_dir = root / provider / timeframe
    else:
        # Legacy layout: {root}/exo/{name}.arrow
        target_dir = root / "exo"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Convert to Arrow Table
    if hasattr(data, "to_arrow"):
        # Polars DataFrame
        table = data.to_arrow()
    elif hasattr(data, "columns"):
        # Pandas DataFrame
        import pandas as pd
        table = pa.Table.from_pandas(data)
    elif isinstance(data, dict):
        table = pa.table(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Use a pandas/polars DataFrame or dict.")

    # Ensure timestamp is TimestampNanosecond(UTC)
    ts_idx = table.schema.get_field_index("timestamp")
    if ts_idx < 0:
        raise ValueError("Data must have a 'timestamp' column")

    ts_type = table.schema.field(ts_idx).type
    if not pa.types.is_timestamp(ts_type):
        raise ValueError(f"'timestamp' column must be a timestamp type, got {ts_type}")

    # Cast to nanos UTC if needed
    target_type = pa.timestamp("ns", tz="UTC")
    if ts_type != target_type:
        ts_col = table.column(ts_idx).cast(target_type)
        table = table.set_column(ts_idx, pa.field("timestamp", target_type), ts_col)

    # Cast value columns to float64
    for i, field in enumerate(table.schema):
        if field.name == "timestamp":
            continue
        if field.type != pa.float64():
            table = table.set_column(
                i, pa.field(field.name, pa.float64()), table.column(i).cast(pa.float64())
            )

    # Write Arrow IPC
    path = target_dir / f"{name}.arrow"
    writer = pa.ipc.new_file(str(path), table.schema)
    writer.write_table(table)
    writer.close()

    print(f"Registered exo '{name}': {table.num_rows} rows, "
          f"columns={[f.name for f in table.schema if f.name != 'timestamp']} -> {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Core types
    "BacktestResult",
    "BatchResultLite",
    "DataStore",
    "Result",
    "SweepResult",
    # Data ingestion
    "ingest",
    "import_csv",
    # Run functions
    "run",
    "run_sweep",
    "run_batch",
    "run_batch_lite",
    "run_json",
    "run_with_parquet",
    "compile_strategy_json",
    # DSL
    "AssetRef",
    "Expr",
    "TimeframeRef",
    "asset",
    "col",
    "exo",
    "lit",
    "param",
    "s",
    "scan",
    "symbol_ref",
    "tf",
    "when",
    # Strategy & config
    "Strategy",
    "BacktestConfig",
    "ExecutionConfig",
    "FeeConfig",
    "VenueFees",
    "OrderConfig",
    # Helpers
    "date_to_ns",
    "time_range",
    "Slippage",
    "Interval",
    "ExecutionPrice",
    "FillModel",
    # Exceptions
    "BacktesterError",
    "DataError",
    "StrategyError",
    "ConfigError",
    # Research
    "run_walk_forward",
    "run_sweep_2d",
    "run_stability",
    "replay",
    "py_run_monte_carlo",
    # Stochastic simulation
    "run_stochastic",
    "StochasticModel",
    # Portfolio
    "Portfolio",
    "run_portfolio",
    # Exogenous data
    "register_exo",
    # Version
    "__version__",
    # Indicators (submodule)
    "indicators",
    # Plotting (lazy, requires matplotlib)
    "plot",
    # Diagnostics (lazy)
    "diagnostics",
]
