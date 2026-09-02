"""manifoldbt: Fast research backtesting with Rust core + Python DSL.

Quickstart::

    import manifoldbt as bt
    from manifoldbt.indicators import sma

    store = bt.DataStore("data", "metadata/metadata.sqlite")  # backend auto-detected
    pos = bt.when(sma(bt.col("close"), 20) > sma(bt.col("close"), 50),
                  bt.lit(1.0), bt.lit(0.0))
    strat = bt.Strategy.create("sma-cross").signal("position", pos).size(pos)
    start, end = bt.time_range("2022-01-01", "2025-01-01")   # UNIX nanoseconds
    cfg = bt.BacktestConfig(universe=["BTCUSDT"], time_range_start=start,
                            time_range_end=end,
                            bar_interval=bt.Interval.hours(1),
                            initial_capital=100_000.0)
    result = bt.run(strat, cfg, store)
    print(result.summary())
    print(result.metrics["sharpe"])

``manifoldbt.guide()`` prints a compact API cheat sheet (data import, config
fields, metrics, cross-asset references, worked recipes, common errors). It
answers most questions that would otherwise need a pile of ``help()`` calls --
including whether the DSL can express a stateful or multi-symbol strategy.
"""
import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import importlib as _importlib

# Avant TOUT chargement du module natif: la roue GPU charge NVRTC par son nom
# et ne le trouverait pas dans site-packages/nvidia/. Sans effet cote CPU.
from manifoldbt import _cuda_libs as _cuda_libs

_cuda_libs.rendre_visible()

from manifoldbt._native import (
    BacktestResult,
    BatchResultLite,
    DataStore,
    activate,
    license_expiry as _license_expiry,
    license_info as _license_info,
    compile_strategy_json,
    run as _run_native,
    run_batch as _run_batch_native,
    run_batch_lite as _run_batch_lite_native,
    run_json,
    run_sweep as _run_sweep_native,
    run_sweep_lite as _run_sweep_lite_native,
    sweep_columns as _sweep_columns_native,
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
    py_import_dataframe as _import_dataframe_native,
)
from manifoldbt._serde import scalar_value_to_json
from manifoldbt.crossasset import prepare_cross_asset as _prepare_cross_asset
from manifoldbt.config import (
    BacktestConfig,
    ExecutionConfig,
    FeeConfig,
    OrderConfig,
    VenueFees,
    entry_price,
    resolve_universe,
)
from manifoldbt.exceptions import (
    BacktesterError,
    ConfigError,
    DataError,
    LicenseError,
    StrategyError,
)
from manifoldbt.expr import AssetRef, Expr, TimeframeRef, asset, choice, col, exo, hold, lit, param, s, scan, symbol_ref, tf, when
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

# Managed compute. Imported eagerly, unlike `plot` and `diagnostics`: it pulls
# nothing but the standard library, and `mbt.cloud` reading as missing until
# something else touched it would be a worse surprise than the microseconds.
from manifoldbt import cloud

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
from manifoldbt import _update as _update

# "0.1.0" is the fallback for a source checkout on `sys.path`: there is no
# installed distribution to read a version from, so there is nothing truthful to
# report. The update check treats that same case as "do not compare".
__version__ = _update.installed_version() or "0.1.0"

# ---------------------------------------------------------------------------
# License banner
# ---------------------------------------------------------------------------
def _print_banner():
    try:
        tier, email = _license_info()
        if tier == "Pro" and email:
            # A trial shows its end date up front. Without this the banner is
            # identical to a paid licence, and the expiry is discovered the day
            # everything stops working, mid-session.
            trial = ""
            try:
                expiry = _license_expiry()
                if expiry:
                    trial = f" (trial ends {expiry[:10]})"
            except Exception:
                pass
            print(f"manifoldbt v{__version__} | \033[38;5;214mPro\033[0m{trial} | {email}")
        else:
            print(f"manifoldbt v{__version__} | \033[36mCommunity\033[0m | upgrade: www.manifoldbt.com")
    except Exception:
        print(f"manifoldbt v{__version__} | \033[36mCommunity\033[0m | upgrade: www.manifoldbt.com")

_print_banner()
del _print_banner

# Right under the banner, and never blocking: prints the answer PyPI gave a
# previous run, then refreshes it on a daemon thread. Off with
# MANIFOLDBT_NO_UPDATE_CHECK=1. Registered before the Pro summary below so that
# atexit's LIFO order puts a late notice last, after everything else this import
# has to say.
_update.start()


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


def check_for_update() -> Optional[str]:
    """Ask PyPI whether a newer manifoldbt exists. Returns its version, or None.

    ``None`` means this install is current (or is a source checkout, whose version
    nothing can be compared against). Unlike the notice printed at import, which
    reads a cached answer, this queries PyPI and blocks until it answers.

        >>> mbt.check_for_update()
        '0.20.0'
    """
    return _update.check_now()


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

    Reported here so that every GPU entry point raises the same clean
    ``LicenseError`` as the other Pro features, instead of each surfacing its own
    error type from deeper in the run. No-op for CPU or for Pro users.
    """
    if isinstance(device, str) and device.lower() in ("cuda", "gpu"):
        _require_pro(feature)


# Community fan-out budget: sweeps and batches may run up to this many backtests
# cumulatively per session for free; beyond it requires Pro. A single run() is
# never affected. Keep in step with the engine's own limit.
_COMMUNITY_MAX_COMBOS = 256


def _grid_combos(param_grid) -> int:
    """Number of Cartesian combinations produced by a sweep param grid."""
    n = 1
    for values in param_grid.values():
        n *= max(1, len(values))
    return n


def _require_pro_over_combos(n_combos: int, what: str) -> None:
    """Raise LicenseError if a fan-out exceeds the Community combination limit.

    Fast-fail, so a call that could never fit the budget reports cleanly in a
    notebook instead of part-way through the run. The budget is consumed
    **cumulatively across the session**: a sequence of smaller calls draws on
    the same allowance as one large one.
    """
    if n_combos <= _COMMUNITY_MAX_COMBOS or _is_pro():
        return
    raise LicenseError(
        f"{what} with {n_combos} runs exceeds the Community limit of "
        f"{_COMMUNITY_MAX_COMBOS} combinations per session. "
        f"Upgrade to Pro at www.manifoldbt.com"
    )


#: Distances de bracket balayables par leur nom, en plus des ``param()``
#: d'expression. Elles ne passent pas par ``param()`` parce qu'une distance de
#: bracket est un champ de configuration, pas un noeud d'expression : rien ne
#: l'evalue. Doit rester aligne sur ``ORDER_SWEEP_PARAMS`` (bt-core,
#: orchestrator.rs), qui fait la substitution par combinaison.
_ORDER_SWEEP_PARAMS = frozenset({"stop_loss", "take_profit", "trailing_stop"})


def _validate_swept_params(strategy: "Strategy", names, what: str) -> None:
    """Reject swept parameter names the strategy never declares.

    Sweeping a name the strategy does not use is a silent no-op: the value is
    merged into a parameter map nothing reads, so every combo runs the same
    backtest and the sweep returns N identical results with no warning. That
    is worse than an error, because an "optimisation" over thousands of combos
    looks like it worked and its best result is meaningless.

    A parameter counts as declared whether it came from ``mbt.param()`` inside
    an expression or from an explicit ``.param()`` call: ``to_json_dict()``
    merges both into ``parameters`` (and is memoised, so this costs nothing).
    """
    declared = set(strategy.to_json_dict().get("parameters") or {})
    unknown = [n for n in names if n not in declared and n not in _ORDER_SWEEP_PARAMS]
    if not unknown:
        return
    known = ", ".join(sorted(declared | _ORDER_SWEEP_PARAMS))
    raise StrategyError(
        f"{what}: parameter(s) {unknown} are not declared by strategy "
        f"'{strategy.name}' (declared: {known}). Sweeping them would run the "
        f"same backtest for every combination. Use mbt.param(\"name\") where "
        f"the value is consumed, e.g. ema(close, mbt.param(\"fast\"))."
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
    """Content-memoised equivalent of ``_prepare_config(...).to_json()``.

    The prepared config no longer depends on the strategy (orders travel in the
    strategy JSON now), so the memo key is just the config content plus the
    metadata DB; the ``strategy`` argument is accepted for call-site symmetry.

    Strategy-dependent validation therefore has to run BEFORE the memo, not
    inside `_prepare_config`: one config reused across several strategies would
    otherwise be validated once, against whichever strategy arrived first.
    """
    _reject_resting_order_without_delay(config, strategy)
    try:
        meta_db = store.metadata_db()
    except Exception:
        meta_db = None
    if meta_db is None:
        return _prepare_config(config, strategy, store).to_json()

    try:
        key = (config.to_json(), meta_db)
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


def _reject_resting_order_without_delay(cfg: BacktestConfig, strategy) -> None:
    """A resting order cannot be priced off the bar it fills on.

    An entry order placed from the signal of bar `t - signal_delay` is gated
    against bar `t` in the same pass, so with ``signal_delay = 0`` the level is
    computed from the very bar whose high and low decide whether it fills. The
    order would have had to exist before that bar opened, and its price did not
    exist yet: the backtest fills at a level nobody could have posted.

    Market orders are unaffected (they take the bar's execution price, which is
    what `signal_delay = 0` is *for*). Only resting orders read the bar twice.
    """
    orders = getattr(strategy, "_orders", None) if strategy is not None else None
    if not orders or not orders.get("limit_entry"):
        return
    execution = getattr(cfg, "execution", None)
    if execution is None or getattr(execution, "signal_delay", 1) != 0:
        return
    trigger = orders["limit_entry"].get("trigger", "Limit")
    raise ValueError(
        f"a resting entry order ({trigger}) needs signal_delay >= 1: with "
        "signal_delay=0 its level is computed from the same bar whose high/low "
        "decide the fill, so the backtest fills at a price that did not exist "
        "when the order was placed. Set "
        "ExecutionConfig(signal_delay=1), or drop the entry order to take a "
        "market fill, which signal_delay=0 is meant for."
    )


def _prepare_config(config: BacktestConfig, strategy, store: DataStore) -> BacktestConfig:
    """Prepare config for execution: resolve symbols, convert deprecated fields."""
    cfg = copy.deepcopy(config)
    _reject_resting_order_without_delay(cfg, strategy)

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

    # Per-strategy SL/TP/trailing orders are NOT merged into the config anymore:
    # they travel inside the strategy JSON (Strategy.to_json -> StrategyDef.orders)
    # so the engine applies them per-strategy. This lets one batch/sweep call run
    # strategies carrying different brackets over a single data load. A bracket
    # set directly on config.execution.orders still applies as the fallback.

    _attach_option_contracts(cfg, store)
    return cfg


def _attach_option_contracts(cfg: BacktestConfig, store: DataStore) -> None:
    """Fill ``cfg.option_contracts`` from what the store recorded at ingest.

    The terms come from the venue, so nothing here is guessed. The one thing the
    caller must supply is ``option_underlyings``: Deribit settles against its own
    index, whose ticker matches no series anyone can ingest, so which price
    stands in for it is a decision, not a lookup. Getting it wrong silently would
    settle every contract against the wrong number, so a missing entry raises.
    """
    if cfg.option_contracts:
        return  # explicitly overridden by the caller
    try:
        available = store.option_contracts()
    except AttributeError:
        return  # store predates option support (mock stores in tests)

    universe = cfg.universe if isinstance(cfg.universe, list) else []
    in_universe = {int(sid) for sid in universe if isinstance(sid, int)}
    underlyings = {int(k): int(v) for k, v in (cfg.option_underlyings or {}).items()}

    # An option whose terms were never recorded is the dangerous case: it
    # prices, it trades, it never expires, and nothing looks wrong. Catch it
    # before the engine sees a plain price series.
    try:
        classes = store.asset_classes()
    except AttributeError:
        classes = {}
    untermed = [
        int(sid)
        for sid, klass in classes.items()
        if klass == "EquityOption"
        and int(sid) in in_universe
        and int(sid) not in {int(k) for k in available}
    ]
    if untermed:
        names = {int(i): t for i, t in store.list_symbols()}
        listed = ", ".join(f"{sid} ({names.get(sid, '?')})" for sid in sorted(untermed))
        raise ValueError(
            f"symbol(s) {listed} are recorded as options but carry no contract terms. "
            "The connector that ingested them does not report a strike and an expiration, "
            "so the engine would hold them forever at their last quoted premium instead of "
            "settling them. Re-ingest from a connector that reports contract terms "
            "(deribit, databento), or set config.option_contracts by hand."
        )

    missing = []
    contracts = {}
    for sid, terms in available.items():
        sid = int(sid)
        if sid not in in_universe:
            continue
        if sid not in underlyings:
            missing.append(sid)
            continue
        contracts[sid] = dict(terms, underlying_id=underlyings[sid])

    if missing:
        names = {int(i): t for i, t in store.list_symbols()}
        listed = ", ".join(f"{sid} ({names.get(sid, '?')})" for sid in sorted(missing))
        raise ValueError(
            f"option symbol(s) {listed} have contract terms but no settlement "
            "underlying. Set config.option_underlyings = {option_id: underlying_id}; "
            "an option cannot be settled against its own last traded premium."
        )
    cfg.option_contracts = contracts


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
    ``"bitstamp"``, ``"deribit"``, ``"yahoo"`` (alias ``"yfinance"``).
    Pro: ``"databento"``, ``"massive"``.

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

    Example (stocks, ETFs, indices, FX and futures via Yahoo Finance)::

        store = bt.ingest(
            provider="yahoo",
            symbol="AAPL",
            symbol_id=1,
            start="2015-01-01T00:00:00Z",
            end="2026-01-01T00:00:00Z",
            interval="1d",
            asset_class="equity",
        )

    Yahoo caps its own history: 1m goes back 30 days, 1h about 2 years, daily
    to the listing date. Prices are dividend-adjusted like ``yfinance``'s
    ``auto_adjust=True``; pass ``dataset="raw"`` for unadjusted quotes.

    Example (a Deribit option, including one that has already expired)::

        store = bt.ingest(
            provider="deribit",
            symbol="BTC-27JUN25-100000-C",
            symbol_id=2,
            start="2025-05-01T00:00:00Z",
            end="2025-07-01T00:00:00Z",
            interval="1d",
            asset_class="option",
        )

    Deribit is the only free connector here that serves expired contracts, which
    is what an option backtest needs. The strike, expiration, side and settlement
    style are read from the venue and stored beside the bars, so the engine can
    settle the contract instead of holding it forever. Prices are quoted in the
    base currency, so such a backtest is denominated in BTC, ``initial_capital``
    included. Set ``config.option_underlyings`` to say which series settles it.

    ``databento`` and ``massive`` (both Pro) report the same terms for US listed
    options: Databento from the ``definition`` schema of a dataset such as
    ``OPRA.PILLAR``, Massive from ``/v3/reference/options/contracts`` on an OSI
    ticker like ``"O:SPY251219C00650000"``. Two things differ from Deribit.
    Positions are counted in units of the underlying, so one 100-multiplier
    contract is a position of 100. And US listed equity options are physically
    settled, which the engine models as cash at intrinsic: exact for an index
    option, an approximation for a single-stock one.
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


def attach_quotes(
    store: DataStore,
    symbol,
    start: str,
    end: str,
    *,
    provider: str = "bybit",
    provider_symbol: Optional[str] = None,
    category: str = "linear",
    cache_dir: Optional[str] = None,
    progress: bool = True,
) -> dict:
    """Fill the ``bid`` / ``ask`` / ``spread`` columns of already-stored bars
    with real top-of-book quotes.

    The bar schema has always carried these columns and the engine has always
    known how to read them (``execution_price="MidPrice"``, per-bar spread for
    cost analysis) — but no ingestion path filled them until now, so they were
    null. This does, from Bybit's public order-book archives: one ~200 MB
    archive per day is streamed and reduced to the best bid/ask standing at
    each bar close, and only those few KB touch the store. Bars outside the
    quoted range keep their current values.

    Bybit's archive is a rolling window of roughly one year, ``linear``
    (USDT perps) and ``spot``. The venue quoted is Bybit: on another venue's
    bars the spread is that of a different book — a reasonable proxy for
    majors, a real approximation for thin alts.

    Args:
        store: The store holding the symbol's bars (see :func:`ingest`).
        symbol: The store ticker (e.g. ``"BTC-USDT:perp"``) or symbol id.
        start: First day, ``YYYY-MM-DD``.
        end: Last day, inclusive, ``YYYY-MM-DD``.
        provider: ``"bybit"`` (the only quote source with free history).
        provider_symbol: Symbol on the provider (e.g. ``"BTCUSDT"``). Derived
            from the ticker when omitted: ``BTC-USDT:perp`` -> ``BTCUSDT``.
        category: ``"linear"`` or ``"spot"``.
        cache_dir: Keep the raw daily archives here and reuse them.
        progress: Print one line per day.

    Returns:
        ``{"days", "quote_bars", "bars_updated", "version"}``.

    Example::

        store = bt.ingest(provider="bybit", symbol="BTCUSDT", symbol_id=1,
                          start="2026-07-01T00:00:00Z", end="2026-08-01T00:00:00Z")
        bt.attach_quotes(store, 1, "2026-07-01", "2026-07-31")
        # bars now carry real bid/ask: MidPrice execution and per-bar
        # spread costs read actual quotes instead of nulls.
    """
    import datetime as _dt

    # No Python-side check: the native side answers for this call, and its
    # refusal is the accurate one. A _require_pro here would stop a Pro user
    # one layer early with the WRONG message -- an upgrade they already own.
    # Same policy as manifoldbt.ticks.
    if provider != "bybit":
        raise ValueError(
            f"unknown quote provider {provider!r}: 'bybit' is the only source "
            "with free order-book history"
        )

    if isinstance(symbol, int):
        symbol_id = symbol
        ticker = None
    else:
        ticker = str(symbol)
        symbol_id = store.resolve_symbol(ticker)
    if provider_symbol is None:
        base = ticker if ticker is not None else ""
        provider_symbol = base.split(":")[0].replace("-", "").replace("/", "").upper()
        if not provider_symbol:
            raise ValueError(
                "provider_symbol is required when symbol is given as an id"
            )

    d0 = _dt.date.fromisoformat(start)
    d1 = _dt.date.fromisoformat(end)
    if d1 < d0:
        raise ValueError(f"end {end!r} is before start {start!r}")
    dates = [
        (d0 + _dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)
    ]

    cb = None
    if progress:
        def cb(i, n, date):
            if i < n:
                print(f"  quotes {provider_symbol} {date} ({i + 1}/{n})", flush=True)

    from manifoldbt._native import py_attach_quotes_bybit as _native_attach

    return _native_attach(
        store.data_root(),
        store.metadata_db(),
        symbol_id,
        provider_symbol,
        category,
        dates,
        cache_dir,
        cb,
    )


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


_BARS_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

# The nullable f64 columns of the canonical bar schema. A caller who has them
# (own quotes on equities, forex, options; a venue's buy/sell split) keeps them;
# a caller who does not gets nulls, which the engine reads as "this venue does
# not publish it". Anything else in the frame is still dropped on purpose:
# the store schema is fixed, and an unknown column would be a silent no-op.
_BARS_OPTIONAL_COLUMNS = ("buy_volume", "sell_volume", "bid", "ask", "spread")


def _df_to_bars_batch(data):
    """Normalise a pandas/polars DataFrame (or dict) to a pyarrow RecordBatch.

    Output contract (what the native import expects): columns
    ``timestamp`` (timestamp[ns, UTC]), ``open/high/low/close/volume`` (f64),
    plus whichever of ``buy_volume/sell_volume/bid/ask/spread`` the frame
    carries (nullable f64, kept in that order after the required ones).
    Naive timestamps are assumed UTC. A pandas DatetimeIndex is promoted to
    the ``timestamp`` column when the column is absent.
    """
    import pyarrow as pa

    # --- to Arrow Table (same dispatch as register_exo) ---
    if hasattr(data, "to_arrow"):
        # Polars DataFrame
        table = data.to_arrow()
    elif hasattr(data, "columns"):
        # Pandas DataFrame
        import pandas as pd
        if "timestamp" not in data.columns and isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index(names="timestamp")
        table = pa.Table.from_pandas(data, preserve_index=False)
    elif isinstance(data, dict):
        table = pa.table(data)
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. Use a pandas/polars DataFrame or dict."
        )

    missing = [c for c in _BARS_REQUIRED_COLUMNS if c not in table.column_names]
    if missing:
        raise DataError(
            f"DataFrame is missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(_BARS_REQUIRED_COLUMNS)}"
        )
    optional = [c for c in _BARS_OPTIONAL_COLUMNS if c in table.column_names]
    table = table.select(list(_BARS_REQUIRED_COLUMNS) + optional)

    # --- timestamp → timestamp[ns, UTC] ---
    ts_type = table.schema.field("timestamp").type
    if not pa.types.is_timestamp(ts_type):
        raise DataError(
            f"'timestamp' column must be a datetime type, got {ts_type}. "
            "For epoch integers, convert first: pd.to_datetime(ts, unit='ms', utc=True)"
        )
    target_ts = pa.timestamp("ns", tz="UTC")
    if ts_type != target_ts:
        table = table.set_column(
            0, pa.field("timestamp", target_ts), table.column(0).cast(target_ts)
        )

    # --- value columns → float64 (the optional ones stay nullable) ---
    for i, name in enumerate(_BARS_REQUIRED_COLUMNS[1:] + tuple(optional), start=1):
        if table.schema.field(i).type != pa.float64():
            table = table.set_column(
                i, pa.field(name, pa.float64()), table.column(i).cast(pa.float64())
            )

    if table.num_rows == 0:
        raise DataError("DataFrame contains no data rows")

    # Single contiguous batch for the zero-copy FFI crossing.
    return table.combine_chunks().to_batches()[0]


def import_dataframe(
    data,
    symbol: str,
    symbol_id: int,
    *,
    interval: str = "1m",
    data_root: str = "data",
    metadata_db: str = "metadata/metadata.sqlite",
    exchange: str = "DATAFRAME",
    asset_class: str = "crypto_spot",
) -> DataStore:
    """Import bars from an in-memory DataFrame into the Arrow IPC store. Free on all tiers.

    The in-memory twin of :func:`import_csv`: edit your data as a DataFrame,
    then import it directly — no intermediate CSV. Returns a :class:`DataStore`
    ready for :func:`run` (same store, metadata and versioning as ``bt.ingest``).
    Reopen it later with ``DataStore(data_root, metadata_db)``: the Arrow IPC
    layout under ``<data_root>/mega`` is detected automatically.

    Accepts a pandas DataFrame, polars DataFrame, or dict of columns with
    ``timestamp`` (datetime; naive values are assumed UTC), ``open``, ``high``,
    ``low``, ``close``, ``volume``. A pandas DatetimeIndex is used as
    ``timestamp`` if that column is absent. Rows must be sorted by timestamp.

    Optional columns are kept when present: ``bid``, ``ask``, ``spread``,
    ``buy_volume``, ``sell_volume`` (missing values allowed). With ``bid`` and
    ``ask`` but no ``spread``, the spread is derived as ``ask - bid``. Bars
    that carry quotes get real ``execution_price="MidPrice"`` fills and real
    per-bar spread costs; bars without them fall back to the close, as before.

    Example::

        df = pd.read_parquet("EURUSD_1m.parquet")
        df["close"] = df["close"].clip(upper=1.5)   # edit in memory
        store = bt.import_dataframe(df, symbol="EURUSD", symbol_id=1,
                                    interval="1m", asset_class="forex")
        result = bt.run(strategy, config, store)

    Args:
        data: pandas/polars DataFrame or dict of columns.
        symbol: Ticker name (e.g. ``"EURUSD"``, ``"BTCUSDT"``).
        symbol_id: Unique integer ID for this symbol in the store.
        interval: Bar interval of the rows (``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``, ...).
        data_root: Store directory (default ``"data"``).
        metadata_db: Metadata SQLite path.
        exchange: Exchange label for metadata (default ``"DATAFRAME"``).
        asset_class: ``crypto_spot``, ``crypto_perp``, ``equity``, ``future``,
            ``option``, ``forex``, or ``index``.
    """
    batch = _df_to_bars_batch(data)
    try:
        return _import_dataframe_native(
            batch,
            symbol=symbol,
            symbol_id=symbol_id,
            interval=interval,
            data_root=data_root,
            metadata_db=metadata_db,
            exchange=exchange,
            asset_class=asset_class,
        )
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


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

def _apply_cross_asset(strategy, config: BacktestConfig, store: DataStore):
    """Rewrite a bare ``symbol_ref()`` strategy to the orchestrator-ready form.

    The engine evaluates SymbolRef only with a dict universe, provider-qualified
    references and no SymbolRef inside position sizing. Strategies written the
    natural way (``symbol_ref("BTCUSDT", "close")`` with a list universe) are
    rewritten here -- see :mod:`manifoldbt.crossasset`. Returns
    ``(strategy_json, config)``; both are the originals when there is nothing
    to rewrite, and resolution failures fall back to the engine's own error.
    """
    try:
        doc = strategy.to_json_dict()
    except Exception:
        return strategy.to_json(), config
    try:
        meta_db = store.metadata_db()
    except Exception:
        meta_db = None
    try:
        prepared = _prepare_cross_asset(doc, config.universe, meta_db)
    except ValueError:
        prepared = None
    if prepared is None:
        return strategy.to_json(), config
    new_doc, dict_universe = prepared
    if dict_universe is None:
        return json.dumps(new_doc), config
    cfg = copy.deepcopy(config)
    cfg.universe = dict_universe
    return json.dumps(new_doc), cfg


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
        strategy_json, config = _apply_cross_asset(strategy, config, store)
        cfg_json = _prepared_config_json(config, strategy, store)
        raw = _run_native(strategy_json, cfg_json, store)
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
        Results come in the engine's enumeration order: axes sorted by
        parameter name, last axis varying fastest, whatever order the dict
        was written in (:func:`manifoldbt.dataframe.grid_combos` lists it).
        ``to_df()`` labels each row from the run's own manifest, so it does
        not depend on that order.
    """
    _require_pro_over_combos(_grid_combos(param_grid), "Parameter sweep")
    _validate_swept_params(strategy, param_grid.keys(), "Parameter sweep")
    try:
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        strategy_json, config = _apply_cross_asset(strategy, config, store)
        cfg_json = _prepared_config_json(config, strategy, store)
        grid_json = json.dumps({
            name: [scalar_value_to_json(v) for v in values]
            for name, values in param_grid.items()
        })
        raw_results = _run_sweep_native(
            strategy_json,
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

    Per-strategy ``stop_loss``/``take_profit``/``trailing_stop`` are honored:
    each strategy's orders travel inside its JSON and the engine applies them
    per-strategy, so a batch of strategies with DIFFERENT brackets still runs
    over a single data load.

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
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        cfg_json = _prepared_config_json(config, None, store)
        raw_results = _run_batch_native(
            [strat.to_json() for strat in strategies],
            cfg_json,
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

    Per-strategy ``stop_loss``/``take_profit``/``trailing_stop`` are honored:
    each strategy's orders travel inside its JSON and the engine applies them
    per-strategy, so a batch of strategies with DIFFERENT brackets still runs
    over a single data load.

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
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        cfg_json = _prepared_config_json(config, None, store)
        return _run_batch_lite_native(
            [strat.to_json() for strat in strategies],
            cfg_json,
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
    device: str = "auto",
    precision: str = "fp64",
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
        device: ``"auto"`` (default), ``"cpu"``, or ``"cuda"``/``"gpu"``.
            The GPU path produces results numerically identical to the CPU
            path. ``"auto"`` picks per sweep: small grids run on the CPU (the
            GPU has a ~50 ms fixed launch floor, so the CPU wins below ~1,000
            combos -- override with ``MBT_GPU_AUTO_MIN_COMBOS``), large grids
            run on the GPU when the build, a device, and a Pro license are
            available, and the CPU otherwise. This is the default because it is
            never slower than the better of the two by more than the launch
            floor and its results match the CPU bit-for-bit, so it is safe to
            leave on: with no GPU, no Pro license, or a Community build it is
            simply the CPU sweep. **Pro-only**: a Community license raises
            ``PermissionError`` for ``device="cuda"`` (``"auto"`` simply stays
            on the CPU; Community keeps the full-speed CPU sweep with no
            restriction). ``"cuda"`` requires a build with ``--features cuda``
            and a CUDA device; for any unsupported strategy/config (or when no
            GPU is present at runtime) it falls back to the CPU sweep with a
            ``UserWarning`` naming the reason, so results are never affected.
            An unknown device string raises ``ValueError`` instead of silently
            running on the CPU.
        precision: ``"fp64"`` (default) runs the GPU sweep in double precision,
            bit-identical to the CPU path. ``"fp32"`` runs the single-asset GPU
            kernel in single precision at the cost of approximate results: a
            signal within ~1e-7 relative of a decision threshold can flip vs f64,
            so occasional combos diverge. Intended as a **scan-only** accelerator
            (rank in fp32, re-run the winner in fp64 for an exact P&L). Note the
            speedup is modest (~1.1x measured on an RTX 3090): the per-bar
            capital/position recurrence is latency-bound, so fp32's throughput
            advantage barely applies. ``"fp32"`` requires ``device="cuda"``.

    Metric resolution:
        The lite path computes risk metrics from one equity point per UTC day
        (this is what makes it fast), whereas :func:`run` uses the full-resolution
        curve. ``final_equity``, ``total_return``, ``sharpe``, ``sortino``,
        ``volatility`` and ``max_drawdown`` are unaffected -- they match ``run``
        exactly. Three annualisation-sensitive metrics differ slightly because
        they are derived from the daily series: ``cagr`` (it starts from the
        first daily equity rather than initial capital), ``calmar`` and
        ``ulcer_index``. The gap is small (< ~0.4% relative on a multi-year daily
        backtest) and is the same for every sweep regardless of orders. Sort and
        rank on it freely; for an exact single-figure P&L, re-run the winning
        combo through :func:`run`.

    Returns:
        One :class:`BatchResultLite` per combo, in the engine's enumeration
        order: axes sorted by parameter NAME, last axis varying fastest. This
        is not the dict's insertion order, so a reshape on the dict's order
        silently transposes the grid. :func:`manifoldbt.dataframe.grid_combos`
        lists the combinations in this order, and
        :func:`manifoldbt.dataframe.results_to_df` labels the results with it.
    """
    _require_pro_over_combos(_grid_combos(param_grid), "Parameter sweep")
    _validate_swept_params(strategy, param_grid.keys(), "Parameter sweep")
    _require_pro_for_gpu(device, "GPU sweep")
    try:
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        strategy_json, config = _apply_cross_asset(strategy, config, store)
        cfg_json = _prepared_config_json(config, strategy, store)
        grid_json = json.dumps({
            name: [scalar_value_to_json(v) for v in values]
            for name, values in param_grid.items()
        })
        # Wrapped in a list subclass: echoing a sweep in a notebook cell
        # printed one BatchResultLite line per combo. Indexing, iteration and
        # len() are unchanged.
        from manifoldbt._reprs import wrap_sweep_lite
        return wrap_sweep_lite(_run_sweep_lite_native(
            strategy_json,
            grid_json,
            cfg_json,
            store,
            max_parallelism,
            device,
            precision,
        ))
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def sweep_columns(
    batch: List["BatchResultLite"],
    names: Union[str, List[str]],
) -> Union["Any", Dict[str, "Any"]]:
    """Extract whole metric columns from a sweep as numpy arrays.

    ``result.metrics`` builds a 21-key dict per combo, so reading one metric off
    a large sweep creates millions of throwaway floats. This walks the results
    once and copies each requested column straight into a numpy array, which is
    ~20x faster: on a 1M-combo sweep, ~1.1s of extraction becomes ~0.05s.

    Args:
        batch: The list returned by :func:`run_sweep_lite`.
        names: One column name, or a list of them. Available: ``final_equity``,
            ``trade_count``, and every :class:`PerformanceMetrics` field
            (``sharpe``, ``sortino``, ``calmar``, ``max_drawdown``, ``alpha``,
            ``beta``, ``tstat_alpha``, ``total_return``, ``cagr``,
            ``volatility``, ``skewness``, ``kurtosis``, ``tail_ratio``,
            ``omega_ratio``, ``ulcer_index``, ``best_day``, ``worst_day``,
            ``avg_daily_return``, ``pct_positive_days``,
            ``max_drawdown_duration_days``, ``tstat_sharpe``).

    Returns:
        A single ``np.ndarray`` if ``names`` is a string, else a dict mapping
        each name to its array. Arrays are float64 and in combo order (the same
        order as ``batch``), so ``np.argmax``/``argsort`` indices map straight
        back onto it. ``trade_count`` comes back as float64 like the rest.

    Note:
        The arrays are read-only views over the returned buffers (no copy). Call
        ``.copy()`` if you need to mutate one.

    Example:
        >>> batch = mbt.run_sweep_lite(strategy, grid, config, store, device="cuda")
        >>> sharpe = mbt.sweep_columns(batch, "sharpe")
        >>> best = batch[int(sharpe.argmax())]
    """
    import numpy as _np

    single = isinstance(names, str)
    wanted = [names] if single else list(names)
    raw = _sweep_columns_native(batch, wanted)
    out = {n: _np.frombuffer(raw[n], dtype=_np.float64) for n in wanted}
    return out[names] if single else out


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
            geometry (str): "anchored" (default), "blocked", "pardo" or
                "custom".
                - "anchored"/"blocked" take ``n_splits`` + ``train_ratio``.
                - "pardo"/"custom" take ``train``/``test`` window specs; the
                  fold count is DERIVED from the window lengths, never chosen.
            n_splits (int): Number of folds (anchored/blocked only).
            train_ratio (float): Training fraction in (0, 1) (anchored/blocked).
            train (dict): pardo: ``{"length": Interval.days(365)}`` (fixed
                sliding window W). custom: ``{"mode": "anchored", "min_length":
                ...}`` or ``{"mode": "rolling", "length": ...}``. Every
                duration also accepts a ``*_bars`` twin (signal bars).
            test (dict): ``{"length": Interval.days(90), "step":
                Interval.days(30)}``. ``step`` defaults to ``length`` (tests
                tile end to end, the only shape whose OOS segments chain into
                one tradable curve); ``step < length`` = overlapping windows,
                flagged by ``folds_overlap``; ``step > length`` is refused.
            optimize_metric (str): e.g. "sharpe", "sortino".
            param_grid (dict): Parameter grid for optimization.
            max_parallelism (int): Max threads.
            device (str): "auto" (default), "cpu" or "cuda".
        config: Backtest configuration.
        store: Data store.

    Returns:
        Dict with ``folds``, ``best_params_per_fold``, ``n_folds``,
        ``folds_overlap``, ``effective_folds`` (independent folds: overlapping
        windows count for less) and ``walk_forward_efficiency`` (Pardo's WFE,
        mean of per-fold ``oos.cagr / is.cagr``).

    Each fold's OOS run is WARMED UP: it simulates from the fold's train start
    with trading suppressed until the test window, so indicators are hot at
    the boundary instead of restarting empty.

    Note: the legacy ``method="Rolling"`` was renamed ``geometry="blocked"``
    (independent blocks separated by gaps, not Pardo's rolling); for Pardo's
    walk-forward use ``geometry="pardo"``.
    """
    # Pro feature: report it here so a notebook gets a clean LicenseError rather
    # than a traceback from deeper in the run.
    _require_pro("Walk-forward optimization")
    _validate_swept_params(strategy, (wf_config.get("param_grid") or {}).keys(),
                           "Walk-forward")
    config = _prepare_config(config, strategy, store)
    wf_json = json.dumps(_convert_param_grid_in_config(wf_config))
    raw = _run_walk_forward_native(strategy.to_json(), wf_json, config.to_json(), store)
    # Wrapped in a dict subclass: the raw dict holds a full equity curve per
    # fold, so echoing it in a cell printed tens of thousands of floats.
    from manifoldbt._reprs import wrap_walk_forward
    return wrap_walk_forward(raw)


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
    _validate_swept_params(
        strategy,
        [n for n in (sweep_config.get("x_param"), sweep_config.get("y_param")) if n],
        "2D parameter sweep")
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
    _validate_swept_params(
        strategy,
        [n for n in (stability_config.get("param_name"),) if n],
        "Parameter stability analysis")
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
    if name == "ticks":
        return _importlib.import_module("manifoldbt.ticks")
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
        # Minuscules obligatoires: les deux ecrivains Rust (ingest.rs) et les
        # deux lecteurs creent ce dossier en minuscules. Ecrire "BINANCE" ici
        # produisait un second dossier, invisible aux lecteurs sur un systeme
        # de fichiers sensible a la casse.
        target_dir = root / provider.lower() / timeframe
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


def guide() -> None:
    """Print a compact API cheat sheet, written for coding agents and new users.

    Everything here is importable from the top-level module unless noted.
    Call ``bt.guide()`` and read the output; it answers most of what a pile of
    ``help()`` calls would, including whether the DSL can express a given
    strategy shape (see the worked recipes at the end -- it can).

    Prints and returns None, like :func:`help`.
    """
    print(_GUIDE)


_GUIDE = """\
manifoldbt cheat sheet
======================

Data
----
store = bt.DataStore(data_root, metadata_db)   # backend auto-detected, incl. Arrow IPC
store.list_symbols()                           # [(id, ticker), ...]
bt.import_dataframe(df, symbol="X", symbol_id=1, interval="1h",
                    data_root=..., metadata_db=...)   # cols: timestamp/open/high/low/close/volume
bt.ingest(provider="binance", symbol="BTCUSDT", symbol_id=1,
          start="2024-01-01", end="2024-06-01", interval="1m")

Strategy (expression DSL -- no per-bar Python)
----------------------------------------------
from manifoldbt.indicators import sma, ema, rsi   # 104 indicator functions
sig  = bt.when(cond, then, otherwise)             # conditions combine with & | ~
pos  = bt.when(z >= bt.lit(2.0), bt.lit(-1.0),
       bt.when(z <= bt.lit(-2.0), bt.lit(1.0),
       bt.when(abs(z) <= bt.lit(0.5), bt.lit(0.0), bt.hold())))  # hold() = keep previous
strat = (bt.Strategy.create("name")
         .signal("position", pos)     # signals are named series
         .size(pos))                  # target fraction of equity; +long / -short
other = bt.symbol_ref("ETHUSDT", "close")   # cross-asset reference; auto-wired by run()
z     = (bt.col("close") / other).zscore(200)   # rolling ops are methods on expressions

Config
------
start, end = bt.time_range("2022-01-01", "2025-01-01")   # UNIX NANOSECONDS -- never raw ints
cfg = bt.BacktestConfig(
    universe=["BTCUSDT"],             # tickers, ids, or {"provider": [...]}
    time_range_start=start, time_range_end=end,
    bar_interval=bt.Interval.hours(1),
    initial_capital=100_000.0,
    warmup_bars=50,
    execution=bt.ExecutionConfig(signal_delay=1,          # bars between signal and fill
                                 execution_price="AtClose",
                                 allow_short=False, max_position_pct=1.0,
                                 position_sizing_mode="FractionOfEquity"),
    fees=bt.FeeConfig(taker_fee_bps=5.0, maker_fee_bps=5.0),   # or FeeConfig.zero()
    slippage=bt.Slippage.none(),      # or Slippage.fixed_bps(2)
)

Run and read results
--------------------
res = bt.run(strat, cfg, store)
res.metrics                     # dict: total_return, sharpe, sortino, max_drawdown,
                                # cagr, calmar, volatility, ... and trade_stats
res.metrics["trade_stats"]      # total_trades counts FILLS; round-trips are under
                                # "round_trips"; also win_rate, profit_factor, ...
res.equity_df(); res.trades_df(); res.daily_returns_series(); res.summary()

Research
--------
bt.run_sweep(strat, {"fast": [10, 20], "slow": [50, 100]}, cfg, store)
bt.run_walk_forward(...)   # and run_stability, run_stochastic, run_portfolio

Worked recipes -- yes, the DSL expresses these
----------------------------------------------
Stateful thresholds (hysteresis). hold() keeps the previous position, so a band
between entry and exit needs no Python loop:

    z = (bt.col("close") / bt.symbol_ref("ETHUSDT", "close")).zscore(200)
    pos = bt.when(z >= bt.lit(2.0), bt.lit(-1.0),          # short the spread
          bt.when(z <= bt.lit(-2.0), bt.lit(1.0),          # long the spread
          bt.when((z >= bt.lit(-0.5)) & (z <= bt.lit(0.5)), bt.lit(0.0),
          bt.hold())))                                     # else: keep position
    strat = bt.Strategy.create("pair").signal("pos", pos).size(bt.col("pos"))
    cfg = bt.BacktestConfig(universe=["BTCUSDT"], ...)     # plain list is fine

Several legs against one anchor. Same expression, several traded symbols: each
one gets its own state and its own position, equity is shared. The anchor is
simply left out of the universe.

    cfg = bt.BacktestConfig(universe=["SOLUSDT", "AVAXUSDT", "DOTUSDT"], ...)
    # each leg's z is computed against symbol_ref("ETHUSDT", ...); ETH is not traded

Cross-sectional (rank/zscore across the universe at each bar). The op must be
the WHOLE signal -- the engine refuses anything wrapped around it rather than
silently ignoring it, and a cross-sectional signal cannot feed another signal:

    strat = bt.Strategy.create("xs").signal("pos", bt.col("close").cs_zscore()).size(bt.col("pos"))

Common errors
-------------
"empty bar dataset ... over time_range [a, b) ns"  -> the window is wrong, not the
    data: time_range values are UNIX nanoseconds; build them with bt.time_range().
"symbol not found"                                 -> store.list_symbols() shows what exists.
"requires orchestrator-level multi-symbol handling" -> a symbol_ref() reached the
    per-symbol evaluator; use bt.run() (it rewrites automatically) or qualify the
    reference as "provider:TICKER" with a dict universe.
"cross-sectional op ... must be the whole signal"  -> give the cs op its own
    signal; thresholding one is not supported (see the recipe above).
"""


__all__ = [
    # Core types
    "BacktestResult",
    "BatchResultLite",
    "DataStore",
    "Result",
    "SweepResult",
    # Data ingestion
    "ingest",
    "attach_quotes",
    "import_csv",
    "import_dataframe",
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
    "choice",
    "tf",
    "when",
    # Strategy & config
    "Strategy",
    "BacktestConfig",
    "ExecutionConfig",
    "FeeConfig",
    "VenueFees",
    "OrderConfig",
    "entry_price",
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
    # NOT exported: py_run_monte_carlo. It is the raw native binding under
    # plot.monte_carlo's friendly layer (which warns before the native cap
    # refuses); listing it in __all__ published an internal as API.
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
    "check_for_update",
    # Agent/API cheat sheet
    "guide",
    # Indicators (submodule)
    "indicators",
    # Managed compute (submodule)
    "cloud",
    # Plotting (lazy, requires plotly)
    "plot",
    # Diagnostics (lazy)
    "diagnostics",
    # Tick-level layer (lazy)
    "ticks",
]
