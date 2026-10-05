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
    _flush_usage as _flush_usage_native,
    license_expiry as _license_expiry,
    license_info as _license_info,
    _grant_couvre,
    compile_strategy_json,
    run as _run_native,
    run_quotes as _run_quotes_native,
    run_quote_sweep as _run_quote_sweep_native,
    run_quote_sweep_lite as _run_quote_sweep_lite_native,
    run_quote_batch as _run_quote_batch_native,
    run_quote_batch_lite as _run_quote_batch_lite_native,
    run_with_fill_rule as _run_with_fill_rule_native,
    run_batch as _run_batch_native,
    run_batch_lite as _run_batch_lite_native,
    run_json,
    run_sweep as _run_sweep_native,
    run_sweep_lite as _run_sweep_lite_native,
    reconcile_fills as _reconcile_fills_native,
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
    AccountPhase,
    AccountRules,
    BacktestConfig,
    ExecutionConfig,
    FeeConfig,
    OrderConfig,
    VenueFees,
    account_sessions,
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
from manifoldbt.expr import AssetRef, Expr, TimeframeRef, asset, cash, choice, clip, col, exo, hold, last_cancel_age, last_fill_age, last_fill_px, lit, live_qty, order_age, order_price, param, position, position_age, queue_ahead, round, s, scan, symbol_ref, tf, when
from manifoldbt.helpers import (
    ExecutionPrice,
    FillModel,
    Interval,
    Slippage,
    date_to_ns,
    time_range,
)
from manifoldbt.portfolio import Portfolio
from manifoldbt.reconcile import Reconciliation, fills_to_columns as _fills_to_columns
from manifoldbt.result import QuoteResult, Result
from manifoldbt.strategy import Strategy
from manifoldbt.sweep import SweepResult
from manifoldbt import indicators
from manifoldbt import book

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

# Fold this session's anonymous usage counters into their file on the way out.
# Without it a script short enough never to reach the periodic flush (most
# scripts) would count everything and persist nothing. Order does not matter:
# this neither prints nor reads anything the hooks around it touch, and it is a
# no-op under MANIFOLDBT_NO_TELEMETRY / DO_NOT_TRACK / CI.
atexit.register(_flush_usage_native)


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


def _require_grant_for_gpu(device, feature: str) -> None:
    """Gate GPU acceleration (``device="cuda"``/``"gpu"``) behind its tier.

    Reported here so that every GPU entry point raises the same clean
    ``LicenseError``, instead of each surfacing its own error type from deeper in
    the run. No-op for CPU, and no-op for a licence that carries it.

    GPU acceleration is a Researcher feature. Note that
    ``license_info()`` reports ``"Pro"`` for a Researcher licence, because
    Researcher includes everything Pro has: the tier string is not the way to
    tell whether a given feature is covered.
    """
    if isinstance(device, str) and device.lower() in ("cuda", "gpu"):
        if _grant_couvre("gpu_sweep"):
            return
        raise LicenseError(
            f"'{feature}' is a Researcher feature; a Pro licence does not unlock "
            f"it. What Researcher includes: www.manifoldbt.com/researcher"
        )


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
#: l'evalue. Doit rester aligne sur la liste des distances que le moteur
#: substitue lui-meme a chaque combinaison.
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


#: Execution prices that `signal_delay = 0` is sound with: both fill at the
#: close of the bar that produced the signal, which is the information the
#: decision was made on. Every other price is a point *inside* that bar, so
#: reaching it would have meant trading before that close.
_CLOSE_EXECUTION_PRICES = frozenset({"AtClose", "NextBarClose"})

#: Columns `ExecutionPrice.custom(name)` resolves against the bar schema,
#: before it looks at the strategy's own signals. Filling on one of these at
#: `signal_delay = 0` hands the strategy a point of its own bar, which is why
#: `custom("low")` measures +358% on the bench where `AtClose` measures +5%.
#: A name absent from this set is either a signal the strategy declared, which
#: is legitimate, or unknown, which the engine rejects on its own.
_BAR_COLUMNS = frozenset({
    "open", "high", "low", "close", "volume", "vwap",
    "bid", "ask", "spread", "buy_volume", "sell_volume", "trade_count",
})


def _reject_intrabar_price_without_delay(cfg: BacktestConfig) -> None:
    """At `signal_delay = 0`, a fill cannot land on a point of the bar it decided on.

    Decide on bar `t`, execute on bar `t` is sound at the close: the fill price
    *is* the information the decision was made on, and it is what
    `signal_delay = 0` exists for. It stops being sound as soon as the fill
    lands somewhere else inside that same bar, because reaching that point
    would have meant trading before the close that produced the signal.
    `AtOpen` fills at a price printed before it; `AtVwap` and `MidPrice` are
    averages over the whole bar; a custom *bar column* like `low` or `high`
    hands the strategy the bar's own extreme, which is perfect intrabar timing.
    `NextBarOpen` counts from the signal row shifted by the delay, so at delay
    zero it names the current bar and behaves exactly like `AtOpen`.

    Measured on a driftless random walk, "long when close > open" returns
    +1913% under `AtOpen`, +361% under `AtVwap`, `MidPrice` and
    `custom("low")`, against +5% under the defaults. The engine said nothing.

    **`custom(<a signal the strategy defines>)` is exempt, and deliberately.**
    There the level is one the DSL computed, and the fill only books if the bar
    actually traded through it: the band case of `test_exec_price_signal`. It
    measures +5% on the same bench, like the close. What that level owes to its
    own bar is the resting-order question, which
    `_reject_resting_order_without_delay` already carries.
    """
    execution = getattr(cfg, "execution", None)
    if execution is None or getattr(execution, "signal_delay", 1) != 0:
        return
    price = getattr(execution, "execution_price", "AtClose")
    # `ExecutionPrice.custom(name)` serialises as {"Custom": name}.
    label = next(iter(price)) if isinstance(price, dict) else price
    if label in _CLOSE_EXECUTION_PRICES:
        return
    if isinstance(price, dict):
        # Refuse only a name that resolves against the BAR schema. A name the
        # strategy declares is a level it chose, and is exempt. A name that is
        # neither stays for the engine to reject, so its own "neither a bar
        # column nor a signal" message reaches the user unchanged.
        if price[label] not in _BAR_COLUMNS:
            return
    shown = f'custom({price[label]!r})' if isinstance(price, dict) else label
    # `ConfigError`, pas `ValueError` : `run()` enveloppe toute `ValueError`
    # dans `_classify_error`, qui classe par mots-cles. Ce message contient
    # "signal", donc il ressortirait en `StrategyError` alors que c'est la
    # configuration d'execution qui est en cause. Lever la bonne classe des le
    # depart la traverse intacte.
    raise ConfigError(
        f"execution_price={shown} needs signal_delay >= 1: with signal_delay=0 "
        "the order fills at a price taken from the same bar whose close "
        "produced the signal, so the fill precedes its own cause. Set "
        "ExecutionConfig(signal_delay=1) to decide on one bar and execute on "
        "the next, or keep signal_delay=0 with execution_price='AtClose' "
        "(the default), which fills at the very close the signal was computed "
        "on."
    )


def _prepare_config(config: BacktestConfig, strategy, store: DataStore) -> BacktestConfig:
    """Prepare config for execution: resolve symbols, convert deprecated fields."""
    cfg = copy.deepcopy(config)
    _reject_resting_order_without_delay(cfg, strategy)
    _reject_intrabar_price_without_delay(cfg)

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


def _is_trade_clock(interval: Any) -> bool:
    """Return True for ``Interval.trades()``, which is a name, not a dict.

    `isinstance` first: an `Expr` answers `==` with a comparison expression
    rather than a bool, and a truthy object here would misread it as the clock.
    """
    return isinstance(interval, str) and interval == "Trades"


def _is_sub_daily(res: Any) -> bool:
    """Return True if an Interval dict represents sub-daily resolution.

    ``Interval.trades()`` answers False, and deliberately: it is not a
    resolution to round down but a SHAPE, legal only under the trade clock. The
    cap below turns a resolution into a coarser one; turning this one into
    ``None`` would replace a named refusal with a silent success.
    """
    if _is_trade_clock(res):
        return False
    if not isinstance(res, dict):
        return False
    if "Seconds" in res or "Minutes" in res:
        return True
    if "Hours" in res and res["Hours"] < 24:
        return True
    return False


def _interval_to_seconds(interval: Any) -> int:
    """Convert an Interval dict to total seconds.

    ``Interval.trades()`` is not a duration and answers 0, which reads
    everywhere here as "finer than anything", the truth for one row per trade.
    """
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

    # The trade clock reads the tape, never a bar dataset: swapping datasets
    # under it would pick a resolution nothing then loads.
    if _is_trade_clock(config.bar_interval):
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
    """Cap output_resolution to daily for Community users (Pro feature).

    Per-event output (``Interval.trades()``) is left exactly as written: the
    engine answers it, either by refusing the shape or by refusing the clock it
    belongs to, and both answers are better than a silent downgrade.
    """
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
    ``"bitstamp"``, ``"deribit"``, ``"yahoo"`` (alias ``"yfinance"``),
    ``"dukascopy"``. Pro: ``"databento"``, ``"massive"``.

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

    Example (FX, metals, indices and CFDs with BOTH sides of the book, free and
    without a key, from Dukascopy Bank)::

        store = bt.ingest(
            provider="dukascopy",
            symbol="EUR/USD",        # or "XAU/USD", "USA500.IDX/USD", "AAPL.US/USD"
            symbol_id=1,
            start="2010-01-01T00:00:00Z",
            end="2026-01-01T00:00:00Z",
            interval="1h",
            asset_class="forex",
        )

    Three intervals are native, ``"1m"``, ``"1h"`` and ``"1d"``; anything else
    is refused, so ingest ``"1m"`` and resample. A request for bars downloads
    bars and nothing else: the bid side, one file per bucket (``dataset="ask"``
    for the other side). Pass ``dataset="both"`` to fetch the second side too
    and get the ``bid``, ``ask`` and ``spread`` columns filled, which almost no
    other connector does, at twice the number of requests.

    What this data IS matters. Dukascopy is a Swiss bank publishing its own
    book, not an exchange tape: ``USA500.IDX/USD`` is a contract for difference
    the bank issues against the CME E-mini future, and ``AAPL.US/USD`` is a
    stock CFD. Prices and spreads are real and tradable at that bank; volume is
    the bank's own and is not market volume. Depth also varies sharply: the FX
    majors and the two precious metals start on 2003-05-04, indices in 2012,
    commodities in 2013, US stocks and crypto in 2017. Minute bars come one
    file per day, so a long minute ingest is thousands of requests and the host
    starts refusing after a burst; the connector paces itself and retries.

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


def ingest_trades(
    provider: str,
    symbol: str,
    symbol_id: int,
    start: str,
    end: str,
    *,
    data_root: str = "data",
    metadata_db: str = "metadata/metadata.sqlite",
    exchange: Optional[str] = None,
    asset_class: Optional[str] = None,
    progress: bool = True,
    category: str = "spot",
    cache_dir: Optional[str] = None,
) -> DataStore:
    """Ingest whole days of trades (a tape) from a venue's public archive.

    The tick-level counterpart of :func:`ingest`, with the same shape: you name
    a provider, a symbol and a range, the provider fetches, the store keeps one
    Arrow file per UTC day under ``{provider}/ticks/{symbol}/``. Nothing is
    fetched that you did not ask for.

    Providers with a public trade archive: ``"bybit"`` (one ``.csv.gz`` per
    day) and ``"binance"`` (spot aggTrades, one ``.zip`` per day). A day the
    venue did not publish raises a named error rather than returning nothing.
    ``start`` and ``end`` are ``YYYY-MM-DD`` (an RFC 3339 instant is accepted;
    its date part is used), ``end`` inclusive.

    ``category`` names the market, with the words of :func:`ingest_book`:

    * ``"spot"`` (the default): the spot archive of either provider.
      Timestamps to the millisecond on Bybit.
    * ``"linear"`` (USDT perps, ``BTCUSDT``) and ``"inverse"``
      (coin-margined, ``BTCUSD``): Bybit's derivatives archive, from the day
      the contract was listed (2019-10-01 for BTCUSD, 2020-03-25 for
      BTCUSDT). Inverse sizes are in contracts (one USD each on BTCUSD), as
      in the book. Asking for one kind of contract under the other's name is
      refused with the right name.

    Bybit stamps its derivative trades in seconds with a decimal fraction,
    converted to nanoseconds exactly (no float on the way). How fine the
    fraction is depends on the day: up to 2021-12-06, a microsecond on the
    inverse contracts and a tenth of a millisecond on the linear ones; from
    2021-12-07 to 2022-12-20, the whole second; from 2022-12-21, a tenth of a
    millisecond. So several prints often share one timestamp (more than half
    of them on a day of XRPUSDT in 2023, and every print of a second on the
    whole-second days): they keep the order the venue matched them in, never
    re-sorted, and a duration shorter than the stamp (a latency, a
    ``time_in_force``, a window in time) cannot tell them apart. Files of the
    days up to 2021-12-06 list the day newest first; they are read back in
    time order. Bybit identifies a derivative trade by a UUID, which the
    tape's integer ``trade_id`` cannot hold: it reads back null.

    One symbol is one market and one id. Each stored day records its
    category, and a store that keeps the ``spot`` tape or book of a symbol
    refuses its ``linear`` tape (and so on), before anything is downloaded.
    The tape and the book of a symbol share its ``symbol_id``, the one
    :func:`ingest_book` used: another id for the same ticker is refused.

    Part of the tick layer: a Researcher licence unlocks it, a Pro one does not;
    the engine says so before touching the network.

    The tape is the first door of the layer: see "Backtesting on the Tape" in
    the strategy authoring guide for what reads it and in what order.

    Args:
        provider: ``"bybit"`` or ``"binance"``.
        symbol: Symbol on the venue (e.g. ``"BTCUSDT"``).
        symbol_id: Store id for the symbol, the same as its book's.
        start: First day, ``YYYY-MM-DD``.
        end: Last day, inclusive.
        data_root: Store root.
        metadata_db: Metadata database.
        exchange: Venue the symbol is filed under; the provider's name by
            default. A ``linear`` or ``inverse`` tape is filed under Bybit.
        asset_class: ``"crypto_spot"`` for ``spot`` and ``"crypto_perp"`` for
            the two others when left out.
        progress: Print one line per day.
        category: ``"spot"``, ``"linear"`` or ``"inverse"``.
        cache_dir: Keep Bybit's raw daily archives here and reuse them.

    Example::

        store = bt.ingest_trades("bybit", "BTCUSDT", symbol_id=1,
                                 start="2026-08-25", end="2026-08-25")
        # The perp's tape, beside its book:
        bt.ingest_trades("bybit", "XRPUSDT", symbol_id=2,
                         start="2023-03-01", end="2023-03-01",
                         category="linear")
        bt.ingest_book("bybit", "XRPUSDT", symbol_id=2,
                       start="2023-03-01", end="2023-03-01",
                       category="linear")
    """
    from manifoldbt._native import py_ingest_trades as _native

    cb = None
    if progress:
        def cb(i, n, day):
            if i < n:
                print(f"  tape {symbol.upper()} {day} ({i + 1}/{n})", flush=True)

    return _native(
        provider, symbol, int(symbol_id), start, end, data_root, metadata_db,
        exchange, asset_class, cb, category, cache_dir,
    )


def ingest_book(
    provider: str,
    symbol: str,
    symbol_id: int,
    start: str,
    end: str,
    *,
    levels: Optional[int] = None,
    data_root: str = "data",
    metadata_db: str = "metadata/metadata.sqlite",
    category: str = "spot",
    cache_dir: Optional[str] = None,
    progress: bool = True,
    book_format: Optional[str] = None,
) -> DataStore:
    """Ingest whole days of order book from a venue's public archive.

    The depth counterpart of :func:`ingest_trades`, with the same shape: you
    name a provider, a symbol and a range, and the store keeps one Arrow file
    per UTC day. Each row is one instant the book changed; reading the book
    back at any instant gives the top ``levels`` of both sides as they stood
    there.

    How the days are stored (``book_format``):

    * ``"deltas"`` (the default for a new symbol): under
      ``{provider}/book_delta/{symbol}/``, each row holds only the levels that
      changed at that instant, compressed. A day of BTCUSDT 200 levels deep
      is about 44 MB instead of 2.7 GB, and loads in a fraction of a second
      at any depth. The engine replays the changes as it reads; a book read
      ten levels deep or less is laid out whole in memory as it loads.
    * ``"ladder"``: under ``{provider}/book/{symbol}/``, the whole top
      ``levels`` ladder on every row. Every read is a lookup, which is the
      cheaper shape at ten levels, and the only shape earlier versions of
      manifoldbt read.

    Readers see the same states either way. A symbol keeps the shape of its
    first stored day: ``None`` follows it, and naming the other shape is
    refused. :func:`convert_book_to_deltas` rewrites a stored book as
    deltas.

    ``"bybit"`` is the only provider: its archive is the one free source of
    historical depth. How deep a day goes depends on the market and the date
    (checked on BTCUSDT, ETHUSDT and BTCUSD; a symbol listed later starts
    later):

    * ``"linear"`` (USDT perps, ``BTCUSDT``) and ``"inverse"`` (coin-margined,
      ``BTCUSD``): 500 levels up to 2025-08-20, 200 levels from 2025-08-21.
      Linear BTCUSDT goes back to 2023-01-18.
    * ``"spot"``: 200 levels, from about 2025-04-30.
    * Options: no archive.

    Each day is read from the file Bybit published for it, whichever depth
    that is. A day it did not publish raises a named error rather than
    returning nothing. One symbol is one market: a store that already holds
    the ``spot`` book or tape of a symbol refuses its ``linear`` book, because
    a depth read from the other market is not an approximation, it is a
    different number. The book and the tape (:func:`ingest_trades`, same
    ``category``) of a symbol share its ``symbol_id``. Inverse books count
    their sizes in contracts (one USD each on BTCUSD), as the venue publishes
    them.

    A day is a download of 100 to 400 MB and several gigabytes once inflated;
    it is streamed straight into the store, never held. ``start`` and ``end``
    are ``YYYY-MM-DD`` (an RFC 3339 instant is accepted; its date part is
    used), ``end`` inclusive.

    Part of the tick layer, a Researcher feature: a Pro licence does not
    unlock it, and the engine says so before touching the network.

    What reads the book: the strategy itself, level by level, on bars or on
    the trade clock (:mod:`manifoldbt.book`: ``bt.book.imbalance(5)``,
    ``bt.book.bid_size(3)``, ...); the quote columns of the trade clock; the
    depth the queue finds in front of a resting order
    (``fill_model={"queue": ...}``); and the mid the fill marks measure
    against. See "Order Book Columns" and "Backtesting on the Tape" in the
    strategy authoring guide.

    Args:
        provider: ``"bybit"``.
        symbol: Symbol on the venue (e.g. ``"BTCUSDT"``).
        symbol_id: Store id for the symbol, as for :func:`ingest`.
        start: First day, ``YYYY-MM-DD``.
        end: Last day, inclusive.
        levels: Levels kept per side, up to 500. ``None`` (the default): the
            depth the symbol's book already has in this store, 200 for a new
            symbol, the depth Bybit publishes every day at. More than 200
            holds only where every day of the range was published 500 levels
            deep: a range with a shallower day in it is refused before
            anything is downloaded, and the error names the day to end the
            range before. A store keeps one depth per symbol. A run reads
            every stored level unless ``BacktestConfig.book_levels`` bounds
            it.
        data_root: Store root.
        metadata_db: Metadata database.
        category: ``"spot"``, ``"linear"`` or ``"inverse"``.
        cache_dir: Keep the raw daily archives here and reuse them, one
            directory per category (``cache_dir/linear/...``); an archive
            left directly in ``cache_dir`` is not reused.
        progress: Print one line per day.
        book_format: ``"deltas"``, ``"ladder"`` or ``None`` (the symbol's
            shape, deltas for a new symbol).

    Example::

        store = bt.ingest_book("bybit", "BTCUSDT", symbol_id=1,
                               start="2026-08-25", end="2026-08-25",
                               category="linear")
        # A day Bybit published 500 levels deep, all of it:
        deep = bt.ingest_book("bybit", "BTCUSDT", symbol_id=1,
                              start="2025-08-20", end="2025-08-20",
                              category="linear", levels=500,
                              data_root="data500",
                              metadata_db="data500/meta.sqlite")
    """
    from manifoldbt._native import py_ingest_book as _native

    cb = None
    if progress:
        def cb(i, n, day):
            if i < n:
                print(f"  book {symbol.upper()} {day} ({i + 1}/{n})", flush=True)

    return _native(
        provider, symbol, int(symbol_id), start, end,
        None if levels is None else int(levels),
        data_root, metadata_db, category, cache_dir, cb, book_format,
    )


def convert_book_to_deltas(
    symbol: str,
    *,
    data_root: str = "data",
    metadata_db: str = "metadata/metadata.sqlite",
) -> list:
    """Rewrite the stored order book of ``symbol`` as deltas, day by day.

    For a book ingested with ``book_format="ladder"``, or by an earlier
    version of manifoldbt: each day is written as the levels that changed at each
    instant (see :func:`ingest_book`), then replayed against the stored
    ladder row by row, every price and size bit for bit, and only then takes
    its place. A day that does not replay exactly is left as it was and the
    conversion stops with an error naming the row. Interrupted, the
    conversion is finished by calling it again; until then the store refuses
    to read that symbol's book, because it holds days in both shapes.

    A day stored 200 levels deep goes from about 2.7 GB to about 44 MB.
    Nothing a backtest reads changes. A converted book is not read by
    earlier versions of manifoldbt.

    Args:
        symbol: The symbol's ticker in the store (e.g. ``"BTCUSDT"``).
        data_root: Store root, as for :func:`ingest_book`.
        metadata_db: Metadata database.

    Returns:
        ``[(day, rows), ...]`` for every day converted, empty when the book
        was stored as deltas already.
    """
    from manifoldbt._native import py_convert_book_to_deltas as _native

    return [tuple(d) for d in _native(symbol, data_root, metadata_db)]


def ingest_mbo(
    path: str,
    symbols,
    date: str,
    *,
    data_root: str = "data",
    metadata_db: str = "metadata/metadata.sqlite",
    exchange: str = "NASDAQ",
    tick_size: float = 0.01,
    progress: bool = True,
) -> DataStore:
    """Ingest a day of order-by-order data (market by order) from a file.

    Where :func:`ingest_book` stores how much rests at each price, this
    stores every order: when it entered the book, each execution,
    cancellation, deletion or replacement of it, and the prints against
    orders the book does not show. The place of an order in its queue is
    then read off the data rather than estimated: a market loaded from these
    days (:meth:`manifoldbt.sim.Market.from_store`, ``Strategy.quote``)
    serves resting orders in the exact order of arrival, and reads its book
    by price exactly as a stored book.

    The file is a day of Nasdaq TotalView-ITCH 5.0: the file Nasdaq publishes
    (``.gz``, read as it inflates), or a stream of its messages kept as they
    are (plain or ZSTD-compressed), such as an extract of some symbols. Every
    symbol asked for is read in one pass. ITCH stamps its messages in
    nanoseconds since midnight in New York on ``date``: they are stored as
    instants (UTC), one Arrow file per symbol and UTC day under
    ``{exchange}/mbo/{symbol}/``. A day that crosses midnight UTC opens on
    the orders resting then. Orders still resting when the session ends leave
    the book then.

    Each day is checked as it is written: an event about an order that is
    not in the book, or taking out more than is left of it, is refused and
    the day left as it was. Ingesting a day again replaces what it covers
    and keeps the rest.

    Args:
        path: The ITCH file.
        symbols: ``{ticker: symbol_id}``, the tickers as ITCH names them.
        date: The session's date, ``YYYY-MM-DD`` (New York).
        data_root: Store root.
        metadata_db: Metadata database.
        exchange: Venue the symbols are filed under.
        tick_size: The price increment recorded for each symbol: a cent,
            Nasdaq's increment for a price of a dollar or more.
        progress: Print one line per symbol.

    Returns:
        The store.

    Example::

        store = bt.ingest_mbo("S120925-v50.txt.gz", {"AAPL": 1, "QQQ": 2},
                              date="2025-12-09")
        res = bt.sim.run(maker, bt.sim.Config(), store, symbols=["AAPL"],
                         start="2025-12-09", end="2025-12-10")
    """
    from manifoldbt._native import py_ingest_mbo as _native

    if isinstance(symbols, dict):
        pairs = [(str(t), int(i)) for t, i in symbols.items()]
    else:
        pairs = [(str(t), int(i)) for t, i in symbols]

    cb = None
    if progress:
        def cb(ticker, events, days):
            print(f"  mbo {ticker}: {events} events, {days} day(s)", flush=True)

    store, _ = _native(
        str(path), pairs, date, data_root, metadata_db, exchange, float(tick_size), cb,
    )
    return store


def bars_from_trades(
    store: DataStore,
    symbol,
    start: str,
    end: str,
    *,
    interval: str = "1s",
) -> dict:
    """Rebuild bars from the tape stored for a symbol, and write them into
    the store at ``interval``.

    A venue's own bars carry one total volume. Bars rebuilt from the tape
    carry what the tape carries: ``buy_volume`` and ``sell_volume`` split by
    the aggressor side, ``vwap`` and ``trade_count``. They are ordinary bars
    for everything downstream: ``run()`` reads them at ``interval`` or
    resamples them coarser, and the DSL addresses the flow columns by name
    (``col("buy_volume")``). Seconds with no trade produce no bar. One second
    is the finest step the store holds.

    Args:
        store: The store holding the symbol's tape (see :func:`ingest_trades`).
        symbol: The store ticker (e.g. ``"BTCUSDT"``) or symbol id.
        start: First instant, ISO date or datetime (``"2026-08-25"``).
        end: Last instant, exclusive (``"2026-08-26"`` for one full day).
        interval: ``"1s"``, ``"1m"``, ``"5m"``, ``"1h"``, ... (default ``"1s"``).

    Returns:
        ``{"bars", "version", "interval_ns", "first_ts", "last_ts"}``.

    Example::

        store = bt.ingest_trades("bybit", "BTCUSDT", symbol_id=1,
                                 start="2026-08-25", end="2026-08-25")
        bt.bars_from_trades(store, "BTCUSDT", "2026-08-25", "2026-08-26")
        config = bt.BacktestConfig(..., bar_interval=Interval.seconds(1))
        result = bt.run(strategy, config, store)
    """
    from manifoldbt.helpers import time_range as _time_range

    # No Python-side licence check: the native side answers, with the
    # accurate refusal. Same policy as manifoldbt.ticks and attach_quotes.
    if isinstance(symbol, int):
        symbol_id = symbol
    else:
        symbol_id = store.resolve_symbol(str(symbol))
    start_ns, end_ns = _time_range(start, end)
    if end_ns <= start_ns:
        raise ValueError(
            f"end {end!r} must be after start {start!r} (end is exclusive: "
            "one full day is start='2026-08-25', end='2026-08-26')"
        )

    from manifoldbt._native import py_bars_from_trades as _native

    try:
        return _native(
            store.data_root(), store.metadata_db(), symbol_id, interval, start_ns, end_ns
        )
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


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
    null. This does, from Bybit's public order-book archives: one archive of
    100 to 400 MB per day is streamed and reduced to the best bid/ask standing
    at each bar close, and only those few KB touch the store. Bars outside the
    quoted range keep their current values.

    Bybit publishes ``linear`` (USDT perps) and ``inverse`` (coin-margined)
    books 500 levels deep up to 2025-08-20 and 200 from 2025-08-21 (linear
    BTCUSDT goes back to 2023-01-18), and ``spot`` books from about
    2025-04-30; each day is read from whichever file it was published as. A
    day it did not publish raises a named error. The venue quoted is Bybit: on
    another venue's bars the spread is that of a different book — a
    reasonable proxy for majors, a real approximation for thin alts.

    Args:
        store: The store holding the symbol's bars (see :func:`ingest`).
        symbol: The store ticker (e.g. ``"BTC-USDT:perp"``) or symbol id.
        start: First day, ``YYYY-MM-DD``.
        end: Last day, inclusive, ``YYYY-MM-DD``.
        provider: ``"bybit"`` (the only quote source with free history).
        provider_symbol: Symbol on the provider (e.g. ``"BTCUSDT"``). Derived
            from the ticker when omitted: ``BTC-USDT:perp`` -> ``BTCUSDT``.
        category: ``"linear"``, ``"inverse"`` or ``"spot"``.
        cache_dir: Keep the raw daily archives here and reuse them, one
            directory per category (``cache_dir/linear/...``); an archive
            left directly in ``cache_dir`` is not reused.
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
        interval: Bar interval of the rows (``"1s"``, ``"1m"``, ``"5m"``, ``"1h"``,
            ``"1d"``, ...). The store holds a tier per interval, down to one second.
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
        interval: Bar interval of the rows (``"1s"``, ``"1m"``, ``"5m"``, ``"1h"``,
            ``"1d"``, ...). The store holds a tier per interval, down to one second.
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
    # Nothing to rewrite without a SymbolRef, and the memoised JSON text says
    # whether there is one without rebuilding the strategy dict and walking
    # it, which was half of the Python cost of a small run. Conservative: the
    # word anywhere in the text (a signal named after it, a description)
    # takes the full path below.
    strategy_json = strategy.to_json()
    if '"SymbolRef"' not in strategy_json:
        return strategy_json, config
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


def _split_fill_callable(config: BacktestConfig):
    """Take a ``fill_model={"python": ...}`` callable out of the config.

    Returns the config the engine serialises, the callable, and whether it
    answers for traversals; or the config unchanged and ``None``. Written
    ``{"python": callable}``, or ``{"python": {"fill": callable,
    "override_traverse": True}}`` to mirror the shape of the DSL route.

    The callable cannot be part of a config -- that is what separates it from a
    rule written in the DSL -- so it travels beside one, on this one entry
    point, and ``config._normalise_fill_model`` refuses it by name everywhere
    else.
    """
    fill_model = getattr(config.execution, "fill_model", None) or {}
    spec = fill_model.get("python")
    if spec is None:
        return config, None, False
    override = False
    if isinstance(spec, dict):
        unknown = set(spec) - {"fill", "override_traverse"}
        if unknown:
            raise TypeError(
                f'unknown fill_model["python"] key(s) {sorted(unknown)}: it takes '
                '"fill" (the callable) and "override_traverse"'
            )
        override = bool(spec.get("override_traverse", False))
        rule = spec.get("fill")
    else:
        rule = spec
    if not callable(rule):
        raise TypeError(
            'fill_model["python"] must be a callable invoked once per event with '
            "the seventeen context fields as positional arguments (or a dict "
            '{"fill": callable, "override_traverse": bool}), got '
            f"{type(rule).__name__}"
        )
    stripped = {k: v for k, v in fill_model.items() if k != "python"}
    config = copy.copy(config)
    config.execution = copy.copy(config.execution)
    config.execution.fill_model = stripped or None
    return config, rule, override

def _execution_grid_json(execution_grid) -> str:
    """An execution grid as the JSON the engine reads.

    The paths are checked in Rust against the config's own shape, so nothing
    here knows what a queue or a latency is. Two things are done at this
    boundary because they are Python's alone: an ``Interval`` written as a
    duration becomes its nanoseconds, and a grid that is not a mapping of
    lists is refused before it becomes a JSON object with nothing in it.
    """
    if not execution_grid:
        return "{}"
    if not isinstance(execution_grid, dict):
        raise TypeError(
            "execution_grid takes a mapping of dotted config paths to lists of "
            'values: {"fill_model.queue.assumed_queue": [1, 2, 5], '
            '"latency.order": [Interval.millis(0), Interval.millis(20)]}'
        )
    from manifoldbt.expr import _interval_to_nanos

    out = {}
    for path, values in execution_grid.items():
        if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
            raise TypeError(
                f"execution_grid[{path!r}] takes a list of values, got "
                f"{type(values).__name__}"
            )
        converted = []
        for v in values:
            nanos = _interval_to_nanos(v) if isinstance(v, dict) else None
            converted.append(nanos if nanos is not None else v)
        if not converted:
            raise ValueError(
                f"execution_grid[{path!r}] has no value: an axis with an empty "
                "list makes the whole sweep empty"
            )
        out[str(path)] = converted
    return json.dumps(out)


def _execution_grid_plan(execution_grid) -> "Tuple[str, int]":
    """The grid as JSON, and how many configs the sweep will run under.

    One call for both, and called BEFORE the run: a grid that is not a mapping
    of lists, or an axis with no value, is the caller's own mistake and reads
    as such rather than as an engine refusal wrapped in a BacktesterError.
    """
    payload = _execution_grid_json(execution_grid)
    if not execution_grid:
        return payload, 1
    n = 1
    for values in execution_grid.values():
        n *= max(1, len(list(values)))
    return payload, n


def _by_day_arg(by_day) -> Optional[bool]:
    """``by_day`` as the engine reads it: ``None`` for ``"auto"``."""
    if isinstance(by_day, str) and by_day == "auto":
        return None
    if isinstance(by_day, bool):
        return by_day
    raise ValueError(f'by_day takes "auto", True or False, got {by_day!r}')


def run(
    strategy: Strategy,
    config: BacktestConfig,
    store: DataStore,
    *,
    by_day: Any = "auto",
    device: str = "cpu",
) -> Result:
    """Run a backtest and return a rich Result.

    Returns a :class:`Result` with DataFrame conversion, summaries,
    and plotting methods. Access the raw Rust object via ``result.raw``.

    A ``fill_model={"python": callable}`` is routed here and nowhere else: the
    callable answers how much of each event reaches each resting order, once
    per event, under the GIL. It is the research route -- slow, and not part of
    the config, so not sweepable. The rule that sweeps is
    ``fill_model={"custom": {"fill": <expression>}}``.

    A strategy that quotes (``Strategy.quote``) runs on the order-level
    simulation instead: woken every ``bar_interval`` (``Interval.millis(n)``
    is accepted there and nowhere else), its quotes posted, kept, repriced
    and withdrawn by the venue of ``bt.sim``, entirely in the engine. The
    result is a :class:`QuoteResult`: this same result plus ``orders_df()``,
    ``fills_df()`` and ``events_df()``.

    A strategy that quotes over more than one UTC day runs DAY BY DAY
    (``by_day="auto"``): the engine holds the book and the tape of a few days
    at a time (the day it plays, the one before it, the next ones prepared
    ahead) and computes the batch of each day from the wake-ups of that day
    and those of the days before it that its signals reach back to, so that a
    month or a year runs in bounded memory. The orders, the fills and the
    metrics are those of the range held whole. What cannot be computed from a
    bounded reach (an exponential mean, a cumulative sum, a forward fill, a
    lead) keeps the range whole under ``"auto"`` and is refused by name under
    ``by_day=True``; ``by_day=False`` always holds it whole. Held day by day,
    ``res.sim``'s equity curve is kept at the output resolution rather than
    at every wake-up. ``device="cuda"`` prepares the days' book on the
    graphics card, with the same result; a build or a machine without CUDA
    prepares them on the CPU and says so in a warning.
    """
    if _quotes(strategy):
        return _run_quotes(strategy, config, store, by_day=by_day, device=device)
    if not (isinstance(by_day, str) and by_day == "auto") or device != "cpu":
        raise TypeError(
            "by_day= and device= are read by a strategy that quotes (Strategy.quote), "
            "whose run holds its market day by day; this strategy runs on bars"
        )
    try:
        config, fill_rule, override_traverse = _split_fill_callable(config)
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        strategy_json, config = _apply_cross_asset(strategy, config, store)
        cfg_json = _prepared_config_json(config, strategy, store)
        if fill_rule is None:
            raw = _run_native(strategy_json, cfg_json, store)
        else:
            raw = _run_with_fill_rule_native(
                strategy_json,
                cfg_json,
                store,
                fill_rule,
                override_traverse,
            )
        return Result(raw)
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def _quote_config_json(config: BacktestConfig, strategy, store: DataStore) -> str:
    """The config of a run of quoting strategies, as the engine reads it.

    One preparation for :func:`run`, the sweeps and the batches, so that a
    combination of a sweep reads exactly the config the same run reads alone.
    """
    config, fill_rule, _ = _split_fill_callable(config)
    if fill_rule is not None:
        raise ValueError(
            'fill_model={"python": ...} is not read by a strategy that quotes: its '
            "resting orders are served by the queue model alone"
        )
    return _prepare_config(config, strategy, store).to_json()


def _quote_results(pairs, config: BacktestConfig) -> List["QuoteResult"]:
    """``(result, venue record)`` pairs from the engine, as :class:`QuoteResult`."""
    from manifoldbt.sim import Config as _SimConfig
    from manifoldbt.sim import SimResult as _SimResult

    initial_cash = float(config.initial_capital)
    return [
        QuoteResult(raw, _SimResult(sim_raw, _SimConfig(initial_cash=initial_cash)))
        for raw, sim_raw in pairs
    ]


def _run_quotes(
    strategy: Strategy,
    config: BacktestConfig,
    store: DataStore,
    *,
    by_day: Any = "auto",
    device: str = "cpu",
) -> "QuoteResult":
    """``mbt.run`` for a strategy that quotes: see :func:`run`."""
    _require_grant_for_gpu(device, "GPU day preparation")
    try:
        cfg_json = _quote_config_json(config, strategy, store)
        pair = _run_quotes_native(
            strategy.to_json(), cfg_json, store, by_day=_by_day_arg(by_day), device=device
        )
        return _quote_results([pair], config)[0]
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def _quotes(strategy) -> bool:
    return bool(getattr(strategy, "_quotes", None))


def _run_quote_sweep(
    strategy: Strategy,
    param_grid: Dict[str, List[Any]],
    config: BacktestConfig,
    store: DataStore,
    max_parallelism: int,
    execution_grid: Optional[Dict[str, List[Any]]],
    exec_json: str,
    lite: bool,
):
    """:func:`run_sweep` and :func:`run_sweep_lite` for a strategy that quotes.

    Each combination is the run :func:`run` makes of the same strategy with
    those values as its defaults, under that execution config: the market is
    loaded once (day by day as :func:`run` holds it), and the combinations
    run in parallel in the engine, with nothing called back into Python.
    """
    try:
        cfg_json = _quote_config_json(config, strategy, store)
        grid_json = json.dumps({
            name: [scalar_value_to_json(v) for v in values]
            for name, values in param_grid.items()
        })
        args = (strategy.to_json(), grid_json, cfg_json, store, None, max_parallelism, exec_json)
        if lite:
            from manifoldbt._reprs import wrap_sweep_lite

            return wrap_sweep_lite(_run_quote_sweep_lite_native(*args))
        pairs = _run_quote_sweep_native(*args)
        return SweepResult(_quote_results(pairs, config), param_grid, execution_grid)
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def _run_quote_batch(
    strategies: List[Strategy],
    config: BacktestConfig,
    store: DataStore,
    max_parallelism: int,
    exec_json: str,
    lite: bool,
):
    """:func:`run_batch` and :func:`run_batch_lite` for strategies that quote."""
    quoting = [_quotes(s) for s in strategies]
    if not all(quoting):
        names = ", ".join(repr(s.name) for s, q in zip(strategies, quoting) if not q)
        raise StrategyError(
            "a batch runs strategies that quote (Strategy.quote) on the order-level "
            "simulation and the others on bars, never the two together: "
            f"{names} do(es) not quote. Run them in two batches"
        )
    try:
        cfg_json = _quote_config_json(config, strategies[0], store)
        args = ([s.to_json() for s in strategies], cfg_json, store, None, max_parallelism,
                exec_json)
        if lite:
            return _run_quote_batch_lite_native(*args)
        return _quote_results(_run_quote_batch_native(*args), config)
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def reconcile(
    result: Result,
    live_fills: Any,
    *,
    store: DataStore,
    symbol_id: Optional[int] = None,
    tolerance: Any = None,
    price_tolerance: Optional[float] = None,
    orders_posted: Optional[int] = None,
) -> Reconciliation:
    """Reconcile a journal of REAL fills with the backtest of the same days.

    An execution model is an assumption -- how long the queue in front of an
    order was, how fast the volume ahead of it cancelled, how late the quote
    reached the book -- and a backtest can only ever be consistent with itself.
    This is the one measurement that settles any of them: the fills a broker
    actually granted over the same days, marked by the same code, counted the
    same way, put beside the backtest's own.

    It also measures the one cost no simulation can produce. A backtest replays
    a tape that never saw your quotes, so its adverse selection is whatever the
    market was going to do anyway; the journal's fills were served by people
    who could see the quote and react to it. The paired difference at each
    horizon is that reactive part, and nothing else measures it.

    Args:
        result: A finished :func:`run`. Its manifest carries the config, so the
            same days, the same grid and the same book are reloaded to mark the
            journal against -- the reference on both sides is one reference.
        live_fills: The journal, as a pandas/polars DataFrame or a dict of
            columns. Minimum: a UTC ``timestamp``, a ``side``, a ``price`` and
            a ``qty``; a numeric ``symbol_id`` if the run held more than one
            symbol. Common broker spellings are read (``time``, ``quantity``,
            ``avg_price``, ``B``/``S``, ``buy``/``sell``, ...); a side that
            reads as neither is refused by name rather than taken for a sell.
            Extra columns (``order_id``, ``posted_at``, ``level``) are the
            caller's to keep; nothing here reads them.
        store: Where the tape and the book of those days live. The store the
            run used, unless the days were re-ingested elsewhere.
        symbol_id: The symbol the journal's fills belong to, when the journal
            carries no numeric one.
        tolerance: How far apart two fills may be and still be the same fill:
            an :class:`Interval` or a number of nanoseconds. One second by
            default. A window wider than the strategy's own requote interval
            starts pairing a real fill with the wrong quote.
        price_tolerance: An optional absolute price tolerance, the venue's tick
            typically. ``None`` (the default) matches on time alone, which is
            usually what you want: the price gap between a real fill and its
            backtest twin is a MEASUREMENT here, and a tolerance narrower than
            that gap would hide it by turning both fills into unmatched ones.
        orders_posted: The quotes the strategy posted, if you know it better
            than the backtest does. Defaults to the run's own
            ``order_activity["orders_posted"]``, which is the shared
            denominator of the two service rates: the two sides ran the same
            decisions, so the same posting count prices both.

    Returns:
        A :class:`Reconciliation`: ``by_day_df()`` / ``by_hour_df()`` for
        service, ``horizons_df()`` for adverse selection on both sides and the
        paired difference, ``pairs_df()`` for the price and time gaps fill by
        fill, and ``verdict`` for one factual sentence per quantity. Nothing in
        it names a setting to change; ``execution_grid`` on :func:`run_sweep`
        is where that choice gets published as a surface.

    The three horizons (+100 ms, +1 s, +10 s) are fixed, here as in
    ``result.fill_marks``: a published markout is only comparable when everyone
    reports the same three.

    Needs the tape of those days in the store, since that is what a mark reads.

    Example::

        result = mbt.run(strategy, config, store)
        rec = mbt.reconcile(result, broker_fills, store=store,
                            symbol_id=1, tolerance=mbt.Interval.seconds(1))
        print(rec.summary())
        rec.horizons_df()
        mbt.plot.reconcile(rec)
    """
    from manifoldbt.expr import _interval_to_nanos

    if tolerance is None:
        tolerance_ns = 1_000_000_000
    else:
        nanos = _interval_to_nanos(tolerance) if isinstance(tolerance, dict) else None
        if nanos is None:
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
                raise TypeError(
                    "reconcile: tolerance takes a duration -- Interval.millis(n), "
                    "Interval.seconds(n), or a number of nanoseconds"
                )
            nanos = int(tolerance)
        tolerance_ns = int(nanos)

    columns = _fills_to_columns(live_fills, symbol_id)
    raw_result = result.raw if isinstance(result, Result) else result
    try:
        raw = _reconcile_fills_native(
            raw_result,
            store,
            columns["ts"],
            columns["side"],
            columns["price"],
            columns["qty"],
            columns["symbol"],
            tolerance_ns,
            price_tolerance,
            None if orders_posted is None else int(orders_posted),
        )
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc
    return Reconciliation(raw)


def run_sweep(
    strategy: Strategy,
    param_grid: Dict[str, List[Any]],
    config: BacktestConfig,
    store: DataStore,
    *,
    max_parallelism: int = 0,
    execution_grid: Optional[Dict[str, List[Any]]] = None,
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
        execution_grid: Mapping of dotted EXECUTION config paths to lists of
            values, swept beside ``param_grid``. The paths are read relative to
            ``config.execution``::

                execution_grid={
                    "fill_model.queue.assumed_queue": [1, 2, 5],
                    "latency.order": [Interval.millis(0), Interval.millis(20)],
                    "fill_model.queue.cancel_ahead_rate": [0.0, 0.1],
                }

            The queue in front of an order, the round trip to the book and the
            rate the volume ahead cancels all make the P&L and none of them is
            knowable from a backtest, so they belong on an axis rather than in
            a decision. A path that names no setting, or a value the model
            refuses, fails by name before a day is read -- an axis that does
            nothing would publish a flat surface, which reads as "this setting
            does not matter". These axes are the SLOWEST of the sweep: the
            results come back as one block of the whole parameter surface per
            execution combination, execution paths sorted, last axis fastest.

    Returns:
        A :class:`SweepResult` with ``.to_df()``, ``.best()``, ``.plot_metric()``.
        Results come in the engine's enumeration order: axes sorted by
        parameter name, last axis varying fastest, whatever order the dict
        was written in (:func:`manifoldbt.dataframe.grid_combos` lists it).
        ``to_df()`` labels each row from the run's own manifest, so it does
        not depend on that order -- including the ``exec_*`` columns, which are
        read out of the config the run actually executed under.

    A strategy that quotes (``Strategy.quote``) is swept the same way, on the
    order-level simulation: the book and the tape are loaded once (over more
    than one UTC day, day by day as :func:`run` holds them, each day read
    once for the ``max_parallelism`` combinations that run together), and each
    combination is exactly the run :func:`run` makes of the strategy with
    those values as its defaults, under that execution config -- the same
    orders, fills and order events, to the bit, whatever ``max_parallelism``.
    Each element is then a :class:`~manifoldbt.result.QuoteResult`, with its
    ``orders_df()``, ``fills_df()`` and ``events_df()``. The natural
    execution axes are ``latency.order``, ``latency.cancel``,
    ``latency.response``, ``latency.feed`` and ``fill_model.queue.*``. Every
    combination keeps its whole order journal, so a large grid is better read
    through :func:`run_sweep_lite`, which keeps the metrics only.
    """
    exec_json, exec_combos = _execution_grid_plan(execution_grid)
    _require_pro_over_combos(
        _grid_combos(param_grid) * exec_combos, "Parameter sweep"
    )
    _validate_swept_params(strategy, param_grid.keys(), "Parameter sweep")
    if _quotes(strategy):
        return _run_quote_sweep(strategy, param_grid, config, store, max_parallelism,
                                execution_grid, exec_json, lite=False)
    try:
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        strategy_json, config = _apply_cross_asset(strategy, config, store)
        cfg_json = _prepared_config_json(config, strategy, store)
        grid_json = json.dumps({
            name: [scalar_value_to_json(v) for v in values]
            for name, values in param_grid.items()
        })
        native_args = (strategy_json, grid_json, cfg_json, store, max_parallelism)
        # The execution grid is passed only when there is one: a sweep without
        # one calls the engine exactly as it always did.
        raw_results = (
            _run_sweep_native(*native_args, exec_json)
            if execution_grid
            else _run_sweep_native(*native_args)
        )
        return SweepResult(raw_results, param_grid, execution_grid)
    except (ValueError, RuntimeError) as exc:
        raise _classify_error(exc) from exc


def run_batch(
    strategies: List[Strategy],
    config: BacktestConfig,
    store: DataStore,
    *,
    max_parallelism: int = 0,
    execution_grid: Optional[Dict[str, List[Any]]] = None,
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
        execution_grid: Execution settings swept beside the strategies, same
            shape as :func:`run_sweep`'s. The execution axes are the slowest:
            the results are one block of the whole batch per execution
            combination, in input order inside each block.

    Returns:
        One :class:`Result` per strategy, in input order; one per strategy and
        execution combination when an ``execution_grid`` is given.

    Strategies that quote (``Strategy.quote``) are batched the same way on
    the order-level simulation, one :class:`~manifoldbt.result.QuoteResult`
    each, each the run :func:`run` makes of it alone. A batch holds strategies
    that quote or strategies that do not, never both.
    """
    exec_json, exec_combos = _execution_grid_plan(execution_grid)
    _require_pro_over_combos(len(strategies) * exec_combos, "Batch backtesting")
    if any(_quotes(s) for s in strategies):
        return _run_quote_batch(strategies, config, store, max_parallelism, exec_json,
                                lite=False)
    try:
        config = _cap_output_resolution(config)
        store = _resolve_store(config, store)
        cfg_json = _prepared_config_json(config, None, store)
        native_args = (
            [strat.to_json() for strat in strategies],
            cfg_json,
            store,
            max_parallelism,
        )
        raw_results = (
            _run_batch_native(*native_args, exec_json)
            if execution_grid
            else _run_batch_native(*native_args)
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

    Strategies that quote (``Strategy.quote``) run on the order-level
    simulation, each without its order journal and its markouts; the metrics
    are those :func:`run` reports for the same strategy, to the bit.
    """
    _require_pro_over_combos(len(strategies), "Batch backtesting")
    if any(_quotes(s) for s in strategies):
        return _run_quote_batch(strategies, config, store, max_parallelism, "{}", lite=True)
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
    execution_grid: Optional[Dict[str, List[Any]]] = None,
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
        curve. ``final_equity``, ``total_return``, ``sharpe``, ``sortino`` and
        ``volatility`` are unaffected -- they match ``run`` exactly.
        ``max_drawdown`` matches to the last bit on a run without exit orders
        and within one ulp (about 1e-16 relative) once a stop, a target or a
        trailing stop fires. Three annualisation-sensitive metrics differ
        slightly because they are derived from the daily series: ``cagr`` (it
        starts from the first daily equity rather than initial capital),
        ``calmar`` and ``ulcer_index``. The gap is small (< ~0.4% relative on a multi-year daily
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
        With an ``execution_grid``, the execution axes are slower still: one
        block of the whole parameter surface per execution combination.

    Execution grid:
        ``execution_grid`` sweeps execution settings beside the parameters,
        exactly as in :func:`run_sweep`. The lite driver walks no tape, so the
        settings that need one -- ``fill_model.queue``, ``execution.latency``,
        the trade clock -- are refused by name for every combination, as they
        already are for a base config; and the CUDA sweep takes one execution
        config, so ``device="cuda"`` with an execution grid is refused too
        (``device="auto"`` simply stays on the CPU). A tape sweep belongs on
        :func:`run_sweep`.

    A strategy that quotes:
        ``Strategy.quote`` strategies are swept on the order-level simulation,
        on the CPU (``device="auto"`` stays there, ``"cuda"`` is refused), in
        ``fp64``. Each combination runs exactly as in :func:`run_sweep`,
        without keeping its order journal or marking its fills, and hands back
        the metrics the full run reports, to the bit -- not a daily
        approximation of them. ``execution_grid`` takes every axis a quoting
        run reads, ``latency.response`` and ``latency.feed`` included.
    """
    exec_json, exec_combos = _execution_grid_plan(execution_grid)
    _require_pro_over_combos(
        _grid_combos(param_grid) * exec_combos, "Parameter sweep"
    )
    _validate_swept_params(strategy, param_grid.keys(), "Parameter sweep")
    if _quotes(strategy):
        if device not in ("auto", "cpu"):
            raise ValueError(
                f"device={device!r}: a strategy that quotes (Strategy.quote) is swept on "
                'the order-level simulation, on the CPU; use device="cpu" or "auto"'
            )
        if precision not in ("fp64", "f64", "double"):
            raise ValueError(
                f"precision={precision!r}: a strategy that quotes is swept in fp64 only"
            )
        return _run_quote_sweep(strategy, param_grid, config, store, max_parallelism,
                                execution_grid, exec_json, lite=True)
    _require_grant_for_gpu(device, "GPU acceleration")
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
        native_args = (
            strategy_json, grid_json, cfg_json, store, max_parallelism, device,
            precision,
        )
        return wrap_sweep_lite(
            _run_sweep_lite_native(*native_args, exec_json)
            if execution_grid
            else _run_sweep_lite_native(*native_args)
        )
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
        batch: The list returned by :func:`run_sweep_lite`, or the one returned
            by :func:`run_sweep` (a ``SweepResult`` iterates into it).
        names: One column name, or a list of them. Available: ``final_equity``,
            ``trade_count``, and every :class:`PerformanceMetrics` field
            (``sharpe``, ``sortino``, ``calmar``, ``max_drawdown``, ``alpha``,
            ``beta``, ``tstat_alpha``, ``total_return``, ``cagr``,
            ``volatility``, ``skewness``, ``kurtosis``, ``tail_ratio``,
            ``omega_ratio``, ``ulcer_index``, ``best_day``, ``worst_day``,
            ``avg_daily_return``, ``pct_positive_days``,
            ``max_drawdown_duration_days``, ``tstat_sharpe``).

            From a FULL sweep, six more say what the execution did, which is
            what an ``execution_grid`` is swept to read:

            * ``service_rate`` -- maker fills over the quotes that were posted;
            * ``queue_decided_fills`` -- fills the queue granted because the
              volume ahead of the order was consumed;
            * ``stale_fills`` -- fills taken by a quote after its cancellation
              was decided and before it took effect;
            * ``adverse_1s_bps`` / ``adverse_10s_bps`` -- mean markout at those
              horizons, negative when the market left in the direction that
              hurts;
            * ``half_spread_captured_bps`` -- what the quote earned at the
              instant it was served.

            Each is ``NaN`` on a run that configured no model behind it (no
            queue, no latency, no ``execution.fill_marks``), and asking a LITE
            sweep for one is refused by name: the lite drivers walk no tape, so
            they produce none of these.

    Returns:
        A single ``np.ndarray`` if ``names`` is a string, else a dict mapping
        each name to its array. Arrays are float64 and in combo order (the same
        order as ``batch``), so ``np.argmax``/``argsort`` indices map straight
        back onto it. ``trade_count`` comes back as float64 like the rest.

    Note:
        The arrays are read-only views over the returned buffers (no copy). Call
        ``.copy()`` if you need to mutate one.

        A metric a combination did not report is NaN: a run shorter than two
        days has no Sharpe, Sortino, volatility, CAGR or other statistic of
        the daily returns. ``np.argmax`` returns the index of the FIRST NaN
        when there is one; rank with ``np.nanargmax`` (which raises when every
        value is NaN) or ``np.argsort`` on the negated array, which puts NaN
        last.

    Example:
        >>> batch = mbt.run_sweep_lite(strategy, grid, config, store, device="cuda")
        >>> sharpe = mbt.sweep_columns(batch, "sharpe")
        >>> best = batch[int(np.nanargmax(sharpe))]
    """
    import numpy as _np

    single = isinstance(names, str)
    wanted = [names] if single else list(names)
    # A SweepResult hands back Result wrappers; the native extractor reads the
    # engine's own objects. Unwrapping here rather than asking the caller to
    # keeps `sweep_columns(list(sweep), ...)` working on both sweeps.
    rows = [r.raw if isinstance(r, Result) else r for r in batch]
    raw = _sweep_columns_native(rows, wanted)
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

    Windows shorter than two days report no metric taken from daily returns
    (Sharpe, Sortino, volatility, CAGR, Calmar, ...: NaN). A NaN never wins a
    fold's selection; a fold where ``optimize_metric`` is NaN for every
    combination is refused with a ``ValueError`` that says so, and an OOS
    window under two days gives NaN ``oos_metrics`` and a ``wfe`` of ``None``.
    Optimise ``total_return`` or ``max_drawdown`` on such windows.

    Note: the legacy ``method="Rolling"`` was renamed ``geometry="blocked"``
    (independent blocks separated by gaps, not Pardo's rolling); for Pardo's
    walk-forward use ``geometry="pardo"``.
    """
    # Pro feature: report it here so a notebook gets a clean LicenseError rather
    # than a traceback from deeper in the run.
    _require_pro("Walk-forward optimization")
    # And the accelerator is a tier of its own. The engine refuses it too, before
    # loading any data, but that refusal surfaces as a ValueError from the native
    # boundary: a caller catching LicenseError would miss it. Asked here, the
    # four GPU entry points all raise the same exception with the same wording.
    _require_grant_for_gpu(wf_config.get("device"), "GPU walk-forward")
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
    _require_grant_for_gpu(device, "GPU stochastic simulation")
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

    Per-strategy ``stop_loss``/``take_profit``/``trailing_stop`` are honored:
    each leg runs with the orders its own JSON carries. See
    :meth:`Portfolio.strategy` for what else the split into legs implies.

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
    if name == "sim":
        return _importlib.import_module("manifoldbt.sim")
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
        # Minuscules obligatoires: les deux ecrivains natifs du moteur et les
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
       bt.when(abs(z) <= bt.lit(0.5), bt.lit(0.0), bt.hold())))  # hold() = keep the POSITION
                                                                 # under a filter, use .ffill() (see recipes)
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
    bar_interval=bt.Interval.hours(1),   # one minute if left out; down to seconds(1)
                                         # on Pro, minutes(1) at the finest on Community
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

Quotes (order-level simulation, one symbol, book and tape in the store)
------------------------------------------------------------------------
from manifoldbt import book
maker = (bt.Strategy.create("maker")    # no .size(): each quote carries its own
         .quote("buy", book.bid_price_at(1), 0.01, enabled=bt.position() < 0.05)
         .quote("sell", book.ask_price_at(1), 0.01, enabled=bt.position() > -0.05))
cfg = bt.BacktestConfig(..., bar_interval=bt.Interval.millis(100),   # the wake-up clock
        execution=bt.ExecutionConfig(latency={"order": bt.Interval.millis(5),
            "cancel": bt.Interval.millis(5), "response": bt.Interval.millis(5),
            "feed": bt.Interval.millis(2)}))
res = bt.run(maker, cfg, store)        # QuoteResult: + orders_df(), fills_df(), events_df()
# a wake-up: NaN price/size or null enabled holds; enabled False or size <= 0
# withdraws; same price keeps the queue; else cancel and repost.
# .quote(..., cooldown=bt.Interval.millis(500)): no post for 500 ms after the
# quote sent a cancellation (a requote then cancels now, posts after the pause).
# state it reads: bt.position(), bt.cash(), bt.live_qty("bid"), bt.order_age("ask"),
# bt.order_price("bid"), bt.last_fill_px("bid"), bt.queue_ahead("ask"),
# bt.position_age(), bt.last_fill_age("bid"), bt.last_cancel_age("ask") (seconds,
# NaN before the event), book.bid_price_at(k), book.bid_levels();
# bt.round (half to even), bt.clip. Sweeps: run_sweep / run_sweep_lite / run_batch,
# each combination exactly the run alone.

Worked recipes -- yes, the DSL expresses these
----------------------------------------------
Stateful thresholds (hysteresis). hold() keeps the previous POSITION, so a band
between entry and exit needs no Python loop:

    z = (bt.col("close") / bt.symbol_ref("ETHUSDT", "close")).zscore(200)
    pos = bt.when(z >= bt.lit(2.0), bt.lit(-1.0),          # short the spread
          bt.when(z <= bt.lit(-2.0), bt.lit(1.0),          # long the spread
          bt.when((z >= bt.lit(-0.5)) & (z <= bt.lit(0.5)), bt.lit(0.0),
          bt.hold())))                                     # else: keep position
    strat = bt.Strategy.create("pair").signal("pos", pos).size(bt.col("pos"))
    cfg = bt.BacktestConfig(universe=["BTCUSDT"], ...)     # plain list is fine

Persistent state UNDER a regime filter -- use .ffill(), not hold(). hold() holds
the position, so a filter that writes 0.0 while it is closed is what hold() holds
when it reopens: exposure only returns on the next threshold crossing (measured:
a filter open 55% of the time left 0.7% exposure, silently). .ffill() carries the
last non-NaN value of the SERIES, so masking it does not destroy it:

    state = bt.when(imb >= thr, bt.lit(1.0),
            bt.when(imb <= -thr, bt.lit(-1.0))).ffill()   # omitted branch = NaN
    pos   = bt.when(regime_open, state, bt.lit(0.0))      # back on the same bar

    # want the other behaviour (exit, re-enter only on a NEW signal)? then keep
    # hold() inside the when: bt.when(regime_open, <...hold()...>, bt.lit(0.0))

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
"bar_interval below one minute requires a Pro license" -> sub-minute simulation is
    Pro; Community runs at one minute at the finest. A config that leaves
    bar_interval out runs at one minute.
"""


__all__ = [
    "AccountPhase",
    "AccountRules",
    "account_sessions",
    # Core types
    "BacktestResult",
    "BatchResultLite",
    "DataStore",
    "Result",
    "QuoteResult",
    "SweepResult",
    # Data ingestion
    "ingest",
    "ingest_trades",
    "ingest_book",
    "ingest_mbo",
    "convert_book_to_deltas",
    "bars_from_trades",
    "attach_quotes",
    "import_csv",
    "import_dataframe",
    # Run functions
    "run",
    "run_sweep",
    "run_batch",
    "reconcile",
    "Reconciliation",
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
    "position",
    "live_qty",
    "order_age",
    "order_price",
    "position_age",
    "last_fill_age",
    "last_cancel_age",
    "last_fill_px",
    "queue_ahead",
    "cash",
    "round",
    "clip",
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
    # The stored order book as columns (submodule)
    "book",
    # Managed compute (submodule)
    "cloud",
    # Plotting (lazy, requires plotly)
    "plot",
    # Diagnostics (lazy)
    "diagnostics",
    # Tick-level layer (lazy)
    "ticks",
    # Order-level simulation (lazy)
    "sim",
]
