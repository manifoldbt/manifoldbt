"""Strategy diagnostics — look-ahead bias detection and systematic risk checks."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

_EMPTY_TS = np.array([], dtype="datetime64[ns]")


def _prepare_for_diagnostics(config, strategy, store):
    """Mirror ``run()``'s config/store preparation for the diagnostics path.

    ``run()`` resolves the config and store before serializing
    (``_cap_output_resolution`` -> ``_resolve_store`` -> ``_prepare_config``).
    Diagnostics must do the same: in particular a dict ``universe`` has to be
    resolved to a ``List[SymbolId]`` first, otherwise ``config.to_json()`` emits
    a JSON map and the Rust loader rejects it ("invalid type: map, expected a
    sequence"). Returns the prepared ``(config, store)``.
    """
    from manifoldbt import (
        _cap_output_resolution,
        _resolve_store,
        _prepare_config,
    )
    config = _cap_output_resolution(config)
    store = _resolve_store(config, store)
    config = _prepare_config(config, strategy, store)
    return config, store


@dataclass
class LookaheadReport:
    """Result of a single look-ahead bias test."""

    passed: bool
    total_trades_base: int
    total_trades_overlap: int
    mismatched: int
    method: str = ""
    details: List[Dict[str, Any]] = field(default_factory=list)

    def assert_clean(self) -> None:
        """Raise AssertionError if look-ahead bias was detected."""
        if not self.passed:
            if self.method == "static":
                msg = (
                    f"Look-ahead bias detected (static): {self.mismatched} "
                    f"expression(s) read bars ahead of the current one"
                )
                if self.details:
                    d = self.details[0]
                    msg += f"\nFirst: {d['field']}: {d['base']}, {d['extended']}"
            else:
                msg = (
                    f"Look-ahead bias detected ({self.method}): "
                    f"{self.mismatched} trades differ out of {self.total_trades_base}"
                )
                if self.details:
                    msg += f"\nFirst mismatch: {self.details[0]}"
            raise AssertionError(msg)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.method == "static":
            lines = [f"  [static] {status}  (reads ahead={self.mismatched})"]
            for d in self.details[:3]:
                lines.append(f"    {d['field']}: {d['base']}, {d['extended']}")
            return "\n".join(lines)
        lines = [
            f"  [{self.method}] {status}  "
            f"(trades={self.total_trades_base}, mismatched={self.mismatched})",
        ]
        if self.details:
            for d in self.details[:3]:
                lines.append(
                    f"    [{d['index']}] {d['field']}: "
                    f"{d['base']} vs {d['extended']}"
                )
        return "\n".join(lines)


@dataclass
class DiagnosticsResult:
    """Combined result of all look-ahead tests."""

    reports: List[LookaheadReport]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.reports)

    def assert_clean(self) -> None:
        """Raise on the first failing sub-test."""
        for r in self.reports:
            r.assert_clean()

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"Lookahead diagnostics: {status}"]
        for r in self.reports:
            parts.append(str(r))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MODES = ("all", "extension", "truncation", "static")


def detect_lookahead(
    strategy,
    config,
    store,
    *,
    mode: str = "all",
    tolerance: float = 1e-9,
) -> DiagnosticsResult:
    """Detect look-ahead bias: a static read of the strategy, then two re-runs.

    Three sub-tests:
      * **static** -- walks the strategy's expressions for ``lead()``: a value
        at bar T taken from bar T+n. Reported by signal, with the period.
      * **extension** -- re-run on the first 2/3 of the data and compare the
        trades with the full run. Catches look-ahead that depends on how much
        data the run was given (a statistic over the whole series, a
        normalisation by the last value).
      * **truncation** -- the same comparison, split at 1/3.

    The re-runs load the data once and slice it (no redundant I/O). The
    verdict is the conjunction: ``.passed`` is False as soon as one sub-test
    fails.

    .. warning::
       **What the re-runs cannot see.** A leak that reads a *fixed* number of
       bars ahead (``close.lead(1)``, or a column precomputed with future bars
       and imported as data) is deterministic: bar T sees the same T+1 whether
       the run stops at 1/3, 2/3 or the end, so the trades of the shorter run
       match the full run and both re-run sub-tests report PASS. Only the last
       bar of a window differs (its T+1 does not exist), which is not enough
       to produce a mismatch. This is a property of every method that re-runs
       the same strategy over a different window, not a matter of tuning: no
       split exposes an intra-series leak. A PASS from ``extension`` and
       ``truncation`` next to a Sharpe in the hundreds is therefore not a
       clean bill of health.

       The ``static`` sub-test is the one that names an explicit ``lead()``.
       For an implicit one (a column built outside the engine) the check that
       works is to perturb the bars after some bar K and require the equity
       up to K to be bit-identical: ``examples/25_lookahead_trap.py`` does it.

    .. note::
       **Scope.** The re-runs re-run the *same strategy* over a different
       window, which is what lets them isolate bias introduced by the engine or
       by the strategy's own use of time. A parameter computed from the data
       *before* the backtest -- ``threshold = df.close.mean()`` in a notebook,
       then handed in as a number -- is unchanged by re-running, so the trades
       match and the verdict is clean: that bias entered upstream, in the
       research step, and is a different thing to check for.

       The boundary is shared by every re-run-based method: none can weigh a
       constant. Treat any parameter derived from data as part of the pipeline
       and re-derive it on the window under test.
       ``examples/25_lookahead_trap.py`` works through both sides with output
       you can read.

    Args:
        strategy: Strategy definition.
        config: BacktestConfig.
        store: DataStore.
        mode: ``"all"`` (default), ``"static"``, ``"extension"``, or
            ``"truncation"``. ``"static"`` needs neither data nor a store.
        tolerance: Float comparison tolerance for quantity/price/fees.

    Returns:
        DiagnosticsResult with ``.passed``, ``.assert_clean()``, ``print()``.

    Raises:
        ValueError: on an unknown ``mode``. It used to run no sub-test at all
            and report PASS over an empty list.
    """
    # Pro feature: report it here so a notebook gets a clean LicenseError rather
    # than a traceback from deeper in the analysis.
    from manifoldbt import _require_pro
    _require_pro("Look-ahead bias detection")

    # Checked natively too; here so a typo fails before any data is touched.
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    from manifoldbt._native import py_detect_lookahead as _native_detect

    # All the analysis lives in Rust (the static walk as much as the re-runs);
    # this is a thin wrapper that rebuilds the report objects from the native
    # JSON.
    if mode == "static":
        raw = _native_detect(strategy.to_json(), None, None, mode, tolerance)
    else:
        if config is None or store is None:
            raise ValueError(
                "the re-run sub-tests need a config and a data store; "
                "only mode='static' runs without them"
            )
        # Resolve config/store exactly like run() (notably dict universe ->
        # ids), otherwise config.to_json() emits a map the Rust loader rejects.
        config, store = _prepare_for_diagnostics(config, strategy, store)
        raw = _native_detect(strategy.to_json(), config.to_json(), store, mode, tolerance)

    reports = [
        LookaheadReport(
            passed=r["passed"],
            total_trades_base=r["total_trades_base"],
            total_trades_overlap=r["total_trades_overlap"],
            mismatched=r["mismatched"],
            method=r["method"],
            details=r["details"],
        )
        for r in raw
    ]
    return DiagnosticsResult(reports=reports)


# ===========================================================================
# Systematic risk checks
# ===========================================================================

@dataclass
class RiskCheckResult:
    """Result of a single risk check (pass / warn / fail)."""

    name: str
    status: str  # "pass", "warn", "fail"
    value: float
    threshold: float
    message: str = ""

    def __str__(self) -> str:
        tag = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[self.status]
        return f"  [{tag}] {self.name}: {self.message}"


@dataclass
class RiskReport:
    """Aggregated systematic risk report for a backtest result.

    Access individual time-series via ``.utilization``, ``.free_margin_ratio``,
    ``.timestamps`` for further analysis or plotting.
    """

    checks: List[RiskCheckResult]
    # Time-series (one value per unique timestamp)
    timestamps: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    utilization: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    free_margin_ratio: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    concentration: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def passed(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    @property
    def clean(self) -> bool:
        return all(c.status == "pass" for c in self.checks)

    def assert_clean(self) -> None:
        """Raise if any check failed."""
        for c in self.checks:
            if c.status == "fail":
                raise AssertionError(f"Risk check failed: {c.name} — {c.message}")

    def __str__(self) -> str:
        n_pass = sum(1 for c in self.checks if c.status == "pass")
        n_warn = sum(1 for c in self.checks if c.status == "warn")
        n_fail = sum(1 for c in self.checks if c.status == "fail")
        header = f"Risk report: {n_pass} pass, {n_warn} warn, {n_fail} fail"
        parts = [header]
        for c in self.checks:
            parts.append(str(c))
        return "\n".join(parts)


def _compute_per_timestamp(pos: dict) -> dict:
    """Aggregate position-level data to per-timestamp metrics.

    Returns dict with keys: timestamps, equity, exposure, utilization,
    free_margin_ratio, concentration (Herfindahl of symbol weights).
    """
    ts_ns = pos["timestamp"].view(np.int64) if pos["timestamp"].dtype.kind == "M" else pos["timestamp"]
    position = pos["position"].astype(np.float64)
    close = pos["close"].astype(np.float64)
    equity = pos["equity"].astype(np.float64)

    market_value = np.abs(position) * close

    unique_ts, inverse = np.unique(ts_ns, return_inverse=True)
    n = len(unique_ts)

    agg_equity = np.empty(n, dtype=np.float64)
    agg_exposure = np.zeros(n, dtype=np.float64)
    agg_hhi = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mask = inverse == i
        agg_equity[i] = equity[mask][0]
        mv = market_value[mask]
        total_mv = mv.sum()
        agg_exposure[i] = total_mv
        if total_mv > 1e-12:
            weights = mv / total_mv
            agg_hhi[i] = (weights ** 2).sum()
        else:
            agg_hhi[i] = 0.0

    safe_eq = np.where(np.abs(agg_equity) > 1e-12, agg_equity, 1e-12)
    utilization = agg_exposure / safe_eq
    free_margin_ratio = 1.0 - utilization

    return {
        "timestamps": unique_ts.view("datetime64[ns]"),
        "equity": agg_equity,
        "exposure": agg_exposure,
        "utilization": utilization,
        "free_margin_ratio": free_margin_ratio,
        "concentration": agg_hhi,
    }


def _linear_slope(y: np.ndarray) -> float:
    """Slope of OLS fit (y vs index). Returns 0 if too few points."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    y_c = y - y.mean()
    denom = (x * x).sum()
    if denom < 1e-15:
        return 0.0
    return float((x * y_c).sum() / denom)


def _check_threshold(name: str, value: float, threshold: float,
                     fail_above: bool, msg: str) -> RiskCheckResult:
    """Build a pass/fail check comparing value against a threshold."""
    if fail_above:
        status = "fail" if value > threshold else "pass"
    else:
        status = "fail" if value < threshold else "pass"
    return RiskCheckResult(name=name, status=status, value=value,
                           threshold=threshold, message=msg)


def _check_trend(utilization: np.ndarray, max_trend: float) -> RiskCheckResult:
    """Check utilization slope over time."""
    slope = _linear_slope(utilization)
    abs_slope = abs(slope)
    if abs_slope > max_trend * 10:
        status = "fail"
    elif abs_slope > max_trend:
        status = "warn"
    else:
        status = "pass"
    direction = "rising" if slope > 0 else "falling"
    return RiskCheckResult(
        name="utilization_trend", status=status, value=slope,
        threshold=max_trend, message=f"slope={slope:.2e}/bar ({direction})",
    )


def _check_concentration(hhi: np.ndarray, n_symbols: int,
                         threshold: float) -> RiskCheckResult:
    """Check Herfindahl concentration index."""
    peak_hhi = float(hhi.max()) if len(hhi) > 0 else 0.0
    if n_symbols <= 1:
        return RiskCheckResult(
            name="concentration", status="pass", value=peak_hhi,
            threshold=threshold, message="single asset (HHI=1.0, skipped)",
        )
    status = "warn" if peak_hhi > threshold else "pass"
    return RiskCheckResult(
        name="concentration", status=status, value=peak_hhi,
        threshold=threshold,
        message=f"HHI={peak_hhi:.3f} (threshold {threshold:.2f}, {n_symbols} assets)",
    )


def risk_check(
    result,
    *,
    max_utilization: float = 0.95,
    min_free_margin: float = 0.05,
    max_exposure_ratio: float = 3.0,
    max_utilization_trend: float = 1e-4,
    max_concentration: float = 0.95,
) -> RiskReport:
    """Run systematic risk checks on a backtest result.

    Analyzes free margin, utilization, leverage, and concentration over
    the full backtest period. Returns a :class:`RiskReport` with
    individual check results and time-series data.

    Args:
        result: BacktestResult from ``bt.run()``.
        max_utilization: Fail if peak utilization exceeds this (default 0.95).
        min_free_margin: Fail if free margin ratio drops below this (default 0.05).
        max_exposure_ratio: Fail if exposure / initial_capital exceeds this (default 3.0).
        max_utilization_trend: Warn if utilization slope per bar exceeds this.
        max_concentration: Warn if peak Herfindahl index exceeds this
            (1.0 = single asset, 0.5 = two equal assets).

    Returns:
        RiskReport with ``.passed``, ``.assert_clean()``, ``print()``.

    Example::

        report = bt.diagnostics.risk_check(result)
        print(report)
        report.assert_clean()
    """
    from manifoldbt._convert import positions_arrays

    pos = positions_arrays(result)
    agg = _compute_per_timestamp(pos)

    utilization = agg["utilization"]
    free_margin = agg["free_margin_ratio"]
    exposure = agg["exposure"]
    hhi = agg["concentration"]

    initial_capital = float(pos["capital"][0]) if len(pos["capital"]) > 0 else 1.0

    peak_util = float(utilization.max()) if len(utilization) > 0 else 0.0
    min_fm = float(free_margin.min()) if len(free_margin) > 0 else 1.0
    exposure_ratio = exposure / max(initial_capital, 1e-12)
    peak_exposure = float(exposure_ratio.max()) if len(exposure_ratio) > 0 else 0.0
    avg_util = float(utilization.mean()) if len(utilization) > 0 else 0.0

    checks: List[RiskCheckResult] = [
        _check_threshold("peak_utilization", peak_util, max_utilization,
                         fail_above=True,
                         msg=f"{peak_util:.1%} (threshold {max_utilization:.0%})"),
        _check_threshold("min_free_margin", min_fm, min_free_margin,
                         fail_above=False,
                         msg=f"{min_fm:.1%} (threshold {min_free_margin:.0%})"),
        _check_threshold("peak_exposure", peak_exposure, max_exposure_ratio,
                         fail_above=True,
                         msg=f"{peak_exposure:.2f}x capital (threshold {max_exposure_ratio:.1f}x)"),
        _check_trend(utilization, max_utilization_trend),
        _check_concentration(hhi, len(np.unique(pos["symbol_id"])),
                             max_concentration),
        RiskCheckResult(name="avg_utilization", status="pass", value=avg_util,
                        threshold=0.0, message=f"{avg_util:.1%}"),
    ]

    return RiskReport(
        checks=checks,
        timestamps=agg["timestamps"],
        utilization=utilization,
        free_margin_ratio=free_margin,
        concentration=hhi,
    )


# ===========================================================================
# Exposure stability across time windows
# ===========================================================================

@dataclass
class ExposureMismatch:
    """A single timestamp where exposure diverges between runs."""

    timestamp: str
    field: str
    base_value: float
    extended_value: float
    diff: float

    def __str__(self) -> str:
        return (
            f"    {self.timestamp}  {self.field}: "
            f"{self.base_value:.6f} vs {self.extended_value:.6f}  "
            f"(diff={self.diff:+.6f})"
        )


@dataclass
class ExposureStabilityReport:
    """Result of exposure stability test across different time windows.

    Compares utilization, exposure, and position sizes at the same
    timestamps when the backtest is run on different periods.
    If values differ, it means position sizing depends on future data.
    """

    passed: bool
    method: str
    overlap_bars: int
    mismatched_bars: int
    max_util_diff: float
    max_exposure_diff: float
    mismatches: List[ExposureMismatch] = field(default_factory=list)

    def assert_clean(self) -> None:
        if not self.passed:
            raise AssertionError(
                f"Exposure stability FAIL ({self.method}): "
                f"{self.mismatched_bars}/{self.overlap_bars} bars differ, "
                f"max util diff={self.max_util_diff:.6f}"
            )

    def __str__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        lines = [
            f"  [{self.method}] {tag}  "
            f"(overlap={self.overlap_bars}, mismatched={self.mismatched_bars}, "
            f"max_util_diff={self.max_util_diff:.6f}, "
            f"max_exposure_diff={self.max_exposure_diff:.4f})",
        ]
        for m in self.mismatches[:5]:
            lines.append(str(m))
        if len(self.mismatches) > 5:
            lines.append(f"    ... and {len(self.mismatches) - 5} more")
        return "\n".join(lines)


@dataclass
class ExposureDiagnosticsResult:
    """Combined result of all exposure stability tests."""

    reports: List[ExposureStabilityReport]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.reports)

    def assert_clean(self) -> None:
        for r in self.reports:
            r.assert_clean()

    def __str__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        parts = [f"Exposure stability: {tag}"]
        for r in self.reports:
            parts.append(str(r))
        return "\n".join(parts)


def _exposure_for_result(result) -> dict:
    """Extract per-timestamp exposure data from a backtest result.

    Returns dict with int64 ns timestamps as keys, values are dicts of
    {utilization, exposure, positions: {symbol_id: qty}}.
    """
    from manifoldbt._convert import positions_arrays

    pos = positions_arrays(result)
    ts_ns = pos["timestamp"].view(np.int64) if pos["timestamp"].dtype.kind == "M" else pos["timestamp"]
    position = pos["position"].astype(np.float64)
    close = pos["close"].astype(np.float64)
    equity = pos["equity"].astype(np.float64)
    sym_ids = pos["symbol_id"]

    market_value = np.abs(position) * close

    unique_ts = np.unique(ts_ns)
    data = {}

    for ts in unique_ts:
        mask = ts_ns == ts
        mv = market_value[mask]
        eq = equity[mask][0]
        total_mv = mv.sum()
        util = total_mv / max(abs(eq), 1e-12)

        sym_pos = {}
        for sid, p in zip(sym_ids[mask], position[mask]):
            sym_pos[int(sid)] = float(p)

        data[int(ts)] = {
            "utilization": float(util),
            "exposure": float(total_mv),
            "equity": float(eq),
            "positions": sym_pos,
        }

    return data


def _compare_exposures(
    base_data: dict,
    ext_data: dict,
    cutoff_ns: int,
    tolerance: float,
    method: str,
) -> ExposureStabilityReport:
    """Compare exposure data at overlapping timestamps."""
    # Only compare timestamps present in BOTH runs and <= cutoff
    base_ts = set(base_data.keys())
    ext_ts = {t for t in ext_data.keys() if t <= cutoff_ns}
    overlap_ts = sorted(base_ts & ext_ts)

    n_overlap = len(overlap_ts)
    if n_overlap == 0:
        return ExposureStabilityReport(
            passed=True, method=method, overlap_bars=0,
            mismatched_bars=0, max_util_diff=0.0, max_exposure_diff=0.0,
        )

    mismatches: List[ExposureMismatch] = []
    max_util_diff = 0.0
    max_exp_diff = 0.0
    mismatched_bars = 0

    for ts in overlap_ts:
        b = base_data[ts]
        e = ext_data[ts]
        ts_str = str(np.datetime64(ts, "ns"))
        bar_mismatch = False

        # Compare utilization
        ud = abs(b["utilization"] - e["utilization"])
        max_util_diff = max(max_util_diff, ud)
        if ud > tolerance:
            bar_mismatch = True
            mismatches.append(ExposureMismatch(
                timestamp=ts_str, field="utilization",
                base_value=b["utilization"], extended_value=e["utilization"],
                diff=e["utilization"] - b["utilization"],
            ))

        # Compare per-symbol positions
        all_syms = set(b["positions"].keys()) | set(e["positions"].keys())
        for sid in sorted(all_syms):
            bp = b["positions"].get(sid, 0.0)
            ep = e["positions"].get(sid, 0.0)
            pd = abs(bp - ep)
            if pd > tolerance:
                bar_mismatch = True
                mismatches.append(ExposureMismatch(
                    timestamp=ts_str, field=f"position[{sid}]",
                    base_value=bp, extended_value=ep, diff=ep - bp,
                ))

        # Compare total exposure
        ed = abs(b["exposure"] - e["exposure"])
        max_exp_diff = max(max_exp_diff, ed)
        if ed > tolerance * max(b["exposure"], 1.0):
            bar_mismatch = True
            mismatches.append(ExposureMismatch(
                timestamp=ts_str, field="exposure",
                base_value=b["exposure"], extended_value=e["exposure"],
                diff=e["exposure"] - b["exposure"],
            ))

        if bar_mismatch:
            mismatched_bars += 1

    return ExposureStabilityReport(
        passed=(mismatched_bars == 0),
        method=method,
        overlap_bars=n_overlap,
        mismatched_bars=mismatched_bars,
        max_util_diff=max_util_diff,
        max_exposure_diff=max_exp_diff,
        mismatches=mismatches,
    )


def check_exposure_stability(
    strategy,
    config,
    store,
    *,
    mode: str = "all",
    tolerance: float = 1e-6,
) -> ExposureDiagnosticsResult:
    """Check that exposure/utilization is identical across time windows.

    Runs the strategy on different sub-periods and compares the
    utilization, exposure, and per-symbol positions at overlapping
    timestamps. Any difference means position sizing leaks future data.

    Data is loaded once and sliced for each sub-test (no redundant I/O).

    Sub-tests:
      * **extension** — compare full period vs first 2/3.
        Catches global normalization (e.g. zscore over entire series).
      * **truncation** — compare full period vs first 1/3.
        Catches rolling window that peeks ahead.

    Args:
        strategy: Strategy definition.
        config: BacktestConfig.
        store: DataStore.
        mode: ``"all"`` (default), ``"extension"``, or ``"truncation"``.
        tolerance: Absolute tolerance for float comparisons.

    Returns:
        ExposureDiagnosticsResult with ``.passed``, ``.assert_clean()``.

    Example::

        report = bt.diagnostics.check_exposure_stability(strategy, config, store)
        print(report)
        report.assert_clean()
    """
    from manifoldbt._native import (
        load_and_align as _load_and_align,
        run_on_aligned as _run_on_aligned,
    )

    # Resolve config/store exactly like run() (notably dict universe -> ids),
    # otherwise config.to_json() emits a map the Rust loader rejects.
    config, store = _prepare_for_diagnostics(config, strategy, store)

    period = config.time_range_end - config.time_range_start

    # Load data ONCE.
    aligned = _load_and_align(config.to_json(), store)

    # Full run on pre-loaded data.
    result_full = _run_on_aligned(strategy.to_json(), config.to_json(), aligned)
    full_data = _exposure_for_result(result_full)

    reports: List[ExposureStabilityReport] = []

    if mode in ("all", "extension"):
        split_ns = config.time_range_start + int(period * 2 / 3)
        short_config = copy.deepcopy(config)
        short_config.time_range_end = split_ns
        try:
            sliced = aligned.slice(config.time_range_start, split_ns)
            result_short = _run_on_aligned(
                strategy.to_json(), short_config.to_json(), sliced,
            )
            short_data = _exposure_for_result(result_short)
            reports.append(_compare_exposures(
                short_data, full_data, split_ns, tolerance, method="extension",
            ))
        except (ValueError, RuntimeError):
            pass  # symbol data missing for sub-period

    if mode in ("all", "truncation"):
        split_ns = config.time_range_start + int(period / 3)
        short_config = copy.deepcopy(config)
        short_config.time_range_end = split_ns
        try:
            sliced = aligned.slice(config.time_range_start, split_ns)
            result_short = _run_on_aligned(
                strategy.to_json(), short_config.to_json(), sliced,
            )
            short_data = _exposure_for_result(result_short)
            reports.append(_compare_exposures(
                short_data, full_data, split_ns, tolerance, method="truncation",
            ))
        except (ValueError, RuntimeError):
            pass  # symbol data missing for sub-period

    return ExposureDiagnosticsResult(reports=reports)
