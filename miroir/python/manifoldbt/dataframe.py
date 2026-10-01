"""Arrow-to-DataFrame conversion utilities.

Supports pandas and polars with automatic backend detection.
All conversions are zero-copy where possible (via PyArrow).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union


def detect_backend() -> str:
    """Auto-detect the best available DataFrame backend.

    Returns:
        ``"pandas"``, ``"polars"``, or ``"arrow"`` (fallback).
    """
    try:
        import pandas  # noqa: F401
        return "pandas"
    except ImportError:
        pass
    try:
        import polars  # noqa: F401
        return "polars"
    except ImportError:
        pass
    return "arrow"


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return detect_backend()
    return backend


def arrow_to_df(
    batch: Any,
    backend: str = "auto",
) -> Any:
    """Convert a PyArrow RecordBatch or Table to a DataFrame.

    Args:
        batch: A ``pyarrow.RecordBatch`` or ``pyarrow.Table``.
        backend: ``"pandas"``, ``"polars"``, or ``"auto"`` (detect).

    Returns:
        A pandas DataFrame or polars DataFrame.

    Raises:
        ImportError: If the requested backend is not installed.
    """
    backend = _resolve_backend(backend)

    if backend == "pandas":
        import pandas as pd
        import pyarrow as pa

        if isinstance(batch, pa.RecordBatch):
            batch = pa.Table.from_batches([batch])
        return batch.to_pandas()

    if backend == "polars":
        import polars as pl
        import pyarrow as pa

        if isinstance(batch, pa.RecordBatch):
            batch = pa.Table.from_batches([batch])
        return pl.from_arrow(batch)

    # Fallback: return as-is
    return batch


def record_to_frame(batch: Any, times: Sequence[str], backend: str) -> Any:
    """A ``pyarrow.RecordBatch`` of the engine's record as a DataFrame.

    ``times`` name the columns that hold nanoseconds since the epoch, turned
    into UTC timestamps here exactly as the frames built from lists turned
    them: ``pd.to_datetime(..., unit="ns", utc=True)`` for pandas, a cast to
    ``Datetime("ns", "UTC")`` for polars.

    For pandas the cast is made on the Arrow side, where a column of
    nanoseconds becomes a column of UTC timestamps without a copy, and
    ``to_pandas`` hands back the same ``datetime64[ns, UTC]`` column
    ``pd.to_datetime`` built in a second pass over it.

    Args:
        batch: The record batch, as the engine hands it over.
        times: The nanosecond columns to read as timestamps.
        backend: ``"pandas"`` or ``"polars"``, already resolved.
    """
    if backend == "pandas":
        import pyarrow as pa

        utc = pa.timestamp("ns", tz="UTC")
        names = batch.schema.names
        columns = [
            batch.column(i).cast(utc) if name in times else batch.column(i)
            for i, name in enumerate(names)
        ]
        return pa.RecordBatch.from_arrays(columns, names=names).to_pandas()
    if backend == "polars":
        import polars as pl

        df = pl.from_arrow(batch)
        return df.with_columns(pl.col(c).cast(pl.Datetime("ns", "UTC")) for c in times)
    raise ValueError(f"record_to_frame: backend {backend!r} is not pandas or polars")


def arrow_to_series(
    array: Any,
    name: str = "value",
    backend: str = "auto",
) -> Any:
    """Convert a PyArrow Array to a pandas Series or polars Series.

    Args:
        array: A ``pyarrow.Array``, ``pyarrow.ChunkedArray``, or ``pyarrow.Float64Array``.
        name: Name for the resulting Series.
        backend: ``"pandas"``, ``"polars"``, or ``"auto"`` (detect).

    Returns:
        A pandas Series or polars Series.
    """
    backend = _resolve_backend(backend)

    if backend == "pandas":
        import pandas as pd

        if hasattr(array, "to_pandas"):
            return pd.Series(array.to_pandas(), name=name)
        return pd.Series(array, name=name)

    if backend == "polars":
        import polars as pl

        try:
            import pyarrow as pa
        except ImportError:
            pa = None
        # Zero-copy: hand the Arrow buffers straight to polars instead of boxing
        # every value into a Python object via to_pylist() (copies the whole
        # column). pl.from_arrow shares the underlying buffers.
        if pa is not None and isinstance(array, (pa.Array, pa.ChunkedArray)):
            return pl.from_arrow(array).rename(name)
        return pl.Series(name=name, values=list(array))

    return array


def grid_combos(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """The combinations of a parameter grid, in the order the engine runs them.

    The engine enumerates a grid with its axes sorted by parameter NAME, the
    last axis varying fastest, whatever order the dict was written in. A
    product in insertion order (``itertools.product(*grid.values())``) only
    agrees with it when the keys happen to be alphabetical; labelling sweep
    rows that way put the right values on the wrong rows.

    Each combo is a dict in the grid's own key order, so a table built from
    them keeps the caller's column order while its rows follow the engine's.
    """
    import itertools

    if not param_grid:
        return []
    axes = sorted(param_grid)
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*(param_grid[axis] for axis in axes)):
        by_name = dict(zip(axes, combo))
        out.append({name: by_name[name] for name in param_grid})
    return out


def _native_parameters(result: Any) -> Optional[Dict[str, Any]]:
    """``result.manifest["parameters"]`` of an engine result, read alone.

    The whole manifest carries the config, and building it once per row was
    most of the cost of labelling a full sweep; ``run_parameters`` is the same
    map, key for key and value for value. ``None`` for anything that is not an
    engine result (or the wrapper of one), which then reads its manifest.
    """
    native, wrappers = _engine_types()
    kind = type(result)
    if kind in wrappers:
        result = result.raw
        kind = type(result)
    if kind is native:
        return result.run_parameters
    return None


_ENGINE_TYPES: Optional[tuple] = None


def _engine_types() -> tuple:
    """``(BacktestResult, (Result, QuoteResult))``, imported once: this module
    is imported by ``manifoldbt.result``, so not at its top."""
    global _ENGINE_TYPES
    if _ENGINE_TYPES is None:
        from manifoldbt._native import BacktestResult
        from manifoldbt.result import QuoteResult, Result

        _ENGINE_TYPES = (BacktestResult, (Result, QuoteResult))
    return _ENGINE_TYPES


def _run_parameters(
    result: Any, param_grid: Dict[str, List[Any]]
) -> Optional[Dict[str, Any]]:
    """The swept parameters a result was actually run with, from its manifest.

    A full run records the parameter map it executed under
    (``manifest["parameters"]``): a label read from there cannot drift from
    the engine's enumeration, whatever order the results are handed over in.
    Returns ``None`` when the object carries no manifest (lite results, plain
    metric holders) or does not name every swept parameter; the caller then
    falls back to the engine's enumeration order.
    """
    from manifoldbt._serde import scalar_value_from_json

    params = _native_parameters(result)
    if params is None:
        manifest = getattr(result, "manifest", None)
        if not isinstance(manifest, dict):
            return None
        params = manifest.get("parameters")
        if not isinstance(params, dict):
            return None
    out: Dict[str, Any] = {}
    for name in param_grid:
        if name not in params:
            return None
        out[name] = scalar_value_from_json(params[name])
    return out


def exec_grid_combos(execution_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """The execution combinations, in the order the engine runs them.

    Same rule as :func:`grid_combos`: axes sorted by PATH, last axis varying
    fastest, whatever order the dict was written in. The execution axes are the
    slowest of a sweep, so combination ``i`` of this list labels the whole
    block of parameter results at ``i * len(grid_combos(param_grid))``.
    """
    import itertools

    if not execution_grid:
        return []
    axes = sorted(execution_grid)
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*(execution_grid[axis] for axis in axes)):
        by_path = dict(zip(axes, combo))
        out.append({path: by_path[path] for path in execution_grid})
    return out


def _run_execution(result: Any, execution_grid: Dict[str, List[Any]]) -> Optional[Dict[str, Any]]:
    """The execution settings a result was actually run under, from its manifest.

    A full run records its whole config (``manifest["config"]["execution"]``),
    so a label read from there cannot drift from the engine's enumeration.
    Returns ``None`` when the object carries no manifest (lite results); the
    caller then falls back to the enumeration order.
    """
    manifest = getattr(result, "manifest", None)
    if not isinstance(manifest, dict):
        return None
    node = manifest.get("config")
    if not isinstance(node, dict):
        return None
    execution = node.get("execution")
    if not isinstance(execution, dict):
        return None
    out: Dict[str, Any] = {}
    for path in execution_grid:
        value: Any = execution
        for segment in str(path).split("."):
            if not isinstance(value, dict) or segment not in value:
                return None
            value = value[segment]
        out[path] = value
    return out


def results_to_df(
    results: Sequence[Any],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    backend: str = "auto",
    execution_grid: Optional[Dict[str, List[Any]]] = None,
) -> Any:
    """Convert a list of BacktestResult (or Result) objects to a metrics DataFrame.

    Each row contains all performance metrics plus parameter values (if provided).

    Args:
        results: Sequence of BacktestResult or Result objects.
        param_grid: Optional parameter grid dict, used to label rows with
            ``param_*`` columns. A result that carries a manifest is labelled
            with the parameters it actually ran with; one that does not (the
            lite path) is labelled by position, in the engine's enumeration
            order (see :func:`grid_combos`).
        backend: ``"pandas"``, ``"polars"``, or ``"auto"``.
        execution_grid: Optional execution grid, used to label rows with
            ``exec_*`` columns. Read from each run's own config when it carries
            a manifest, by position otherwise -- and the execution axes are the
            SLOWEST, so position ``i`` takes execution combination
            ``i // len(grid_combos(param_grid))``.

    Returns:
        A DataFrame with one row per result and columns for each metric,
        parameter and execution setting.
    """
    backend = _resolve_backend(backend)

    rows: List[Dict[str, Any]] = []

    param_combos = grid_combos(param_grid) if param_grid else None
    exec_combos = exec_grid_combos(execution_grid) if execution_grid else None
    block = max(1, len(param_combos) if param_combos else 1)
    # Positions that have a combination: the whole parameter surface once per
    # execution combination. Past them, a row gets no invented label.
    positioned = (len(param_combos) if param_combos is not None else 0) * (
        len(exec_combos) if exec_combos else 1)

    for i, result in enumerate(results):
        # Support both raw BacktestResult and Result wrapper. One read: the
        # getter builds a fresh dict, and `hasattr` followed by the attribute
        # built it twice per row.
        metrics = getattr(result, "metrics", {})
        row: Dict[str, Any] = {}

        # Add parameters
        if param_grid:
            labels = _run_parameters(result, param_grid)
            if labels is None and param_combos is not None and i < positioned:
                labels = param_combos[i % len(param_combos)]
            if labels is not None:
                for k, v in labels.items():
                    row[f"param_{k}"] = v

        # Add execution settings
        if execution_grid:
            exec_labels = _run_execution(result, execution_grid)
            if exec_labels is None and exec_combos:
                exec_labels = exec_combos[(i // block) % len(exec_combos)]
            if exec_labels is not None:
                for k, v in exec_labels.items():
                    row[f"exec_{k}"] = v

        # Flatten metrics dict
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, dict):
                    # Nested (e.g. trade_stats)
                    for sub_k, sub_v in v.items():
                        row[sub_k] = sub_v
                else:
                    row[k] = v

        rows.append(row)

    if backend == "pandas":
        import pandas as pd
        return pd.DataFrame(rows)

    if backend == "polars":
        import polars as pl
        return pl.DataFrame(rows)

    return rows
