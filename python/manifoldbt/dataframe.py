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


def results_to_df(
    results: Sequence[Any],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    backend: str = "auto",
) -> Any:
    """Convert a list of BacktestResult (or Result) objects to a metrics DataFrame.

    Each row contains all performance metrics plus parameter values (if provided).

    Args:
        results: Sequence of BacktestResult or Result objects.
        param_grid: Optional parameter grid dict (used to label rows with param values).
            When provided, the Cartesian product is expanded to match result order.
        backend: ``"pandas"``, ``"polars"``, or ``"auto"``.

    Returns:
        A DataFrame with one row per result and columns for each metric + parameter.
    """
    import itertools

    backend = _resolve_backend(backend)

    rows: List[Dict[str, Any]] = []

    # Expand parameter grid into list of param dicts
    param_combos: Optional[List[Dict[str, Any]]] = None
    if param_grid:
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        param_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    for i, result in enumerate(results):
        # Support both raw BacktestResult and Result wrapper
        metrics = result.metrics if hasattr(result, "metrics") else {}
        row: Dict[str, Any] = {}

        # Add parameters
        if param_combos is not None and i < len(param_combos):
            for k, v in param_combos[i].items():
                row[f"param_{k}"] = v

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
