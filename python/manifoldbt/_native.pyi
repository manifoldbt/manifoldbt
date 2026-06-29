"""Type stubs for the Rust-built _native extension module."""
from typing import Any, Dict, List, Optional

import pyarrow as pa


class DataStore:
    """Bar data store (Parquet by default, or Arrow IPC via ``arrow_dir``) with SQLite metadata."""

    def __init__(
        self,
        data_root: str,
        metadata_db: str = "metadata/metadata.sqlite",
        dataset: str = "bars_1m",
        mega: Optional[str] = None,
        arrow_dir: Optional[str] = None,
    ) -> None: ...
    def dataset(self) -> str: ...
    def data_root(self) -> str: ...
    def metadata_db(self) -> str: ...
    def active_version(self, dataset: str) -> str: ...
    def list_versions(self, dataset: str) -> List[str]: ...
    def resolve_symbol(self, ticker: str) -> int: ...
    def list_symbols(self) -> List[tuple[int, str]]: ...


class BacktestResult:
    """Arrow-backed backtest results (zero-copy from Rust)."""

    @property
    def manifest(self) -> Dict[str, Any]: ...
    @property
    def metrics(self) -> Dict[str, Any]: ...
    @property
    def equity_curve(self) -> pa.Array: ...
    @property
    def positions(self) -> pa.RecordBatch: ...
    @property
    def trades(self) -> pa.RecordBatch: ...
    @property
    def daily_returns(self) -> pa.Array: ...
    @property
    def warnings(self) -> List[str]: ...
    @property
    def trade_count(self) -> int: ...


class AlignedData:
    """Pre-loaded and aligned bar data for fast repeated backtests."""

    @property
    def num_bars(self) -> int: ...
    @property
    def num_symbols(self) -> int: ...
    def slice(self, start_ns: int, end_ns: int) -> "AlignedData": ...


class BatchResultLite:
    """Lightweight batch result with metrics only (no Arrow output)."""

    @property
    def strategy_name(self) -> str: ...
    @property
    def final_equity(self) -> float: ...
    @property
    def trade_count(self) -> int: ...
    @property
    def metrics(self) -> Dict[str, Any]: ...


def compile_strategy_json(strategy_json: str) -> str: ...
def run_json(strategy_json: str, config_json: str, store: DataStore) -> str: ...
def run(strategy_json: str, config_json: str, store: DataStore) -> BacktestResult: ...
def run_sweep(
    strategy_json: str,
    param_grid_json: str,
    config_json: str,
    store: DataStore,
    max_parallelism: int = 0,
) -> List[BacktestResult]: ...
def run_batch(
    strategy_jsons: List[str],
    config_json: str,
    store: DataStore,
    max_parallelism: int = 0,
) -> List[BacktestResult]: ...
def run_batch_lite(
    strategy_jsons: List[str],
    config_json: str,
    store: DataStore,
    max_parallelism: int = 0,
) -> List[BatchResultLite]: ...
def run_with_parquet(
    strategy_json: str,
    config_json: str,
    parquet_path: str,
    version_id: str,
) -> BacktestResult: ...
def load_and_align(config_json: str, store: DataStore) -> AlignedData: ...
def run_on_aligned(
    strategy_json: str,
    config_json: str,
    aligned: AlignedData,
) -> BacktestResult: ...
def py_run_walk_forward(
    strategy_json: str,
    wf_config_json: str,
    config_json: str,
    store: DataStore,
) -> Dict[str, Any]: ...
def py_run_sweep_2d(
    strategy_json: str,
    sweep_config_json: str,
    config_json: str,
    store: DataStore,
) -> Dict[str, Any]: ...
def py_run_stability(
    strategy_json: str,
    stability_config_json: str,
    config_json: str,
    store: DataStore,
) -> Dict[str, Any]: ...
def py_replay(
    manifest_json: str,
    strategy_json: str,
    store: DataStore,
) -> BacktestResult: ...
def py_run_monte_carlo(
    result: BacktestResult,
    mc_config_json: str,
) -> Dict[str, Any]: ...
def py_run_stochastic(
    sim_config_json: str,
) -> Dict[str, Any]: ...
