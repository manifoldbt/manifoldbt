"""Type stubs for the Rust-built _native extension module."""
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa


class DataStore:
    """Bar data store (Parquet, or Arrow IPC via ``arrow_dir``) with SQLite metadata.

    A store written by ``import_dataframe`` / ``import_csv`` / ``ingest`` lives in
    Arrow IPC under ``<data_root>/mega`` and is detected automatically: reopening
    with just ``DataStore(data_root, metadata_db)`` works.
    """

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
    @property
    def fill_fragility(self) -> Optional[Dict[str, int]]: ...
    @property
    def order_activity(self) -> Optional[Dict[str, int]]: ...
    @property
    def tape_resolution(self) -> Optional[Dict[str, int]]: ...
    @property
    def fill_marks(self) -> Optional[Dict[str, Any]]: ...
    def fill_marks_detail(self) -> Optional[Dict[str, List[Any]]]: ...
    def fill_marks_arrow(self) -> Optional[pa.RecordBatch]: ...


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
def run_with_fill_rule(
    strategy_json: str,
    config_json: str,
    store: DataStore,
    callback: Any,
    override_traverse: bool = False,
) -> BacktestResult: ...
def run_sweep(
    strategy_json: str,
    param_grid_json: str,
    config_json: str,
    store: DataStore,
    max_parallelism: int = 0,
    exec_grid_json: str = "{}",
) -> List[BacktestResult]: ...
def run_batch(
    strategy_jsons: List[str],
    config_json: str,
    store: DataStore,
    max_parallelism: int = 0,
    exec_grid_json: str = "{}",
) -> List[BacktestResult]: ...
def reconcile_fills(
    result: BacktestResult,
    store: DataStore,
    live_ts_ns: List[int],
    live_side: List[int],
    live_price: List[float],
    live_qty: List[float],
    live_symbol_id: List[int],
    tolerance_ns: int,
    price_tolerance: Optional[float] = None,
    orders_posted: Optional[int] = None,
) -> Dict[str, Any]: ...
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

# Anonymous usage counters. Private: they exist so a user can see what the
# install ping sends, and so the tests can assert on it.
def _usage_snapshot() -> Dict[str, Any]: ...
def _flush_usage() -> None: ...

# Order-level simulation (manifoldbt.sim)

class Market:
    @staticmethod
    def from_store(
        store: DataStore, symbol: Union[str, int], start_ns: int, end_ns: int,
        levels: Optional[int] = None, by_day: bool = False, device: str = "cpu",
        prefetch: bool = True,
    ) -> Market:
        """The stored book and tape of ``symbol`` over ``[start_ns, end_ns)``.

        ``levels`` keeps at most that many levels per side, all of them when
        the book is stored shallower; every stored level when ``None`` (the
        default). ``by_day`` holds the market a UTC day at a time: the
        simulation loads each day when it reaches it (the next ones prepared
        ahead unless ``prefetch`` is off) and lets go of the days behind it,
        with the run of the market held whole, bit for bit. ``device``
        (``"cpu"`` or ``"cuda"``) says where the days' book is prepared.

        A symbol stored order by order (``ingest_mbo``) is read from its
        events: its book by price, 1000 levels deep unless ``levels`` says
        otherwise, and its tape are read off them, and its venue serves a
        resting order in its exact place in the queue."""
        ...
    @staticmethod
    def from_ladders(
        name: str,
        tick_size: float,
        states: List[Tuple[int, List[Tuple[float, float]], List[Tuple[float, float]]]],
        trades: Optional[List[Tuple[int, float, float, str]]] = None,
    ) -> Market: ...
    @property
    def name(self) -> str: ...
    @property
    def tick_size(self) -> float: ...
    @property
    def book_states(self) -> int: ...
    @property
    def trades(self) -> int: ...
    @property
    def span(self) -> Optional[Tuple[int, int]]: ...
    @property
    def by_day(self) -> bool: ...
    @property
    def by_order(self) -> bool: ...
    @property
    def mbo_events(self) -> int: ...

class Book:
    symbol: str
    ts_ns: Optional[int]
    tick: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    best_bid_qty: Optional[float]
    best_ask_qty: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    def qty_at(self, price: float) -> float: ...
    def levels(
        self, n: Optional[int] = None
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]: ...
    def depth_through(self, price: float) -> float: ...

class Order:
    id: int
    symbol: str
    side: str
    price: Optional[float]
    qty: float
    filled: float
    avg_price: Optional[float]
    status: str
    tif: str
    expire_at: Optional[int]
    cancel_pending: bool
    replace_pending: bool
    sent_ns: int
    queue_ahead: Optional[float]
    remaining: float

class Wakeup:
    kind: str
    symbol: Optional[str]
    order_id: Optional[int]

class Sim:
    end: int
    symbols: List[str]
    def now(self) -> int: ...
    def elapse(self, dt: Any) -> bool: ...
    def wait(
        self, trade: bool = False, book: bool = False, order: bool = False,
        timeout: Any = None,
    ) -> Wakeup: ...
    def book(self, symbol: Union[str, int]) -> Book: ...
    def trades(self, symbol: Union[str, int]) -> List[Tuple[int, float, float, str]]: ...
    def orders(
        self, symbol: Union[str, int, None] = None, side: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Order]: ...
    def order(self, id: int) -> Optional[Order]: ...
    def position(self, symbol: Union[str, int]) -> float: ...
    def cash(self) -> float: ...
    def post(
        self, symbol: Union[str, int], side: str, price: Optional[float], qty: float,
        tif: str = "GTC", expire_at: Optional[int] = None,
    ) -> int: ...
    def cancel(self, id: int) -> None: ...
    def replace(self, id: int, price: Optional[float] = None, qty: Optional[float] = None) -> None: ...

class SimResult:
    symbols: List[str]
    position: Dict[str, float]
    cash: float
    fees: float
    fill_marks: Dict[str, Any]
    fill_fragility: Dict[str, Any]
    levels_snapped: int
    def orders_columns(self) -> Dict[str, List[Any]]: ...
    def fills_columns(self) -> Dict[str, List[Any]]: ...
    def events_columns(self) -> Dict[str, List[Any]]: ...
    def equity_columns(self) -> Tuple[List[int], List[float]]: ...
    def fill_marks_detail(self) -> Dict[str, List[Any]]: ...
    # The same tables as record batches, handed over without a copy; the
    # orders' `sent_ns` is named `sent_at` there, as in `orders_df`.
    def orders_arrow(self) -> pa.RecordBatch: ...
    def fills_arrow(self) -> pa.RecordBatch: ...
    def events_arrow(self) -> pa.RecordBatch: ...
    def equity_arrow(self) -> pa.RecordBatch: ...
    def fill_marks_arrow(self) -> pa.RecordBatch: ...
    def metrics_totals(
        self, compensated: bool
    ) -> Tuple[int, int, int, int, int, float, float, float, Optional[float]]: ...

def sim_run(strategy: Any, markets: List[Market], config_json: str) -> SimResult: ...
def run_quotes(
    strategy_json: str,
    config_json: str,
    store: Optional[DataStore] = None,
    markets: Optional[List[Market]] = None,
    by_day: Optional[bool] = None,
    device: str = "cpu",
    prefetch: bool = True,
) -> Tuple[BacktestResult, SimResult]: ...
def run_quote_sweep(
    strategy_json: str,
    param_grid_json: str,
    config_json: str,
    store: Optional[DataStore] = None,
    markets: Optional[List[Market]] = None,
    max_parallelism: int = 0,
    exec_grid_json: str = "{}",
) -> List[Tuple[BacktestResult, SimResult]]: ...
def run_quote_sweep_lite(
    strategy_json: str,
    param_grid_json: str,
    config_json: str,
    store: Optional[DataStore] = None,
    markets: Optional[List[Market]] = None,
    max_parallelism: int = 0,
    exec_grid_json: str = "{}",
) -> List[BatchResultLite]: ...
def run_quote_batch(
    strategy_jsons: List[str],
    config_json: str,
    store: Optional[DataStore] = None,
    markets: Optional[List[Market]] = None,
    max_parallelism: int = 0,
    exec_grid_json: str = "{}",
) -> List[Tuple[BacktestResult, SimResult]]: ...
def run_quote_batch_lite(
    strategy_jsons: List[str],
    config_json: str,
    store: Optional[DataStore] = None,
    markets: Optional[List[Market]] = None,
    max_parallelism: int = 0,
    exec_grid_json: str = "{}",
) -> List[BatchResultLite]: ...
