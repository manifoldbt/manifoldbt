"""Type stubs for the Rust-built _native extension module."""
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa


def calibrate_gpu_json() -> str: ...


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
    def resting_orders_detail(self) -> Optional[Dict[str, List[Any]]]: ...


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
        prefetch: bool = True, tape_only: bool = False,
    ) -> Market:
        """The stored book and tape of ``symbol`` over ``[start_ns, end_ns)``.

        ``tape_only=True`` reads the stored tape alone, for a venue whose book
        history is not published (a Polymarket market ingested with
        ``ingest_trades``): ``sim.book()`` stays empty and orders execute
        against the prints under ``Config(prints_update_book=...)``. It takes
        neither ``levels``, ``by_day`` nor ``device``.

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
    ) -> Market:
        """A market written out by hand: ``states`` as ``(ts_ns, bids, asks)``
        with each side a list of ``(price, size)`` from the best, ``trades`` as
        ``(ts_ns, price, qty, "buy" | "sell")``.

        Refused by name: states or prints out of time order, levels out of
        price order, a price or a size that is not finite and positive, and a
        ``tick_size`` that is neither ``0`` (no tick) nor a multiple of
        ``1e-8``, the grid the venue compares prices on. Not
        checked, and accepted: a crossed or locked book (an order that crosses
        it executes at once against the displayed opposite side); prints and no
        book state (every book read is empty and a market order expires); a
        level off the tick grid (``Book.levels()`` shows it as written, while an
        order priced there is put on the grid, on its passive side, and does not
        queue behind that level).

        Several states at one instant are accepted and the last one stands:
        the book has one state per instant. Several prints at one instant are
        kept in the order given."""
        ...
    @staticmethod
    def from_arrays(
        name: str,
        tick_size: float,
        *,
        quotes: Optional[Any] = None,
        trades: Optional[Any] = None,
    ) -> Market:
        """A market from columns, without a Python object per event.

        ``quotes`` is the book: ``ts_ns``, ``bid_px``, ``bid_qty``,
        ``ask_px``, ``ask_qty``. A side column of numbers is the top of book.
        A book several levels deep lists each side best first, bids falling
        and asks rising, in one of three shapes: list columns (a row the
        levels of its side; a fixed-size list, an Arrow or polars list, a
        pandas column of lists), a 2-D numpy array in a dict (a row of levels
        a row), or one column a level, ``bid_px_0``, ``bid_qty_0``,
        ``bid_px_1``, ... (or ``bid_px_00``, ...), read when there is no
        ``bid_px``. A side ends at its first level without a price (NaN,
        null, zero): pad a shallower row so. The book keeps as many levels as
        the deepest side. ``trades`` is the
        tape: ``ts_ns``, ``price``, ``qty`` and, when the tape has one,
        ``aggressor``. Each is a pandas or polars DataFrame, a pyarrow Table
        or RecordBatch, or a dict of numpy or pyarrow arrays; other columns
        are ignored. Either may be left out (an empty book, an empty tape).

        ``ts_ns`` is integer nanoseconds since the epoch, UTC, or a timestamp
        column of any unit; ``timestamp`` is read when there is no ``ts_ns``
        (a pandas index of that name too). Prices and sizes are any numeric
        type. ``aggressor`` is ``"buy"`` / ``"sell"`` or a number, ``1`` for a
        buy, ``-1`` or ``2`` for a sell; a null, a NaN or an empty string is
        not known. A ``0`` is refused rather than guessed (a column of ``1``
        for a buy and ``0`` for a sell would read its sells as unknown).

        Rows come in time order, several at one instant allowed:

        * of several quotes at one instant the last one stands, the book once
          every update stamped there is in;
        * several prints at one instant are kept in the order given; at an
          instant holding both, the venue applies the prints first, then the
          book;
        * a quote with a side missing (a best price that is zero, negative,
          NaN or null) or crossed (best bid above best ask) shows an empty
          book until the next quote; a locked quote (bid equal to ask)
          stands;
        * a print whose aggressor is not known takes the tick rule: above the
          previous price a buy, below it a sell, at the same price the side of
          the last price move, a buy before the first move.

        Refused by name: rows out of time order, a column missing, of another
        length or of a type it cannot read (a float ``ts_ns``, a boolean
        ``aggressor``), a null ``ts_ns``, a size below zero on a valid quote,
        a print whose price or size is not finite and positive, a level of
        a deep side priced after the end of its side, out of the side's
        order, or whose price comes without its size, and a
        ``tick_size`` as in :meth:`from_ladders`. The market is the one
        :meth:`from_ladders` builds from the same quotes and prints."""
        ...
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
    # A stop order's trigger, what sets it off ("last" or "bid_ask"), what a
    # stop-limit does once set off ("rest" or "ioc"), and the venue instant it
    # was set off; None for any other order (and before it is set off). A
    # bracket's stop and target carry their level and legs_on; every leg,
    # the exit at a time included, the instant it was set off.
    trigger_price: Optional[float]
    trigger_on: Optional[str]
    after_trigger: Optional[str]
    triggered_at: Optional[int]
    # A leg of a bracket: the id of its parent, and what it does ("stop",
    # "target" or "time"); None for any other order.
    parent: Optional[int]
    role: Optional[str]
    remaining: float

class Wakeup:
    # Why wait() returned: "trade", "book", "order", "timeout" or "end".
    # False once the data has ended, so `while sim.wait(...)` stops there.
    kind: str
    symbol: Optional[str]
    order_id: Optional[int]
    def __bool__(self) -> bool: ...

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
        tif: str = "GTC", expire_at: Optional[int] = None, *,
        trigger: Optional[float] = None, trigger_on: str = "last",
        after_trigger: str = "rest",
    ) -> int:
        """Send an order and get its id. ``side``, ``tif`` and the symbol are
        read without regard to case. ``price=None`` is a market order; a price
        is put on the tick grid first. A GTD order whose ``expire_at`` is at or
        before its arrival expires without executing. A post after the data is
        over is accepted: the order reaches the venue only if the clock moves
        again, and stays ``in_flight`` otherwise.

        With ``trigger``, a stop order. It reaches the venue as any order and
        waits there, ``armed``, out of the book, until the market reaches the
        trigger: ``trigger_on="last"``, a print at or through it, whichever
        side crossed; ``"bid_ask"``, a stored state whose best ask (a buy
        stop) or best bid (a sell stop) is at or through it, an empty side
        reaching nothing. Set off at an instant, it executes at the end of it,
        after every market event stamped there and before the actions arriving
        then: a market order of the same ``tif`` when ``price`` is None, a
        limit order at ``price`` otherwise, which rests when it does not
        execute (``after_trigger="rest"``) or expires (``"ioc"``). ``tif`` is
        its life while it waits, ``"GTC"`` or ``"GTD"``. The trigger is put on
        the tick grid away from the market (a buy stop's up, a sell stop's
        down). A stop whose trigger the market has already reached when it
        arrives is set off at once, or refused under
        ``Config(stop_in_market="reject")``. Cancelled as any order; a stop
        waiting for its trigger is not replaced."""
        ...
    def post_bracket(
        self, symbol: Union[str, int], side: str, qty: float, *,
        price: Optional[float] = None, trigger: Optional[float] = None,
        trigger_on: str = "last", tif: str = "GTC", expire_at: Optional[int] = None,
        stop: Optional[float] = None, target: Optional[float] = None,
        legs_on: str = "last", exit_after: Any = None, exit_at: Any = None,
    ) -> int:
        """Send the parent of a bracket, and the exits the venue holds for the
        position it opens; get the parent's id.

        The parent is an order as ``post`` sends it: a market order
        (``price=None``), a limit at ``price``, or a stop (``trigger``,
        ``trigger_on``; with ``price``, a stop-limit). When it ends having
        filled something (filled whole, or expired or cancelled after a fill
        in part), the venue arms its legs at that instant, once the market of
        the instant is in, with no latency: each an order of the other side
        for the size the parent filled. ``stop`` is a stop-market, set off at
        or through it; ``target`` a market-if-touched order, set off at or
        beyond it, the other way; both by ``legs_on`` (``"last"``: a print;
        ``"bid_ask"``: the stored best bid of a leg that sells, the best ask
        of one that buys). ``exit_after`` (a duration) or ``exit_at`` (a venue
        instant in nanoseconds, a list of them, or ``(instants, n)``: the n-th
        of them strictly after the arming) is an exit at a time, a market
        order. A stop or a target the market already reaches when the legs
        are armed is set off at once. A parent that ends with nothing filled
        arms no leg.

        The first leg set off takes the others off at the same instant
        (``cancelled``); at one instant the market comes first, so a stop or
        a target set off then goes before the exit due then. A leg executes
        at the end of its instant as a market order, taker fills; what the
        book does not fill of it waits for the venue's next event, a print or
        a book state, and executes again at the end of that instant
        (``carried`` in the journal), until filled. The legs are orders of
        their own (``parent``, ``role``), in ``sim.orders()`` once the venue's
        answer about them arrives; cancelling one takes that leg alone off,
        and a leg is not replaced. For a position bought the stop must be
        below the target, above it for one sold. Each level is put on the
        tick grid away from the market."""
        ...
    def cancel(self, id: int) -> None:
        """Send a cancellation. A cancellation that reaches the venue before its
        order (a cancel latency shorter than the entry latency) is refused, and
        the order then rests and can be served."""
        ...
    def replace(self, id: int, price: Optional[float] = None, qty: Optional[float] = None) -> None:
        """Change the price, the total size, or both. The same price with a size
        smaller or unchanged keeps the order's place; anything else sends it to
        the back. A total size that does not exceed what filled cancels the rest:
        the order ends ``cancelled``, ``qty`` unchanged."""
        ...

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
def sim_run_plan(
    plan_json: str,
    config_json: str,
    markets: List[Market],
    values_json: str = "{}",
    full: bool = True,
) -> SimResult:
    """Run an order plan (``mbt.sim.Plan``, serialised) over its one market,
    its parameters at their defaults or at ``values_json`` (``{name:
    value}``), without calling back into Python. ``full`` keeps the record of
    ``sim_run`` (the journal, the equity at each move of the clock, the
    markouts); off, the orders and the fills."""
    ...

class PlanSweepResult:
    """One row of metrics per variant of a sweep of an order plan, in the
    order of the enumeration, and what it takes to run any variant again."""

    #: ``"direct"`` or ``"sim"``.
    engine: str
    engine_reason: str
    def __len__(self) -> int: ...
    def metrics_arrow(self) -> pa.RecordBatch: ...
    def fills(self, i: int) -> Optional[SimResult]: ...
    def config_json(self, i: int) -> str: ...
    def params_json(self, i: int) -> str: ...
    def run(self, i: int, full: bool = True) -> SimResult: ...

def sim_sweep(
    plan_json: str,
    param_grid_json: str,
    config_json: str,
    markets: List[Market],
    max_parallelism: int = 0,
    exec_grid_json: str = "{}",
    keep: str = "metrics",
    engine: str = "auto",
) -> PlanSweepResult:
    """Every variant of an order plan over its one market, in parallel, the
    GIL released: the parameter grid (``{name: [values]}``, names sorted, the
    last the fastest) crossed with the execution grid (``{path: [values]}``
    over the keys of ``mbt.sim.Config``, the slowest axes). ``keep`` is
    ``"metrics"`` or ``"fills"``; ``engine`` ``"auto"``, ``"sim"`` or
    ``"direct"``, the way it took in ``engine`` and why in
    ``engine_reason``. A market held day by day is read a day at a time for
    every variant, to the bits of the market held whole."""
    ...
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
