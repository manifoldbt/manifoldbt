"""NautilusTrader adapter.

NautilusTrader is an event-driven engine: it replays bars one at a time through a
simulated exchange and calls a Python strategy on each of them. The other three
engines in this harness are vectorised or compiled whole-series loops, so this is
the one adapter where the *design* differs, not only the speed. What is compared
is therefore the same workload, expressed the way a Nautilus user would write it:
the low-level ``BacktestEngine``, a ``Strategy`` subclass, built-in indicators,
market orders, and the venue's own matching engine.

What is timed, and what is not
------------------------------
Everything that can be prepared once is prepared before timing: the instrument,
the ``Bar`` objects (``BarDataWrangler``), the venue, the account, the strategy,
and the ``add_data`` call that hands the bars to the engine. That is the same
courtesy the other adapters get (a prepared store, prepared Series, prepared
arrays), and it is the data-loading exclusion the README states.

The timed closure is what a user repeats: ``engine.reset()``, ``engine.run()``,
then reading the headline numbers off the engine. ``reset()`` followed by ``run()``
is Nautilus's own way of replaying the same data again, and it is checked rather
than assumed. The strategy clears its own indicators and state in ``on_reset``
(without that, the second replay starts from the first one's moving averages and
returns a different equity, which was measured), the parity gate reads the first
call, and every later call is compared with it: a replay that drifted from the
first would stop the run instead of timing a different computation.

Execution conventions, chosen to line up with the reference
-----------------------------------------------------------
* **Fill at the close of the signal bar.** The strategy submits a market order from
  ``on_bar``; the simulated exchange has already moved the market to that bar's
  close, and the order is matched before the next bar arrives, so the fill lands on
  the same bar at the close. That is manifoldbt's ``signal_delay=0`` plus
  ``execution_price="AtClose"``, with no setting needed for it.
* **No volume-limited fills.** Nautilus builds its book from a bar by splitting the
  bar's volume over four synthetic updates (open, high, low, close) and uses that
  quantity as the liquidity available. A market order larger than a quarter of the
  bar's volume is *partially* filled and the remainder is booked one tick worse.
  The generated bars carry a random volume between 100 and 10,000 units, so with
  full-equity sizing (about 1,000 units) that rule fires on a large share of
  fills. The other three engines have no volume constraint at all, so the venue is
  given a fill model (``_BenchFillModel``) that answers every market order from a
  book with effectively unlimited depth at the bar's close. That is the same
  mechanism Nautilus ships for it (``BestPriceFillModel``), written out here
  because the bracket workload needs stop and limit orders left on the standard
  logic, which ``BestPriceFillModel`` does not do.
* **Slippage** in basis points is not a Nautilus setting (``prob_slippage`` moves a
  fill by one *tick*, at random). The fill model shifts the book instead: asks sit
  ``slippage_bps`` above the close and bids ``slippage_bps`` below it, which is the
  documented extension point for custom slippage and produces exactly what
  ``Slippage.fixed_bps`` produces on the reference.
* **Fees** are the instrument's taker fee (market orders always take), charged on
  notional by the venue's default ``MakerTakerFeeModel``.
* **Account and sizing.** A margin account with leverage 1 in a high-precision
  quote currency (USDT, 8 decimals). Not USD: USD carries 2 decimals, and Nautilus
  rounds each realised PnL to the currency's precision, so a cent of rounding per
  trade would put the final equity ~1e-6 of capital away from a float64 engine
  after a few thousand trades. Prices and quantities are fixed point with 9
  decimals on both platforms (the standard-precision Windows wheel allows 9, the
  Linux wheel 16, and using the same on both keeps the results identical to the
  digit); the rounding that introduces is 1e-12 to 2e-10 of capital depending on
  how often the strategy trades, and is reported by the gate, not hidden.
  ``FractionOfEquity`` is the cash balance at the signal bar divided by the close,
  rounded *down* to the quantity precision; units sizing is the fixed quantity.
* **Indicators** are Nautilus's own ``SimpleMovingAverage``,
  ``ExponentialMovingAverage`` and ``RelativeStrengthIndex`` (Wilder), registered
  on the bar type so the engine updates them before ``on_bar`` runs. The EMA
  matches the reference bit for bit and the SMA to ~1e-13. The RSI is seeded
  differently (see README.md, "The RSI seed"): it agrees to the last bit after
  ~800 bars and to within 4.5e-2 before that, and the parity gate is what says
  whether that moved a decision on the benchmark data.

Brackets
--------
The stop and the target are Nautilus's native contingent orders
(``OrderFactory.bracket``): a stop-market and a limit, linked one-cancels-other,
armed when the market entry fills. Their levels are computed from the signal bar's
close, which is the fill price of the entry. The venue's standard logic decides the
fills: a stop that the bar trades through fills at its trigger price, a limit that
the bar reaches fills at its limit price.

Nautilus runs the exchange *before* it hands the bar to the strategy, so a stop
that fires intrabar leaves the account flat by the time ``on_bar`` runs, and if the
entry condition still holds that same bar's close, the strategy enters again on
that bar. That is the two-orders-on-the-exit-bar behaviour of the reference, not
vectorbt's "next bar" and not raptorbt's "wait for a new crossing". The one thing
that still differs is how the re-entry is sized, which is why the workload is
``documented`` for this engine (see ``workloads.py`` and ``diagnose`` below).
"""
from __future__ import annotations

import importlib.metadata as md
from decimal import Decimal
from typing import Any, Callable, Dict, List

import data as data_mod
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.indicators import (
    ExponentialMovingAverage,
    RelativeStrengthIndex,
    SimpleMovingAverage,
)
from nautilus_trader.indicators.averages import MovingAverageType
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.data import BarType, BookOrder
from nautilus_trader.model.enums import AccountType, BookType, OmsType, OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from workloads import CAPITAL, WORKLOADS

NAME = "nautilus_trader"

VENUE = Venue("BENCH")
QUOTE = USDT
PRICE_PRECISION = 9
SIZE_PRECISION = 9

# Depth the fill model offers at the close. The largest order any workload places
# is full equity at the lowest price a 10M-bar random walk reaches (~20,000
# units), so a million is "unlimited" with room to spare, as in Nautilus's own
# BestPriceFillModel.
UNLIMITED = 1_000_000


def probe() -> Dict[str, Any]:
    # From the installed metadata, as for raptorbt: the version a benchmark
    # reports should be the one pip resolved, not a constant in an __init__.
    try:
        version = md.version("nautilus_trader")
    except Exception:
        import nautilus_trader

        version = getattr(nautilus_trader, "__version__", "unknown")
    return {"engine": NAME, "version": version}


# --------------------------------------------------------------------------- #
# The venue's view of the market
# --------------------------------------------------------------------------- #
class _BenchFillModel(FillModel):
    """Unlimited depth at the close for market orders, standard logic otherwise.

    ``get_orderbook_for_fill_simulation`` is called by the matching engine for
    every fill. Returning an ``OrderBook`` restricts the fill to that book's
    liquidity; returning ``None`` leaves it on the venue's standard logic, which is
    what stop and limit orders get here: their fill price is the trigger or limit
    price the bar crossed, and that is the behaviour being compared.
    """

    def __init__(self, slippage_bps: float = 0.0) -> None:
        super().__init__()
        self._shift = float(slippage_bps) / 10_000.0

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid, best_ask):
        if order.order_type != OrderType.MARKET or best_bid is None or best_ask is None:
            return None
        if self._shift:
            bid = instrument.make_price(best_bid.as_double() * (1.0 - self._shift))
            ask = instrument.make_price(best_ask.as_double() * (1.0 + self._shift))
        else:
            bid, ask = best_bid, best_ask
        depth = Quantity(UNLIMITED, instrument.size_precision)
        book = OrderBook(instrument.id, BookType.L2_MBP)
        book.add(BookOrder(OrderSide.BUY, bid, depth, 1), 0, 0)
        book.add(BookOrder(OrderSide.SELL, ask, depth, 2), 0, 0)
        return book


def _instrument(symbol: str, fee_bps: float) -> CurrencyPair:
    fee = Decimal(repr(fee_bps)) / Decimal(10_000)
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol(symbol), VENUE),
        raw_symbol=Symbol(symbol),
        base_currency=BTC,
        quote_currency=QUOTE,
        price_precision=PRICE_PRECISION,
        size_precision=SIZE_PRECISION,
        price_increment=Price(10 ** -PRICE_PRECISION, PRICE_PRECISION),
        size_increment=Quantity(10 ** -SIZE_PRECISION, SIZE_PRECISION),
        ts_event=0,
        ts_init=0,
        maker_fee=fee,
        taker_fee=fee,
        # No margin requirement: the account is leverage 1 and the strategy never
        # asks for more than its cash, so there is nothing for the risk engine to
        # veto and nothing for it to round.
        margin_init=Decimal(0),
        margin_maint=Decimal(0),
    )


# --------------------------------------------------------------------------- #
# The strategy
# --------------------------------------------------------------------------- #
class _Leg:
    """Indicators and state of one instrument."""

    __slots__ = ("instrument", "bar_type", "fast", "slow", "rsi", "long")

    def __init__(self, instrument, bar_type, p, kind):
        self.instrument = instrument
        self.bar_type = bar_type
        if kind == "ema_rsi":
            self.fast = ExponentialMovingAverage(p["fast"])
            self.slow = ExponentialMovingAverage(p["slow"])
            self.rsi = RelativeStrengthIndex(p["rsi_period"], MovingAverageType.WILDER)
        else:
            self.fast = SimpleMovingAverage(p["fast"])
            self.slow = SimpleMovingAverage(p["slow"])
            self.rsi = None
        self.long = False

    def reset(self):
        self.fast.reset()
        self.slow.reset()
        if self.rsi is not None:
            self.rsi.reset()
        self.long = False


class _StratConfig(StrategyConfig, frozen=True):
    pass


class _Bench(Strategy):
    def __init__(self, legs: List[_Leg], p: Dict[str, Any], kind: str) -> None:
        super().__init__(_StratConfig())
        self._legs = {leg.bar_type: leg for leg in legs}
        self._by_instrument = {leg.instrument.id: leg for leg in legs}
        self._kind = kind
        self._units = p.get("units")
        self._alloc = p.get("alloc", 1.0)
        self._sl = p["sl_pct"] / 100.0 if "sl_pct" in p else None
        self._tp = p["tp_pct"] / 100.0 if "tp_pct" in p else None
        self._rsi_lo = p.get("rsi_lo", 30.0) / 100.0
        self._rsi_hi = p.get("rsi_hi", 70.0) / 100.0
        self.fills = 0
        self.round_trips = 0
        self.fees = 0.0
        # Only the untimed diagnosis switches this on.
        self.journal = None

    # -- lifecycle ---------------------------------------------------------
    def on_start(self):
        for leg in self._legs.values():
            self.register_indicator_for_bars(leg.bar_type, leg.fast)
            self.register_indicator_for_bars(leg.bar_type, leg.slow)
            if leg.rsi is not None:
                self.register_indicator_for_bars(leg.bar_type, leg.rsi)
            self.subscribe_bars(leg.bar_type)

    def on_reset(self):
        for leg in self._legs.values():
            leg.reset()
        self.fills = 0
        self.round_trips = 0
        self.fees = 0.0

    # -- the strategy ------------------------------------------------------
    def on_bar(self, bar):
        leg = self._legs[bar.bar_type]
        if self._kind == "ema_rsi":
            level = (
                leg.rsi.initialized
                and leg.fast.value > leg.slow.value
                and self._rsi_lo < leg.rsi.value < self._rsi_hi
            )
        else:
            level = leg.slow.initialized and leg.fast.value > leg.slow.value

        if level:
            if not leg.long:
                self._enter(leg, bar)
        elif leg.long:
            self._exit(leg)

    def _enter(self, leg, bar):
        instrument = leg.instrument
        close = bar.close.as_double()
        if self._units is not None:
            quantity = instrument.make_qty(self._units)
        else:
            cash = self.portfolio.account(VENUE).balance_total(QUOTE).as_double()
            quantity = instrument.make_qty(self._alloc * cash / close, round_down=True)

        if self._sl is not None:
            orders = self.order_factory.bracket(
                instrument.id,
                OrderSide.BUY,
                quantity,
                sl_trigger_price=instrument.make_price(close * (1.0 - self._sl)),
                tp_price=instrument.make_price(close * (1.0 + self._tp)),
            )
            self.submit_order_list(orders)
        else:
            self.submit_order(self.order_factory.market(instrument.id, OrderSide.BUY, quantity))
        leg.long = True

    def _exit(self, leg):
        if self._sl is not None:
            # The bracket's stop and target are still resting when the signal
            # turns off, and they have to go with the position.
            self.cancel_all_orders(leg.instrument.id)
        self.close_all_positions(leg.instrument.id)
        leg.long = False

    # -- accounting --------------------------------------------------------
    def on_order_filled(self, event):
        self.fills += 1
        self.fees += event.commission.as_double()
        bought = event.order_side == OrderSide.BUY
        self._by_instrument[event.instrument_id].long = bought
        if self.journal is not None:
            self.journal.append((event.ts_event, bought, float(event.last_qty)))

    def on_position_closed(self, event):
        # A round trip is a position that went flat. Counting sell fills instead
        # would count a stop that the venue filled in two pieces twice.
        self.round_trips += 1


def _bar_type(symbol: str) -> BarType:
    return BarType.from_str("{}.{}-1-MINUTE-LAST-EXTERNAL".format(symbol, VENUE.value))


def _frames(key: str, df) -> Dict[str, Any]:
    """Symbol -> frame. The universe is the reference's: same generator, same seed."""
    assets = int(WORKLOADS[key].params.get("assets", 1))
    if assets > 1:
        return {"A%d" % sid: frame for sid, frame in data_mod.make_universe(len(df), assets).items()}
    return {"BENCH": df}


def _build(key: str, df):
    p = WORKLOADS[key].params
    kind = "ema_rsi" if key == "ema_rsi_fees" else "sma"
    fee_bps = float(p.get("fee_bps", 0.0))

    legs, bars = [], []
    for symbol, frame in _frames(key, df).items():
        instrument = _instrument(symbol, fee_bps)
        bar_type = _bar_type(symbol)
        bars.append((instrument, BarDataWrangler(bar_type, instrument).process(
            frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]])))
        legs.append(_Leg(instrument, bar_type, p, kind))

    engine = BacktestEngine(config=BacktestEngineConfig(
        # Errors only: at INFO every order event is formatted and written, which
        # is a cost of the log and not of the engine, and no benchmark of a
        # simulator should be timing its own console.
        logging=LoggingConfig(log_level="ERROR"),
    ))
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=QUOTE,
        default_leverage=Decimal(1),
        starting_balances=[Money(CAPITAL, QUOTE)],
        fill_model=_BenchFillModel(float(p.get("slippage_bps", 0.0))),
    )
    for instrument, leg_bars in bars:
        engine.add_instrument(instrument)
        engine.add_data(leg_bars)

    strategy = _Bench(legs, p, kind)
    engine.add_strategy(strategy)
    return engine, strategy


def _final_equity(engine) -> float:
    """Cash plus the open position marked at the last price, in the quote currency."""
    portfolio = engine.portfolio
    cash = portfolio.account(VENUE).balance_total(QUOTE).as_double()
    unrealized = portfolio.unrealized_pnls(VENUE).get(QUOTE)
    return cash + (unrealized.as_double() if unrealized is not None else 0.0)


def prepare(key: str, df, workdir: str | None = None) -> Callable[[], Dict[str, Any]]:
    """Untimed setup; returns the closure the harness times."""
    p = WORKLOADS[key].params
    wants_metrics = bool(p.get("metrics"))
    engine, strategy = _build(key, df)

    first: List[tuple] = []

    def run() -> Dict[str, Any]:
        engine.reset()
        engine.run()
        final_equity = _final_equity(engine)
        # The gate reads the first call only, so every replay after it is held to
        # the first one here. Both are deterministic, and a difference means the
        # reset left state behind: fail loudly rather than time another run.
        outcome = (final_equity, strategy.round_trips, strategy.fills)
        if not first:
            first.append(outcome)
        elif outcome != first[0]:
            raise RuntimeError(
                "nautilus_trader replay drifted from the first run: {} against {}. "
                "engine.reset() left state behind, so the timed runs would not be "
                "the run the parity gate checked.".format(outcome, first[0]))
        out = {
            "total_return": final_equity / CAPITAL - 1.0,
            "final_equity": final_equity,
            "round_trips": strategy.round_trips,
            "fills": strategy.fills,
            "total_fees": strategy.fees,
        }
        if wants_metrics:
            stats = engine.get_result().stats_returns
            out.update({
                # Nautilus has no drawdown statistic. Left absent rather than
                # rebuilt from an equity curve the harness would have to sample
                # bar by bar: this column shows what the engine hands a user.
                "max_drawdown": None,
                "sharpe": stats.get("Sharpe Ratio (252 days)"),
                "sortino": stats.get("Sortino Ratio (252 days)"),
                "volatility": stats.get("Returns Volatility (252 days)"),
            })
        return out

    return run


def diagnose(key: str, df, workdir: str | None = None) -> Dict[str, Any]:
    """Untimed measurement of how far the bracket divergence goes on this side.

    Entries and exits are the same trades at the same prices as the reference's.
    What differs is the quantity of a re-entry on the exit bar, so that is what is
    counted: how many of them there are, and how far each one's quantity sits from
    the position that was just closed on the same bar (which is the quantity the
    reference re-enters with, because it sizes from the equity at the close).
    """
    note = WORKLOADS[key].notes.get(NAME)
    if note is None or note.status != "documented":
        return {}
    engine, strategy = _build(key, df)
    strategy.journal = []
    engine.reset()
    engine.run()

    sold: Dict[int, float] = {}
    bought: Dict[int, float] = {}
    for ts, is_buy, quantity in strategy.journal:
        book = bought if is_buy else sold
        book[ts] = book.get(ts, 0.0) + quantity
    gaps = sorted(abs(quantity - sold[ts]) / sold[ts] for ts, quantity in bought.items()
                  if ts in sold and sold[ts] > 0.0)
    return {
        "round_trips": strategy.round_trips,
        "final_equity": _final_equity(engine),
        "reentries_on_exit_bar": len(gaps),
        "reentry_quantity_gap_median": gaps[len(gaps) // 2] if gaps else 0.0,
        "reentry_quantity_gap_max": gaps[-1] if gaps else 0.0,
    }


def sweep_runner(df, fast_vals, slow_vals) -> Callable[[], Dict[tuple, float]]:
    """A parameter grid the only way Nautilus spells one: a loop of backtests.

    There is no fan-out entry point. The engine is built once, with the data in
    it, and each cell swaps in a new pair of moving averages, resets the engine
    and replays the same bars: the repeat-run path the engine documents, and the
    cheapest one, since it pays for the instrument, the venue and the data once
    rather than once per cell. Nothing is shared between cells beyond that: each
    cell recomputes its own averages inside the event loop, because that is where
    a Nautilus indicator lives.
    """
    engine, strategy = _build("sma_cross", df)
    leg = next(iter(strategy._legs.values()))

    def call() -> Dict[tuple, float]:
        out = {}
        for fast in fast_vals:
            for slow in slow_vals:
                leg.fast = SimpleMovingAverage(fast)
                leg.slow = SimpleMovingAverage(slow)
                engine.reset()
                engine.run()
                out[(fast, slow)] = _final_equity(engine) / CAPITAL - 1.0
        return out

    return call
