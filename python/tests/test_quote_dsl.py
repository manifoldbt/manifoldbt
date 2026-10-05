"""``Strategy.quote``: a strategy that quotes, written in the DSL and run by
``mbt.run`` on the order-level simulation.

The venue's rules for quotes are pinned by the engine's own tests (hold,
withdraw, keep, reprice, the maker of example 30 against its native port). This file checks the Python side: the builder and its JSON,
the refusals by name, and that a run through ``mbt.run`` gives exactly what
``bt.sim.run`` gives the same maker written as a function, with the same
columns.
"""
import json
import math

import pytest

import manifoldbt as bt
from manifoldbt import _native, book
from manifoldbt.helpers import Interval
from manifoldbt.result import QuoteResult

MS = 1_000_000
CLIP = 0.01
MAX_INVENTORY = 0.05


def _layer_open():
    try:
        bt.sim.Market.from_ladders("X", 0.1, [(0, [(1.0, 1.0)], [(2.0, 1.0)])])
    except PermissionError:
        return False
    return True


needs_layer = pytest.mark.skipif(
    not _layer_open(),
    reason="order-level simulation locked (runs in the engine's own debug builds)",
)


# ---------------------------------------------------------------------------
# The maker of example 30, both ways
# ---------------------------------------------------------------------------


def dsl_maker():
    inventory = bt.position()
    clips = bt.round(inventory / CLIP)
    bids, asks = book.bid_levels(), book.ask_levels()
    book_ok = (bids > 0) & (asks > 0)
    bid = book.bid_price_at(bt.clip(clips, 0, bids - 1) + 1)
    ask = book.ask_price_at(bt.clip(-clips, 0, asks - 1) + 1)
    return (
        bt.Strategy.create("dsl_maker")
        .quote("buy", bt.when(book_ok, bid), CLIP, tif="GTX",
               enabled=inventory < MAX_INVENTORY - 1e-12)
        .quote("sell", bt.when(book_ok, ask), CLIP, tif="GTX",
               enabled=inventory > -MAX_INVENTORY + 1e-12)
    )


def py_maker(sim):
    """Example 30's `maker`, verbatim but for the symbol and the clock."""
    live_states = ("in_flight", "resting", "partially_filled")
    quotes = {"buy": None, "sell": None}
    while sim.elapse(Interval.millis(100)):
        bids, asks = sim.book("BTCUSDT").levels()
        if not bids or not asks:
            continue
        inventory = sim.position("BTCUSDT")
        clips = round(inventory / CLIP)
        bid_level = min(max(clips, 0), len(bids) - 1)
        ask_level = min(max(-clips, 0), len(asks) - 1)
        wanted = {
            "buy": bids[bid_level][0] if inventory < MAX_INVENTORY - 1e-12 else None,
            "sell": asks[ask_level][0] if inventory > -MAX_INVENTORY + 1e-12 else None,
        }
        for side, price in wanted.items():
            oid = quotes[side]
            order = sim.order(oid) if oid is not None else None
            live = order is not None and order.status in live_states
            if live and price is not None and math.isclose(order.price, price):
                continue
            if live and not order.cancel_pending:
                sim.cancel(order.id)
            quotes[side] = None
            if price is not None:
                quotes[side] = sim.post("BTCUSDT", side, price, CLIP, tif="GTX")


LATENCY = {"order": 5 * MS, "cancel": 5 * MS, "response": 5 * MS, "feed": 2 * MS}
QUEUE = {"depth_source": "book", "assumed_queue": 2.0}


def quote_config(**over):
    execution = dict(latency=LATENCY, fill_model={"queue": QUEUE})
    execution.update(over.pop("execution", {}))
    kw = dict(
        universe=[1],
        time_range_start=0,
        time_range_end=86_400_000_000_000,
        bar_interval=Interval.millis(100),
        initial_capital=1_000.0,
        fees=bt.FeeConfig(maker_fee_bps=0.0, taker_fee_bps=4.0),
        execution=bt.ExecutionConfig(**execution),
    )
    kw.update(over)
    return bt.BacktestConfig(**kw)


def sim_config():
    return bt.sim.Config(
        latency={"entry": 5 * MS, "cancel": 5 * MS, "response": 5 * MS, "feed": 2 * MS},
        queue=QUEUE, taker_fee_bps=4.0, initial_cash=1_000.0,
    )


def walk_market(seed=7, n=2_000):
    """A book that walks a tick at a time, thins, empties a side now and then,
    and prints on both sides: a day in miniature."""
    import random

    rng = random.Random(seed)
    mid, t, states, trades = 1_000_000, 0, [], []
    for _ in range(n):
        t += (1 + rng.randrange(60)) * MS
        mid += rng.choice((-1, 0, 1))
        bids = [((mid - 1 - k) / 10, 0.001 * (1 + rng.randrange(50))) for k in range(1 + rng.randrange(10))]
        asks = [((mid + 1 + k) / 10, 0.001 * (1 + rng.randrange(50))) for k in range(1 + rng.randrange(10))]
        r = rng.randrange(40)
        if r == 0:
            bids = []
        elif r == 1:
            asks = []
        states.append((t, bids, asks))
        for _ in range(rng.randrange(3)):
            ts = t + rng.randrange(5) * MS
            q = 0.002 * (1 + rng.randrange(50))
            if rng.random() < 0.5 and bids:
                trades.append((ts, bids[rng.randrange(min(2, len(bids)))][0], q, "sell"))
            elif asks:
                trades.append((ts, asks[rng.randrange(min(2, len(asks)))][0], q, "buy"))
    trades.sort(key=lambda p: p[0])
    return bt.sim.Market.from_ladders("BTCUSDT", 0.1, states, trades)


def run_on(market, strategy=None, config=None):
    strategy = strategy or dsl_maker()
    config = config or quote_config()
    raw, sim_raw = _native.run_quotes(strategy.to_json(), config.to_json(), None, [market])
    sim = bt.sim.SimResult(sim_raw, bt.sim.Config(initial_cash=config.initial_capital))
    return QuoteResult(raw, sim)


def _bits(x):
    return None if x is None else float(x).hex()


def _table(cols):
    return [tuple(_bits(v) if isinstance(v, float) else v for v in row)
            for row in zip(*cols.values())]


@needs_layer
def test_the_dsl_maker_gives_the_python_maker_s_journal_to_the_bit():
    market = walk_market()
    res = run_on(market)
    ref = bt.sim.run(py_maker, sim_config(), markets=[market])
    assert len(res.sim.raw.orders_columns()["order_id"]) > 50
    for name in ("orders_columns", "fills_columns", "events_columns"):
        a = getattr(res.sim.raw, name)()
        b = getattr(ref.raw, name)()
        assert list(a) == list(b), name
        assert _table(a) == _table(b), name
    ts_a, eq_a = res.sim.raw.equity_columns()
    ts_b, eq_b = ref.raw.equity_columns()
    assert ts_a == ts_b
    assert [e.hex() for e in eq_a] == [e.hex() for e in eq_b]
    assert (res.sim.cash.hex(), res.sim.fees.hex()) == (ref.cash.hex(), ref.fees.hex())


def _squared_maker(square):
    """The maker above, its level pushed back by the SQUARE of the inventory
    in clips: `**` inside a quote, read at each wake-up."""
    inventory = bt.position()
    clips = bt.round(inventory / CLIP)
    bids, asks = book.bid_levels(), book.ask_levels()
    book_ok = (bids > 0) & (asks > 0)
    depth = square(clips)
    bid = book.bid_price_at(bt.clip(depth, 0, bids - 1) + 1)
    ask = book.ask_price_at(bt.clip(depth, 0, asks - 1) + 1)
    return (
        bt.Strategy.create("squared_maker")
        .quote("buy", bt.when(book_ok, bid), CLIP, tif="GTX",
               enabled=inventory < MAX_INVENTORY - 1e-12)
        .quote("sell", bt.when(book_ok, ask), CLIP, tif="GTX",
               enabled=inventory > -MAX_INVENTORY + 1e-12)
    )


@needs_layer
def test_a_power_in_a_quote_runs_as_the_product_written_by_hand():
    powered = _squared_maker(lambda c: c ** 2)
    by_hand = _squared_maker(lambda c: c * c)
    assert powered.to_json() == by_hand.to_json()
    market = walk_market()
    a, b = run_on(market, powered), run_on(market, by_hand)
    assert len(a.sim.raw.orders_columns()["order_id"]) > 50
    for name in ("orders_columns", "fills_columns", "events_columns"):
        assert _table(getattr(a.sim.raw, name)()) == _table(getattr(b.sim.raw, name)()), name


@needs_layer
def test_the_result_is_a_result_with_the_venue_s_tables_beside_it():
    res = run_on(walk_market())
    ref = bt.sim.run(py_maker, sim_config(), markets=[walk_market()])
    pd = pytest.importorskip("pandas")
    for name in ("orders_df", "fills_df", "events_df"):
        a, b = getattr(res, name)("pandas"), getattr(ref, name)("pandas")
        assert list(a.columns) == list(b.columns), name
        pd.testing.assert_frame_equal(a, b)
    # The same Result as bt.run: one trade per fill, metrics, the activity.
    fills = res.fills_df("pandas")
    trades = res.trades_df("pandas")
    assert len(trades) == len(fills)
    assert list(trades["fill_price"]) == list(fills["price"])
    assert set(res.metrics) >= {"sharpe", "max_drawdown", "total_return", "trade_stats"}
    activity = res.order_activity
    assert activity["orders_posted"] == len(res.orders_df("pandas"))
    assert {"stale_fills", "overlapping_live_orders"} <= set(activity)
    assert res.fill_marks["anchor"] == "fill"
    # Equity at one second: the wake-ups are 100 ms apart.
    eq = res.equity_df("pandas")
    steps = eq["timestamp"].diff().dropna().dt.total_seconds()
    assert steps.median() == pytest.approx(1.0)
    assert eq["equity"].iloc[-1] == pytest.approx(ref.metrics["final_equity"])


# ---------------------------------------------------------------------------
# The builder and its JSON
# ---------------------------------------------------------------------------


def test_a_strategy_without_quotes_serialises_as_it_always_did():
    s = bt.Strategy.create("s").signal("x", bt.col("close") > 1).size(bt.col("x"))
    assert "quotes" not in s.to_json_dict()
    assert "quotes" not in json.loads(s.to_json())


def test_quote_writes_its_fields_and_collects_its_params():
    s = bt.Strategy.create("q").quote(
        "Buy", book.bid_price_at(1) - bt.param("edge", default=0.1), 2, tif="gtc",
        enabled=bt.position() < 1,
    )
    doc = s.to_json_dict()
    (q,) = doc["quotes"]
    assert (q["side"], q["tif"]) == ("buy", "GTC")
    assert q["size"] == {"Literal": {"Float64": 2.0}}
    assert "enabled" in q
    assert "edge" in doc["parameters"]
    s2 = bt.Strategy.create("q").quote("sell", 100.0, 1.0)
    assert "enabled" not in s2.to_json_dict()["quotes"][0]
    assert s2.to_json_dict()["quotes"][0]["tif"] == "GTX", "post-only by default"


def test_a_cooldown_is_written_in_nanoseconds_and_only_when_there_is_one():
    def q(**kw):
        return bt.Strategy.create("q").quote("buy", 100.0, 1.0, **kw)

    assert "cooldown_ns" not in q().to_json_dict()["quotes"][0]
    # Zero is no pause: the JSON of the quote without one, byte for byte.
    assert q(cooldown=0).to_json() == q().to_json()
    assert q(cooldown=Interval.millis(0)).to_json() == q().to_json()
    assert q(cooldown=Interval.millis(500)).to_json_dict()["quotes"][0]["cooldown_ns"] == 500 * MS
    assert q(cooldown=Interval.seconds(2)).to_json_dict()["quotes"][0]["cooldown_ns"] == 2_000 * MS
    assert q(cooldown=250_000).to_json_dict()["quotes"][0]["cooldown_ns"] == 250_000
    with pytest.raises(ValueError, match="cooldown of quote 1 must be a duration of zero or more"):
        q(cooldown=-1)
    for bad in (True, 0.5, "500ms"):
        with pytest.raises(TypeError, match="cooldown of quote 1 takes a duration"):
            q(cooldown=bad)
    with pytest.raises(TypeError, match="simulation clock"):
        q(cooldown=Interval.trades())


def test_size_and_quotes_refuse_each_other():
    with pytest.raises(ValueError, match="sizes each quote itself"):
        bt.Strategy.create("a").size(bt.lit(1.0)).quote("buy", 100.0, 1.0)
    with pytest.raises(ValueError, match="sizes each quote itself"):
        bt.Strategy.create("a").quote("buy", 100.0, 1.0).size(bt.lit(1.0))


def test_a_side_or_a_time_in_force_is_checked_where_it_is_written():
    with pytest.raises(ValueError, match='side must be "buy" or "sell"'):
        bt.Strategy.create("a").quote("long", 100.0, 1.0)
    with pytest.raises(ValueError, match="tif must be"):
        bt.Strategy.create("a").quote("buy", 100.0, 1.0, tif="GTD")


# ---------------------------------------------------------------------------
# Refused by name
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_store(tmp_path):
    return bt.DataStore(
        data_root=str(tmp_path / "data"), metadata_db=str(tmp_path / "meta.sqlite")
    )


def _refused(strategy, config, store):
    with pytest.raises(Exception) as err:
        bt.run(strategy, config, store)
    return str(err.value)


@needs_layer
@pytest.mark.parametrize(
    "over, words",
    [
        (dict(universe=[1, 2]), "runs on one symbol"),
        (dict(execution=dict(signal_delay=1)), "execution.signal_delay"),
        (dict(slippage={"FixedBps": {"bps": 1.0}}), "slippage"),
        (dict(execution=dict(position_sizing_mode="Units")), "execution.position_sizing_mode"),
        (dict(execution=dict(tick_size=0.1)), "execution.tick_size"),
        (
            dict(execution=dict(fill_model={
                "queue": QUEUE,
                "adverse": {"model": "snipe", "threshold_ticks": 2.0,
                             "extra_cancel_latency": Interval.millis(5)},
            })),
            "execution.fill_model.adverse (snipe)",
        ),
        (dict(bar_interval=Interval.trades()), "Interval.trades()"),
        (dict(resample_to=Interval.seconds(1)), "resample_to"),
    ],
)
def test_a_setting_a_quoting_strategy_does_not_read_is_refused_by_name(over, words, empty_store):
    m = _refused(dsl_maker(), quote_config(**over), empty_store)
    assert words in m, m


@needs_layer
def test_brackets_are_refused_by_name(empty_store):
    s = dsl_maker().stop_loss(pct=2.0)
    assert "sets stop_loss" in _refused(s, quote_config(), empty_store)


def test_every_bar_path_refuses_a_millisecond_clock(empty_store):
    s = bt.Strategy.create("s").signal("x", bt.col("close") > 1).size(bt.col("x"))
    m = _refused(s, quote_config(execution=dict(latency=None, fill_model=None)), empty_store)
    assert "bar_interval=Interval.millis(100) is the wake-up clock" in m, m


def test_the_bar_engine_refuses_response_and_feed(empty_store):
    s = bt.Strategy.create("s").signal("x", bt.col("close") > 1).size(bt.col("x"))
    cfg = quote_config(
        bar_interval=Interval.minutes(1),
        execution=dict(latency={"order": MS, "feed": MS}, fill_model=None),
    )
    m = _refused(s, cfg, empty_store)
    assert "execution.latency.feed is read only by a strategy that quotes" in m, m


def test_the_bar_compiler_refuses_a_strategy_that_quotes_naming_where_it_runs():
    with pytest.raises(Exception, match=r"quotes \(Strategy.quote\): it runs on the "
                                        r"order-level simulation, through mbt.run, run_sweep"):
        _native.compile_strategy_json(dsl_maker().to_json())


@pytest.mark.skipif(_layer_open(), reason="the order-level layer is open here")
def test_a_strategy_that_quotes_is_behind_the_same_door_as_bt_sim(empty_store):
    with pytest.raises(PermissionError, match="Order-level simulation"):
        bt.run(dsl_maker(), quote_config(), empty_store)


# ---------------------------------------------------------------------------
# Prices on the tick grid, and what a quoting run counts
# ---------------------------------------------------------------------------


def _offered_at_116994_6(trades):
    """5 BTC offered at 116994.6, tick 0.1 (``repro/grille_1.py``)."""
    return bt.sim.Market.from_ladders(
        "BTCUSDT", tick_size=0.1,
        states=[(1 * MS, [(116994.3, 1.0)], [(116994.4, 1.0), (116994.6, 5.0), (116994.7, 1.0)])],
        trades=trades,
    )


@needs_layer
def test_a_sim_price_a_few_ulps_under_the_grid_waits_in_its_queue():
    computed = 116994.4 + 0.2
    assert computed != 116994.6
    seen = []
    for price in (computed, 116994.6):
        def strat(sim, price=price):
            sim.elapse(1_500_000 - sim.now())
            oid = sim.post("BTCUSDT", "sell", price, 0.005, tif="GTX")
            seen.append(sim.order(oid).price)
            sim.elapse(5 * MS)

        market = _offered_at_116994_6([(3 * MS, 116994.6, 0.01, "buy")])
        res = bt.sim.run(strat, markets=[market])
        assert res.fills_df(backend="dict")["qty"] == [], price
        assert res.levels_snapped == 0
        assert res.fill_fragility["queue"]["traverse_fills"] == 0
    assert seen == [116994.6, 116994.6]


@needs_layer
def test_a_sim_price_between_two_ticks_goes_to_the_passive_side():
    def strat(sim):
        sim.elapse(1_500_000 - sim.now())
        buy = sim.post("BTCUSDT", "buy", 116994.37, 0.005, tif="GTX")
        sell = sim.post("BTCUSDT", "sell", 116994.43, 0.005, tif="GTX")
        assert (sim.order(buy).price, sim.order(sell).price) == (116994.3, 116994.5)
        sim.elapse(MS)

    res = bt.sim.run(strat, markets=[_offered_at_116994_6([])])
    assert res.levels_snapped == 2


def _ask_plus(ticks, rounded):
    price = book.ask_price_at(1) + ticks * 0.1
    if rounded:
        price = price.round_to(0.1)
    return bt.Strategy.create("grille").quote("sell", price, 0.005, tif="GTX")


@needs_layer
def test_a_quote_computed_off_the_grid_runs_as_the_quote_rounded_onto_it():
    market = walk_market(seed=3, n=1_500)
    a = run_on(market, _ask_plus(2, rounded=False))
    b = run_on(market, _ask_plus(2, rounded=True))
    assert len(a.sim.raw.orders_columns()["order_id"]) > 20
    for name in ("orders_columns", "fills_columns", "events_columns"):
        assert _table(getattr(a.sim.raw, name)()) == _table(getattr(b.sim.raw, name)()), name
    assert a.order_activity == b.order_activity
    assert "levels_snapped" not in a.order_activity


@needs_layer
def test_a_quoting_run_reports_its_queue_and_its_refusals():
    res = run_on(walk_market())
    frag = res.fill_fragility
    assert frag is not None, "a quoting run has fill_fragility, as a bar run under a queue"
    q = frag["queue"]
    assert set(q) >= {
        "queue_decided_fills", "traverse_fills", "partial_fills",
        "book_unknown_at_post", "fills_from_book_cross",
    }
    channels = res.fills_df(backend="dict")["channel"]
    assert q["queue_decided_fills"] == channels.count("queue")
    assert q["fills_from_book_cross"] == channels.count("book_cross")
    assert q["traverse_fills"] == channels.count("traverse") + channels.count("book_cross")
    assert frag["maker_fills"] == len(channels)
    assert frag["touch_only_fills"] == 0
    statuses = res.orders_df(backend="dict")["status"]
    activity = res.order_activity
    assert activity.get("post_only_rejected", 0) == statuses.count("rejected") > 0
    # Per quote: this maker never posts a quote again before the cancellation
    # of its previous order came back, so every overlap is a requote.
    assert activity["overlapping_live_orders"] == activity["requotes"]
    # The execution columns of a sweep read them.
    cols = bt.sweep_columns([res], ["service_rate", "queue_decided_fills"])
    assert cols["queue_decided_fills"][0] == q["queue_decided_fills"]
    assert cols["service_rate"][0] == pytest.approx(
        frag["maker_fills"] / activity["orders_posted"]
    )


@needs_layer
def test_a_quote_deeper_than_the_levels_read_is_refused_as_such():
    deep = bt.Strategy.create("profond").quote("buy", book.bid_price_at(1) - 3.0, 0.01)
    config = quote_config(execution={"fill_model": {"queue": {"depth_source": "book"}}})
    with pytest.raises(Exception, match="levels read from the store") as err:
        run_on(walk_market(), deep, config)
    assert "ingest the book" not in str(err.value)
    res = run_on(walk_market(), deep, quote_config())
    assert res.fill_fragility["queue"]["book_unknown_at_post"] > 0


# ---------------------------------------------------------------------------
# The ages and the order's own price: a maker that pulls and pauses, a taker
# that holds for a bounded time, each against its bt.sim twin
# ---------------------------------------------------------------------------

SEUIL, MAX_INV, EPS = 0.03, 0.025, 1e-12
PAUSE_S = 0.5
WAKE = Interval.millis(50)


def dsl_file_courte(cooldown=False):
    """The maker of the HFT harness's ``file_courte``: a clip at each best price,
    pulled (never moved) when the queue ahead passes SEUIL, when the best price
    leaves it or when the inventory is full; a side pulled pauses PAUSE_S,
    written with ``last_cancel_age`` or, ``cooldown=True``, as the quote's
    ``cooldown``."""
    from manifoldbt.indicators import abs_val

    pos = bt.position()
    bb, ba = book.bid_price_at(1), book.ask_price_at(1)
    book_ok = (bb > 0) & (ba > bb)
    strategy = bt.Strategy.create("file_courte")
    for side, qs, own, inv_ok in (("buy", "bid", bb, pos < MAX_INV - EPS),
                                  ("sell", "ask", ba, pos > -MAX_INV + EPS)):
        has = bt.live_qty(qs) > 0
        pull = (~inv_ok | ~(own > 0) | (abs_val(bt.order_price(qs) - own) > 1e-6)
                | (bt.queue_ahead(qs) > SEUIL))
        wanted = (has & ~pull) | (~has & book_ok & inv_ok)
        if cooldown:
            strategy = strategy.quote(
                side, bt.when(has, bt.order_price(qs), own), CLIP, tif="GTX",
                enabled=wanted, cooldown=Interval.millis(int(PAUSE_S * 1000)))
        else:
            paused = bt.last_cancel_age(qs) < PAUSE_S
            strategy = strategy.quote(
                side, bt.when(has, bt.order_price(qs), own), CLIP, tif="GTX",
                enabled=~paused & wanted)
    return strategy


def py_file_courte(sim):
    """``run_mbt_sim`` of the harness, on the wake-up grid of the quotes."""
    live_states = ("in_flight", "resting", "partially_filled")
    oid = {"buy": None, "sell": None}
    pulled_at = {"buy": None, "sell": None}
    while sim.elapse(WAKE):
        now = sim.now()
        bids, asks = sim.book("BTCUSDT").levels(1)
        bb = bids[0][0] if bids else None
        ba = asks[0][0] if asks else None
        pos = sim.position("BTCUSDT")
        for side, px, inv_ok in (("buy", bb, pos < MAX_INV - EPS),
                                 ("sell", ba, pos > -MAX_INV + EPS)):
            if oid[side] is not None:
                o = sim.order(oid[side])
                if o.status in live_states:
                    if o.cancel_pending:
                        continue
                    qa = o.queue_ahead
                    if (not inv_ok or px is None or abs(o.price - px) > 1e-6
                            or (qa is not None and qa > SEUIL)):
                        sim.cancel(o.id)
                        pulled_at[side] = now
                    continue
                oid[side] = None
            if px is None or bb is None or ba is None or ba <= bb or not inv_ok:
                continue
            if pulled_at[side] is not None and now - pulled_at[side] < PAUSE_S * 1e9:
                continue
            oid[side] = sim.post("BTCUSDT", side, px, CLIP, tif="GTX")


HOLD_S, POS_MAX, SIG = 2.0, 0.03, 2


def dsl_preneur():
    """A taker that leaves at the touch after HOLD_S, its holding time read
    from the state: no timer order."""
    pos = bt.position()
    nb, na = book.bid_levels(), book.ask_levels()
    sig = nb - na
    book_ok = (nb > 0) & (na > 0)
    old = bt.position_age() >= HOLD_S - 1e-6
    out_long = book_ok & (pos > 1e-9) & ((sig < -SIG) | old)
    out_short = book_ok & (pos < -1e-9) & ((sig > SIG) | old)
    free = ~out_long & ~out_short
    bid1, ask1 = book.bid_price_at(1), book.ask_price_at(1)
    return (
        bt.Strategy.create("preneur")
        .quote("sell", bid1, pos, tif="IOC", enabled=out_long)
        .quote("buy", ask1, -pos, tif="IOC", enabled=out_short)
        .quote("buy", ask1, CLIP, tif="IOC",
               enabled=book_ok & (sig > SIG) & (pos < POS_MAX - 1e-9) & free)
        .quote("sell", bid1, CLIP, tif="IOC",
               enabled=book_ok & (sig < -SIG) & (pos > -POS_MAX + 1e-9) & free)
    )


def py_preneur(sim):
    """The same taker for bt.sim, its holding time counted as the harness
    counts it: from the first wake-up that sees the position."""
    t_pos = None
    while sim.elapse(Interval.millis(100)):
        now = sim.now()
        pos = sim.position("BTCUSDT")
        if abs(pos) > 1e-9:
            t_pos = now if t_pos is None else t_pos
        else:
            t_pos = None
        bids, asks = sim.book("BTCUSDT").levels()
        if not bids or not asks:
            continue
        sig = len(bids) - len(asks)
        old = t_pos is not None and now - t_pos >= HOLD_S * 1e9
        if pos > 1e-9 and (sig < -SIG or old):
            sim.post("BTCUSDT", "sell", bids[0][0], pos, tif="IOC")
        elif pos < -1e-9 and (sig > SIG or old):
            sim.post("BTCUSDT", "buy", asks[0][0], -pos, tif="IOC")
        elif sig > SIG and pos < POS_MAX - 1e-9:
            sim.post("BTCUSDT", "buy", asks[0][0], CLIP, tif="IOC")
        elif sig < -SIG and pos > -POS_MAX + 1e-9:
            sim.post("BTCUSDT", "sell", bids[0][0], CLIP, tif="IOC")


def _same_journal(res, ref):
    for name in ("orders_columns", "fills_columns", "events_columns"):
        a = getattr(res.sim.raw, name)()
        b = getattr(ref.raw, name)()
        assert list(a) == list(b), name
        assert _table(a) == _table(b), name
    ts_a, eq_a = res.sim.raw.equity_columns()
    ts_b, eq_b = ref.raw.equity_columns()
    assert ts_a == ts_b
    assert [e.hex() for e in eq_a] == [e.hex() for e in eq_b]


@needs_layer
@pytest.mark.parametrize("cooldown", [False, True])
@pytest.mark.parametrize("seed", [3, 11])
def test_a_maker_that_pulls_and_pauses_gives_its_bt_sim_twin_s_journal(seed, cooldown):
    market = walk_market(seed, 3_000)
    res = run_on(market, dsl_file_courte(cooldown), quote_config(bar_interval=WAKE))
    ref = bt.sim.run(py_file_courte, sim_config(), markets=[market])
    events = res.sim.raw.events_columns()["event"]
    assert events.count("cancel_sent") > 100
    assert res.order_activity["requotes"] == 0
    _same_journal(res, ref)


@needs_layer
@pytest.mark.parametrize("seed", [5, 17])
def test_a_taker_that_leaves_on_the_position_age_gives_its_bt_sim_twin_s_journal(seed):
    market = walk_market(seed, 3_000)
    res = run_on(market, dsl_preneur())
    ref = bt.sim.run(py_preneur, sim_config(), markets=[market])
    assert len(res.sim.raw.orders_columns()["order_id"]) > 100
    _same_journal(res, ref)
