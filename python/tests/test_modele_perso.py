"""``fill_model={"custom": ...}`` and ``fill_model={"python": ...}``.

A service rule of the caller's own: an expression of the DSL evaluated once per
event for each resting order, or -- for research -- a Python callable. This file
covers the three halves a user meets:

* the SHAPES, which need no tape and no store: a malformed ``custom`` block is
  named as malformed, and a callable left where a config is serialised is
  refused rather than dropped;
* the REFUSALS a wrong rule earns from the engine: a rule without a queue, a
  rule reading a column that is not a fill-context field, a rule carrying an
  operator with memory;
* the COUNTERS and the sweep, which need a store holding a tape AND a book,
  since the whole point is deciding what a print gives an order behind a real
  queue. A tape enters a store only through ``bt.ingest_trades``, which fetches
  a venue's archive over the network, so that half skips by name unless
  ``MANIFOLDBT_TEST_TAPE_STORE`` points at a store that already holds one, as
  ``<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD>``.

The rule itself is pinned by the engine's own tests: the grammar and the
evaluator, and the loop that consumes it -- including the one claim this whole route stands on, that the
native queue rule REWRITTEN as an expression books exactly the same fills.
"""
import os

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import manifoldbt as bt  # noqa: E402
from manifoldbt.expr import col, lit, param, when  # noqa: E402
from manifoldbt.indicators import max_val, min_val, sma  # noqa: E402
from manifoldbt.exceptions import BacktesterError  # noqa: E402
from manifoldbt.helpers import Interval, Slippage  # noqa: E402

# A rule the caller wrote is one more model behind the shared interface, so its
# counters live where every other model's do: inside ``fill_fragility["queue"]``.
CUSTOM_KEYS = {
    "custom_decided_fills",
    "custom_rule_events",
    "custom_declined_events",
    "custom_truncated_answers",
}


def _custom(res):
    """The rule's four counters, or a clear failure saying it never ran."""
    q = (res.fill_fragility or {}).get("queue") or {}
    missing = CUSTOM_KEYS - set(q)
    assert not missing, f"the rule reported no {sorted(missing)}: {q}"
    return {k: q[k] for k in CUSTOM_KEYS}

QUEUE = {"depth_source": "book", "assumed_queue": 2.0}

# The native queue rule, written in the DSL: the volume ahead burns first and
# only the remainder reaches the order.
REGLE_FILE = min_val(
    max_val(col("print_qty") - col("queue_ahead"), lit(0.0)),
    col("order_size_left"),
)


def _store(tmp_path):
    n = 8
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-03-01", periods=n, freq="1min", tz="UTC"),
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1_000.0),
        }
    )
    root = tmp_path / "store"
    os.makedirs(root, exist_ok=True)
    store = bt.import_dataframe(
        df,
        symbol="TEST",
        symbol_id=1,
        interval="1m",
        data_root=os.path.join(root, "data"),
        metadata_db=os.path.join(root, "meta.sqlite"),
    )
    return store, int(df["timestamp"].iloc[-1].value)


def _config(last_ns, fill_model):
    return bt.BacktestConfig(
        universe=[1],
        time_range_start=0,
        time_range_end=last_ns + 60_000_000_000,
        bar_interval=Interval.minutes(1),
        initial_capital=10_000.0,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            execution_price="AtClose",
            allow_short=False,
            fill_model=fill_model,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
        warmup_bars=0,
    )


def _strategy(**params):
    s = (
        bt.Strategy.create("perso")
        .signal("sig", lit(1.0))
        .size(col("sig"))
        .limit_entry(price=99.5)
    )
    for name, default in params.items():
        s = s.param(name, default=default)
    return s


# ---------------------------------------------------------------------------
# The shape of the two configs
# ---------------------------------------------------------------------------


def test_the_rule_must_be_an_expression(tmp_path):
    store, last_ns = _store(tmp_path)
    cfg = _config(last_ns, {"queue": QUEUE, "custom": {"fill": 1.0}})
    with pytest.raises(TypeError, match="must be an expression"):
        bt.run(_strategy(), cfg, store)


def test_a_custom_block_without_a_rule_says_so(tmp_path):
    store, last_ns = _store(tmp_path)
    cfg = _config(last_ns, {"queue": QUEUE, "custom": {"override_traverse": True}})
    with pytest.raises(TypeError, match='needs a "fill"'):
        bt.run(_strategy(), cfg, store)


def test_a_typo_in_the_custom_block_is_not_a_default(tmp_path):
    store, last_ns = _store(tmp_path)
    cfg = _config(
        last_ns, {"queue": QUEUE, "custom": {"fill": lit(0.0), "override_traverses": True}}
    )
    with pytest.raises(TypeError, match="override_traverses"):
        bt.run(_strategy(), cfg, store)


def test_a_callable_that_is_not_callable_is_named(tmp_path):
    store, last_ns = _store(tmp_path)
    cfg = _config(last_ns, {"queue": QUEUE, "python": 3})
    with pytest.raises(TypeError, match="must be a callable"):
        bt.run(_strategy(), cfg, store)


def test_a_callable_cannot_be_serialised_into_a_sweep(tmp_path):
    """It is not part of a config, and dropping it would answer another question."""
    store, last_ns = _store(tmp_path)

    def rule(*_args):
        return 0.0

    cfg = _config(last_ns, {"queue": QUEUE, "python": rule})
    with pytest.raises(TypeError, match="not part of a config"):
        bt.run_sweep(_strategy(k=1.0), {"k": [1.0, 2.0]}, cfg, store)


# ---------------------------------------------------------------------------
# What the engine refuses, and why
# ---------------------------------------------------------------------------


def test_a_rule_without_a_queue_is_refused_by_name(tmp_path):
    """And BEFORE the tape is asked for: the rule has nothing to read from."""
    store, last_ns = _store(tmp_path)
    cfg = _config(last_ns, {"custom": {"fill": REGLE_FILE}})
    with pytest.raises(BacktesterError, match="fill_model.custom needs fill_model.queue"):
        bt.run(_strategy(), cfg, store)


def test_a_rule_reading_a_strategy_signal_is_refused_and_the_fields_are_listed(tmp_path):
    store, last_ns = _store(tmp_path)
    cfg = _config(last_ns, {"queue": QUEUE, "custom": {"fill": col("sig")}})
    with pytest.raises(BacktesterError) as exc:
        bt.run(_strategy(), cfg, store)
    why = str(exc.value)
    assert "'sig'" in why
    assert "queue_ahead" in why, "the refusal lists the fields a rule may read"
    assert "lookahead" in why


def test_a_rule_with_memory_is_refused_by_the_name_of_its_operator(tmp_path):
    store, last_ns = _store(tmp_path)
    cfg = _config(
        last_ns, {"queue": QUEUE, "custom": {"fill": sma(col("print_qty"), 5)}}
    )
    with pytest.raises(BacktesterError, match="memory"):
        bt.run(_strategy(), cfg, store)


def test_a_param_the_strategy_does_not_declare_is_refused_by_name(tmp_path):
    store, last_ns = _store(tmp_path)
    cfg = _config(last_ns, {"queue": QUEUE, "custom": {"fill": param("slope")}})
    with pytest.raises(BacktesterError, match="slope"):
        bt.run(_strategy(), cfg, store)


def test_a_rule_without_prints_is_refused_where_the_queue_is(tmp_path):
    """A bar has a high and a low, not an order of trades."""
    store, last_ns = _store(tmp_path)
    cfg = _config(last_ns, {"queue": QUEUE, "custom": {"fill": REGLE_FILE}})
    with pytest.raises(BacktesterError, match="needs the prints"):
        bt.run(_strategy(), cfg, store)


def test_a_rule_is_refused_on_the_lite_path(tmp_path):
    store, last_ns = _store(tmp_path)
    cfg = _config(last_ns, {"custom": {"fill": REGLE_FILE}})
    with pytest.raises(BacktesterError, match="fill_model.custom"):
        bt.run_sweep_lite(_strategy(k=1.0), {"k": [1.0, 2.0]}, cfg, store)


# ---------------------------------------------------------------------------
# End to end, against a store that really holds a tape and a book
# ---------------------------------------------------------------------------

_TAPE = os.environ.get("MANIFOLDBT_TEST_TAPE_STORE", "")
_needs_tape = pytest.mark.skipif(
    not _TAPE,
    reason="set MANIFOLDBT_TEST_TAPE_STORE=<root>|<meta.sqlite>|<symbol_id>|<YYYY-MM-DD> "
    "to run a service rule end to end (a tape and a book enter a store only "
    "through bt.ingest_trades / bt.ingest_book, which need the network), and "
    "hold the tick-layer key",
)


def _tape_store():
    root, meta, sid, day = _TAPE.split("|")
    store = bt.DataStore(
        data_root=root, metadata_db=meta, arrow_dir=os.path.join(root, "mega")
    )
    start = int(bt.date_to_ns(day))
    return store, int(sid), start, start + 86_400_000_000_000


def _tape_config(sid, start, end, fill_model, marks=False):
    return bt.BacktestConfig(
        universe=[sid],
        time_range_start=start,
        time_range_end=end,
        bar_interval=Interval.seconds(1),
        # DAILY, and on purpose: a wheel without the tick layer returns one row
        # a day whatever is asked, so a test that read an intraday position
        # would pass here and fail in the public CI. Fills carry their own
        # stamps.
        output_resolution=Interval.days(1),
        initial_capital=100_000.0,
        warmup_bars=120,
        execution=bt.ExecutionConfig(
            signal_delay=1,
            allow_short=False,
            position_sizing_mode="Units",
            fill_model={**fill_model, "fill_resolution": "ticks"},
            fill_marks=marks,
            tick_size=0.1,
        ),
        fees=bt.FeeConfig.zero(),
        slippage=Slippage.none(),
    )


def _maker(**params):
    s = (
        bt.Strategy.create("maker")
        .signal("sig", lit(1.0))
        .size(col("sig") * lit(0.01))
        .limit_entry(offset_bps=2.0, time_in_force={"GTB": 1})
        .take_profit(pct=0.02, side="long")
        .stop_loss(pct=0.06, side="long")
    )
    for name, default in params.items():
        s = s.param(name, default=default)
    return s


@_needs_tape
def test_the_native_rule_rewritten_in_the_dsl_books_the_same_fills():
    """The claim the whole route stands on, on a real day."""
    store, sid, start, end = _tape_store()
    natif = bt.run(_maker(), _tape_config(sid, start, end, {"queue": QUEUE}), store)
    perso = bt.run(
        _maker(),
        _tape_config(sid, start, end, {"queue": QUEUE, "custom": {"fill": REGLE_FILE}}),
        store,
    )
    pd.testing.assert_frame_equal(natif.trades_df(), perso.trades_df())
    # And it really went through the rule rather than falling back to the queue.
    custom = _custom(perso)
    assert custom["custom_decided_fills"] > 0
    assert (
        custom["custom_decided_fills"]
        == natif.fill_fragility["queue"]["queue_decided_fills"]
    ), "the rewrite decided exactly the fills the native rule decided"


@_needs_tape
def test_a_rule_reports_its_four_counters():
    store, sid, start, end = _tape_store()
    res = bt.run(
        _maker(),
        _tape_config(sid, start, end, {"queue": QUEUE, "custom": {"fill": REGLE_FILE}}),
        store,
    )
    custom = _custom(res)
    assert all(isinstance(v, int) and v >= 0 for v in custom.values())
    assert custom["custom_rule_events"] >= custom["custom_decided_fills"]
    assert (
        custom["custom_declined_events"] + custom["custom_decided_fills"]
        == custom["custom_rule_events"]
    )
    # Fills, never intraday positions: the count is the same on a wheel that
    # returns one output row a day.
    assert len(res.trades_df()) > 0


@_needs_tape
def test_two_runs_of_the_same_rule_are_identical():
    store, sid, start, end = _tape_store()
    cfg = _tape_config(
        sid,
        start,
        end,
        # A rule that draws on `u`: the one shape a run could differ on twice.
        {"queue": QUEUE, "custom": {"fill": when(col("u") < lit(0.5), REGLE_FILE, lit(0.0))}},
    )
    a = bt.run(_maker(), cfg, store)
    b = bt.run(_maker(), cfg, store)
    assert a.fill_fragility == b.fill_fragility
    pd.testing.assert_frame_equal(a.trades_df(), b.trades_df())


@_needs_tape
def test_a_param_inside_the_rule_is_a_sweep_axis():
    """Three combos, three different runs: the axis really moves.

    Distinct, not monotone. A rule that serves more ends order lives sooner,
    which changes how many quotes the strategy posts at all, so the totals are
    a feedback loop and not a curve. What a swept name must guarantee is that
    it is not a silent no-op -- which is exactly what ``_validate_swept_params``
    exists to catch and what this pins.
    """
    store, sid, start, end = _tape_store()
    regle = when(col("u") < param("p"), REGLE_FILE, lit(0.0))
    cfg = _tape_config(
        sid, start, end, {"queue": QUEUE, "custom": {"fill": regle, "override_traverse": True}}
    )
    sweep = bt.run_sweep(_maker(p=1.0), {"p": [0.25, 0.5, 1.0]}, cfg, store)
    assert len(sweep) == 3
    served = [_custom(r)["custom_decided_fills"] for r in sweep]
    assert len(set(served)) == 3, f"the axis ran the same backtest twice: {served}"
    fills = [len(r.trades_df()) for r in sweep]
    assert len(set(fills)) > 1, f"and it must move the fills too: {fills}"


@_needs_tape
def test_the_callable_is_called_with_the_seventeen_fields_and_fills_the_same():
    store, sid, start, end = _tape_store()
    seen = []

    def rule(*args):
        if not seen:
            seen.append(args)
        left = args[11] - args[0]  # print_qty - queue_ahead
        return min(left, args[1]) if left > 0.0 else 0.0

    dsl = bt.run(
        _maker(),
        _tape_config(sid, start, end, {"queue": QUEUE, "custom": {"fill": REGLE_FILE}}),
        store,
    )
    py = bt.run(
        _maker(), _tape_config(sid, start, end, {"queue": QUEUE, "python": rule}), store
    )
    assert seen, "the callable was never invoked"
    assert len(seen[0]) == 17, "seventeen positional fields, in the documented order"
    assert all(isinstance(v, float) for v in seen[0])
    pd.testing.assert_frame_equal(dsl.trades_df(), py.trades_df())


@_needs_tape
def test_a_callable_that_raises_stops_the_run_by_name():
    store, sid, start, end = _tape_store()

    def rule(*_args):
        raise ValueError("calibration not loaded")

    cfg = _tape_config(sid, start, end, {"queue": QUEUE, "python": rule})
    with pytest.raises(BacktesterError, match="calibration not loaded"):
        bt.run(_maker(), cfg, store)


@_needs_tape
def test_a_rule_answering_nan_stops_the_run_and_names_the_event():
    store, sid, start, end = _tape_store()
    cfg = _tape_config(
        sid, start, end, {"queue": QUEUE, "custom": {"fill": col("mid") / lit(0.0)}}
    )
    with pytest.raises(BacktesterError, match="returned NaN at the event stamped"):
        bt.run(_maker(), cfg, store)


@_needs_tape
def test_a_param_of_the_rule_and_an_execution_grid_sweep_together():
    """The two grids cross: one moves the rule, the other moves the config.

    ``param_grid`` carries a ``param()`` that lives INSIDE the rule;
    ``execution_grid`` carries a dotted config path that names the rule's own
    setting. Nothing in the engine knows they belong to the same object, which
    is the point: a rule is data, so it is swept like any other data.
    """
    store, sid, start, end = _tape_store()
    regle = when(col("u") < param("p"), REGLE_FILE, lit(0.0))
    cfg = _tape_config(sid, start, end, {"queue": QUEUE, "custom": {"fill": regle}})
    sweep = bt.run_sweep(
        _maker(p=1.0),
        {"p": [0.25, 1.0]},
        cfg,
        store,
        execution_grid={"fill_model.custom.override_traverse": [False, True]},
    )
    assert len(sweep) == 4, "two axes of two, crossed"
    served = sorted(_custom(r)["custom_rule_events"] for r in sweep)
    assert served[0] < served[-1], (
        "taking the traversals over shows the rule far more events, "
        f"got {served}"
    )
