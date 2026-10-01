"""Workload definitions: the numbers every engine reads, in one place.

Nothing here is engine-specific. Each adapter (``engine_mbt``, ``engine_vbt``,
``engine_rbt``) reads the same constants, so a workload cannot drift between two
sides by someone editing one file and forgetting another.

What a workload *does* carry per engine is a note: the places where an engine is
known in advance to disagree, or is unable to run the workload at all. Those are
written down here, next to the parameters they apply to, rather than in the
adapters, so a reader can see the whole map of "who runs what, and where they
part ways" without opening three files.

Sizing policy, and why it changes when fees are on
--------------------------------------------------
With ``FractionOfEquity`` sizing and a non-zero fee, engines size a position
differently: manifoldbt charges the fee on top of a full-equity notional,
vectorbt reserves it out of cash first. Both are defensible product decisions,
and comparing them would compare *policy*, not speed or correctness. The fee
workload therefore sizes in fixed units, which isolates the fee arithmetic
itself. This mirrors the choice already made in the cross-engine parity suite
shipped with the library. It is also the reason raptorbt sits that workload out:
it has no fixed-quantity sizing to offer (see its note below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

CAPITAL = 100_000.0

# Bar interval of the generated data. Both engines are told the same thing:
# manifoldbt through `Interval.minutes(1)`, vectorbt through `freq="1min"`
# (annualisation only; it does not touch the simulation).
FREQ = "1min"


@dataclass(frozen=True)
class Note:
    """What is known in advance about one engine on one workload.

    ``documented``
        The engine is known to disagree with the reference, and ``why`` says on
        what. It still gets timed, but in the annex, never in the headline
        table, and always with the reason printed next to the number.
    ``unsupported``
        The engine cannot express this workload at all. It is not run, and the
        report says so rather than leaving a blank a reader would read as a
        loss. An unsupported entry is a finding about the engine's API, so
        ``why`` has to be specific enough to be checked.
    """

    status: str
    why: str


@dataclass(frozen=True)
class Workload:
    key: str
    title: str
    why: str
    params: Dict[str, Any] = field(default_factory=dict)
    # Longest series this workload is valid on, or None for no ceiling. A
    # workload can stop being a comparison before it stops running: see the fee
    # workload, whose account this exists to keep alive.
    max_bars: int | None = None
    # Engine name -> Note. An engine with no entry here is expected to agree
    # with the reference down to float-reordering noise, and a disagreement is
    # a failure that withholds the timing.
    notes: Dict[str, Note] = field(default_factory=dict)


WORKLOADS: Dict[str, Workload] = {
    w.key: w
    for w in (
        Workload(
            key="sma_cross",
            title="SMA 30/150 crossover, long-only, no cost",
            why="The canonical baseline. Unambiguous indicator, no fee policy, "
                "no stop semantics: if the engines disagree here, nothing else "
                "in the suite is worth reading.",
            params=dict(fast=30, slow=150, alloc=1.0),
        ),
        Workload(
            key="ema_rsi_fees",
            title="EMA 12/26 crossover + RSI(14) filter, 5 bps taker fee",
            why="A realistic signal stack with a real cost model, sized in fixed "
                "units so the fee arithmetic is comparable across engines. The "
                "unit count is small relative to capital on purpose: at 1-minute "
                "resolution this strategy turns over often enough that a larger "
                "size would spend the whole account on fees, and comparing two "
                "engines on a wiped-out account compares rounding noise.",
            params=dict(fast=12, slow=26, rsi_period=14, rsi_lo=30.0, rsi_hi=70.0,
                        units=5.0, fee_bps=5.0),
            # The account has to survive, or the engines are being compared on
            # rounding noise around zero. Measured: -15% of capital at 1M bars,
            # -74% at 5M, and exactly -100% at 10M, where fees reach 99,611 of
            # the 100,000 started with. Past that the two disagree by thousands
            # of round-trips while both sit at zero equity, which is a fact
            # about a bankrupt strategy and not about either engine. The ceiling
            # is set where the comparison still means something.
            max_bars=1_000_000,
            notes={
                "raptorbt": Note(
                    "unsupported",
                    "Two blockers, either one sufficient. Sizing: raptorbt has no "
                    "fixed-quantity mode. `position_sizes` is a fraction of equity "
                    "(measured: 0.5 buys exactly half the equity of the bar before "
                    "the entry), `lot_size` rounds a computed size down to a "
                    "multiple, and `alloted_capital` fixes the notional, not the "
                    "quantity. Reproducing `units=5` would mean feeding a fraction "
                    "derived from an equity curve that does not exist until the run "
                    "is over. Indicator: `raptorbt.ema` seeds on a simple mean of "
                    "the first `period` bars and emits from bar `period-1`, while "
                    "manifoldbt seeds on the first observation and emits from bar 0 "
                    "(measured: 11 leading NaN for span 12, and ema[11] equal to "
                    "sma(12)[11] to the last bit). The two are the same recursion "
                    "with a different warmup, so the signal differs early and the "
                    "round-trip count with it. Its `sma` and `rsi` do match, which "
                    "is why the other three workloads run.",
                ),
            },
        ),
        Workload(
            key="sma_cross_metrics",
            title="SMA 30/150 crossover, with a performance summary",
            why="The same simulation as `sma_cross`, but both engines are asked "
                "for what a user actually reads: max drawdown, Sharpe, Sortino "
                "and volatility alongside the return. manifoldbt computes them "
                "inside run() whether you ask or not; vectorbt defers the equity "
                "curve until a risk metric needs it, and every one of them pays "
                "for materialising it. This is a scope difference, not a trick: "
                "`sma_cross` above is the same work without the summary, and the "
                "two are reported side by side so the reader can see what the "
                "summary costs each engine.",
            params=dict(fast=30, slow=150, alloc=1.0, metrics=True),
        ),
        Workload(
            key="sma_cross_costs",
            title="SMA 30/150 with a 5 bps fee and 2 bps of slippage",
            why="The same signal as the headline workload, run against a cost "
                "model. Costs were the easiest objection to make of a benchmark "
                "that had them on one workload out of four, and the answer is "
                "not to sprinkle them everywhere: measured, a fee or a slippage "
                "on FractionOfEquity sizing makes the engines disagree by 1e-4 "
                "of capital against a 1e-9 tolerance, because they resolve the "
                "same policy differently. In fixed units the arithmetic is "
                "comparable and both costs land exactly, so this is where they "
                "belong.",
            params=dict(fast=30, slow=150, units=5.0, fee_bps=5.0, slippage_bps=2.0),
            notes={
                "raptorbt": Note(
                    "unsupported",
                    "Same blocker as the other fixed-quantity workload: raptorbt "
                    "has no units sizing, and a cost model on top of a fraction "
                    "of equity compares policy rather than arithmetic. See "
                    "`ema_rsi_fees` for the measurements behind that.",
                ),
            },
        ),
        Workload(
            key="multi_asset",
            title="Five assets in one book, SMA 30/150, 5 bps fee",
            why="A single-asset benchmark measures a loop; a portfolio measures "
                "the thing people actually run. It is also where the two designs "
                "differ in kind rather than in speed: manifoldbt walks a universe "
                "against one shared book, while vectorbt broadcasts a column per "
                "asset and has to be told to share cash at all. The five series "
                "are independent rather than correlated, so a sizing bug cannot "
                "cancel itself out across the book.",
            params=dict(fast=30, slow=150, units=2.0, fee_bps=5.0, assets=5),
            # Five columns of a materialised simulation, not one. vectorbt adds
            # 1.55 GB on a 10M-bar single asset, so five of them would ask a
            # 16 GB runner for around 8 GB on top of 2.4 GB of generated frames.
            # The ceiling is set where the point is certain to be measuring the
            # engine rather than the swap file.
            max_bars=1_000_000,
            notes={
                "raptorbt": Note(
                    "unsupported",
                    "No multi-instrument portfolio on the entry point this "
                    "harness drives: `run_single_backtest` is one instrument, and "
                    "`run_multi_backtest` broadcasts over strategies rather than "
                    "over assets. 0.9.0 does add `run_portfolio_backtest`, but "
                    "its allocation model is a different one and would need its "
                    "own parity work before any timing from it could be "
                    "published. The units blocker applies here too.",
                ),
            },
        ),
        Workload(
            key="bracket_sl_tp",
            title="SMA 10/50 entry with a 15 bps stop / 30 bps target bracket",
            why="Brackets are where two engines most easily disagree: the stop "
                "level, the fill on the triggering bar, and whether a re-entry "
                "is allowed on the bar after an exit.",
            params=dict(fast=10, slow=50, alloc=1.0, sl_pct=0.15, tp_pct=0.30),
            notes={
                "vectorbt": Note(
                    "documented",
                    "Re-entry on the exit bar. When a bracket fires intrabar and "
                    "the entry condition still holds at that bar's close, "
                    "manifoldbt books two orders on that bar (the stop or target "
                    "exit, then a fresh entry at the close); vectorbt processes one "
                    "order per bar and re-enters on the next bar instead. Neither "
                    "is wrong, and on controlled bars the bracket fills themselves "
                    "match exactly (see the cross-engine parity suite shipped with "
                    "the library). The harness counts the affected round-trips so "
                    "the size of the divergence is measured, not asserted.",
                ),
                "raptorbt": Note(
                    "documented",
                    "Same fork in the road, taken further: raptorbt does not "
                    "re-arm at all. Once a bracket closes a position, the entry "
                    "level being still true is not enough to open another one; it "
                    "waits for the level to go false and true again. So on the same "
                    "bars the three engines book a different number of round-trips "
                    "from the same signal, manifoldbt re-entering on the exit bar, "
                    "vectorbt on the next bar, raptorbt not until the next crossing. "
                    "The count of affected round-trips is measured rather than "
                    "asserted, and it is the same population for both challengers.",
                ),
            },
        ),
    )
}

def expectation(key: str, engine: str) -> str:
    """What this engine is expected to do on this workload, before it runs."""
    note = WORKLOADS[key].notes.get(engine)
    return note.status if note else "exact"


def why(key: str, engine: str) -> str:
    note = WORKLOADS[key].notes.get(engine)
    return note.why if note else ""


def supported(key: str, engine: str) -> bool:
    return expectation(key, engine) != "unsupported"


def unsupported_by(key: str) -> Dict[str, str]:
    """Engines that sit this workload out, and the reason each one gives."""
    return {
        name: note.why
        for name, note in WORKLOADS[key].notes.items()
        if note.status == "unsupported"
    }


DEFAULT_KEYS = list(WORKLOADS)

# Two workloads that the report compares directly against each other, so they
# must be measured inside ONE interleaved loop rather than in two blocks minutes
# apart. Comparing medians across blocks is exactly the mistake the interleaving
# exists to prevent: absolute timings drift between blocks, and the drift once
# produced a table claiming the version doing MORE work was the faster one.
SCOPE_PAIR = ("sma_cross", "sma_cross_metrics")
