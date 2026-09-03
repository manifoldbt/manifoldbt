"""Portfolio builder for multi-strategy backtesting.

Example::

    portfolio = (
        bt.Portfolio()
        .strategy(trend_strategy, weight=0.4)
        .strategy(mr_strategy, weight=0.3)
        .strategy(arb_strategy, weight=0.3)
        .max_drawdown(pct=20.0)
        .max_gross_exposure(pct=150.0)
        .rebalance_periodic(every_n_bars=30)
    )

    result = bt.run_portfolio(portfolio, config, store)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from manifoldbt.strategy import Strategy


class Portfolio:
    """Fluent builder for multi-strategy portfolio definitions."""

    def __init__(self) -> None:
        self._strategies: List[Dict[str, Any]] = []
        self._risk_rules: List[Dict[str, Any]] = []
        self._rebalance: Dict[str, Any] = {"type": "None"}

    def strategy(self, strategy: Strategy, weight: float = 1.0) -> "Portfolio":
        """Add a strategy with its capital allocation weight.

        Each strategy runs as an independent backtest on ``initial_capital *
        weight``; the portfolio equity is the sum of the legs. What that means
        for per-strategy settings:

        * ``stop_loss`` / ``take_profit`` / ``trailing_stop`` set on the
          strategy are honored, exactly as in :func:`run`. A stop that must
          apply to one direction only is written by splitting the legs into
          two strategies and setting the stop on one of them.
        * ``max_position_pct`` clamps on the LEG's equity, not the
          portfolio's: a leg at weight 0.5 with ``max_position_pct=1.0`` can
          never hold more than half the portfolio.
        * With the default ``FractionOfEquity`` sizing each leg compounds on
          its own equity. For the portfolio to equal the sum of the legs run
          separately (or one strategy trading both legs), use
          ``position_sizing_mode="FractionOfInitialCapital"``.

        Args:
            strategy: A Strategy instance.
            weight: Fraction of total capital (0.0 to 1.0). Weights must sum
                to at most 1.0.
        """
        self._strategies.append({
            "name": strategy.name,
            "strategy_json": strategy.to_json(),
            "weight": weight,
        })
        return self

    # -- Risk rules -----------------------------------------------------------

    def max_drawdown(self, pct: float) -> "Portfolio":
        """Kill all positions if portfolio drawdown exceeds threshold.

        Args:
            pct: Maximum drawdown percentage (e.g. 20.0 = -20%).
        """
        self._risk_rules.append({
            "type": "MaxDrawdown",
            "threshold_pct": pct,
        })
        return self

    def strategy_kill_switch(self, strategy: str, max_loss_pct: float) -> "Portfolio":
        """Kill a specific strategy if its P&L drops below threshold.

        Args:
            strategy: Strategy name.
            max_loss_pct: Maximum loss percentage (e.g. 10.0 = -10%).
        """
        self._risk_rules.append({
            "type": "StrategyKillSwitch",
            "strategy": strategy,
            "max_loss_pct": max_loss_pct,
        })
        return self

    def max_gross_exposure(self, pct: float) -> "Portfolio":
        """Cap total gross exposure as fraction of equity.

        Args:
            pct: Maximum gross exposure percentage (e.g. 150.0 = 1.5x leverage).
        """
        self._risk_rules.append({
            "type": "MaxGrossExposure",
            "max_pct": pct,
        })
        return self

    def max_net_exposure(self, pct: float) -> "Portfolio":
        """Cap total net exposure as fraction of equity.

        Args:
            pct: Maximum net exposure percentage (e.g. 50.0 = 50% net long/short).
        """
        self._risk_rules.append({
            "type": "MaxNetExposure",
            "max_pct": pct,
        })
        return self

    # -- Rebalancing ----------------------------------------------------------

    def rebalance_periodic(self, every_n_bars: int) -> "Portfolio":
        """Rebalance allocations back to target weights every N bars.

        Args:
            every_n_bars: Rebalance interval in bars.
        """
        self._rebalance = {
            "type": "Periodic",
            "every_n_bars": every_n_bars,
        }
        return self

    def rebalance_threshold(self, drift_pct: float) -> "Portfolio":
        """Rebalance when any strategy's weight drifts > threshold from target.

        Args:
            drift_pct: Maximum drift percentage before rebalancing.
        """
        self._rebalance = {
            "type": "Threshold",
            "drift_pct": drift_pct,
        }
        return self

    def no_rebalance(self) -> "Portfolio":
        """Never rebalance — allocations drift with P&L."""
        self._rebalance = {"type": "None"}
        return self

    # -- Serialization --------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to JSON for the Rust engine."""
        return json.dumps({
            "strategies": self._strategies,
            "risk_rules": self._risk_rules,
            "rebalance": self._rebalance,
        })

    def __repr__(self) -> str:
        strats = ", ".join(
            f"{s['name']}({s['weight']:.0%})" for s in self._strategies
        )
        return f"Portfolio([{strats}])"
