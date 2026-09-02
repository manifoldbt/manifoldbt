"""Make bare ``symbol_ref()`` strategies runnable without orchestrator lore.

The engine evaluates a ``SymbolRef`` only through its multi-symbol
orchestrator, and that path engages solely when the config universe is the
**dict** form (``{"binance": ["BTCUSDT", "ETHUSDT"]}``), every reference is
provider-**qualified** (``symbol_ref("binance:BTCUSDT", ...)``) and no
``SymbolRef`` sits inside the position-sizing rule (sizing is evaluated
per-symbol, where the op is refused).

Nobody should have to know that to write a pairs strategy. This module
detects the natural form -- ``symbol_ref("PAIRB", "close")`` with
``universe=["PAIRA"]`` -- and rewrites it to the orchestrator-ready form:
references qualified with each symbol's provider (its exchange, lower-cased,
as recorded at import), referenced symbols added to a dict universe, and any
sizing-level reference hoisted into a synthetic signal.

A strategy that already uses qualified references or a dict universe is left
untouched: the explicit form keeps working, and is never second-guessed.
"""
from __future__ import annotations

import copy
import json
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def collect_symbol_refs(node: Any, out: set) -> None:
    """Gather every ``SymbolRef`` symbol string in a StrategyDef tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "SymbolRef" and isinstance(value, list) and value:
                out.add(str(value[0]))
            collect_symbol_refs(value, out)
    elif isinstance(node, list):
        for item in node:
            collect_symbol_refs(item, out)


def _rewrite_symbol_refs(node: Any, rename: Dict[str, str]) -> Any:
    """Deep copy with each SymbolRef symbol string remapped via ``rename``."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "SymbolRef" and isinstance(value, list) and value:
                symbol = str(value[0])
                out[key] = [rename.get(symbol, symbol)] + [
                    _rewrite_symbol_refs(v, rename) for v in value[1:]
                ]
            else:
                out[key] = _rewrite_symbol_refs(value, rename)
        return out
    if isinstance(node, list):
        return [_rewrite_symbol_refs(item, rename) for item in node]
    return node


def _lift_sizing_refs(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Hoist any SymbolRef out of ``position_sizing`` into a synthetic signal.

    Signals are evaluated by the orchestrator, where a SymbolRef resolves;
    position sizing is evaluated per-symbol, where the engine refuses it
    ("this op should not be evaluated per-symbol"). Identical references
    share one signal.
    """
    sizing = doc.get("position_sizing")
    if sizing is None:
        return doc
    signals = dict(doc.get("signals") or {})
    lifted: Dict[str, str] = {}
    counter = 0

    def hoist(node: Any) -> Any:
        nonlocal counter
        if isinstance(node, dict):
            if "SymbolRef" in node and isinstance(node["SymbolRef"], list):
                key = json.dumps(node["SymbolRef"], sort_keys=True)
                name = lifted.get(key)
                if name is None:
                    name = f"_xref_{counter}"
                    counter += 1
                    lifted[key] = name
                    signals[name] = node
                return {"Column": name}
            return {k: hoist(v) for k, v in node.items()}
        if isinstance(node, list):
            return [hoist(item) for item in node]
        return node

    new_sizing = hoist(sizing)
    if not lifted:
        return doc
    return {**doc, "position_sizing": new_sizing, "signals": signals}


def _symbol_rows(metadata_db: str) -> List[Tuple[int, str, str]]:
    """``(id, ticker, exchange)`` for every symbol in the metadata catalog."""
    con = sqlite3.connect(f"file:{metadata_db}?mode=ro", uri=True)
    try:
        return [
            (int(r[0]), str(r[1]), str(r[2] or ""))
            for r in con.execute("SELECT id, ticker, exchange FROM symbols")
        ]
    finally:
        con.close()


def _resolve_row(
    entry: Any, rows: List[Tuple[int, str, str]]
) -> Tuple[str, str]:
    """Resolve a ticker or id to ``(canonical_ticker, provider)``.

    Provider is the symbol's exchange lower-cased, which is the key the
    engine's dict-universe resolver matches (COLLATE NOCASE).
    """
    if isinstance(entry, int) and not isinstance(entry, bool):
        for sid, ticker, exchange in rows:
            if sid == int(entry):
                return ticker, (exchange.strip().lower() or "binance")
        raise ValueError(f"symbol id {entry} is not in the metadata store")
    name = str(entry)
    matches = [
        (ticker, exchange)
        for _, ticker, exchange in rows
        if ticker.upper() == name.upper()
    ]
    if not matches:
        raise ValueError(f"symbol '{name}' not found in the metadata store")
    if len(matches) > 1:
        raise ValueError(
            f"symbol '{name}' matches several store entries; qualify the "
            f"reference as 'provider:{name}'"
        )
    ticker, exchange = matches[0]
    return ticker, (exchange.strip().lower() or "binance")


def prepare_cross_asset(
    strategy_doc: Dict[str, Any], universe: Any, metadata_db: Optional[str]
) -> Optional[Tuple[Dict[str, Any], Dict[str, List[str]]]]:
    """Turn a bare cross-asset strategy into the orchestrator-ready form.

    Returns ``(rewritten_strategy_doc, dict_universe_or_None)`` when something
    had to change: bare references are provider-qualified and the dict universe
    rebuilt (``dict_universe`` set), and any SymbolRef inside position sizing is
    hoisted into a synthetic signal -- that one applies even to fully qualified
    strategies, because sizing is evaluated per-symbol where the op is refused.
    ``dict_universe`` is ``None`` when the caller's universe must be kept.
    Returns ``None`` when there is nothing to do.
    """
    refs: set = set()
    collect_symbol_refs(strategy_doc, refs)
    if not refs:
        return None

    # A SymbolRef inside position sizing is refused per-symbol even when it is
    # provider-qualified, so the hoist applies to EVERY cross-asset strategy,
    # not only the bare-name form.
    new_doc = _lift_sizing_refs(copy.deepcopy(strategy_doc))
    lifted = new_doc is not strategy_doc and new_doc != strategy_doc

    bare = sorted(r for r in refs if ":" not in r)
    dict_universe: Optional[Dict[str, List[str]]] = None
    if bare and not isinstance(universe, dict) and metadata_db is not None:
        rows = _symbol_rows(metadata_db)
        traded = [
            _resolve_row(entry, rows)
            for entry in (universe if isinstance(universe, (list, tuple)) else [universe])
        ]
        referenced = [_resolve_row(name, rows) for name in bare]

        by_provider: Dict[str, List[str]] = defaultdict(list)
        seen: set = set()
        for ticker, provider in traded + referenced:
            if (provider, ticker) not in seen:
                seen.add((provider, ticker))
                by_provider[provider].append(ticker)

        rename = {
            raw: f"{prov}:{tick}" for raw, (tick, prov) in zip(bare, referenced)
        }
        new_doc = _rewrite_symbol_refs(new_doc, rename)
        dict_universe = dict(by_provider)

    if dict_universe is None and not lifted:
        return None  # already orchestrator-ready -- do not second-guess
    return new_doc, dict_universe
