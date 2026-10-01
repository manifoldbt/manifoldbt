"""Rich progress display for data ingestion."""
from __future__ import annotations

import time


def make_progress_display(symbol: str, provider: str):
    """Create a rich progress display for a single symbol.

    Returns (display, callback).
    """
    try:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TextColumn,
            TimeRemainingColumn,
        )
        from rich.console import Console

        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.fields[status]}"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        task_id = progress.add_task(
            f"[bold blue]{symbol} ({provider})", total=10000, status="connecting..."
        )
        progress.start()

        def callback(phase: str, fetched: int, pct: float, msg: str):
            if phase == "done":
                progress.update(task_id, completed=10000, status=f"done - {fetched:,} bars")
                progress.stop()
            elif phase == "store":
                progress.update(task_id, completed=9900, status="writing...")
            else:
                progress.update(
                    task_id,
                    completed=max(1, int(pct * 9800)),
                    status=f"{fetched:,} bars",
                )

        return progress, callback

    except ImportError:
        return _make_fallback(symbol, provider)


def make_multi_progress(symbols: list[tuple[str, int]], provider: str):
    """Create a rich progress display for multiple symbols.

    All symbols are shown upfront: current in blue, pending in dim grey.
    Returns (display, dict of {symbol: callback}).
    """
    try:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TextColumn,
            TimeRemainingColumn,
        )
        from rich.console import Console

        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.fields[status]}"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )

        task_ids = {}
        for i, (sym, _sid) in enumerate(symbols):
            if i == 0:
                desc = f"[bold blue]{sym} ({provider})"
                status = "connecting..."
            else:
                desc = f"[dim]{sym} ({provider})[/dim]"
                status = "waiting"
            task_ids[sym] = progress.add_task(desc, total=10000, status=status)

        progress.start()

        def make_callback(sym: str, idx: int):
            tid = task_ids[sym]

            def callback(phase: str, fetched: int, pct: float, msg: str):
                if phase == "done":
                    progress.update(
                        tid,
                        description=f"[green]{sym} ({provider})[/green]",
                        completed=10000,
                        status=f"{fetched:,} bars",
                    )
                    # Activate next symbol if any.
                    next_idx = idx + 1
                    if next_idx < len(symbols):
                        next_sym = symbols[next_idx][0]
                        next_tid = task_ids[next_sym]
                        progress.update(
                            next_tid,
                            description=f"[bold blue]{next_sym} ({provider})",
                            status="connecting...",
                        )
                elif phase == "store":
                    progress.update(tid, completed=9900, status="writing...")
                else:
                    progress.update(
                        tid,
                        completed=max(1, int(pct * 9800)),
                        status=f"{fetched:,} bars",
                    )

            return callback

        callbacks = {sym: make_callback(sym, i) for i, (sym, _sid) in enumerate(symbols)}
        return progress, callbacks

    except ImportError:
        return _make_multi_fallback(symbols, provider)


def _make_fallback(symbol: str, provider: str):
    """Plain print fallback when rich is not installed."""
    t0 = time.perf_counter()
    last_pct = [-1]

    def callback(phase: str, fetched: int, pct: float, msg: str):
        elapsed = time.perf_counter() - t0
        current = int(pct * 100)
        if phase == "done":
            print(f"\r  {symbol} ({provider}): done - {fetched:,} bars [{elapsed:.1f}s]      ")
        elif current > last_pct[0] + 4 or phase == "store":
            last_pct[0] = current
            label = "writing..." if phase == "store" else f"{fetched:,} bars"
            print(f"\r  {symbol} ({provider}): {current}% {label} [{elapsed:.1f}s]", end="", flush=True)

    return None, callback


def _make_multi_fallback(symbols: list[tuple[str, int]], provider: str):
    """Plain print fallback for multiple symbols."""
    callbacks = {}
    for sym, _sid in symbols:
        _, cb = _make_fallback(sym, provider)
        callbacks[sym] = cb
    return None, callbacks
