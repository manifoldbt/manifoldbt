"""Native frameless windows for charts (pywebview), matplotlib-style.

``show=True`` queues a chart as a borderless native window; a single
``manifoldbt.plot.show()`` (or the automatic one at interpreter exit) opens
ALL queued windows together, so you can have the equity in one window and
the return distribution in another, side by side.

Each window runs in its own child process with a dedicated WebView2 profile:
WebView2 windows sharing one process share one UI thread (several heavy
charts freeze it), and separate processes sharing the default user-data
folder collide on startup. One process + one profile per window avoids both,
keeps every window frameless and responsive, and lets show() be called again
later. show() blocks until all windows are closed (like ``plt.show()``).

Without pywebview each queued chart opens in a browser tab instead. Install
the window backend with ``pip install manifoldbt[window]``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import webbrowser
from html import escape
from pathlib import Path
from typing import List, Tuple

_SHELL = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
 html,body{{margin:0;height:100%;background:#0c0c0f;overflow:hidden}}
 #chart{{height:100vh}} .plotly-graph-div{{width:100%!important;height:100%!important}}
 #drag{{position:fixed;top:0;left:0;right:0;height:26px;z-index:5}}
 #close{{position:fixed;top:8px;right:10px;z-index:6;width:26px;height:26px;
   display:flex;align-items:center;justify-content:center;font-family:Arial,sans-serif;
   font-size:17px;color:#8a8a8a;cursor:pointer;border-radius:5px;
   background:rgba(17,17,22,0.45);transition:all .12s}}
 #close:hover{{background:#ef4444;color:#fff}}
</style></head><body>
 <div id="drag" class="pywebview-drag-region"></div>
 <div id="close" onclick="pywebview.api.close()" title="Close (Alt+F4)">&times;</div>
 <div id="chart">{div}</div>
 <script>
  function fit(){{var g=document.querySelector('.plotly-graph-div');
    if(g&&window.Plotly)Plotly.relayout(g,{{width:window.innerWidth,height:window.innerHeight}});}}
  window.addEventListener('resize',fit);
  window.addEventListener('load',function(){{fit();requestAnimationFrame(fit);
    setTimeout(fit,120);setTimeout(fit,400);}});
 </script>
</body></html>"""

# One frameless window; runs inside a dedicated child process.
_CHILD = """
import ctypes, os, sys
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("manifoldbt.plot")
except Exception:
    pass
import webview
title, url, x, y, w, h, icon = sys.argv[1:8]
holder = {}
class Api:
    def close(self):
        win = holder.get("w")
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
holder["w"] = webview.create_window(
    title, url, frameless=True, easy_drag=False, js_api=Api(),
    width=int(w), height=int(h), x=int(x), y=int(y),
    background_color="#0c0c0f",
)
kwargs = dict(storage_path=os.environ.get("MANIFOLDBT_WV_STORAGE"), private_mode=False)
try:
    webview.start(icon=icon, **kwargs) if icon and os.path.exists(icon) else webview.start(**kwargs)
except TypeError:
    webview.start(**kwargs)  # some backends reject the icon kwarg
"""

# Charts registered by show=True, waiting for the next show() / atexit call.
_pending: "List[Tuple[str, str, Tuple[int, int]]]" = []


def _fig_div(fig) -> str:
    """Responsive chart div with plotly.js inlined (instant + offline)."""
    fig.update_layout(width=None, height=None, autosize=True)
    return fig.to_html(
        full_html=False, include_plotlyjs=True,
        default_width="100%", default_height="100%",
        config={"displayModeBar": False, "responsive": True},
    )


def _write_tmp(html: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                      mode="w", encoding="utf-8")
    tmp.write(html)
    tmp.close()
    return Path(tmp.name).resolve()


def queue_window(fig, *, title: str = "Chart",
                 size: Tuple[int, int] = (1280, 720)) -> None:
    """Register *fig* to be shown as a native window on the next show()."""
    _pending.append((_fig_div(fig), title, size))


# Backward-compatible alias (was the immediate opener).
open_in_window = queue_window


def show() -> None:
    """Open every chart queued by ``show=True``, each in its own frameless window.

    Blocks until all windows are closed (like ``matplotlib.pyplot.show``).
    A no-op if nothing is queued; can be called again after more charts are
    queued. Falls back to browser tabs when pywebview is not installed.
    """
    if not _pending:
        return
    pending = list(_pending)
    _pending.clear()

    try:
        import webview  # noqa: F401 — only to detect the backend
    except ImportError:
        for div, title, _ in pending:
            _open_browser(div, title)
        return

    icon = Path(__file__).parent / "_assets" / "manifoldbt.ico"
    procs = []
    for i, (div, title, size) in enumerate(pending):
        html = _SHELL.format(title=escape(title), div=div)
        path = _write_tmp(html)
        env = dict(os.environ)
        # Dedicated WebView2 profile: concurrent windows sharing the default
        # user-data folder fail to start (window class/profile collision).
        storage = tempfile.mkdtemp(prefix="manifoldbt_win_")
        env["MANIFOLDBT_WV_STORAGE"] = storage
        env["WEBVIEW2_USER_DATA_FOLDER"] = storage
        procs.append(subprocess.Popen(
            [sys.executable, "-c", _CHILD, title, path.as_uri(),
             str(80 + i * 60), str(80 + i * 60),
             str(size[0]), str(size[1]), str(icon)],
            env=env,
        ))
    for p in procs:
        try:
            p.wait()
        except KeyboardInterrupt:
            for q in procs:
                if q.poll() is None:
                    q.terminate()
            break


def _open_browser(div: str, title: str) -> None:
    # No pywebview: plain browser tab. The OS/browser chrome provides closing.
    html = _SHELL.format(title=escape(title),
                         div=div).replace('pywebview.api.close()', 'window.close()')
    webbrowser.open(_write_tmp(html).as_uri())


def _atexit_show() -> None:
    # Scripts "just work": show whatever is still queued when the process exits.
    show()


import atexit  # noqa: E402
atexit.register(_atexit_show)
