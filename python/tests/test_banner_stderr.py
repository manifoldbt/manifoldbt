"""The import banner goes to stderr, so a script's stdout carries only what the
script prints; it is coloured only on a terminal, never under NO_COLOR, and
MANIFOLDBT_QUIET=1 silences it.

The banner runs once, at import: each case reads a fresh process.
"""
import os
import subprocess
import sys


def _import(**env):
    full = dict(os.environ)
    full.pop("MANIFOLDBT_QUIET", None)
    full.pop("NO_COLOR", None)
    full["MANIFOLDBT_NO_UPDATE_CHECK"] = "1"
    full.update(env)
    return subprocess.run(
        [sys.executable, "-c", "import manifoldbt"],
        capture_output=True, text=True, check=True, env=full,
    )


def test_stdout_stays_empty_and_the_banner_is_on_stderr():
    proc = _import()
    assert proc.stdout == "", proc.stdout
    assert "manifoldbt v" in proc.stderr, proc.stderr


def test_no_colour_when_stderr_is_not_a_terminal():
    # capture_output gives the child a pipe, not a terminal.
    assert "\033[" not in _import().stderr


def test_quiet_silences_the_banner():
    proc = _import(MANIFOLDBT_QUIET="1")
    assert proc.stdout == "" and "manifoldbt v" not in proc.stderr, proc.stderr


def test_quiet_zero_reads_as_not_quiet():
    assert "manifoldbt v" in _import(MANIFOLDBT_QUIET="0").stderr


def test_no_color_is_honoured_on_a_terminal(monkeypatch):
    from manifoldbt import _console

    class _Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr(_console.sys, "stderr", _Terminal())
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert _console.paint("x", "36") == "\033[36mx\033[0m"
    monkeypatch.setenv("NO_COLOR", "1")
    assert _console.paint("x", "36") == "x"
