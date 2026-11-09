"""What the library says on its own: the import banner, the update notice and
the summary of Pro features at exit.

None of it is a result, so it goes to stderr and leaves stdout to the
caller's own output, which scripts pipe and capture into files. Colour only
when stderr is a terminal and ``NO_COLOR`` is unset (https://no-color.org);
``MANIFOLDBT_QUIET=1`` silences the banner and the update notice.
"""
import os
import sys


def _truthy(name: str) -> bool:
    """Unset, empty, ``0`` and ``false`` mean no."""
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false")


def quiet() -> bool:
    return _truthy("MANIFOLDBT_QUIET")


def _colour() -> bool:
    if "NO_COLOR" in os.environ:
        return False
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def paint(text: str, code: str) -> str:
    """``text`` in the ANSI colour ``code`` when colour is on, as is otherwise."""
    return f"\033[{code}m{text}\033[0m" if _colour() else text


def say(line: str = "") -> None:
    """Print one line on stderr; never fails the caller."""
    try:
        print(line, file=sys.stderr)
    except Exception:
        pass
