"""Tell the user when a newer manifoldbt has been published on PyPI.

Same idea as pip's own "A new release of pip is available" notice, and the same
three rules, in order of importance:

1. It cannot slow an import down. The notice printed at import comes from the
   answer cached by a previous run; the HTTP request runs on a daemon thread and
   nothing ever waits on it.
2. It cannot fail the caller. Every path here swallows its own exception; a
   broken cache file, a captive-portal proxy or a PyPI outage must be invisible.
3. It is opt-out, via ``MANIFOLDBT_NO_UPDATE_CHECK=1``, and it stays quiet on CI
   runners where nobody reads the notice and nobody can act on it.

This is not telemetry and shares nothing with the engine's install ping: the
request is a plain GET of a public JSON document, carries no identifier, and its
body is thrown away except for one version string. ``DO_NOT_TRACK`` is therefore
*not* read here -- there is nothing to track -- while ``MANIFOLDBT_NO_TELEMETRY``
is, because someone who set it meant "this library does not touch the network on
its own", which covers this too.
"""
import atexit
import json
import os
import re
import sys
import threading
import time
from typing import Optional

PACKAGE = "manifoldbt"

_PYPI_URL = "https://pypi.org/pypi/{}/json".format(PACKAGE)

# One question a day per machine, like pip's own check but tighter: releases here
# are frequent enough that a week-old answer would usually be the wrong one.
_CHECK_INTERVAL_SECONDS = 24 * 60 * 60

# The request runs off the critical path, so this bound only limits how long a
# dead network keeps a daemon thread alive.
_TIMEOUT_SECONDS = 3.0

# Process state. `_announced` is what keeps the notice to one per process: the
# cached answer prints it at import, and the background thread must not repeat it
# at exit.
_started = False
_announced = False
_current = None  # type: Optional[str]
_pending = None  # type: Optional[str]


# ---------------------------------------------------------------------------
# Installed version
# ---------------------------------------------------------------------------
def installed_version() -> Optional[str]:
    """Version of the installed distribution, or None if there is none.

    None means "do not compare": a source checkout on ``sys.path``, or metadata a
    packaging tool mangled. Claiming an update against a version we had to invent
    would be worse than saying nothing.
    """
    try:
        from importlib.metadata import version

        return version(PACKAGE)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Opt-out
# ---------------------------------------------------------------------------
def _truthy(name: str) -> bool:
    """Read an env var the conventional way: unset, empty, ``0`` and ``false`` mean no."""
    value = os.environ.get(name, "").strip().lower()
    return value not in ("", "0", "false")


def _disabled() -> bool:
    return (
        _truthy("MANIFOLDBT_NO_UPDATE_CHECK")
        or _truthy("MANIFOLDBT_NO_TELEMETRY")
        or _truthy("CI")
    )


# ---------------------------------------------------------------------------
# Cached answer
# ---------------------------------------------------------------------------
def _cache_dir() -> str:
    """Local state directory, chosen to match the engine's own (``dirs::data_local_dir``)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, PACKAGE)


def _cache_file() -> str:
    return os.path.join(_cache_dir(), "update-check.json")


def _load() -> dict:
    try:
        with open(_cache_file(), "r", encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _save(state: dict) -> None:
    """Write the cache atomically.

    Two interpreters importing at once is the normal case (a sweep script and the
    notebook it was copied from), and a half-written file would be read as "never
    checked" by every run after it.
    """
    import tempfile

    tmp = None
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        # A unique name per writer, rather than `<target>.tmp`: two threads of
        # the same process (the import-time check and an explicit
        # `check_for_update()`) would share that one and each would replace the
        # file the other was still writing.
        handle, tmp = tempfile.mkstemp(
            dir=_cache_dir(), prefix="update-check.", suffix=".tmp"
        )
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(state, out)
        os.replace(tmp, _cache_file())
        tmp = None
    except Exception:
        pass
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)  # never leave litter in the user's data directory
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(
    r"""^\s*v?
        (?P<release>\d+(?:\.\d+)*)
        (?:[-_.]?(?P<pre>a|b|c|rc|alpha|beta|pre|preview)[-_.]?(?P<pre_n>\d*))?
        (?:[-_.]?(?P<post>post|rev|r)[-_.]?(?P<post_n>\d*))?
        (?:[-_.]?dev[-_.]?(?P<dev_n>\d*))?
        \s*$""",
    re.VERBOSE | re.IGNORECASE,
)

_PRE_RANK = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}

# The stages of one release, ordered as PEP 440 orders them:
#   1.0.dev1  <  1.0rc1  <  1.0  <  1.0.post1
_DEV, _PRE, _FINAL, _POST = 0, 1, 2, 3


def _parse(version: str):
    """Sort key for a PEP 440 version, or None if it is not one we understand.

    Deliberately not a full PEP 440 implementation: ``packaging`` is not a
    dependency of this package, and importing it only when present would make the
    comparison depend on which environment we happen to be in. Anything this does
    not recognise (an epoch, a nonsense string) returns None, and an unparseable
    version simply means no notice.
    """
    if not isinstance(version, str):
        return None
    # A local segment (`+cuda12`) never orders two published releases.
    match = _VERSION_RE.match(version.split("+", 1)[0])
    if match is None:
        return None

    release = tuple(int(part) for part in match.group("release").split("."))
    if len(release) < 4:
        release += (0,) * (4 - len(release))  # 1.2 and 1.2.0 are the same release

    def number(group: str) -> int:
        raw = match.group(group)
        return int(raw) if raw else 0

    if match.group("dev_n") is not None and not match.group("pre") and not match.group("post"):
        stage = (_DEV, number("dev_n"))
    elif match.group("pre"):
        stage = (_PRE, _PRE_RANK[match.group("pre").lower()], number("pre_n"))
    elif match.group("post"):
        stage = (_POST, number("post_n"))
    else:
        stage = (_FINAL,)
    return release, stage


def _is_prerelease(version: str) -> bool:
    key = _parse(version)
    return key is not None and key[1][0] < _FINAL


def _newer(current: str, latest: str) -> Optional[str]:
    """``latest``, if it is an upgrade worth telling ``current`` about. Else None."""
    here, there = _parse(current), _parse(latest)
    if here is None or there is None or there <= here:
        return None
    # Never push a pre-release at someone on a stable build: `pip install -U`
    # would not install it, so the notice would be an instruction that fails.
    if _is_prerelease(latest) and not _is_prerelease(current):
        return None
    return latest


# ---------------------------------------------------------------------------
# PyPI
# ---------------------------------------------------------------------------
def _fetch(etag: Optional[str]) -> Optional[dict]:
    """Ask PyPI for the latest release. None on any failure whatsoever."""
    # Imported here, on the background thread that asks PyPI: urllib.request
    # pulls http.client and the email package, some 20 ms that every
    # `import manifoldbt` paid up front for a check it does not wait for.
    import urllib.error
    import urllib.request

    headers = {"Accept": "application/json", "User-Agent": "manifoldbt-update-check"}
    if etag:
        # The full document is a few hundred kilobytes (every file of every
        # release, plus the README); a daily check that has not missed a release
        # costs a 304 and no body at all.
        headers["If-None-Match"] = etag
    request = urllib.request.Request(_PYPI_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
            info = payload.get("info") or {}
            return {
                "latest": info.get("version"),
                "yanked": bool(info.get("yanked")),
                "etag": response.headers.get("ETag"),
            }
    except urllib.error.HTTPError as err:
        if err.code == 304:
            return {"unchanged": True}
        return None
    except Exception:
        return None


def _refresh(state: dict) -> Optional[str]:
    """Query PyPI, update the cache in place, return the latest version known.

    Returns the cached value on failure: an unreachable network is not evidence
    that the last answer was wrong.
    """
    cached = state.get("latest") if isinstance(state.get("latest"), str) else None
    result = _fetch(state.get("etag") if cached else None)
    now = time.time()

    if result is None or result.get("unchanged"):
        # Record the attempt either way. Without this, a machine that is offline
        # (or behind a proxy that eats the request) fires one request per import,
        # forever.
        state["last_check"] = now
        _save(state)
        return cached

    latest = result.get("latest")
    if not isinstance(latest, str) or result.get("yanked"):
        # A yanked latest is a release nobody should be told to install, and a
        # response with no version is a PyPI we do not recognise.
        state["last_check"] = now
        _save(state)
        return cached

    _save({"last_check": now, "latest": latest, "etag": result.get("etag")})
    return latest


# ---------------------------------------------------------------------------
# Notice
# ---------------------------------------------------------------------------
def notice(current: str, latest: str) -> str:
    return "\033[38;5;214mUpdate available:\033[0m {} {} (you have {}) -- pip install -U {}".format(
        PACKAGE, latest, current, PACKAGE
    )


def _announce(current: Optional[str], latest: Optional[str]) -> bool:
    """Print the notice, at most once per process. True if it printed."""
    global _announced
    if _announced or not current or not latest:
        return False
    if _newer(current, latest) is None:
        return False
    _announced = True
    print(notice(current, latest))
    return True


def _background(state: dict) -> None:
    global _pending
    latest = _refresh(state)
    if latest and not _announced:
        # Too late for the banner: hand it to the exit hook rather than print
        # from a background thread into the middle of the user's own output.
        _pending = latest


def _flush() -> None:
    _announce(_current, _pending)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def start() -> Optional[threading.Thread]:
    """Print a cached update notice, and refresh the cache in the background.

    Called once, from ``manifoldbt/__init__.py``, right after the banner. Returns
    immediately in every case; the thread it hands back (when it started one) is
    there so a test can join it, and nothing else should wait on it.
    """
    global _started, _current
    if _started or _disabled():
        return None
    _started = True

    _current = installed_version()
    if _current is None:
        return None

    state = _load()
    _announce(_current, state.get("latest"))

    try:
        last_check = float(state.get("last_check") or 0)
    except (TypeError, ValueError):
        last_check = 0.0
    if time.time() - last_check < _CHECK_INTERVAL_SECONDS:
        return None

    thread = threading.Thread(
        target=_background,
        args=(state,),
        name="manifoldbt-update-check",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        return None  # a runtime that refuses threads keeps the cached answer
    atexit.register(_flush)
    return thread


def check_now() -> Optional[str]:
    """Ask PyPI right now: the newer version, or None if this install is current.

    Blocking, unlike the import-time check, and it refreshes the cache that the
    notice reads from. This is what ``manifoldbt.check_for_update()`` calls.
    """
    current = installed_version()
    if current is None:
        return None
    return _newer(current, _refresh(_load()) or "")
