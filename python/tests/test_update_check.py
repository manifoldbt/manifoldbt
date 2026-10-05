"""The PyPI update notice.

Everything here runs offline: the one function that touches the network
(``_fetch``) is either replaced or driven through a fake ``urlopen``. A test that
really queried PyPI would be a test that fails on a plane.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request  # manifoldbt._update imports it only where it asks PyPI

import pytest

from manifoldbt import _update


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private cache, no opt-out in the environment, and fresh process state."""
    # Importing manifoldbt started a real check on a daemon thread. Let it finish
    # before the cache directory is redirected, or it lands in this test's
    # tmp_path and the assertions below read someone else's answer.
    for thread in threading.enumerate():
        if thread.name == "manifoldbt-update-check":
            thread.join(timeout=10)
    monkeypatch.setattr(_update, "_cache_dir", lambda: str(tmp_path))
    for var in ("MANIFOLDBT_NO_UPDATE_CHECK", "MANIFOLDBT_NO_TELEMETRY", "CI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_update, "_started", False)
    monkeypatch.setattr(_update, "_announced", False)
    monkeypatch.setattr(_update, "_current", None)
    monkeypatch.setattr(_update, "_pending", None)
    yield


def _write_cache(state):
    _update._save(state)


def _read_cache():
    with open(_update._cache_file(), encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "current, latest, expected",
    [
        ("0.19.1", "0.19.2", "0.19.2"),
        ("0.19.1", "0.20.0", "0.20.0"),
        ("0.9.0", "0.10.0", "0.10.0"),  # not a string comparison
        ("0.19.1", "0.19.1", None),
        ("0.19.2", "0.19.1", None),  # a local build ahead of PyPI says nothing
        ("0.19", "0.19.0", None),  # 0.19 and 0.19.0 are the same release
        ("0.19.1", "0.20.0rc1", None),  # `pip install -U` would not take it
        ("0.20.0rc1", "0.20.0rc2", "0.20.0rc2"),  # unless you asked for pre-releases
        ("0.20.0rc1", "0.20.0", "0.20.0"),
        ("0.19.1", "0.19.1.post1", "0.19.1.post1"),
        ("0.19.1+cuda12", "0.19.2", "0.19.2"),  # a local segment does not order releases
        ("0.19.1", "not-a-version", None),
        ("not-a-version", "0.19.2", None),
        ("0.19.1", "1!0.1.0", None),  # an epoch is not understood, so: silence
    ],
)
def test_newer_decides_what_is_worth_saying(current, latest, expected):
    assert _update._newer(current, latest) == expected


def test_prerelease_ordering_follows_pep440():
    order = ["1.0.dev1", "1.0a1", "1.0b1", "1.0rc1", "1.0", "1.0.post1", "1.0.1"]
    keys = [_update._parse(v) for v in order]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_unreadable_cache_reads_as_never_checked(tmp_path):
    with open(_update._cache_file(), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert _update._load() == {}


def test_save_leaves_no_temporary_file_behind(tmp_path):
    _update._save({"latest": "0.19.2"})
    assert _update._load() == {"latest": "0.19.2"}
    assert os.listdir(str(tmp_path)) == ["update-check.json"]


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------
def test_refresh_records_the_answer(monkeypatch):
    monkeypatch.setattr(
        _update, "_fetch", lambda etag: {"latest": "0.20.0", "etag": '"abc"'}
    )
    assert _update._refresh({}) == "0.20.0"
    cached = _read_cache()
    assert cached["latest"] == "0.20.0"
    assert cached["etag"] == '"abc"'
    assert cached["last_check"] > 0


def test_offline_keeps_the_last_answer_and_still_burns_the_day(monkeypatch):
    """An unreachable network must not mean one request per import for the rest
    of the day, and it is not evidence that yesterday's answer was wrong."""
    monkeypatch.setattr(_update, "_fetch", lambda etag: None)
    state = {"latest": "0.20.0", "last_check": 0.0}
    assert _update._refresh(state) == "0.20.0"
    cached = _read_cache()
    assert cached["latest"] == "0.20.0"
    assert cached["last_check"] > time.time() - 60


def test_unchanged_response_keeps_the_cached_version_and_etag(monkeypatch):
    seen = {}

    def fetch(etag):
        seen["etag"] = etag
        return {"unchanged": True}

    monkeypatch.setattr(_update, "_fetch", fetch)
    assert _update._refresh({"latest": "0.20.0", "etag": '"abc"'}) == "0.20.0"
    assert seen["etag"] == '"abc"'  # the 304 was actually asked for
    assert _read_cache()["etag"] == '"abc"'


def test_a_yanked_release_is_never_recommended(monkeypatch):
    monkeypatch.setattr(
        _update, "_fetch", lambda etag: {"latest": "0.20.0", "yanked": True}
    )
    assert _update._refresh({"latest": "0.19.9"}) == "0.19.9"
    assert _read_cache()["latest"] == "0.19.9"


# ---------------------------------------------------------------------------
# The HTTP call itself
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, body, headers=None):
        self._body = body.encode("utf-8")
        self.headers = headers or {}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_reads_info_version_from_the_pypi_json_api(monkeypatch):
    captured = {}

    def urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        body = json.dumps({"info": {"version": "0.20.0", "yanked": False}})
        return _FakeResponse(body, {"ETag": '"xyz"'})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _update._fetch(None) == {
        "latest": "0.20.0",
        "yanked": False,
        "etag": '"xyz"',
    }
    assert captured["url"] == "https://pypi.org/pypi/manifoldbt/json"
    assert "if-none-match" not in captured["headers"]  # nothing cached to revalidate


def test_fetch_revalidates_with_the_stored_etag(monkeypatch):
    captured = {}

    def urlopen(request, timeout=None):
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        raise urllib.error.HTTPError(request.full_url, 304, "Not Modified", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _update._fetch('"xyz"') == {"unchanged": True}
    assert captured["headers"]["if-none-match"] == '"xyz"'


def test_fetch_swallows_everything_else(monkeypatch):
    def urlopen(request, timeout=None):
        raise OSError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _update._fetch(None) is None


def test_fetch_swallows_a_server_error(monkeypatch):
    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 503, "nope", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _update._fetch(None) is None


# ---------------------------------------------------------------------------
# start(): what the user actually sees at import
# ---------------------------------------------------------------------------
def _never_called(*args, **kwargs):
    raise AssertionError("the network was touched")


def test_a_fresh_cache_prints_the_notice_without_asking_pypi(monkeypatch, capsys):
    monkeypatch.setattr(_update, "installed_version", lambda: "0.19.1")
    monkeypatch.setattr(_update, "_fetch", _never_called)
    _write_cache({"latest": "0.20.0", "last_check": time.time()})

    _update.start()

    out = capsys.readouterr().out
    assert "0.20.0" in out and "0.19.1" in out
    assert "pip install -U manifoldbt" in out


def test_being_up_to_date_prints_nothing(monkeypatch, capsys):
    monkeypatch.setattr(_update, "installed_version", lambda: "0.20.0")
    monkeypatch.setattr(_update, "_fetch", _never_called)
    _write_cache({"latest": "0.20.0", "last_check": time.time()})

    _update.start()

    assert capsys.readouterr().out == ""


def test_a_source_checkout_is_never_compared(monkeypatch, capsys):
    monkeypatch.setattr(_update, "installed_version", lambda: None)
    monkeypatch.setattr(_update, "_fetch", _never_called)
    _write_cache({"latest": "99.0.0", "last_check": time.time()})

    _update.start()

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "var", ["MANIFOLDBT_NO_UPDATE_CHECK", "MANIFOLDBT_NO_TELEMETRY", "CI"]
)
def test_opt_out_silences_it_entirely(monkeypatch, capsys, var):
    monkeypatch.setenv(var, "1")
    monkeypatch.setattr(_update, "installed_version", lambda: "0.19.1")
    monkeypatch.setattr(_update, "_fetch", _never_called)
    _write_cache({"latest": "0.20.0", "last_check": time.time()})

    _update.start()

    assert capsys.readouterr().out == ""


def test_opt_out_reads_zero_as_opting_in(monkeypatch, capsys):
    monkeypatch.setenv("MANIFOLDBT_NO_UPDATE_CHECK", "0")
    monkeypatch.setattr(_update, "installed_version", lambda: "0.19.1")
    monkeypatch.setattr(_update, "_fetch", _never_called)
    _write_cache({"latest": "0.20.0", "last_check": time.time()})

    _update.start()

    assert "0.20.0" in capsys.readouterr().out


def test_a_stale_cache_asks_pypi_in_the_background(monkeypatch, capsys):
    """The first run after a release: nothing to print at import, so the answer
    comes back on the daemon thread and the exit hook says it."""
    monkeypatch.setattr(_update, "installed_version", lambda: "0.19.1")
    monkeypatch.setattr(_update, "_fetch", lambda etag: {"latest": "0.20.0"})
    _write_cache({"latest": "0.19.1", "last_check": time.time() - 48 * 3600})

    thread = _update.start()
    assert capsys.readouterr().out == ""  # the banner is not delayed for this
    thread.join(timeout=5)

    _update._flush()  # what atexit runs
    assert "0.20.0" in capsys.readouterr().out
    assert _read_cache()["latest"] == "0.20.0"


def test_the_notice_is_printed_once_per_process(monkeypatch, capsys):
    monkeypatch.setattr(_update, "installed_version", lambda: "0.19.1")
    monkeypatch.setattr(_update, "_fetch", lambda etag: {"latest": "0.20.0"})
    _write_cache({"latest": "0.20.0", "last_check": time.time() - 48 * 3600})

    thread = _update.start()
    assert "0.20.0" in capsys.readouterr().out
    thread.join(timeout=5)

    _update._flush()
    assert capsys.readouterr().out == ""


def test_start_is_idempotent(monkeypatch, capsys):
    monkeypatch.setattr(_update, "installed_version", lambda: "0.19.1")
    monkeypatch.setattr(_update, "_fetch", _never_called)
    _write_cache({"latest": "0.20.0", "last_check": time.time()})

    _update.start()
    capsys.readouterr()
    _update.start()

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# check_for_update()
# ---------------------------------------------------------------------------
def test_check_now_queries_and_answers(monkeypatch):
    monkeypatch.setattr(_update, "installed_version", lambda: "0.19.1")
    monkeypatch.setattr(_update, "_fetch", lambda etag: {"latest": "0.20.0"})
    assert _update.check_now() == "0.20.0"


def test_check_now_is_none_when_current(monkeypatch):
    monkeypatch.setattr(_update, "installed_version", lambda: "0.20.0")
    monkeypatch.setattr(_update, "_fetch", lambda etag: {"latest": "0.20.0"})
    assert _update.check_now() is None


def test_check_now_is_none_offline(monkeypatch):
    monkeypatch.setattr(_update, "installed_version", lambda: "0.19.1")
    monkeypatch.setattr(_update, "_fetch", lambda etag: None)
    assert _update.check_now() is None


def test_public_alias_is_exported():
    import manifoldbt as bt

    assert "check_for_update" in bt.__all__
    assert callable(bt.check_for_update)
