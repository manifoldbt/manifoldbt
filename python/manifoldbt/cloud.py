"""Run a backtest or a sweep on managed capacity instead of this machine.

The call is the local one with `cloud.` in front:

    result = mbt.run_sweep(strategy, grid, config, store)          # here
    job    = mbt.cloud.run_sweep(strategy, grid, config, store)    # there

That is deliberate. The payload is built by the same objects and the same
serialization a local run uses, so a job that runs on a worker is the run you
would have got here, through the same code path. Nothing about the strategy
changes because it moved.

Authentication is an API key from the Firm portal, read from `MANIFOLDBT_API_KEY`
or passed to `configure()`. The key is never logged, and never travels anywhere
but the Authorization header.

Submission returns as soon as the job is queued. `wait()` blocks until a worker
has finished it; without it you can close the notebook and come back:

    job = mbt.cloud.run_sweep(strategy, grid, config, store)
    print(job.id)                  # later, from anywhere
    job = mbt.cloud.job(job.id)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from manifoldbt._serde import scalar_value_to_json
from manifoldbt.exceptions import BacktesterError

DEFAULT_BASE_URL = "https://www.manifoldbt.com"

# Terminal states. Anything else means a worker still has it, or nobody has
# taken it yet.
_DONE = ("succeeded", "failed")

# Consecutive failed polls tolerated before `wait()` gives up. Five, because a
# deploy or a restart on the other side is measured in seconds and the job it is
# waiting on is unaffected by any of it.
_MAX_POLL_HICCUPS = 5

_api_key: Optional[str] = None
_base_url: Optional[str] = None


class CloudError(BacktesterError):
    """Raised when the cloud API refuses a request or a job fails.

    `status` carries the HTTP code when there was one, and None when the server
    could not be reached at all. `transient` reads it: a refusal is final, a
    server error or an unreachable host is worth waiting out.
    """

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status

    @property
    def transient(self) -> bool:
        return self.status is None or self.status >= 500


class QuotaExceeded(CloudError):
    """Raised when the org's monthly compute allowance is spent.

    The cap is hard by default: a job that would run into an exhausted
    allowance is refused at submission rather than billed as a surprise.
    """

    def __init__(self, used: float, included: float):
        self.used = used
        self.included = included
        super().__init__(
            f"monthly compute allowance spent: {used:.0f} of {included:.0f} credits used. "
            "It resets at the start of the billing month, or ask for a higher allowance.",
            402,
        )


def configure(api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
    """Set the credentials for this session.

    Both fall back to the environment (`MANIFOLDBT_API_KEY`,
    `MANIFOLDBT_CLOUD_URL`), so a notebook that never calls this still works
    when the environment is set.
    """
    global _api_key, _base_url
    if api_key is not None:
        _api_key = api_key.strip()
    if base_url is not None:
        _base_url = base_url.rstrip("/")


def _key() -> str:
    key = _api_key or os.environ.get("MANIFOLDBT_API_KEY", "").strip()
    if not key:
        raise CloudError(
            "no API key. Pass it with mbt.cloud.configure(api_key=...) or set "
            "MANIFOLDBT_API_KEY. Keys are created in the Firm portal, under Keys."
        )
    return key


def _url(path: str) -> str:
    base = _base_url or os.environ.get("MANIFOLDBT_CLOUD_URL", DEFAULT_BASE_URL)
    return f"{base.rstrip('/')}{path}"


def _call(path: str, method: str = "GET", body: Optional[dict] = None, timeout: int = 60):
    request = urllib.request.Request(
        _url(path),
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {_key()}",
            "User-Agent": "manifoldbt-sdk",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError:
            payload = {}
        _raise_for(err.code, payload)
        raise  # unreachable, _raise_for always raises
    except urllib.error.URLError as err:
        # No status at all: DNS, a dropped connection, a laptop lid. Transient
        # by construction, which is what lets `wait()` ride it out.
        raise CloudError(f"cannot reach {_url(path)}: {err.reason}") from err


def _raise_for(status: int, payload: dict):
    if status == 401:
        raise CloudError(
            "the API key was refused. Check it, or create a new one in the portal.", 401
        )
    if status == 402:
        raise QuotaExceeded(payload.get("used", 0), payload.get("included", 0))
    if status == 403:
        raise CloudError(
            "this organisation has not finished portal setup: sign in to the portal once, "
            "answer the setup questions, then retry.",
            403,
        )
    if status == 413:
        raise CloudError(
            "the payload is too large. A job carries a strategy, never a dataset.", 413
        )
    raise CloudError(
        f"the cloud API answered {status}: {payload.get('error', 'no detail')}", status
    )


def _payload(strategy, config, store, param_grid=None, rank_by="sharpe", top_n=50) -> dict:
    # Imported here, not at module scope: `manifoldbt/__init__` imports this
    # module, so a top-level import would be circular.
    from manifoldbt import _prepared_config_json

    body = {
        "strategy": strategy.to_json_dict(),
        "config": json.loads(_prepared_config_json(config, strategy, store)),
    }
    if param_grid is not None:
        body["param_grid"] = {
            name: [scalar_value_to_json(v) for v in values]
            for name, values in param_grid.items()
        }
        body["rank_by"] = rank_by
        body["top_n"] = top_n
    return body


class CloudJob:
    """A job on managed capacity: its state, and its result once it has one."""

    def __init__(self, row: dict):
        self._row = row

    # -- identity ----------------------------------------------------------
    @property
    def id(self) -> int:
        return self._row["id"]

    @property
    def kind(self) -> str:
        return self._row.get("kind", "sweep")

    @property
    def status(self) -> str:
        """Last known status. Call `refresh()` for the current one."""
        return self._row.get("status", "queued")

    @property
    def done(self) -> bool:
        return self.status in _DONE

    @property
    def error(self) -> Optional[str]:
        return self._row.get("error")

    @property
    def credits(self) -> Optional[float]:
        return self._row.get("credits_charged")

    @property
    def seconds(self) -> Optional[float]:
        return self._row.get("seconds")

    # -- result ------------------------------------------------------------
    @property
    def result(self) -> Optional[dict]:
        """The raw result document, or None while the job is still running."""
        return self._row.get("result")

    @property
    def top(self) -> List[dict]:
        """Ranked combos of a sweep: `[{"rank", "params", "metrics", ...}]`."""
        return list((self.result or {}).get("top") or [])

    @property
    def metrics(self) -> Optional[dict]:
        """Metrics of a single backtest."""
        return (self.result or {}).get("metrics")

    @property
    def truncated(self) -> bool:
        """Whether the ranked table is a head, not the whole sweep.

        A result travels inline today, so a large sweep comes back as its best
        rows. `combos` says how many were actually run.
        """
        return bool((self.result or {}).get("truncated"))

    @property
    def combos(self) -> int:
        return int((self.result or {}).get("combos") or 0)

    # -- lifecycle ---------------------------------------------------------
    def refresh(self) -> "CloudJob":
        _, payload = _call(f"/api/jobs/{self.id}")
        self._row = (payload or {}).get("job") or self._row
        return self

    def wait(self, timeout: float = 3600, poll: float = 1.0, quiet: bool = False) -> "CloudJob":
        """Block until the job finishes, then return it.

        Raises `CloudError` if the job failed, because a failed job that
        returned quietly would be read as an empty result.
        """
        deadline = time.time() + timeout
        interval = poll
        seen = None
        hiccups = 0
        while time.time() < deadline:
            try:
                self.refresh()
                hiccups = 0
            except CloudError as err:
                # The job is running on a worker, not in this process: a server
                # error or a dropped connection while polling says nothing about
                # it. Giving up here would abandon compute that is being billed.
                # A refusal (a revoked key, an unknown id) is different and ends
                # the wait immediately.
                if not err.transient:
                    raise
                hiccups += 1
                if hiccups >= _MAX_POLL_HICCUPS:
                    raise CloudError(
                        f"lost contact with the cloud API for {hiccups} polls in a row: {err}. "
                        f"Job {self.id} keeps running: fetch it later with mbt.cloud.job({self.id}).",
                        err.status,
                    ) from err
                time.sleep(min(interval * 2, 10.0))
                continue
            if not quiet and self.status != seen:
                seen = self.status
                print(f"job {self.id}: {seen}")
            if self.done:
                if self.status == "failed":
                    raise CloudError(f"job {self.id} failed: {self.error or 'no reason given'}")
                return self
            time.sleep(interval)
            # Ease off: a sweep that takes ten minutes should not be polled a
            # thousand times, and the first seconds are when an answer is most
            # likely.
            interval = min(interval * 1.5, 5.0)
        raise CloudError(
            f"job {self.id} was still {self.status} after {timeout:.0f}s. "
            "It keeps running: fetch it later with mbt.cloud.job(id)."
        )

    def summary(self) -> str:
        lines = [f"job {self.id} ({self.kind}) -- {self.status}"]
        if self.credits is not None:
            lines.append(f"  {self.seconds:.2f}s, {self.credits:g} credits")
        if self.status == "failed":
            lines.append(f"  {self.error}")
        elif self.kind == "sweep" and self.top:
            shown = len(self.top)
            head = f"  {self.combos} combos"
            if self.truncated:
                head += f", best {shown} returned"
            lines.append(head)
            for row in self.top[:5]:
                params = ", ".join(f"{k}={v}" for k, v in (row.get("params") or {}).items())
                metrics = row.get("metrics") or {}
                lines.append(f"    {params:<28} sharpe={metrics.get('sharpe', float('nan')):.4f}")
        elif self.metrics:
            lines.append(f"  sharpe={self.metrics.get('sharpe', float('nan')):.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<CloudJob {self.id} {self.status}>"


def run(strategy, config, store, *, wait: bool = False, timeout: float = 3600) -> CloudJob:
    """Submit a single backtest. Same arguments as `mbt.run`."""
    return _submit("backtest", _payload(strategy, config, store), wait, timeout)


def run_sweep(
    strategy,
    param_grid: Dict[str, List[Any]],
    config,
    store,
    *,
    rank_by: str = "sharpe",
    top_n: int = 50,
    wait: bool = False,
    timeout: float = 3600,
) -> CloudJob:
    """Submit a parameter sweep. Same arguments as `mbt.run_sweep`.

    `rank_by` names the metric the returned table is sorted by, highest first,
    and `top_n` how many rows come back.
    """
    body = _payload(strategy, config, store, param_grid, rank_by, top_n)
    return _submit("sweep", body, wait, timeout)


def _submit(kind: str, body: dict, wait: bool, timeout: float) -> CloudJob:
    _, payload = _call("/api/jobs", "POST", {"kind": kind, "payload": body})
    job = CloudJob((payload or {}).get("job") or {})
    return job.wait(timeout=timeout) if wait else job


def job(job_id: int) -> CloudJob:
    """Fetch a job by id, from any session."""
    return CloudJob({"id": int(job_id)}).refresh()


def jobs(limit: int = 25) -> List[CloudJob]:
    """The org's recent jobs, newest first."""
    _, payload = _call("/api/jobs")
    return [CloudJob(row) for row in ((payload or {}).get("jobs") or [])[:limit]]


def usage() -> dict:
    """Credits used and included this month, per kind."""
    _, payload = _call("/api/jobs")
    return (payload or {}).get("usage") or {}
