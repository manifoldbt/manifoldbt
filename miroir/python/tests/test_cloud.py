"""Tests for the cloud client, against a stub server.

No network, no licence, no data store: what is under test here is the contract
with the API and the behaviour of `wait()`, which is where the client can lose a
job that is being billed.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import manifoldbt as mbt
from manifoldbt.cloud import CloudError, QuotaExceeded


class _Stub:
    """A one-endpoint API whose answers the test scripts in advance."""

    def __init__(self):
        self.submit_status = 202
        self.submit_body = {"job": {"id": 7, "kind": "sweep", "status": "queued"}}
        # Answers to successive GETs, as (status, body). The last one repeats.
        self.polls = [(200, {"job": {"id": 7, "kind": "sweep", "status": "succeeded"}})]
        self.requests = []
        self._poll_index = 0

    def next_poll(self):
        status, body = self.polls[min(self._poll_index, len(self.polls) - 1)]
        self._poll_index += 1
        return status, body


@pytest.fixture
def stub():
    state = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # keep pytest output readable

        def _send(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append((self.path, self.headers.get("Authorization"), body))
            self._send(state.submit_status, state.submit_body)

        def do_GET(self):
            state.requests.append((self.path, self.headers.get("Authorization"), None))
            status, body = state.next_poll()
            self._send(status, body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    mbt.cloud.configure(api_key="mbt_live_test", base_url=f"http://127.0.0.1:{server.server_port}")
    yield state
    server.shutdown()
    server.server_close()
    mbt.cloud.configure(api_key="", base_url=mbt.cloud.DEFAULT_BASE_URL)


def submit(**kwargs):
    """Submit a pre-built payload, skipping the store-dependent serialization."""
    return mbt.cloud._submit("sweep", {"strategy": {}, "config": {}}, **kwargs)


def test_submission_returns_a_queued_job_without_waiting(stub):
    job = submit(wait=False, timeout=5)
    assert job.id == 7
    assert job.status == "queued"
    assert not job.done
    # One call: submission must not poll behind your back.
    assert len(stub.requests) == 1
    path, auth, body = stub.requests[0]
    assert path == "/api/jobs"
    assert auth == "Bearer mbt_live_test"
    assert body["kind"] == "sweep"


def test_wait_returns_the_finished_job(stub):
    stub.polls = [
        (200, {"job": {"id": 7, "status": "queued"}}),
        (200, {"job": {"id": 7, "status": "running"}}),
        (200, {"job": {"id": 7, "kind": "sweep", "status": "succeeded",
                       "credits_charged": 0.42, "seconds": 30.0,
                       "result": {"combos": 2, "top": [
                           {"rank": 1, "params": {"period": 20}, "metrics": {"sharpe": 1.5}}]}}}),
    ]
    job = submit(wait=True, timeout=10)
    assert job.status == "succeeded"
    assert job.credits == 0.42
    assert job.combos == 2
    assert job.top[0]["params"] == {"period": 20}
    assert not job.truncated
    assert "sharpe=1.5000" in job.summary()


def test_a_failed_job_raises_instead_of_returning_empty(stub):
    # A failed job returning quietly would read as "no results", and the reason
    # the engine gave would never be seen.
    stub.polls = [(200, {"job": {"id": 7, "status": "failed", "error": "strategy did not compile"}})]
    with pytest.raises(CloudError, match="strategy did not compile"):
        submit(wait=True, timeout=10)


def test_wait_rides_out_a_server_hiccup(stub):
    # The job runs on a worker: a 500 while polling says nothing about it, and
    # abandoning the wait would walk away from compute already being billed.
    stub.polls = [
        (500, {"error": "gateway is having a moment"}),
        (502, {"error": "still"}),
        (200, {"job": {"id": 7, "status": "succeeded", "result": {"combos": 1, "top": []}}}),
    ]
    job = submit(wait=True, timeout=30)
    assert job.status == "succeeded"


def test_wait_gives_up_after_too_many_hiccups(stub):
    stub.polls = [(500, {"error": "down"})]
    with pytest.raises(CloudError, match="lost contact"):
        submit(wait=True, timeout=60)


def test_a_revoked_key_ends_the_wait_at_once(stub):
    # Unlike a 500, a refusal will be a refusal on every retry.
    stub.polls = [(401, {"error": "nope"})]
    with pytest.raises(CloudError, match="API key was refused"):
        submit(wait=True, timeout=60)


def test_an_exhausted_allowance_says_what_is_left(stub):
    stub.submit_status = 402
    stub.submit_body = {"error": "Monthly compute allowance exhausted", "used": 5000, "included": 5000}
    with pytest.raises(QuotaExceeded) as excinfo:
        submit(wait=False, timeout=5)
    assert excinfo.value.used == 5000
    assert excinfo.value.included == 5000
    assert "5000" in str(excinfo.value)


def test_a_missing_key_names_where_to_get_one():
    mbt.cloud.configure(api_key="", base_url="http://127.0.0.1:1")
    with pytest.raises(CloudError, match="MANIFOLDBT_API_KEY"):
        mbt.cloud.job(1)


def test_an_unreachable_server_is_transient_not_a_refusal():
    mbt.cloud.configure(api_key="k", base_url="http://127.0.0.1:1")
    with pytest.raises(CloudError) as excinfo:
        mbt.cloud.job(1)
    assert excinfo.value.transient
    assert excinfo.value.status is None


def test_a_truncated_sweep_says_so(stub):
    stub.polls = [(200, {"job": {"id": 7, "kind": "sweep", "status": "succeeded",
                                 "result": {"combos": 20000, "truncated": True,
                                            "top": [{"rank": 1, "params": {"p": 1},
                                                     "metrics": {"sharpe": 2.0}}]}}})]
    job = submit(wait=True, timeout=10)
    assert job.truncated
    assert job.combos == 20000
    # The summary must not let 20000 combos look like one row of results.
    assert "20000 combos" in job.summary()
    assert "best 1 returned" in job.summary()


def test_the_grid_is_serialized_the_way_the_engine_reads_it():
    # Scalars are tagged on the wire ({"Int64": 20}), which is how the engine's
    # ScalarValue deserializes. A bare 20 is refused by the worker.
    from manifoldbt._serde import scalar_value_to_json

    grid = {"period": [10, 20], "size": [0.5]}
    wire = {k: [scalar_value_to_json(v) for v in vals] for k, vals in grid.items()}
    assert wire == {"period": [{"Int64": 10}, {"Int64": 20}], "size": [{"Float64": 0.5}]}
