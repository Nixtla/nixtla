"""Wire-level tests for the asynchronous job protocol used by simulate/explain.

The API accepts the request with a job id (202), the client polls the job's
status until it is terminal and reads the result from the status envelope.
"""

import json
from http import HTTPStatus
from unittest.mock import MagicMock

import httpx
import numpy as np
import orjson
import pandas as pd
import pytest
import zstandard as zstd

import nixtla.nixtla_client as client_module
from nixtla import ApiError, AsyncJobError, NixtlaClient


class FakeApi:
    """In-memory stand-in for the async job routes of the API.

    `statuses` is the sequence of statuses reported by successive polls of a
    job; the last one repeats. `submit_responses` optionally overrides the
    responses of successive submits (a list of `(status_code, body, headers)`
    or exceptions); once exhausted the job is accepted.
    """

    def __init__(
        self,
        task="simulate",
        statuses=("pending", "running", "succeeded"),
        result=None,
        error=None,
        submit_responses=None,
        poll_responses=None,
    ):
        self.task = task
        self.prefix = {"simulate": "sm", "explain": "ex"}[task]
        self.statuses = list(statuses)
        self.result = result
        self.error = error
        self.submit_responses = list(submit_responses or [])
        self.poll_responses = list(poll_responses or [])
        self.requests: list[httpx.Request] = []
        self.jobs: dict[str, int] = {}
        self.cancelled: list[str] = []
        self.transport = httpx.MockTransport(self.handle)

    # -- helpers ---------------------------------------------------------- #
    @property
    def submits(self):
        return [r for r in self.requests if r.url.path.endswith("/async")]

    @property
    def polls(self):
        return [r for r in self.requests if "/jobs/" in r.url.path]

    @staticmethod
    def decode(request):
        body = request.content
        if request.headers.get("content-encoding") == "zstd":
            body = zstd.ZstdDecompressor().decompressobj().decompress(body)
        return orjson.loads(body)

    def make_client(self, **kwargs):
        client = NixtlaClient(
            api_key="test",
            max_retries=kwargs.pop("max_retries", 3),
            retry_interval=kwargs.pop("retry_interval", 0),
            **kwargs,
        )
        client._get_model_params = MagicMock(return_value=(28, 7))
        client._make_client = lambda **kw: httpx.Client(transport=self.transport, **kw)
        return client

    # -- routes ----------------------------------------------------------- #
    def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == f"/v2/{self.task}/async":
            if self.submit_responses:
                response = self.submit_responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                status_code, body, headers = response
                return httpx.Response(status_code, json=body, headers=headers)
            job_id = f"{self.prefix}-{len(self.jobs):032x}"
            self.jobs[job_id] = 0
            return httpx.Response(HTTPStatus.ACCEPTED, json={"job_id": job_id})
        if request.method == "GET" and path.startswith(f"/v2/{self.task}/jobs/"):
            if self.poll_responses:
                status_code, body = self.poll_responses.pop(0)
                return httpx.Response(status_code, json=body)
            job_id = path.rsplit("/", 1)[1]
            if job_id not in self.jobs:
                return httpx.Response(
                    HTTPStatus.NOT_FOUND,
                    json={"detail": f"Async job '{job_id}' not found."},
                )
            idx = min(self.jobs[job_id], len(self.statuses) - 1)
            self.jobs[job_id] += 1
            status = self.statuses[idx]
            return httpx.Response(
                HTTPStatus.OK,
                json={
                    "job_id": job_id,
                    "status": status,
                    "result": self.result if status == "succeeded" else None,
                    "error": self.error if status in ("failed", "cancelled") else None,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:01Z",
                },
            )
        if request.method == "POST" and path.startswith("/v2/async/jobs/"):
            job_id = path.split("/")[4]
            self.cancelled.append(job_id)
            return httpx.Response(HTTPStatus.ACCEPTED, json={"job_id": job_id})
        return httpx.Response(HTTPStatus.NOT_FOUND, json={"detail": path})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []

    def wait_for_poll(seconds, cancellation_event):
        sleeps.append(seconds)
        return cancellation_event is not None and cancellation_event.is_set()

    monkeypatch.setattr(client_module, "_sleep", sleeps.append)
    monkeypatch.setattr(client_module, "_wait_for_poll", wait_for_poll)
    return sleeps


def _series_df(n_series=1, n=6):
    return pd.DataFrame(
        {
            "unique_id": np.repeat([f"id-{i}" for i in range(n_series)], n),
            "ds": list(pd.date_range("2024-01-01", periods=n, freq="D")) * n_series,
            "y": np.arange(n_series * n, dtype=float),
        }
    )


def _simulate_result(n_series, n_paths, h):
    return {
        "samples": list(range(n_paths * n_series * h)),
        "n_paths": n_paths,
        "h": h,
        "sizes": [h] * n_series,
        "coupled": False,
    }


def _explain_df():
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 8,
            "ds": pd.date_range("2024-01-01", periods=8, freq="D"),
            "y": np.arange(8, dtype=float),
            "driver": np.arange(8, dtype=float) * 2,
            "noise": np.arange(8, dtype=float)[::-1],
        }
    )


# --------------------------------------------------------------------------- #
# Happy paths                                                                  #
# --------------------------------------------------------------------------- #


def test_simulate_submits_then_polls_until_succeeded(no_sleep):
    api = FakeApi(result=_simulate_result(n_series=1, n_paths=2, h=3))
    client = api.make_client()

    out = client.simulate(df=_series_df(), h=3, freq="D", n_paths=2, model="timegpt-1")

    assert len(out) == 6
    assert out["TimeGPT"].tolist() == list(range(6))
    (submit,) = api.submits
    assert submit.url.path == "/v2/simulate/async"
    assert submit.headers["nixtla-model"] == "timegpt-1"
    assert submit.headers["authorization"] == "Bearer test"
    body = api.decode(submit)
    assert body["h"] == 3 and body["n_paths"] == 2 and body["model"] == "timegpt-1"
    assert "job_options" not in body
    job_id = next(iter(api.jobs))
    assert [p.url.path for p in api.polls] == [f"/v2/simulate/jobs/{job_id}"] * 3
    # slept between the three polls, backing off from the initial interval
    assert len(no_sleep) == 2
    assert no_sleep[1] > no_sleep[0]
    assert api.cancelled == []


def test_explain_submits_without_model_header_and_reads_result():
    api = FakeApi(
        task="explain",
        result={"weights": [0.75, 0.25], "feature_names": None, "method": "granger"},
    )
    client = api.make_client()

    out = client.explain(_explain_df(), features=["driver", "noise"])

    assert out["feature"].tolist() == ["driver", "noise"]
    assert out["weight"].tolist() == [0.75, 0.25]
    (submit,) = api.submits
    assert submit.url.path == "/v2/explain/async"
    assert "nixtla-model" not in submit.headers
    body = api.decode(submit)
    assert body["method"] == "granger"
    assert "model" not in body
    assert api.polls[0].url.path.startswith("/v2/explain/jobs/ex-")


def test_timeout_seconds_is_forwarded_as_job_options():
    api = FakeApi(result=_simulate_result(1, 1, 2))
    client = api.make_client()

    client.simulate(df=_series_df(), h=2, freq="D", n_paths=1, timeout_seconds=120)

    assert api.decode(api.submits[0])["job_options"] == {"timeout_seconds": 120}


@pytest.mark.parametrize("bad", [0, -5, True, 1.5, "10"])
def test_invalid_timeout_seconds_is_rejected_before_any_request(bad):
    api = FakeApi(result=_simulate_result(1, 1, 2))
    client = api.make_client()

    with pytest.raises(ValueError, match="timeout_seconds"):
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1, timeout_seconds=bad)
    with pytest.raises(ValueError, match="timeout_seconds"):
        client.explain(_explain_df(), timeout_seconds=bad)
    assert api.requests == []


def test_large_submit_body_is_zstd_compressed():
    api = FakeApi(result=_simulate_result(1, 1, 1))
    client = api.make_client()
    # > 1MB of history triggers compression; keep the model's input size above
    # the history length so the client does not trim it before sending.
    n = 300_000
    client._get_model_params = MagicMock(return_value=(n, 7))
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2000-01-01", periods=n, freq="min"),
            "y": np.random.default_rng(0).normal(size=n),
        }
    )

    client.simulate(df=df, h=1, freq="min", n_paths=1, model="timegpt-1")

    (submit,) = api.submits
    assert submit.headers["content-encoding"] == "zstd"
    assert len(api.decode(submit)["series"]["y"]) == n


def test_partitioned_simulate_runs_one_job_per_partition():
    api = FakeApi(result=_simulate_result(n_series=1, n_paths=2, h=2))
    client = api.make_client()

    out = client.simulate(
        df=_series_df(n_series=3),
        h=2,
        freq="D",
        n_paths=2,
        num_partitions=3,
        seed=1,
    )

    assert len(out) == 3 * 2 * 2
    assert len(api.submits) == 3
    assert len(api.jobs) == 3
    assert {api.decode(s)["seed"] for s in api.submits} == {1, 2, 3}
    for job_id in api.jobs:
        assert sum(p.url.path.endswith(job_id) for p in api.polls) == 3


# --------------------------------------------------------------------------- #
# Terminal failures and malformed envelopes                                    #
# --------------------------------------------------------------------------- #


def test_failed_job_raises_async_job_error_with_server_message():
    api = FakeApi(
        statuses=("running", "failed"),
        error="InvalidInputException: n_paths too large",
    )
    client = api.make_client()

    with pytest.raises(AsyncJobError, match="n_paths too large") as excinfo:
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    err = excinfo.value
    assert err.task == "simulate"
    assert err.status == "failed"
    assert err.job_id.startswith("sm-")
    assert err.error == "InvalidInputException: n_paths too large"
    assert api.cancelled == []


def test_cancelled_job_raises_async_job_error_even_without_message():
    api = FakeApi(task="explain", statuses=("cancelled",), error=None)
    client = api.make_client()

    with pytest.raises(AsyncJobError, match="cancelled") as excinfo:
        client.explain(_explain_df())

    assert excinfo.value.status == "cancelled"
    assert excinfo.value.error is None


def test_succeeded_job_without_result_raises():
    api = FakeApi(statuses=("succeeded",), result=None)
    client = api.make_client()

    with pytest.raises(RuntimeError, match="returned no result"):
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)


def test_unknown_status_raises_instead_of_polling_forever():
    api = FakeApi(statuses=("exploded",))
    client = api.make_client()

    with pytest.raises(RuntimeError, match="Unexpected status"):
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)
    assert len(api.polls) == 1


def test_unexpected_submit_body_raises():
    api = FakeApi(submit_responses=[(HTTPStatus.ACCEPTED, {"job_id": "fc-wrong"}, {})])
    client = api.make_client()

    with pytest.raises(RuntimeError, match="Unexpected response"):
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)
    assert api.polls == []


# --------------------------------------------------------------------------- #
# Submit errors and retries                                                    #
# --------------------------------------------------------------------------- #


def test_submit_422_is_not_retried():
    api = FakeApi(
        submit_responses=[(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": "bad"}, {})] * 3
    )
    client = api.make_client()

    with pytest.raises(ApiError) as excinfo:
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert excinfo.value.status_code == 422
    assert len(api.submits) == 1


def test_submit_read_timeout_is_not_retried_to_avoid_duplicate_jobs():
    api = FakeApi(submit_responses=[httpx.ReadTimeout("lost response")])
    client = api.make_client()

    with pytest.raises(httpx.ReadTimeout):
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert len(api.submits) == 1


def test_submit_connect_error_is_retried():
    api = FakeApi(
        submit_responses=[httpx.ConnectError("refused")],
        result=_simulate_result(1, 1, 2),
    )
    client = api.make_client()

    out = client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert len(out) == 2
    assert len(api.submits) == 2


def test_submit_429_is_retried_and_carries_retry_after():
    api = FakeApi(
        submit_responses=[
            (
                HTTPStatus.TOO_MANY_REQUESTS,
                {"detail": "Your team already has the maximum of 5 async jobs"},
                {"Retry-After": "0"},
            )
        ],
        result=_simulate_result(1, 1, 2),
    )
    client = api.make_client()

    out = client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert len(out) == 2
    assert len(api.submits) == 2


def test_submit_429_exhausting_retries_raises_api_error():
    api = FakeApi(
        submit_responses=[
            (HTTPStatus.TOO_MANY_REQUESTS, {"detail": "capped"}, {"Retry-After": "0"})
        ]
        * 5
    )
    client = api.make_client(max_retries=2)

    with pytest.raises(ApiError) as excinfo:
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 0.0
    assert len(api.submits) == 2


def test_submit_unavailable_503_fails_fast_with_actionable_message():
    api = FakeApi(
        task="explain",
        submit_responses=[
            (
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"detail": "Async jobs are not available in this deployment."},
                {},
            )
        ]
        * 3,
    )
    client = api.make_client()

    with pytest.raises(ApiError) as excinfo:
        client.explain(_explain_df())

    assert excinfo.value.status_code == 503
    assert "asynchronous job" in str(excinfo.value)
    assert "not available in this deployment" in str(excinfo.value)
    assert len(api.submits) == 1


def test_submit_other_503_is_not_retried():
    api = FakeApi(
        submit_responses=[(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "upstream"}, {})]
        * 3
    )
    client = api.make_client()

    with pytest.raises(ApiError) as excinfo:
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert excinfo.value.status_code == 503
    assert len(api.submits) == 1


# --------------------------------------------------------------------------- #
# Poll errors                                                                  #
# --------------------------------------------------------------------------- #


def test_transient_503_while_polling_is_retried():
    api = FakeApi(
        poll_responses=[
            (HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "Status temporarily unavailable"})
        ],
        statuses=("succeeded",),
        result=_simulate_result(1, 1, 2),
    )
    client = api.make_client()

    out = client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert len(out) == 2
    assert len(api.polls) == 2


def test_404_while_polling_raises_immediately():
    api = FakeApi(poll_responses=[(HTTPStatus.NOT_FOUND, {"detail": "not found"})] * 3)
    client = api.make_client()

    with pytest.raises(ApiError) as excinfo:
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert excinfo.value.status_code == 404
    assert len(api.polls) == 1


def test_wait_timeout_cancels_the_job_and_raises_timeout_error(monkeypatch):
    api = FakeApi(statuses=("pending",))
    client = api.make_client(async_job_wait_timeout=2)
    clock = iter(range(0, 1000))
    monkeypatch.setattr(client_module.time, "monotonic", lambda: float(next(clock)))

    with pytest.raises(TimeoutError, match="did not finish within 2 seconds") as excinfo:
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    job_id = next(iter(api.jobs))
    assert job_id in str(excinfo.value)
    assert api.cancelled == [job_id]


def test_success_response_received_after_wait_deadline_is_discarded():
    api = FakeApi(statuses=("succeeded",), result=_simulate_result(1, 1, 2))
    original_handle = api.handle

    def handle(request):
        if request.method == "GET" and "/jobs/" in request.url.path:
            client_module.time.sleep(0.02)
        return original_handle(request)

    api.transport = httpx.MockTransport(handle)
    client = api.make_client(async_job_wait_timeout=0.001)

    with pytest.raises(TimeoutError, match="did not finish within 0.001 seconds"):
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert api.cancelled == [next(iter(api.jobs))]


def test_poll_retries_do_not_run_past_wait_deadline():
    api = FakeApi(
        poll_responses=[
            (HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "temporarily unavailable"})
        ]
    )
    original_handle = api.handle

    def handle(request):
        if request.method == "GET" and "/jobs/" in request.url.path:
            client_module.time.sleep(0.02)
        return original_handle(request)

    api.transport = httpx.MockTransport(handle)
    client = api.make_client(async_job_wait_timeout=0.001, max_retries=3)

    with pytest.raises(TimeoutError, match="did not finish within 0.001 seconds"):
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert sum(request.method == "GET" for request in api.polls) == 1
    assert api.cancelled == [next(iter(api.jobs))]


def test_wait_timeout_none_polls_until_terminal():
    api = FakeApi(statuses=("pending",) * 30 + ("succeeded",), result=_simulate_result(1, 1, 2))
    client = api.make_client(async_job_wait_timeout=None)

    out = client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert len(out) == 2
    assert len(api.polls) == 31


def test_keyboard_interrupt_while_polling_requests_cancellation():
    api = FakeApi(statuses=("pending",))
    client = api.make_client()
    original_handle = api.handle

    def handle(request):
        if request.url.path.startswith("/v2/simulate/jobs/"):
            api.requests.append(request)
            raise KeyboardInterrupt
        return original_handle(request)

    api.transport = httpx.MockTransport(handle)

    with pytest.raises(KeyboardInterrupt):
        client.simulate(df=_series_df(), h=2, freq="D", n_paths=1)

    assert api.cancelled == [next(iter(api.jobs))]


def test_partition_failure_signals_other_workers_to_stop():
    client = NixtlaClient(api_key="test")
    started = client_module.Event()
    stopped = client_module.Event()

    def run(_client, _task, payload, *, cancellation_event, **_kwargs):
        if payload["position"] == 0:
            assert started.wait(1)
            raise RuntimeError("partition failed")
        started.set()
        assert cancellation_event.wait(1)
        stopped.set()
        raise client_module.CancelledError

    client._run_async_job = run
    with pytest.raises(RuntimeError, match="partition failed"):
        client._dispatch_async_jobs(
            MagicMock(),
            "simulate",
            [{"position": 0}, {"position": 1}],
        )

    assert stopped.is_set()


def test_partition_cancellation_signal_cancels_submitted_job():
    api = FakeApi(statuses=("pending",))
    client = api.make_client()
    job_id = "sm-00000000000000000000000000000000"
    api.jobs[job_id] = 0
    cancellation_event = client_module.Event()
    cancellation_event.set()

    with client._make_client(**client._client_kwargs) as http:
        with pytest.raises(client_module.CancelledError):
            client._poll_async_job(
                http,
                "simulate",
                job_id,
                deadline=None,
                cancellation_event=cancellation_event,
            )

    assert api.cancelled == [job_id]


@pytest.mark.parametrize("status_code", [HTTPStatus.ACCEPTED, HTTPStatus.NOT_FOUND, HTTPStatus.CONFLICT])
def test_cancel_is_best_effort_and_quiet_for_expected_statuses(status_code, caplog):
    def handle(request):
        return httpx.Response(status_code, json={"job_id": "sm-1"})

    client = NixtlaClient(api_key="test")
    with httpx.Client(transport=httpx.MockTransport(handle), base_url="http://t") as http:
        with caplog.at_level("WARNING"):
            client._cancel_async_job(http, "sm-1")
    assert caplog.records == []


def test_cancel_never_raises(caplog):
    def handle(request):
        raise httpx.ConnectError("down")

    client = NixtlaClient(api_key="test")
    with httpx.Client(transport=httpx.MockTransport(handle), base_url="http://t") as http:
        with caplog.at_level("WARNING"):
            client._cancel_async_job(http, "sm-1")
    assert "Could not cancel job sm-1" in caplog.text


# --------------------------------------------------------------------------- #
# Helpers and constructor validation                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({}, None),
        ({"Retry-After": "30"}, 30.0),
        ({"retry-after": "1.5"}, 1.5),
        ({"Retry-After": "-2"}, 0.0),
        ({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}, None),
    ],
)
def test_parse_retry_after(headers, expected):
    assert client_module._parse_retry_after(httpx.Headers(headers)) == expected


def test_api_error_defaults_are_backwards_compatible():
    err = ApiError(status_code=500, body="boom")
    assert err.retry_after is None
    assert str(err) == "status_code: 500, body: boom"


def test_async_job_error_message_and_attributes():
    err = AsyncJobError(job_id="ex-1", task="explain", status="failed", error="bad input")
    assert str(err) == "explain job 'ex-1' failed: bad input"
    assert isinstance(err, RuntimeError)
    quiet = AsyncJobError(job_id="sm-1", task="simulate", status="cancelled")
    assert "no error message was reported" in str(quiet)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"async_job_wait_timeout": 0},
        {"async_job_wait_timeout": -1},
        {"async_job_poll_interval": 0},
    ],
)
def test_constructor_rejects_non_positive_async_settings(kwargs):
    with pytest.raises(ValueError):
        NixtlaClient(api_key="test", **kwargs)


def test_sync_endpoints_still_use_make_request():
    """The forecast path is unchanged: one POST, 200, optional `data` unwrapping."""

    def handle(request):
        assert request.url.path == "/v2/forecast"
        return httpx.Response(200, json={"data": {"mean": [1.0]}})

    client = NixtlaClient(api_key="test")
    with httpx.Client(transport=httpx.MockTransport(handle), base_url="http://t") as http:
        out = client._make_request(http, "v2/forecast", {"model": "m"}, False)
    assert out == {"mean": [1.0]}


def test_oversized_payload_guidance_uses_the_task_name(monkeypatch):
    class _Oversized:
        def __len__(self):
            return 201 * 2**20

    monkeypatch.setattr(client_module.orjson, "dumps", lambda *a, **k: _Oversized())
    client = NixtlaClient(api_key="test")

    with pytest.raises(ValueError, match="cannot be partitioned"):
        client._encode_payload({"series": {}}, False, task="explain")
    with pytest.raises(ValueError, match="num_partitions"):
        client._encode_payload({"series": {}, "multivariate": False}, False, task="simulate")
    with pytest.raises(ValueError, match="cannot be partitioned"):
        client._encode_payload({"series": {}, "multivariate": True}, False, task="simulate")
    with pytest.raises(ValueError, match="num_partitions"):
        client._encode_payload({"series": {}}, False, task="forecast")


def test_status_envelope_is_json_from_the_fixture():
    """Guard the fixture itself: it must speak the documented envelope."""
    api = FakeApi(statuses=("succeeded",), result={"weights": [1.0], "method": "granger"}, task="explain")
    with httpx.Client(transport=api.transport, base_url="http://t") as http:
        job_id = http.post("/v2/explain/async", content=b"{}").json()["job_id"]
        envelope = json.loads(http.get(f"/v2/explain/jobs/{job_id}").content)
    assert set(envelope) == {"job_id", "status", "result", "error", "created_at", "updated_at"}
