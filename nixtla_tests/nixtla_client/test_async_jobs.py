from unittest.mock import MagicMock

import httpx
import orjson
import pandas as pd
import pytest

from nixtla.nixtla_client import (
    ApiError,
    AsyncJobCancelledError,
    AsyncJobError,
    AsyncJobTimeoutError,
    Job,
    NixtlaClient,
)


def _client(**kwargs):
    return NixtlaClient(api_key="dummy", **kwargs)


def _small_df(n=20):
    return pd.DataFrame(
        {
            "unique_id": "id_0",
            "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
            "y": range(n),
        }
    )


def _multi_series_df(n_series=2, n=20):
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"id_{i}",
                    "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                    "y": range(n),
                }
            )
            for i in range(n_series)
        ],
        ignore_index=True,
    )


def _mock_response(status_code, body):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = orjson.dumps(body)
    return resp


def _polling_stubs(statuses, job_id="fc-abc123"):
    """Build fake `_make_request`/`_get_request` replacements simulating an
    async submit followed by the given sequence of job-status bodies. The
    last entry in `statuses` repeats if polled past the end of the list."""
    calls = {"n": 0}

    def fake_make_request(client, endpoint, payload, multithreaded_compress=True):
        assert endpoint.endswith("/async")
        return {"job_id": job_id}

    def fake_get_request(client, endpoint, params=None, timeout=None):
        assert endpoint.endswith(f"/jobs/{job_id}")
        i = min(calls["n"], len(statuses) - 1)
        calls["n"] += 1
        return {"job_id": job_id, **statuses[i]}

    return fake_make_request, fake_get_request, calls


# ---------------------------------------------------------------------------
# _run_async_job: submit + poll mechanics
# ---------------------------------------------------------------------------


def test_run_async_job_success():
    client = _client()
    fake_make_request, fake_get_request, calls = _polling_stubs(
        [
            {"status": "pending"},
            {"status": "running"},
            {"status": "succeeded", "result": {"mean": [1, 2, 3]}},
        ]
    )
    client._make_request = fake_make_request
    client._get_request = fake_get_request

    result = client._run_async_job(
        MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=5
    )

    assert result == {"mean": [1, 2, 3]}
    assert calls["n"] == 3


def test_run_async_job_failed():
    client = _client()
    fake_make_request, fake_get_request, _ = _polling_stubs(
        [{"status": "running"}, {"status": "failed", "error": {"detail": "boom"}}]
    )
    client._make_request = fake_make_request
    client._get_request = fake_get_request

    with pytest.raises(AsyncJobError) as excinfo:
        client._run_async_job(
            MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=5
        )

    assert excinfo.value.job_id == "fc-abc123"
    assert excinfo.value.error == {"detail": "boom"}


def test_run_async_job_cancelled():
    client = _client()
    fake_make_request, fake_get_request, _ = _polling_stubs(
        [{"status": "running"}, {"status": "cancelled"}]
    )
    client._make_request = fake_make_request
    client._get_request = fake_get_request

    with pytest.raises(AsyncJobCancelledError) as excinfo:
        client._run_async_job(
            MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=5
        )

    assert excinfo.value.job_id == "fc-abc123"


def test_run_async_job_unexpected_status():
    client = _client()
    fake_make_request, fake_get_request, _ = _polling_stubs([{"status": "weird"}])
    client._make_request = fake_make_request
    client._get_request = fake_get_request

    with pytest.raises(AsyncJobError, match="unexpected job status"):
        client._run_async_job(
            MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=5
        )


def test_run_async_job_timeout(monkeypatch):
    client = _client()
    fake_make_request, fake_get_request, _ = _polling_stubs([{"status": "running"}])
    client._make_request = fake_make_request
    client._get_request = fake_get_request
    monkeypatch.setattr(NixtlaClient, "_cancel_job", lambda self, client, job_id: None)

    with pytest.raises(AsyncJobTimeoutError) as excinfo:
        client._run_async_job(
            MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=0.05
        )

    assert excinfo.value.job_id == "fc-abc123"


def test_run_async_job_cancels_the_job_on_timeout(monkeypatch):
    """`_run_async_job` never surfaces the job_id, so a client-side timeout must
    cancel the job here or nobody ever can."""
    client = _client()
    fake_make_request, fake_get_request, _ = _polling_stubs([{"status": "running"}])
    client._make_request = fake_make_request
    client._get_request = fake_get_request

    calls = []
    monkeypatch.setattr(
        NixtlaClient, "_cancel_job", lambda self, client, job_id: calls.append(job_id)
    )

    with pytest.raises(AsyncJobTimeoutError):
        client._run_async_job(
            MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=0.05
        )

    assert calls == ["fc-abc123"]


@pytest.mark.parametrize(
    "statuses, expected_error",
    [
        ([{"status": "failed", "error": {"detail": "boom"}}], AsyncJobError),
        ([{"status": "cancelled"}], AsyncJobCancelledError),
    ],
    ids=["failed", "cancelled"],
)
def test_run_async_job_does_not_cancel_on_terminal_states(
    monkeypatch, statuses, expected_error
):
    """Failed/cancelled jobs are already terminal -- cancelling them is pointless."""
    client = _client()
    fake_make_request, fake_get_request, _ = _polling_stubs(statuses)
    client._make_request = fake_make_request
    client._get_request = fake_get_request

    calls = []
    monkeypatch.setattr(
        NixtlaClient, "_cancel_job", lambda self, client, job_id: calls.append(job_id)
    )

    with pytest.raises(expected_error):
        client._run_async_job(
            MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=5
        )

    assert calls == []


def test_run_async_job_timeout_survives_a_failing_cancel(monkeypatch, caplog):
    """A failed cancel must be logged, not raised: it would mask the timeout."""
    client = _client()
    fake_make_request, fake_get_request, _ = _polling_stubs([{"status": "running"}])
    client._make_request = fake_make_request
    client._get_request = fake_get_request

    def fake_cancel_job(self, client, job_id):
        raise ApiError(status_code=500, body={"detail": "boom"})

    monkeypatch.setattr(NixtlaClient, "_cancel_job", fake_cancel_job)

    with caplog.at_level("WARNING"):
        with pytest.raises(AsyncJobTimeoutError) as excinfo:
            client._run_async_job(
                MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=0.05
            )

    assert excinfo.value.job_id == "fc-abc123"
    assert "Failed to cancel job fc-abc123" in caplog.text


def test_run_async_job_fails_fast_on_non_retriable_poll_error():
    """A permanent error (e.g. bad job_id, auth failure) while polling should
    surface immediately, not be retried until poll_timeout elapses."""
    client = _client()
    fake_make_request, _, _ = _polling_stubs([{"status": "running"}])
    calls = {"n": 0}

    def fake_get_request(client, endpoint, params=None, timeout=None):
        calls["n"] += 1
        raise ApiError(status_code=404, body={"detail": "job not found"})

    client._make_request = fake_make_request
    client._get_request = fake_get_request

    with pytest.raises(ApiError) as excinfo:
        client._run_async_job(
            MagicMock(), "v2/forecast", {}, poll_interval=10, poll_timeout=3600
        )

    assert excinfo.value.status_code == 404
    assert calls["n"] == 1


def test_run_async_job_retries_transient_poll_error():
    """A transient network error while polling (not wrapped as ApiError by
    _get_request) should be retried like any other not-yet-terminal poll,
    not crash the whole call."""
    client = _client()
    fake_make_request, _, _ = _polling_stubs([{"status": "running"}])
    responses = iter(
        [
            httpx.ReadTimeout("timed out"),
            {"job_id": "fc-abc123", "status": "succeeded", "result": {"mean": [1, 2, 3]}},
        ]
    )

    def fake_get_request(client, endpoint, params=None, timeout=None):
        resp = next(responses)
        if isinstance(resp, Exception):
            raise resp
        return resp

    client._make_request = fake_make_request
    client._get_request = fake_get_request

    result = client._run_async_job(
        MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=5
    )

    assert result == {"mean": [1, 2, 3]}


def test_make_request_with_retries_retries_on_transient_error():
    client = _client(max_retries=3, retry_interval=0, max_wait_time=10)
    mock_http_client = MagicMock()
    mock_http_client.post.side_effect = [
        _mock_response(503, {"detail": "server error"}),
        _mock_response(202, {"job_id": "fc-abc123"}),
    ]

    resp = client._make_request_with_retries(
        mock_http_client, "v2/forecast/async", {"model": "timegpt-2.1"}
    )

    assert resp == {"job_id": "fc-abc123"}
    assert mock_http_client.post.call_count == 2


def test_make_request_accepts_202():
    client = _client()
    mock_http_client = MagicMock()
    mock_http_client.post.return_value = _mock_response(202, {"job_id": "fc-abc123"})

    resp = client._make_request(
        mock_http_client, "v2/forecast/async", {}, multithreaded_compress=True
    )

    assert resp == {"job_id": "fc-abc123"}


def test_make_request_still_rejects_other_status_codes():
    client = _client()
    mock_http_client = MagicMock()
    mock_http_client.post.return_value = _mock_response(500, {"detail": "oops"})

    with pytest.raises(ApiError) as excinfo:
        client._make_request(
            mock_http_client, "v2/forecast/async", {}, multithreaded_compress=True
        )

    assert excinfo.value.status_code == 500


# ---------------------------------------------------------------------------
# submit_finetune_job / submit_forecast_job / submit_cross_validation_job
# ---------------------------------------------------------------------------

# (method_name, endpoint, call_kwargs factory, _get_model_params return value or
# None if the method doesn't call it)
SUBMIT_JOB_CASES = [
    pytest.param(
        "submit_finetune_job",
        "v2/finetune",
        lambda: {"df": _small_df(), "freq": "D"},
        None,
        id="finetune",
    ),
    pytest.param(
        "submit_forecast_job",
        "v2/forecast",
        lambda: {"df": _small_df(), "h": 5},
        (100, 12),
        id="forecast",
    ),
    pytest.param(
        "submit_cross_validation_job",
        "v2/cross_validation",
        lambda: {"df": _small_df(), "h": 5},
        (10_000, 12),
        id="cross_validation",
    ),
]


def _stub_model_params(monkeypatch, model_params):
    if model_params is not None:
        monkeypatch.setattr(
            NixtlaClient, "_get_model_params", lambda self, model, freq: model_params
        )


def _stub_job_status(monkeypatch, status):
    monkeypatch.setattr(
        NixtlaClient,
        "_get_job_data",
        lambda self, client, endpoint, job_id: {"status": status},
    )


@pytest.mark.parametrize("method_name, endpoint, make_call_kwargs, model_params", SUBMIT_JOB_CASES)
def test_submit_job_returns_job(monkeypatch, method_name, endpoint, make_call_kwargs, model_params):
    calls = []

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        calls.append(endpoint)
        return "job-1"

    _stub_model_params(monkeypatch, model_params)
    _stub_job_status(monkeypatch, "pending")
    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    client = _client()

    job = getattr(client, method_name)(**make_call_kwargs())

    assert isinstance(job, Job)
    assert job.job_id == "job-1"
    assert job.status == "pending"
    assert calls == [endpoint]


def _finetune_poll_response():
    return {"status": "succeeded", "result": {"finetuned_model_id": "abc123"}}


def _forecast_poll_response():
    return {
        "status": "succeeded",
        "result": {"mean": list(range(5)), "intervals": None, "weights_x": None},
    }


def _cross_validation_poll_response():
    n, h = 20, 5
    return {
        "status": "succeeded",
        "result": {
            "idxs": list(range(n - h, n)),
            "sizes": [h],
            "mean": list(range(h)),
            "intervals": None,
        },
    }


def _check_finetune_result(result):
    assert result == "abc123"


def _check_point_forecast_df(result):
    assert len(result) == 5
    assert result["TimeGPT"].tolist() == list(range(5))


WAIT_JOB_CASES = [
    pytest.param(
        "submit_finetune_job",
        lambda: {"df": _small_df(), "freq": "D"},
        None,
        _finetune_poll_response,
        _check_finetune_result,
        id="finetune",
    ),
    pytest.param(
        "submit_forecast_job",
        lambda: {"df": _small_df(), "h": 5},
        (100, 12),
        _forecast_poll_response,
        _check_point_forecast_df,
        id="forecast",
    ),
    pytest.param(
        "submit_cross_validation_job",
        lambda: {"df": _small_df(n=20), "h": 5},
        (10_000, 12),
        _cross_validation_poll_response,
        _check_point_forecast_df,
        id="cross_validation",
    ),
]


@pytest.mark.parametrize(
    "method_name, make_call_kwargs, model_params, poll_response_fn, check_result", WAIT_JOB_CASES
)
def test_submit_job_wait_returns_result(
    monkeypatch, method_name, make_call_kwargs, model_params, poll_response_fn, check_result
):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        return "job-1"

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout, **kwargs):
        return poll_response_fn()

    _stub_model_params(monkeypatch, model_params)
    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_poll_job", fake_poll_job)
    client = _client()

    job = getattr(client, method_name)(**make_call_kwargs())
    result = job.wait(poll_interval=1, poll_timeout=2)

    check_result(result)
    assert job.status == "succeeded"
    assert job.result is result


def test_job_cancel_calls_cancel_job(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        return "ft-job-1"

    calls = []

    def fake_cancel_job(self, client, job_id):
        calls.append(job_id)

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_cancel_job", fake_cancel_job)
    client = _client()

    job = client.submit_finetune_job(df=_small_df(), freq="D")
    job.cancel()

    assert calls == ["ft-job-1"]
    assert job.status == "cancelled"


def test_job_status_queries_server_and_caches_once_terminal(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        return "ft-job-1"

    calls = []
    statuses = iter(["running", "succeeded"])

    def fake_get_job_data(self, client, endpoint, job_id):
        calls.append(job_id)
        return {"status": next(statuses)}

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_get_job_data", fake_get_job_data)
    client = _client()

    job = client.submit_finetune_job(df=_small_df(), freq="D")

    assert job.status == "running"
    assert job.status == "succeeded"
    assert len(calls) == 2  # "running" isn't terminal, so it wasn't cached

    assert job.status == "succeeded"
    assert len(calls) == 2  # terminal status is now cached, no further calls


def test_job_wait_raises_after_cancelled_status(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        return "ft-job-1"

    def fake_cancel_job(self, client, job_id):
        pass

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout, **kwargs):
        raise AsyncJobCancelledError(job_id=job_id)

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_cancel_job", fake_cancel_job)
    monkeypatch.setattr(NixtlaClient, "_poll_job", fake_poll_job)
    client = _client()

    job = client.submit_finetune_job(df=_small_df(), freq="D")
    job.cancel()

    with pytest.raises(AsyncJobCancelledError) as excinfo:
        job.wait(poll_interval=1, poll_timeout=2)

    assert excinfo.value.job_id == "ft-job-1"


def _raise(exc):
    raise exc


def _recording_cancel(calls):
    """A `_cancel_job` stub that records the job ids it was asked to cancel."""

    def fake_cancel_job(self, client, job_id):
        calls.append(job_id)

    return fake_cancel_job


def _timing_out_job(monkeypatch, cancel_job):
    """A submitted `Job` whose polling always times out, with `_cancel_job` stubbed."""
    monkeypatch.setattr(
        NixtlaClient,
        "_submit_job",
        lambda self, client, endpoint, payload, multithreaded_compress=True: "ft-job-1",
    )
    monkeypatch.setattr(
        NixtlaClient,
        "_poll_job",
        lambda self, client, endpoint, job_id, poll_interval, poll_timeout: _raise(
            AsyncJobTimeoutError(job_id=job_id, poll_timeout=poll_timeout)
        ),
    )
    monkeypatch.setattr(NixtlaClient, "_cancel_job", cancel_job)
    return _client().submit_finetune_job(df=_small_df(), freq="D")


def test_job_wait_cancels_on_timeout_by_default(monkeypatch):
    """Giving up on a job should stop it consuming server-side compute."""
    calls = []
    job = _timing_out_job(monkeypatch, _recording_cancel(calls))

    with pytest.raises(AsyncJobTimeoutError):
        job.wait(poll_interval=0, poll_timeout=0.01)

    assert calls == ["ft-job-1"]
    assert job.status == "cancelled"


def test_job_wait_leaves_the_job_running_when_opted_out(monkeypatch):
    """`poll_timeout` bounds local polling only, so `cancel_on_timeout=False`
    supports waiting in short increments and resuming."""
    calls = []
    job = _timing_out_job(monkeypatch, _recording_cancel(calls))

    for _ in range(2):
        with pytest.raises(AsyncJobTimeoutError):
            job.wait(poll_interval=0, poll_timeout=0.01, cancel_on_timeout=False)

    assert calls == []
    assert job._status is None  # still resumable


def test_job_wait_cancels_on_timeout_when_set_explicitly(monkeypatch):
    calls = []
    job = _timing_out_job(monkeypatch, _recording_cancel(calls))

    with pytest.raises(AsyncJobTimeoutError):
        job.wait(poll_interval=0, poll_timeout=0.01, cancel_on_timeout=True)

    assert calls == ["ft-job-1"]
    assert job.status == "cancelled"


def test_job_wait_cancel_on_timeout_leaves_status_unresolved_if_cancel_fails(
    monkeypatch, caplog
):
    """A cancel that the server rejected must not be cached as `cancelled`;
    `status` should go back to the server for the truth."""

    def fake_cancel_job(self, client, job_id):
        raise ApiError(status_code=500, body={"detail": "boom"})

    job = _timing_out_job(monkeypatch, fake_cancel_job)
    monkeypatch.setattr(
        NixtlaClient,
        "_get_job_data",
        lambda self, client, endpoint, job_id: {"status": "running"},
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(AsyncJobTimeoutError):
            job.wait(poll_interval=0, poll_timeout=0.01, cancel_on_timeout=True)

    assert "Failed to cancel job ft-job-1" in caplog.text
    assert job._status is None
    assert job.status == "running"


def test_job_wait_cancel_on_timeout_inside_context_manager_cancels_once(monkeypatch):
    """`wait` marks the status terminal, so `__exit__` must not cancel again."""
    calls = []
    job = _timing_out_job(monkeypatch, _recording_cancel(calls))

    with pytest.raises(AsyncJobTimeoutError):
        with job:
            job.wait(poll_interval=0, poll_timeout=0.01, cancel_on_timeout=True)

    assert calls == ["ft-job-1"]


# ---------------------------------------------------------------------------
# Job as a context manager
# ---------------------------------------------------------------------------


def test_job_context_manager_cancels_on_exception(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        return "ft-job-1"

    calls = []

    def fake_cancel_job(self, client, job_id):
        calls.append(job_id)

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_cancel_job", fake_cancel_job)
    client = _client()

    with pytest.raises(ValueError):
        with client.submit_finetune_job(df=_small_df(), freq="D") as job:
            raise ValueError("boom")

    assert calls == ["ft-job-1"]
    assert job.status == "cancelled"


def test_job_context_manager_no_cancel_on_normal_exit(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        return "ft-job-1"

    calls = []

    def fake_cancel_job(self, client, job_id):
        calls.append(job_id)

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_cancel_job", fake_cancel_job)
    client = _client()

    with client.submit_finetune_job(df=_small_df(), freq="D") as job:
        pass

    assert calls == []
    assert job._status is None


def test_job_context_manager_no_cancel_if_already_terminal(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        return "ft-job-1"

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout, **kwargs):
        return _finetune_poll_response()

    calls = []

    def fake_cancel_job(self, client, job_id):
        calls.append(job_id)

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_poll_job", fake_poll_job)
    monkeypatch.setattr(NixtlaClient, "_cancel_job", fake_cancel_job)
    client = _client()

    with pytest.raises(ValueError):
        with client.submit_finetune_job(df=_small_df(), freq="D") as job:
            job.wait(poll_interval=1, poll_timeout=2)
            raise ValueError("boom")

    assert calls == []
    assert job.status == "succeeded"


def test_job_context_manager_logs_and_swallows_cancel_failure(monkeypatch, caplog):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        return "ft-job-1"

    def fake_cancel_job(self, client, job_id):
        raise ApiError(status_code=500, body={"detail": "boom"})

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_cancel_job", fake_cancel_job)
    client = _client()

    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError, match="original error"):
            with client.submit_finetune_job(df=_small_df(), freq="D") as job:
                raise ValueError("original error")

    assert "Failed to cancel job ft-job-1" in caplog.text
    assert job._status is None


# ---------------------------------------------------------------------------
# _cancel_job
# ---------------------------------------------------------------------------


def test_cancel_job_accepts_terminal_success_codes():
    client = _client()
    mock_http_client = MagicMock()
    for status_code in (200, 202, 204):
        resp = MagicMock()
        resp.status_code = status_code
        mock_http_client.post.return_value = resp
        client._cancel_job(mock_http_client, "fc-abc123")
    mock_http_client.post.assert_called_with("v2/async/jobs/fc-abc123/cancel")


def test_cancel_job_raises_on_other_status_codes():
    client = _client()
    mock_http_client = MagicMock()
    resp = MagicMock()
    resp.status_code = 404
    resp.json.return_value = {"detail": "job not found"}
    mock_http_client.post.return_value = resp

    with pytest.raises(ApiError) as excinfo:
        client._cancel_job(mock_http_client, "fc-abc123")

    assert excinfo.value.status_code == 404
    assert excinfo.value.body == {"detail": "job not found"}


# ---------------------------------------------------------------------------
# job_timeout_seconds -> job_options threading on submit_*_job
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name, endpoint, make_call_kwargs, model_params", SUBMIT_JOB_CASES)
def test_submit_job_threads_job_timeout_seconds(
    monkeypatch, method_name, endpoint, make_call_kwargs, model_params
):
    payloads = []

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        payloads.append(payload)
        return "job-1"

    _stub_model_params(monkeypatch, model_params)
    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    client = _client()

    getattr(client, method_name)(**make_call_kwargs(), job_timeout_seconds=120)
    getattr(client, method_name)(**make_call_kwargs())

    assert payloads[0]["job_options"] == {"timeout_seconds": 120}
    assert "job_options" not in payloads[1]


@pytest.mark.parametrize(
    "method_name, endpoint, make_call_kwargs, model_params", SUBMIT_JOB_CASES
)
@pytest.mark.parametrize("bad_timeout", [0, -1])
def test_submit_job_rejects_a_non_positive_job_timeout(
    monkeypatch, method_name, endpoint, make_call_kwargs, model_params, bad_timeout
):
    """The server refuses these, so the client should not spend a round-trip finding out."""

    def boom(*args, **kwargs):
        raise AssertionError("validation should fail before any HTTP call")

    _stub_model_params(monkeypatch, model_params)
    monkeypatch.setattr(NixtlaClient, "_submit_job", boom)

    with pytest.raises(ValueError, match="job_timeout_seconds must be positive"):
        getattr(_client(), method_name)(
            **make_call_kwargs(), job_timeout_seconds=bad_timeout
        )


# ---------------------------------------------------------------------------
# job_timeout_seconds on the async forecast/cross_validation paths
# ---------------------------------------------------------------------------


def _capture_submitted_payloads(monkeypatch, h=5):
    """Record every payload reaching `_submit_job`, with polling stubbed out.

    The stubbed result is shaped per endpoint because forecast and cross_validation parse their
    responses differently; these tests only assert on what was *sent*, but the call still has to
    return without blowing up in `parse_result`.
    """
    payloads = []

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True, **kwargs):
        payloads.append((endpoint, payload))
        return "job-1"

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout, **kwargs):
        if endpoint == "v2/cross_validation":
            n = len(payloads[-1][1]["series"]["y"])
            result = {
                "idxs": list(range(n - h, n)),
                "sizes": [h],
                "mean": list(range(h)),
                "intervals": None,
            }
        else:
            result = {"mean": list(range(h)), "intervals": None, "weights_x": None}
        return {"status": "succeeded", "result": result}

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_poll_job", fake_poll_job)
    monkeypatch.setattr(
        NixtlaClient, "_get_model_params", lambda self, model, freq: (10_000, 12)
    )
    return payloads


def test_run_async_job_folds_job_options_into_the_payload(monkeypatch):
    payloads = _capture_submitted_payloads(monkeypatch)
    client = _client()

    original = {"idx": 0}
    client._run_async_job(
        MagicMock(), "v2/forecast", original, 0, 1, job_timeout_seconds=300
    )

    assert payloads[0][1]["job_options"] == {"timeout_seconds": 300}
    # The caller's dict is reused to derive other requests, so it must not be mutated.
    assert original == {"idx": 0}


def test_run_async_job_omits_job_options_when_unset(monkeypatch):
    payloads = _capture_submitted_payloads(monkeypatch)
    _client()._run_async_job(MagicMock(), "v2/forecast", {"idx": 0}, 0, 1)
    assert "job_options" not in payloads[0][1]


def test_make_partitioned_requests_forwards_the_job_timeout():
    client = _client()
    seen = []
    events = []

    def fake_run_async_job(
        client,
        endpoint,
        payload,
        poll_interval,
        poll_timeout,
        multithreaded_compress=True,
        job_timeout_seconds=None,
        task=None,
        cancellation_event=None,
    ):
        assert task is None
        assert cancellation_event is not None
        events.append(cancellation_event)
        seen.append(job_timeout_seconds)
        return {"mean": [payload["idx"]], "intervals": None, "weights_x": None}

    client._run_async_job = fake_run_async_job
    client._make_partitioned_requests(
        MagicMock(),
        "v2/forecast",
        [{"idx": i} for i in range(3)],
        _is_async_job=True,
        _poll_interval=1,
        _poll_timeout=2,
        _job_timeout_seconds=300,
    )

    # Per job, not per call: every partition is its own job and gets the full budget.
    assert seen == [300, 300, 300]

    assert len({id(event) for event in events}) == 1


def test_forecast_async_job_carries_the_job_timeout(monkeypatch):
    payloads = _capture_submitted_payloads(monkeypatch)

    _client().forecast(df=_small_df(), h=5, _job_timeout_seconds=300, _is_async_job=True)

    assert payloads[0][1]["job_options"] == {"timeout_seconds": 300}


def test_forecast_add_history_applies_the_timeout_to_both_jobs(monkeypatch):
    payloads = _capture_submitted_payloads(monkeypatch)

    _client().forecast(
        df=_small_df(),
        h=5,
        add_history=True,
        _job_timeout_seconds=300,
        _is_async_job=True,
    )

    # The forecast job and the in-sample cross_validation job are separate jobs.
    assert [endpoint for endpoint, _ in payloads] == [
        "v2/forecast",
        "v2/cross_validation",
    ]
    assert all(p["job_options"] == {"timeout_seconds": 300} for _, p in payloads)


def test_forecast_partitioned_async_job_carries_the_job_timeout(monkeypatch):
    payloads = _capture_submitted_payloads(monkeypatch)

    _client().forecast(
        df=_multi_series_df(n_series=2),
        h=5,
        num_partitions=2,
        _job_timeout_seconds=300,
        _is_async_job=True,
    )

    assert len(payloads) == 2
    assert all(p["job_options"] == {"timeout_seconds": 300} for _, p in payloads)


def test_cross_validation_async_job_carries_the_job_timeout(monkeypatch):
    payloads = _capture_submitted_payloads(monkeypatch)

    _client().cross_validation(
        df=_small_df(), h=5, _job_timeout_seconds=300, _is_async_job=True
    )

    assert payloads[0][1]["job_options"] == {"timeout_seconds": 300}


@pytest.mark.parametrize("method_name", ["forecast", "cross_validation"])
def test_job_timeout_without_async_job_raises(method_name):
    # A synchronous request creates no job, so silently ignoring the value would be worse.
    with pytest.raises(ValueError, match="requires _is_async_job"):
        getattr(_client(), method_name)(df=_small_df(), h=5, _job_timeout_seconds=300)


@pytest.mark.parametrize("method_name", ["forecast", "cross_validation"])
@pytest.mark.parametrize("bad_timeout", [0, -1])
def test_forecast_and_cv_reject_a_non_positive_job_timeout(method_name, bad_timeout):
    with pytest.raises(ValueError, match="job_timeout_seconds must be positive"):
        getattr(_client(), method_name)(
            df=_small_df(), h=5, _job_timeout_seconds=bad_timeout, _is_async_job=True
        )


# ---------------------------------------------------------------------------
# num_partitions + async job fan-out (local pandas/polars DataFrames)
# ---------------------------------------------------------------------------


def test_make_partitioned_requests_dispatches_async_jobs():
    client = _client()
    calls = []

    def fake_run_async_job(
        client,
        endpoint,
        payload,
        poll_interval,
        poll_timeout,
        multithreaded_compress=True,
        job_timeout_seconds=None,
        **kwargs,
    ):
        calls.append((endpoint, poll_interval, poll_timeout, multithreaded_compress))
        return {"mean": [payload["idx"]], "intervals": None, "weights_x": None}

    client._run_async_job = fake_run_async_job

    payloads = [{"idx": i} for i in range(3)]
    resp = client._make_partitioned_requests(
        MagicMock(),
        "v2/forecast",
        payloads,
        _is_async_job=True,
        _poll_interval=1,
        _poll_timeout=2,
    )

    assert len(calls) == 3
    assert all(c == ("v2/forecast", 1, 2, False) for c in calls)
    assert sorted(resp["mean"].tolist()) == [0, 1, 2]
    assert resp["intervals"] is None
    assert resp["weights_x"] is None


def test_make_partitioned_requests_propagates_async_job_error():
    client = _client()

    def fake_run_async_job(
        client,
        endpoint,
        payload,
        poll_interval,
        poll_timeout,
        multithreaded_compress=True,
        job_timeout_seconds=None,
        **kwargs,
    ):
        if payload["idx"] == 1:
            raise AsyncJobError(job_id="fc-bad", error="boom")
        return {"mean": [0], "intervals": None, "weights_x": None}

    client._run_async_job = fake_run_async_job

    payloads = [{"idx": i} for i in range(3)]
    with pytest.raises(AsyncJobError):
        client._make_partitioned_requests(
            MagicMock(),
            "v2/forecast",
            payloads,
            _is_async_job=True,
            _poll_interval=0,
            _poll_timeout=1,
        )


def test_forecast_num_partitions_with_async_job(monkeypatch):
    h = 5
    calls = []

    def fake_get_model_params(self, model, freq):
        return 100, 12

    def fake_run_async_job(
        self,
        client,
        endpoint,
        payload,
        poll_interval,
        poll_timeout,
        multithreaded_compress=True,
        job_timeout_seconds=None,
        **kwargs,
    ):
        calls.append(endpoint)
        return {"mean": list(range(h)), "intervals": None, "weights_x": None}

    monkeypatch.setattr(NixtlaClient, "_get_model_params", fake_get_model_params)
    monkeypatch.setattr(NixtlaClient, "_run_async_job", fake_run_async_job)
    client = _client()

    out = client.forecast(
        df=_multi_series_df(n_series=2),
        h=h,
        num_partitions=2,
        _is_async_job=True,
        _poll_interval=1,
        _poll_timeout=2,
    )

    assert calls == ["v2/forecast", "v2/forecast"]
    assert len(out) == h * 2


def test_cross_validation_num_partitions_with_async_job(monkeypatch):
    h = 5
    calls = []

    def fake_get_model_params(self, model, freq):
        return 10_000, 12

    def fake_run_async_job(
        self,
        client,
        endpoint,
        payload,
        poll_interval,
        poll_timeout,
        multithreaded_compress=True,
        job_timeout_seconds=None,
        **kwargs,
    ):
        calls.append(endpoint)
        n = len(payload["series"]["y"])
        return {
            "idxs": list(range(n - h, n)),
            "sizes": [h],
            "mean": list(range(h)),
            "intervals": None,
        }

    monkeypatch.setattr(NixtlaClient, "_get_model_params", fake_get_model_params)
    monkeypatch.setattr(NixtlaClient, "_run_async_job", fake_run_async_job)
    client = _client()

    out = client.cross_validation(
        df=_multi_series_df(n_series=2),
        h=h,
        num_partitions=2,
        _is_async_job=True,
        _poll_interval=1,
        _poll_timeout=2,
    )

    assert calls == ["v2/cross_validation", "v2/cross_validation"]
    assert len(out) == h * 2


@pytest.mark.parametrize("method_name", ["submit_forecast_job", "submit_cross_validation_job"])
def test_submit_job_with_unrecognized_df_type_still_raises(method_name):
    # submit_forecast_job/submit_cross_validation_job don't support distributed
    # (dask/spark/ray) dataframes in this version — an arbitrary non-pandas/polars
    # object should raise a clear ValueError rather than doing something undefined.
    client = _client()
    with pytest.raises(ValueError, match=f"{method_name} only supports"):
        getattr(client, method_name)(df=[1, 2, 3], h=5)


@pytest.mark.parametrize(
    "task", ["forecast", "finetune", "cross_validation", "simulate", "explain"]
)
@pytest.mark.parametrize("failure", ["read_timeout", "gateway_error"])
def test_submit_job_does_not_retry_ambiguous_failures(task, failure):
    client = _client(max_retries=3, retry_interval=0)
    requests = []

    def handle(request):
        requests.append(request)
        if failure == "read_timeout":
            raise httpx.ReadTimeout("response lost")
        return httpx.Response(503, json={"detail": "upstream unavailable"})

    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as http:
        with pytest.raises((httpx.ReadTimeout, ApiError)):
            client._submit_job(http, f"v2/{task}", {})

    assert len(requests) == 1
    assert requests[0].url.path == f"/v2/{task}/async"


@pytest.mark.parametrize(
    "task", ["forecast", "finetune", "cross_validation", "simulate", "explain"]
)
@pytest.mark.parametrize("failure", ["connect_error", "rate_limit"])
def test_submit_job_retries_failures_before_acceptance(task, failure):
    client = _client(max_retries=3, retry_interval=0)
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "connect_error":
                raise httpx.ConnectError("connection failed")
            return httpx.Response(
                429, json={"detail": "job cap"}, headers={"retry-after": "0"}
            )
        return httpx.Response(202, json={"data": {"job_id": "opaque-id"}})

    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as http:
        assert client._submit_job(http, f"v2/{task}", {}) == "opaque-id"

    assert len(requests) == 2


@pytest.mark.parametrize(
    "http_timeout,expected", [(None, [5, 3]), (1, [1, 1]), (10, [5, 3])]
)
def test_poll_job_bounds_http_requests_to_remaining_timeout(
    monkeypatch, http_timeout, expected
):
    import nixtla.nixtla_client as client_module

    now = [0.0]
    timeouts = []
    waits = []
    monkeypatch.setattr(client_module.time, "monotonic", lambda: now[0])

    def wait(seconds, cancellation_event):
        waits.append(seconds)
        now[0] += seconds
        return False

    monkeypatch.setattr(client_module, "_wait_for_poll", wait)

    def handle(request):
        timeouts.append(request.extensions["timeout"]["read"])
        if len(timeouts) == 1:
            return httpx.Response(200, json={"status": "running"})
        return httpx.Response(
            200, json={"status": "succeeded", "result": {"mean": [1]}}
        )

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        base_url="http://test",
        timeout=http_timeout,
    ) as http:
        result = _client()._poll_job(http, "v2/forecast", "job-1", 2, 5)

    assert result["result"] == {"mean": [1]}
    assert timeouts == expected
    assert waits == [2]


def test_job_wait_can_resume_after_actual_poll_timeout(monkeypatch):
    import nixtla.nixtla_client as client_module

    now = [0.0]
    requests = []
    monkeypatch.setattr(client_module.time, "monotonic", lambda: now[0])

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            now[0] += 2
            return httpx.Response(200, json={"status": "running"})
        return httpx.Response(
            200, json={"status": "succeeded", "result": {"mean": [1]}}
        )

    client = _client()
    client._make_client = lambda **kwargs: httpx.Client(
        transport=httpx.MockTransport(handle), **kwargs
    )
    job = Job(
        client=client,
        job_id="job-1",
        endpoint="v2/forecast",
        get_result=lambda job_data, *_: job_data["result"],
    )

    with pytest.raises(AsyncJobTimeoutError):
        job.wait(poll_timeout=1, cancel_on_timeout=False)
    assert job._status is None
    assert job.wait(poll_timeout=1) == {"mean": [1]}
    assert all(request.method == "GET" for request in requests)


@pytest.mark.parametrize("task", ["forecast", "cross_validation", "simulate"])
@pytest.mark.parametrize("sibling_state", ["running", "submit_retry", "poll_retry"])
def test_partition_failure_cancels_siblings_and_stops_queued_submissions(
    monkeypatch, task, sibling_state
):
    import nixtla.nixtla_client as client_module

    monkeypatch.setattr(client_module, "_MAX_CONCURRENT_ASYNC_JOBS", 2)
    sibling_started = client_module.Event()
    submitted = []
    cancelled = []
    sibling_polls = []
    wait_outcomes = []

    def bounded_wait(seconds, cancellation_event):
        signalled = cancellation_event is not None and cancellation_event.wait(1)
        wait_outcomes.append(signalled)
        # Always release the worker, then assert it observed cancellation.
        # A broken implementation must fail quickly rather than time out minutes later.
        return True

    monkeypatch.setattr(client_module, "_wait_for_poll", bounded_wait)

    def handle(request):
        if request.url.path.endswith("/async"):
            position = orjson.loads(request.content)["position"]
            submitted.append(position)
            if position == 1 and sibling_state == "submit_retry":
                sibling_started.set()
                return httpx.Response(
                    429, json={"detail": "job cap"}, headers={"retry-after": "30"}
                )
            return httpx.Response(202, json={"job_id": f"job-{position}"})
        if request.url.path.endswith("/cancel"):
            cancelled.append(request.url.path.split("/")[-2])
            return httpx.Response(202, json={})
        if request.url.path.endswith("/jobs/job-0"):
            assert sibling_started.wait(2)
            return httpx.Response(200, json={"status": "failed", "error": "boom"})
        sibling_polls.append(request.url.path)
        sibling_started.set()
        if sibling_state == "poll_retry":
            return httpx.Response(503, json={"detail": "temporarily unavailable"})
        return httpx.Response(200, json={"status": "running"})

    client = _client(retry_interval=30)
    payloads = [{"position": i} for i in range(4)]
    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as http:
        with pytest.raises(AsyncJobError) as excinfo:
            if task == "simulate":
                client._make_partitioned_simulate_requests(
                    http, payloads, n_paths=1, h=1
                )
            else:
                client._make_partitioned_requests(
                    http,
                    f"v2/{task}",
                    payloads,
                    _is_async_job=True,
                    _poll_interval=30,
                    _poll_timeout=120,
                )

    assert excinfo.value.job_id == "job-0"
    assert excinfo.value.error == "boom"
    assert sorted(submitted) == [0, 1]
    assert cancelled == ([] if sibling_state == "submit_retry" else ["job-1"])

    assert wait_outcomes == [True]
    assert len(sibling_polls) == (0 if sibling_state == "submit_retry" else 1)


@pytest.mark.parametrize("status_code", [401, 404])
def test_job_wait_preserves_permanent_errors_after_deadline(monkeypatch, status_code):
    import nixtla.nixtla_client as client_module

    now = [0.0]
    original = ApiError(status_code=status_code, body={"detail": "original error"})
    client = _client()
    client._cancel_job_best_effort = MagicMock()
    monkeypatch.setattr(client_module.time, "monotonic", lambda: now[0])

    def fail(*args, **kwargs):
        now[0] = 2.0
        raise original

    client._get_job_data = fail
    job = Job(
        client=client,
        job_id="job-1",
        endpoint="v2/forecast",
        get_result=lambda *_: None,
    )
    with pytest.raises(ApiError) as excinfo:
        job.wait(poll_timeout=1)
    assert excinfo.value is original
    client._cancel_job_best_effort.assert_not_called()


@pytest.mark.parametrize("task", [None, "simulate", "explain"])
@pytest.mark.parametrize(
    "failure",
    ["retries_exhausted", "unknown_status", "no_result", "failed", "cancelled"],
)
def test_async_runner_cleans_up_only_jobs_with_unknown_terminal_state(task, failure):
    client = _client(max_retries=2, retry_interval=0)
    client._submit_job = MagicMock(return_value="job-1")
    client._cancel_job_best_effort = MagicMock()
    server_error = {"detail": "server error"}
    if failure == "retries_exhausted":
        original = ApiError(status_code=503, body=server_error)
        client._get_job_data = MagicMock(side_effect=original)
        expected = ApiError
    else:
        status = {"unknown_status": "unknown", "no_result": "succeeded"}.get(
            failure, failure
        )
        client._get_job_data = MagicMock(
            return_value={"status": status, "error": server_error}
        )
        expected = AsyncJobCancelledError if failure == "cancelled" else AsyncJobError
    # Finite retry settings keep this test short for all policies.
    if task is None and failure == "retries_exhausted":
        # Legacy polling retries until the deadline. Use a permanent transport
        # failure here; exhausted retries are exercised by the two new tasks.
        original = ApiError(status_code=401, body=server_error)
        client._get_job_data.side_effect = original

    with pytest.raises(expected) as excinfo:
        client._run_async_job(
            MagicMock(),
            "v2/forecast" if task is None else f"v2/{task}",
            {},
            0,
            1,
            task=task,
        )
    if failure in ("retries_exhausted", "unknown_status"):
        client._cancel_job_best_effort.assert_called_once()
    else:
        client._cancel_job_best_effort.assert_not_called()
    if failure == "failed":
        assert excinfo.value.error is server_error
        assert excinfo.value.status == "failed"
    if failure == "retries_exhausted":
        assert excinfo.value is original


@pytest.mark.parametrize(
    "value", [None, float("nan"), float("inf"), -float("inf"), 0, -1, True, "1"]
)
def test_job_wait_rejects_invalid_timeout_before_http(value):
    client = _client()
    client._make_client = MagicMock()
    job = Job(
        client=client,
        job_id="job-1",
        endpoint="v2/forecast",
        get_result=lambda *_: None,
    )
    with pytest.raises(ValueError, match="poll_timeout"):
        job.wait(poll_timeout=value)
    client._make_client.assert_not_called()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True, "1"])
def test_job_wait_rejects_invalid_interval_before_http(value):
    client = _client()
    client._make_client = MagicMock()
    job = Job(
        client=client,
        job_id="job-1",
        endpoint="v2/forecast",
        get_result=lambda *_: None,
    )
    with pytest.raises(ValueError, match="poll_interval"):
        job.wait(poll_interval=value)
    client._make_client.assert_not_called()


def test_legacy_polling_rejects_an_unbounded_timeout():
    client = _client()
    client._get_job_data = MagicMock()
    with pytest.raises(ValueError, match="poll_timeout"):
        client._poll_job(MagicMock(), "v2/forecast", "job-1", 0, None)
    client._get_job_data.assert_not_called()


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("retry_after", ["86400", "NaN", "Infinity"])
def test_submit_retry_budget_bounds_waits_and_prevents_late_resubmission(
    monkeypatch, binary, retry_after
):
    import nixtla.nixtla_client as client_module

    now = [0.0]
    waits = []
    submitted = []
    monkeypatch.setattr(client_module.time, "monotonic", lambda: now[0])

    def wait(seconds, cancellation_event):
        waits.append(seconds)
        now[0] += seconds
        return False

    monkeypatch.setattr(client_module, "_wait_for_poll", wait)

    def handle(request):
        submitted.append(request)
        now[0] += 1.0  # The first attempt consumes part of the retry budget.
        return httpx.Response(
            429, json={"detail": "job cap"}, headers={"Retry-After": retry_after}
        )

    client = _client(max_retries=3, retry_interval=30, max_wait_time=5)
    client._make_client = lambda **kwargs: httpx.Client(
        transport=httpx.MockTransport(handle), **kwargs
    )
    with pytest.raises(ApiError) as excinfo:
        if binary:
            client._submit_and_wrap_binary_job("v2/execute_step", "{}", b"payload")
        else:
            with client._make_client(**client._client_kwargs) as http:
                client._submit_job(http, "v2/forecast", {})
    assert excinfo.value.status_code == 429
    assert waits == [4.0]
    assert now[0] == 5.0
    assert len(submitted) == 1


@pytest.mark.parametrize("failure", ["read_timeout", "gateway", "connection"])
def test_binary_submission_uses_safe_retry_policy(failure):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "read_timeout":
                raise httpx.ReadTimeout("response lost")
            if failure == "connection":
                raise httpx.ConnectError("connection failed")
            return httpx.Response(503, json={"detail": "upstream unavailable"})
        return httpx.Response(202, json={"job_id": "es-1"})

    client = _client(max_retries=3, retry_interval=0)
    client._make_client = lambda **kwargs: httpx.Client(
        transport=httpx.MockTransport(handle), **kwargs
    )
    if failure == "connection":
        assert (
            client._submit_and_wrap_binary_job(
                "v2/execute_step", "{}", b"payload"
            ).job_id
            == "es-1"
        )
        assert len(requests) == 2
    else:
        with pytest.raises((ApiError, httpx.ReadTimeout)):
            client._submit_and_wrap_binary_job("v2/execute_step", "{}", b"payload")
        assert len(requests) == 1


def test_collecting_cancelled_future_first_preserves_original_failure(monkeypatch):
    import nixtla.nixtla_client as client_module

    collected = []

    def cancelled_first(futures):
        for future in reversed(list(futures)):
            collected.append(future)
            yield future

    monkeypatch.setattr(client_module, "as_completed", cancelled_first)
    original = RuntimeError("first partition failed")
    queued = MagicMock(return_value={})
    with pytest.raises(RuntimeError) as excinfo:
        _client()._collect_concurrent_results(
            [lambda: _raise(original), queued],
            max_workers=1,
            cancellation_event=client_module.Event(),
        )
    assert excinfo.value is original
    assert isinstance(collected[0].exception(), client_module.CancelledError)
    queued.assert_not_called()


@pytest.mark.parametrize("status_code", [404, 409])
def test_job_wait_does_not_cache_cancelled_after_noop_cleanup(
    monkeypatch, caplog, status_code
):
    def cancel(*args):
        raise ApiError(status_code=status_code, body={"detail": "not cancellable"})

    job = _timing_out_job(monkeypatch, cancel)
    monkeypatch.setattr(
        NixtlaClient, "_get_job_data", lambda *args: {"status": "succeeded"}
    )
    caplog.clear()
    with caplog.at_level("WARNING"):
        with pytest.raises(AsyncJobTimeoutError):
            job.wait(poll_timeout=1)
    assert job._status is None
    assert caplog.records == []
    assert job.status == "succeeded"


def test_transient_poll_failures_warn_once_per_job(caplog):
    import logging

    client = _client()
    responses = iter(
        [
            ApiError(status_code=503, body="temporary"),
            {"status": "running"},
            ApiError(status_code=503, body="temporary"),
            {"status": "succeeded", "result": {}},
        ]
    )

    def get(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    client._get_job_data = get
    with caplog.at_level("DEBUG", logger="nixtla.nixtla_client"):
        client._poll_job(MagicMock(), "v2/forecast", "job-1", 0, 1)
    levels = [
        record.levelno
        for record in caplog.records
        if record.message.startswith("Polling attempt")
    ]
    assert levels == [logging.WARNING, logging.DEBUG]
