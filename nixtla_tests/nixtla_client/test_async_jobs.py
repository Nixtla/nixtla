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

    def fake_get_request(client, endpoint, params=None):
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


def test_run_async_job_timeout():
    client = _client()
    fake_make_request, fake_get_request, _ = _polling_stubs([{"status": "running"}])
    client._make_request = fake_make_request
    client._get_request = fake_get_request

    with pytest.raises(AsyncJobTimeoutError) as excinfo:
        client._run_async_job(
            MagicMock(), "v2/forecast", {}, poll_interval=0, poll_timeout=0.05
        )

    assert excinfo.value.job_id == "fc-abc123"


def test_run_async_job_fails_fast_on_non_retriable_poll_error():
    """A permanent error (e.g. bad job_id, auth failure) while polling should
    surface immediately, not be retried until poll_timeout elapses."""
    client = _client()
    fake_make_request, _, _ = _polling_stubs([{"status": "running"}])
    calls = {"n": 0}

    def fake_get_request(client, endpoint, params=None):
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

    def fake_get_request(client, endpoint, params=None):
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


def test_run_async_job_submit_retries_on_transient_error():
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

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
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
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        return "job-1"

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout):
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
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
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
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
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
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        return "ft-job-1"

    def fake_cancel_job(self, client, job_id):
        pass

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout):
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


# ---------------------------------------------------------------------------
# Job as a context manager
# ---------------------------------------------------------------------------


def test_job_context_manager_cancels_on_exception(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
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
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
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
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        return "ft-job-1"

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout):
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
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
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

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        payloads.append(payload)
        return "job-1"

    _stub_model_params(monkeypatch, model_params)
    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    client = _client()

    getattr(client, method_name)(**make_call_kwargs(), job_timeout_seconds=120)
    getattr(client, method_name)(**make_call_kwargs())

    assert payloads[0]["job_options"] == {"timeout_seconds": 120}
    assert "job_options" not in payloads[1]


# ---------------------------------------------------------------------------
# num_partitions + async job fan-out (local pandas/polars DataFrames)
# ---------------------------------------------------------------------------


def test_make_partitioned_requests_dispatches_async_jobs():
    client = _client()
    calls = []

    def fake_run_async_job(
        client, endpoint, payload, poll_interval, poll_timeout, multithreaded_compress=True
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
        client, endpoint, payload, poll_interval, poll_timeout, multithreaded_compress=True
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
        self, client, endpoint, payload, poll_interval, poll_timeout, multithreaded_compress=True
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
        self, client, endpoint, payload, poll_interval, poll_timeout, multithreaded_compress=True
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
