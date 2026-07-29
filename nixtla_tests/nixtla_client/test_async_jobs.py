from unittest.mock import MagicMock

import httpx
import orjson
import pandas as pd
import pytest

from nixtla.nixtla_client import (
    ApiError,
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


def test_submit_finetune_job_returns_job(monkeypatch):
    calls = []

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        calls.append(endpoint)
        return "ft-job-1"

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    client = _client()

    job = client.submit_finetune_job(df=_small_df(), freq="D")

    assert isinstance(job, Job)
    assert job.job_id == "ft-job-1"
    assert job.status == "pending"
    assert calls == ["v2/finetune"]


def test_finetune_job_wait_returns_model_id(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        return "ft-job-1"

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout):
        assert (endpoint, job_id, poll_interval, poll_timeout) == (
            "v2/finetune",
            "ft-job-1",
            1,
            2,
        )
        return {"finetuned_model_id": "abc123"}

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_poll_job", fake_poll_job)
    client = _client()

    job = client.submit_finetune_job(df=_small_df(), freq="D")
    result = job.wait(poll_interval=1, poll_timeout=2)

    assert result == "abc123"
    assert job.status == "succeeded"
    assert job.result == "abc123"


def test_submit_forecast_job_returns_job(monkeypatch):
    h = 5
    calls = []

    def fake_get_model_params(self, model, freq):
        return 100, 12

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        calls.append(endpoint)
        return "fc-job-1"

    monkeypatch.setattr(NixtlaClient, "_get_model_params", fake_get_model_params)
    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    client = _client()

    job = client.submit_forecast_job(df=_small_df(), h=h)

    assert isinstance(job, Job)
    assert job.job_id == "fc-job-1"
    assert calls == ["v2/forecast"]


def test_forecast_job_wait_returns_dataframe(monkeypatch):
    h = 5

    def fake_get_model_params(self, model, freq):
        return 100, 12

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        return "fc-job-1"

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout):
        assert endpoint == "v2/forecast"
        return {"mean": list(range(h)), "intervals": None, "weights_x": None}

    monkeypatch.setattr(NixtlaClient, "_get_model_params", fake_get_model_params)
    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_poll_job", fake_poll_job)
    client = _client()

    job = client.submit_forecast_job(df=_small_df(), h=h)
    out = job.wait(poll_interval=1, poll_timeout=2)

    assert len(out) == h
    assert out["TimeGPT"].tolist() == list(range(h))
    assert job.status == "succeeded"
    assert job.result is out


def test_submit_cross_validation_job_returns_job(monkeypatch):
    h = 5
    calls = []

    def fake_get_model_params(self, model, freq):
        return 10_000, 12

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        calls.append(endpoint)
        return "cv-job-1"

    monkeypatch.setattr(NixtlaClient, "_get_model_params", fake_get_model_params)
    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    client = _client()

    job = client.submit_cross_validation_job(df=_small_df(), h=h)

    assert isinstance(job, Job)
    assert job.job_id == "cv-job-1"
    assert calls == ["v2/cross_validation"]


def test_cross_validation_job_wait_returns_dataframe(monkeypatch):
    h = 5
    n = 20

    def fake_get_model_params(self, model, freq):
        return 10_000, 12

    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        return "cv-job-1"

    def fake_poll_job(self, client, endpoint, job_id, poll_interval, poll_timeout):
        assert endpoint == "v2/cross_validation"
        return {
            "idxs": list(range(n - h, n)),
            "sizes": [h],
            "mean": list(range(h)),
            "intervals": None,
        }

    monkeypatch.setattr(NixtlaClient, "_get_model_params", fake_get_model_params)
    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_poll_job", fake_poll_job)
    client = _client()

    job = client.submit_cross_validation_job(df=_small_df(n=n), h=h)
    out = job.wait(poll_interval=1, poll_timeout=2)

    assert len(out) == h
    assert out["TimeGPT"].tolist() == list(range(h))


def test_job_cancel_calls_cancel_job(monkeypatch):
    def fake_submit_job(self, client, endpoint, payload, multithreaded_compress=True):
        return "ft-job-1"

    calls = []

    def fake_cancel_job(self, client, endpoint, job_id):
        calls.append((endpoint, job_id))

    monkeypatch.setattr(NixtlaClient, "_submit_job", fake_submit_job)
    monkeypatch.setattr(NixtlaClient, "_cancel_job", fake_cancel_job)
    client = _client()

    job = client.submit_finetune_job(df=_small_df(), freq="D")
    job.cancel()

    assert calls == [("v2/finetune", "ft-job-1")]
    assert job.status == "cancelled"


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
        client._cancel_job(mock_http_client, "v2/forecast", "fc-abc123")
    mock_http_client.post.assert_called_with("v2/forecast/jobs/fc-abc123/cancel")


def test_cancel_job_raises_on_other_status_codes():
    client = _client()
    mock_http_client = MagicMock()
    resp = MagicMock()
    resp.status_code = 404
    resp.json.return_value = {"detail": "job not found"}
    mock_http_client.post.return_value = resp

    with pytest.raises(ApiError) as excinfo:
        client._cancel_job(mock_http_client, "v2/forecast", "fc-abc123")

    assert excinfo.value.status_code == 404
    assert excinfo.value.body == {"detail": "job not found"}


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


def test_submit_forecast_job_with_unrecognized_df_type_still_raises():
    # submit_forecast_job doesn't support distributed (dask/spark/ray)
    # dataframes in this version — an arbitrary non-pandas/polars object
    # should raise a clear ValueError rather than doing something undefined.
    client = _client()
    with pytest.raises(ValueError, match="submit_forecast_job only supports"):
        client.submit_forecast_job(df=[1, 2, 3], h=5)


def test_submit_cross_validation_job_with_unrecognized_df_type_still_raises():
    client = _client()
    with pytest.raises(ValueError, match="submit_cross_validation_job only supports"):
        client.submit_cross_validation_job(df=[1, 2, 3], h=5)
