"""Live-API coverage for the async job surface.

`test_async_jobs.py`, `test_simulate_explain_async_jobs.py` and
`test_execute_step_job.py` pin the client side of this protocol against mocks:
they prove the right bytes go out and the right objects come back for a
*hypothetical* server. Nothing there proves the real server accepts those
payloads, nor that its responses parse.

These tests close that gap. Each one submits a deliberately tiny job to the
live API and asserts on the shape of what comes back, never on exact values.

Deliberately absent: `submit_simulate_job` and `submit_explain_job`. The
blocking `simulate()` / `explain()` are themselves async jobs -- each is
`_submit_and_wrap_job(...)` followed by `with job: job.wait(...)` -- so
`test_simulate.py::test_simulate_live_endpoint_is_reproducible` and
`test_explain.py::test_explain_live_endpoint_returns_normalized_weights`
already exercise `v2/simulate/async` and `v2/explain/async` end to end. A
submit-flavoured copy here would repeat the same wire path for a second live
job and no extra coverage; the handle-without-waiting surface those methods
add is pure client-side logic, already pinned in
`test_simulate_explain_async_jobs.py`.
"""

import uuid
from http import HTTPStatus

import pytest

from nixtla import (
    ApiError,
    AsyncJobCancelledError,
    AsyncJobTimeoutError,
    JobStatus,
    StepResult,
    ref,
)

pytestmark = pytest.mark.integration

# `Job.wait()` defaults to a status check every 15s with an hour-long budget.
# Every job here is sized to finish in seconds, so poll fast and give up early
# rather than stalling CI on a job that wedged server-side.
POLL = {"poll_interval": 2.0, "poll_timeout": 300.0}

MODEL = "timegpt-2.1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _submit_slow_job(client, ts_data_set1):
    """A job heavy enough to still be running a second after submission.

    The cancellation and resume tests need a job that has not finished yet;
    refitting five cross-validation windows reliably outlasts the moment they
    need, and leaves no server-side artifact to clean up.
    """
    return client.submit_cross_validation_job(
        df=ts_data_set1.train,
        h=ts_data_set1.h,
        freq=ts_data_set1.freq,
        n_windows=5,
        finetune_steps=5,
        refit=True,
        model=MODEL,
    )


def _cancel_accepted(job) -> bool:
    """Request cancellation, tolerating a job that finished first.

    A job can reach a terminal state before the cancel lands; the server then
    answers 404/409 and `Job.cancel()` surfaces that as `ApiError`. Returns
    whether the server accepted the cancellation.
    """
    try:
        job.cancel()
    except ApiError as exc:
        if exc.status_code in (HTTPStatus.NOT_FOUND, HTTPStatus.CONFLICT):
            return False
        raise
    return True


def _skip_if_undeployed(exc: ApiError, route: str) -> None:
    if exc.status_code == HTTPStatus.NOT_FOUND:
        pytest.skip(f"{route} is not deployed on this base URL")
    raise exc


@pytest.fixture
def finetuned_model_id(nixtla_test_client):
    """A fresh model id, deleted afterwards however the test ends.

    Deliberately not `helpers.states.model_ids_object`: that state is shared
    and ordering-dependent, and is owned by `test_finetune_and_forecast.py`.
    """
    model_id = str(uuid.uuid4())
    yield model_id
    try:
        nixtla_test_client.delete_finetuned_model(model_id)
    except Exception:
        print(f"{model_id} not found, skipping deletion.")


# ---------------------------------------------------------------------------
# one happy path per submit_*_job method
# ---------------------------------------------------------------------------


def test_submit_forecast_job_returns_forecasts(nixtla_test_client, ts_data_set1):
    job = nixtla_test_client.submit_forecast_job(
        df=ts_data_set1.train,
        h=ts_data_set1.h,
        freq=ts_data_set1.freq,
        model=MODEL,
    )
    # Submitting caches no status, so this is answered by the server.
    assert isinstance(job.status, JobStatus)

    result = job.wait(**POLL)

    assert job.status is JobStatus.SUCCEEDED
    assert job.result is result
    n_ids = ts_data_set1.train["unique_id"].nunique()
    assert len(result) == n_ids * ts_data_set1.h
    assert result.columns.tolist() == ["unique_id", "ds", "TimeGPT"]
    assert result["TimeGPT"].notna().all()
    assert set(result["unique_id"]) == set(ts_data_set1.train["unique_id"])


def test_submit_cross_validation_job_returns_windows(nixtla_test_client, ts_data_set1):
    n_windows = 2
    job = nixtla_test_client.submit_cross_validation_job(
        df=ts_data_set1.train,
        h=ts_data_set1.h,
        freq=ts_data_set1.freq,
        n_windows=n_windows,
        model=MODEL,
    )
    result = job.wait(**POLL)

    n_ids = ts_data_set1.train["unique_id"].nunique()
    assert len(result) == n_ids * ts_data_set1.h * n_windows
    assert {"unique_id", "ds", "cutoff", "y", "TimeGPT"} <= set(result.columns)
    assert result["cutoff"].nunique() == n_windows
    assert result["TimeGPT"].notna().all()


def test_submit_anomaly_detection_job_returns_online_anomalies(
    nixtla_test_client, anomaly_online_df
):
    """The async route is `v2/anomaly_detection/async` while the blocking
    `detect_anomalies_online()` posts to `v2/online_anomaly_detection` (the
    server is mid-rename, see `nixtla_client.py`). Only a live call can catch
    those two drifting apart, so assert on the online-detection result shape:
    one row per detection step, with the univariate threshold columns.
    """
    df, n_series, detection_size = anomaly_online_df
    job = nixtla_test_client.submit_anomaly_detection_job(
        df=df,
        h=20,
        detection_size=detection_size,
        threshold_method="univariate",
        freq="W-SUN",
        level=99,
        model=MODEL,
    )
    result = job.wait(**POLL)

    assert len(result) == n_series * detection_size
    assert {
        "unique_id",
        "ds",
        "y",
        "TimeGPT",
        "anomaly",
        "anomaly_score",
        "TimeGPT-hi-99",
        "TimeGPT-lo-99",
    } == set(result.columns)
    assert result["anomaly"].dtype == bool


def test_submit_finetune_job_returns_a_usable_model_id(
    nixtla_test_client, ts_data_set1, finetuned_model_id
):
    job = nixtla_test_client.submit_finetune_job(
        df=ts_data_set1.train,
        freq=ts_data_set1.freq,
        finetune_steps=2,
        output_model_id=finetuned_model_id,
        model=MODEL,
    )
    result = job.wait(**POLL)

    assert result == finetuned_model_id
    # The id is only meaningful if the server actually registered the model.
    assert nixtla_test_client.finetuned_model(result).id == finetuned_model_id


def test_submit_execute_step_job_chains_two_steps(nixtla_test_client, ts_data_set1):
    """Two chained TSMP steps, the second fed from the first's `.data`.

    `.data` is passed verbatim rather than `.to_pandas()`: the tables carry
    their resource identity in arrow schema metadata, which a pandas round
    trip would drop.
    """
    # TSMP enforces its dataframe schema server-side and requires a string id;
    # `generate_series` hands out an integer one.
    df = ts_data_set1.train.astype({"unique_id": str})
    try:
        step1 = nixtla_test_client.submit_execute_step_job(
            "make_forecast_input",
            {
                "data": ref("panel"),
                "id_col": "unique_id",
                "time_col": "ds",
                "target_col": "y",
                "freq": ts_data_set1.freq,
            },
            data={"panel": df},
        ).wait(**POLL)
    except ApiError as exc:
        _skip_if_undeployed(exc, "v2/execute_step/async")

    assert isinstance(step1, StepResult)
    assert "result" in step1
    assert step1["result"].num_rows > 0
    assert step1.metadata["func_name"] == "make_forecast_input"

    step2 = nixtla_test_client.submit_execute_step_job(
        "forecast",
        {"resource": ref("result"), "models": [MODEL], "h": ts_data_set1.h},
        data=step1.data,
    ).wait(**POLL)

    assert isinstance(step2, StepResult)
    assert step2["result"].num_rows > 0


# ---------------------------------------------------------------------------
# Job lifecycle against a real server
# ---------------------------------------------------------------------------


def test_job_timeout_seconds_is_accepted_by_the_server(
    nixtla_test_client, ts_data_set1
):
    """`job_timeout_seconds` travels as a `job_options` envelope on the
    request body. A mock accepts any envelope; only the server can say
    whether this one is well-formed.

    The server rejects a value above its own per-task ceiling with a 422, so
    this stays well inside it -- 600 is already refused for forecast.
    """
    job = nixtla_test_client.submit_forecast_job(
        df=ts_data_set1.train,
        h=ts_data_set1.h,
        freq=ts_data_set1.freq,
        model=MODEL,
        job_timeout_seconds=120,
    )
    result = job.wait(**POLL)

    assert len(result) == ts_data_set1.train["unique_id"].nunique() * ts_data_set1.h


def test_job_wait_is_resumable_after_a_short_timeout(nixtla_test_client, ts_data_set1):
    """`cancel_on_timeout=False` exists so a caller can poll in short
    increments and resume. That only means anything against a server-side job
    that is genuinely still running when the first wait gives up.
    """
    job = _submit_slow_job(nixtla_test_client, ts_data_set1)

    with pytest.raises(AsyncJobTimeoutError):
        job.wait(poll_interval=0.0, poll_timeout=1.0, cancel_on_timeout=False)

    result = job.wait(**POLL)

    n_ids = ts_data_set1.train["unique_id"].nunique()
    assert len(result) == n_ids * ts_data_set1.h * 5
    assert job.status is JobStatus.SUCCEEDED


def test_job_can_be_cancelled(nixtla_test_client, ts_data_set1):
    """Covers `POST v2/async/jobs/{job_id}/cancel`, the one route shared by
    every task. `Job.__exit__` reaches the same route through
    `_cancel_job_best_effort`; its extra branching is client-side only and
    pinned in `test_async_jobs.py`, so it is not repeated live here.
    """
    job = _submit_slow_job(nixtla_test_client, ts_data_set1)

    if _cancel_accepted(job):
        # Re-queries the server, which must now report the terminal state.
        with pytest.raises(AsyncJobCancelledError):
            job.wait(**POLL)
    else:
        # It finished before the cancel landed; nothing was left to cancel.
        assert job.status.is_terminal
