"""Live-API tests for `client.jobs`: the submit -> poll -> result loop for real.

Everything in `test_jobs.py` is mocked, which pins the client's own logic but cannot
see a server-side rule the client does not model. Both constraints these tests
encode -- the per-task ceiling on `job_timeout_seconds`, and TSMP's insistence on a
string id column -- were found by running against a live gateway, not by the mocks.

Payloads are kept deliberately small and few: the server caps a team's in-flight
jobs and refuses further submits with 429.
"""

import time

import numpy as np
import pandas as pd
import pytest

from nixtla import ApiError, JobCancelledError, JobStatus, JobTimeoutError
from nixtla._steps import ref
from nixtla_tests.helpers.states import model_ids_object

pytestmark = pytest.mark.integration

# Every task's server-side ceiling is 300s, and that is also the default a job gets
# when no budget is named. Only the one test that exercises the option passes it, with
# enough headroom that a cold sandbox start does not eat the margin; asking for less
# elsewhere buys nothing and makes the suite flaky.
JOB_TIMEOUT = 290

# The listing is served from an index that is only eventually consistent with the jobs
# themselves -- it trails a submit by about a second, and trails a job reaching a
# terminal state by about as much. So no single `list()` call is ever a safe assertion
# about one job: everything here polls, and asserts against the snapshot that matched.
LISTING_LAG_TIMEOUT = 60.0

# Bound on how far down the listing to read. Rows are newest first and the job under
# test was created seconds ago, so only jobs created after it sit in front -- a few
# from each concurrent CI matrix leg. This caps the walk without risking a truncation
# that hides the job.
LISTING_SCAN_LIMIT = 500

# Statuses a job under test can legitimately be in once it has been submitted. Polling
# the open-only default would race: a forecast finishes in well under the time the
# index takes to settle, and the job would drop out before it could be observed.
LISTING_STATUSES = ["pending", "running", "succeeded"]


@pytest.fixture(scope="module")
def jobs_df():
    """One short, clean daily series -- enough to forecast, cheap to upload."""
    n = 120
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "unique_id": "series-0",
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": 10 + np.arange(n) * 0.5 + rng.normal(scale=0.5, size=n),
        }
    )


def _list_until_present(client, job_id, status=LISTING_STATUSES):
    """Poll the listing until `job_id` appears, and return that whole snapshot.

    Returning the snapshot rather than just the row matters: a job can change state
    between two calls, so every assertion about it has to be made against the one
    listing that actually contained it.

    No `task` filter here on purpose -- `task` is applied client-side, so `limit` would
    count matching rows and stop bounding how much of the listing gets walked.
    """
    deadline = time.monotonic() + LISTING_LAG_TIMEOUT
    while True:
        rows = client.jobs.list(status=status, limit=LISTING_SCAN_LIMIT)
        if any(row.job_id == job_id for row in rows):
            return rows
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"{job_id} never appeared in the listing (status={status}) "
                f"within {LISTING_LAG_TIMEOUT}s"
            )
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# lifecycle: submit -> poll -> result, one test per task
# ---------------------------------------------------------------------------


def test_forecast_job_runs_to_completion(nixtla_test_client, jobs_df):
    # The one call that names a budget, so the option is exercised end to end.
    job = nixtla_test_client.jobs.forecast(
        jobs_df, h=4, freq="D", job_timeout_seconds=JOB_TIMEOUT
    )

    assert job.job_id.startswith("fc-")
    assert job.task == "forecast"

    out = job.wait()

    assert job.status == JobStatus.SUCCEEDED
    assert list(out.columns) == ["unique_id", "ds", "TimeGPT"]
    assert len(out) == 4
    assert out["unique_id"].eq("series-0").all()
    assert out["TimeGPT"].notna().all()


def test_cross_validation_job_runs_to_completion(nixtla_test_client, jobs_df):
    job = nixtla_test_client.jobs.cross_validation(jobs_df, h=4, freq="D", n_windows=2)

    assert job.job_id.startswith("cv-")
    out = job.wait()

    assert job.status == JobStatus.SUCCEEDED
    assert {"unique_id", "ds", "cutoff", "y", "TimeGPT"} <= set(out.columns)
    # Two windows of four steps, for the one series.
    assert len(out) == 8
    assert out["cutoff"].nunique() == 2


def test_detect_anomalies_job_runs_to_completion(nixtla_test_client, jobs_df):
    job = nixtla_test_client.jobs.detect_anomalies(
        jobs_df,
        h=4,
        detection_size=10,
        freq="D",
    )

    assert job.job_id.startswith("ad-")
    out = job.wait()

    assert job.status == JobStatus.SUCCEEDED
    assert {"unique_id", "ds", "y", "TimeGPT", "anomaly"} <= set(out.columns)
    assert out["anomaly"].isin([True, False]).all()


def test_finetune_job_returns_a_model_id(nixtla_test_client, jobs_df):
    job = nixtla_test_client.jobs.finetune(jobs_df, freq="D", finetune_steps=1)

    assert job.job_id.startswith("ft-")
    model_id = job.wait()

    assert job.status == JobStatus.SUCCEEDED
    # Not a dataframe: this is the one task whose result is a bare string.
    assert isinstance(model_id, str) and model_id
    # Hand it to the `nixtla_test_client` teardown so the model is deleted.
    model_ids_object.model_id2 = model_id

    # Recovered by id, the same job yields the raw payload the string came out of.
    raw = nixtla_test_client.jobs.retrieve(job.job_id).wait()
    assert raw["finetuned_model_id"] == model_id


def test_execute_step_job_returns_tables(nixtla_test_client, jobs_df):
    # TSMP validates the frame's schema server-side and requires a string id column;
    # a numeric `unique_id` fails the job rather than the request.
    df = jobs_df.assign(unique_id=jobs_df["unique_id"].astype(str))

    job = nixtla_test_client.jobs.execute_step(
        "make_forecast_input",
        {
            "data": ref("panel"),
            "id_col": "unique_id",
            "time_col": "ds",
            "target_col": "y",
            "freq": "D",
        },
        data={"panel": df},
    )

    assert job.job_id.startswith("es-")
    result = job.wait()

    assert job.status == JobStatus.SUCCEEDED
    # Binary task: the result carries its own Arrow schema rather than a JSON body.
    assert result.data
    assert result["result"].num_rows > 0


# ---------------------------------------------------------------------------
# cancellation, timeout and cleanup against a real polling loop
# ---------------------------------------------------------------------------


def test_cancelling_a_job_stops_it(nixtla_test_client, jobs_df):
    job = nixtla_test_client.jobs.forecast(jobs_df, h=4, freq="D")

    job.cancel()

    assert job.status == JobStatus.CANCELLED
    with pytest.raises(JobCancelledError):
        job.wait()


def test_poll_timeout_leaves_the_job_running_and_wait_resumes_it(
    nixtla_test_client, jobs_df
):
    # `cancel_on_timeout=False` is the resumable path: giving up on waiting is not
    # the same as giving up on the job. No mock covers this, because the job has to
    # keep making progress between the two waits.
    job = nixtla_test_client.jobs.forecast(jobs_df, h=4, freq="D")

    with pytest.raises(JobTimeoutError):
        job.wait(poll_timeout=0.001, cancel_on_timeout=False)

    assert job.status != JobStatus.CANCELLED

    out = job.wait()

    assert job.status == JobStatus.SUCCEEDED
    assert len(out) == 4


def test_context_manager_cancels_on_an_exception(nixtla_test_client, jobs_df):
    job = nixtla_test_client.jobs.forecast(jobs_df, h=4, freq="D")

    with pytest.raises(RuntimeError, match="boom"):
        with job:
            raise RuntimeError("boom")

    assert job.status == JobStatus.CANCELLED


def test_job_timeout_seconds_over_the_server_cap_is_refused(
    nixtla_test_client, jobs_df
):
    # The ceiling is per-task and server-side only; the client validates nothing but
    # positivity, so this has to reach the gateway to fail.
    with pytest.raises(ApiError) as excinfo:
        nixtla_test_client.jobs.forecast(
            jobs_df, h=4, freq="D", job_timeout_seconds=600
        )

    assert excinfo.value.status_code == 422


# ---------------------------------------------------------------------------
# recovering a job: `list()` and `retrieve()`
# ---------------------------------------------------------------------------


def test_a_submitted_job_is_listed_and_filterable(nixtla_test_client, jobs_df):
    job = nixtla_test_client.jobs.forecast(jobs_df, h=4, freq="D")

    rows = _list_until_present(nixtla_test_client, job.job_id)

    row = next(r for r in rows if r.job_id == job.job_id)
    assert row.task_name == "forecast"
    assert row.created_at
    assert row.status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED)

    # `task` narrows the same view. Safe to assert membership in a second call now:
    # the job is already known to be listed under one of `LISTING_STATUSES`, and it
    # cannot leave that set.
    forecasts = nixtla_test_client.jobs.list(
        status=LISTING_STATUSES, task="forecast", limit=LISTING_SCAN_LIMIT
    )
    assert {r.task_name for r in forecasts} == {"forecast"}
    assert job.job_id in {r.job_id for r in forecasts}

    job.wait()

    # The succeeded view has to be polled too: the index trails a job's terminal
    # transition just as it trails its submit, so it is not there the instant
    # `wait()` returns.
    _list_until_present(nixtla_test_client, job.job_id, status=["succeeded"])


def test_listing_paginates_without_repeating_rows(nixtla_test_client):
    # `limit` caps the walk so this stays cheap however many jobs the team has.
    rows = nixtla_test_client.jobs.list(
        status=["pending", "running", "succeeded"], limit=25
    )

    ids = [r.job_id for r in rows]
    assert len(ids) == len(set(ids))
    assert len(ids) <= 25


def test_retrieve_recovers_a_job_and_returns_the_raw_result(
    nixtla_test_client, jobs_df
):
    job = nixtla_test_client.jobs.forecast(jobs_df, h=4, freq="D")
    parsed = job.wait()

    # Everything below goes through the id alone, as a fresh process would.
    recovered = nixtla_test_client.jobs.retrieve(job.job_id)
    assert recovered.task == "forecast"

    result = recovered.wait()

    assert recovered.status == JobStatus.SUCCEEDED
    # The raw response, not the frame `jobs.forecast(...).wait()` builds: the series
    # ids and column names never reached the server, so they cannot come back.
    assert isinstance(result, dict)
    assert len(result["mean"]) == 4
    # Same numbers, one labelled and one positional -- which is exactly what a caller
    # holding their input frame needs in order to put the labels back on.
    np.testing.assert_allclose(result["mean"], parsed["TimeGPT"].to_numpy())
