"""Live-API tests for `client.jobs`: the submit -> poll -> result loop for real.

`test_jobs.py` is mocked throughout, so it cannot see a server-side rule the client
does not model. Payloads here are kept small and few: the server caps a team's
in-flight jobs and refuses further submits with 429.
"""

import time

import numpy as np
import pandas as pd
import pytest

from nixtla import ApiError, JobCancelledError, JobStatus, JobTimeoutError
from nixtla._steps import ref
from nixtla_tests.helpers.states import model_ids_object

pytestmark = pytest.mark.integration

# The server caps every task at 300s and defaults to it, so only the one test that
# exercises the option passes a budget, just under the cap.
JOB_TIMEOUT = 290

# The listing index is only eventually consistent: it trails submit and completion by
# about a second, and it re-drops rows it has already served, so a job seen in one call
# can be missing from the next. No single `list()` call is a safe assertion about one
# job -- everything here polls and asserts against the snapshot that matched.
LISTING_LAG_TIMEOUT = 60.0

# Caps the walk: `limit` bounds rows fetched, so this is at most three requests per
# poll even with `task=` set. Rows are newest first and the job under test is seconds
# old, so only a handful of concurrent CI jobs can sit in front of it -- but if more
# than 500 newer jobs ever do, raise this rather than dropping the bound.
LISTING_SCAN_LIMIT = 500

# Polling the open-only default would race: a forecast finishes before the index
# settles, so the job would drop out before it could be observed.
LISTING_STATUSES = ["pending", "running", "succeeded"]


@pytest.fixture(scope="module")
def jobs_df():
    """One short daily series: enough to forecast, cheap to upload."""
    n = 120
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "unique_id": "series-0",
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": 10 + np.arange(n) * 0.5 + rng.normal(scale=0.5, size=n),
        }
    )


def _list_until_present(client, job_id, status=LISTING_STATUSES, task=None):
    """Poll the listing until `job_id` appears, and return that whole snapshot.

    The snapshot, not the row: a job can change state between calls, so every
    assertion has to be made against the listing that actually contained it.
    """
    deadline = time.monotonic() + LISTING_LAG_TIMEOUT
    while True:
        rows = client.jobs.list(status=status, task=task, limit=LISTING_SCAN_LIMIT)
        if any(row.job_id == job_id for row in rows):
            return rows
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"{job_id} never appeared in the listing "
                f"(status={status}, task={task}) within {LISTING_LAG_TIMEOUT}s"
            )
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# lifecycle: submit -> poll -> result, one test per task
# ---------------------------------------------------------------------------


def test_forecast_job_runs_to_completion(nixtla_test_client, jobs_df):
    # The one call that names a budget, exercising the option end to end.
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
    # The one task whose result is a bare string, not a dataframe.
    assert isinstance(model_id, str) and model_id
    model_ids_object.model_id2 = model_id  # so the teardown deletes the model

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
    # Binary task: the result carries its own Arrow schema, not a JSON body.
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
    # giving up on the job. No mock covers it; the job must progress between waits.
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
    # The ceiling is server-side only -- the client validates nothing but
    # positivity -- so this has to reach the gateway to fail.
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

    # filterable by task
    forecasts = _list_until_present(nixtla_test_client, job.job_id, task="forecast")
    assert {r.task_name for r in forecasts} == {"forecast"}

    job.wait()

    # filterable by status
    _list_until_present(nixtla_test_client, job.job_id, status=["succeeded"])


def test_listing_paginates_without_repeating_rows(nixtla_test_client):
    # `limit` keeps this cheap however many jobs the team has.
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
    # The raw response, not the frame `jobs.forecast(...).wait()` builds.
    assert isinstance(result, dict)
    assert len(result["mean"]) == 4
    # Same numbers, one labelled and one positional.
    np.testing.assert_allclose(result["mean"], parsed["TimeGPT"].to_numpy())
