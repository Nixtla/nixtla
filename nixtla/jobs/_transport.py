"""Transport for server-side asynchronous jobs: submit, poll, cancel, fetch.

Every task the API runs asynchronously goes through here -- both the `client.jobs`
namespace, which hands back a `Job`, and the blocking methods that are async jobs
underneath (`simulate()`, `explain()`, and `forecast()`/`cross_validation()` when
they fan out across partitions).

These are free functions taking the `NixtlaClient` as their first argument rather
than methods on it, so the client keeps no job machinery of its own. They reach
into it only for the HTTP session (`_make_client`, `_client_kwargs`), the request
helpers (`_make_request`, `_get_request`) and the retry budget (`_retry_settings`).
"""

from concurrent.futures import CancelledError
from functools import partial
from http import HTTPStatus
from threading import Event
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import httpcore
import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
)

from .._http import (
    _is_retriable_error,
    _RESULT_NOT_READY_CODES,
    ApiError,
    logger,
)
from ._job import (
    _MIN_POLL_RETRY_INTERVAL,
    _poll_intervals,
    _validate_poll_settings,
    JobCancelledError,
    JobError,
    JobTimeoutError,
    Job,
    JobStatus,
)
from .._steps import (
    CONTENT_TYPE as _STEP_CONTENT_TYPE,
    METADATA_HEADER as _STEP_METADATA_HEADER,
    StepResult,
    build_result as _build_step_result,
)

if TYPE_CHECKING:
    from ..nixtla_client import NixtlaClient


def _task_name(endpoint: str) -> str:
    """The task an API route runs: `"v2/cross_validation/async"` -> `"cross_validation"`.

    Every route is named after its task, so the endpoint a call already holds is the
    label to put on its payload-size guidance and on the errors its job raises.
    """
    return endpoint.removeprefix("v2/").removesuffix("/async")


def _validate_job_timeout_seconds(job_timeout_seconds: Optional[int]) -> None:
    """Reject a job timeout the server would refuse, before spending a round-trip on it.

    Kept identical in wording to the check `steps.build_request` runs for `execute_step`, which
    validates separately because that module is a self-contained codec. Every async task shares
    this one check so the same bad value is reported the same way everywhere.
    """
    if job_timeout_seconds is None:
        return
    if (
        isinstance(job_timeout_seconds, bool)
        or not isinstance(job_timeout_seconds, int)
        or job_timeout_seconds <= 0
    ):
        raise ValueError(
            f"job_timeout_seconds must be positive, got {job_timeout_seconds!r}"
        )


def _with_job_options(
    payload: dict[str, Any], job_timeout_seconds: Optional[int]
) -> dict[str, Any]:
    """Return `payload` carrying the job's server-side timeout, or unchanged when none is set.

    Never mutates the argument: `forecast` reuses one payload to derive its add_history request,
    and the partitioned path hands the same dict shape to several jobs.
    """
    if job_timeout_seconds is None:
        return payload
    return {**payload, "job_options": {"timeout_seconds": job_timeout_seconds}}


def _wait_for_poll(seconds: float, cancellation_event: Optional[Event]) -> bool:
    """Wait for the next poll, returning whether cancellation was requested."""
    if cancellation_event is None:
        time.sleep(seconds)
        return False
    return cancellation_event.wait(seconds)


def _submit_retry_strategy(
    max_retries: int,
    retry_interval: int,
    max_wait_time: int,
    cancellation_event: Optional[Event] = None,
):
    """Retry policy for submitting an async job.

    A submission that reached the server may already have created a job even
    when the response was lost, so only failures that are known to precede
    acceptance are retried: refusals because of the in-flight job cap (429) and
    errors while connecting. Read timeouts and gateway errors are not retried.
    """

    def should_retry(exc: BaseException) -> bool:
        connect_exceptions = (
            httpcore.ConnectError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
        )
        return isinstance(exc, connect_exceptions) or (
            isinstance(exc, ApiError)
            and exc.status_code == HTTPStatus.TOO_MANY_REQUESTS
        )

    last_error: Optional[BaseException] = None

    def before_attempt(retry_state: RetryCallState) -> None:
        if (
            retry_state.attempt_number > 1
            and time.monotonic() - retry_state.start_time >= max_wait_time
        ):
            assert last_error is not None
            raise last_error

    def wait_for(retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        delay = float(retry_interval)
        if isinstance(exc, ApiError) and exc.retry_after is not None:
            delay = exc.retry_after
        remaining = max(
            max_wait_time - (time.monotonic() - retry_state.start_time), 0.0
        )
        return min(delay, remaining)

    def after_retry(retry_state: RetryCallState) -> None:
        nonlocal last_error
        assert retry_state.outcome is not None
        last_error = retry_state.outcome.exception()
        logger.warning(
            f"Submission attempt {retry_state.attempt_number} failed with error: {last_error}"
        )

    def sleep(seconds: float) -> None:
        if _wait_for_poll(seconds, cancellation_event):
            raise CancelledError

    return Retrying(
        retry=retry_if_exception(should_retry),
        wait=wait_for,
        before=before_attempt,
        after=after_retry,
        stop=stop_after_attempt(max_retries) | stop_after_delay(max_wait_time),
        sleep=sleep,
        reraise=True,
    ).wraps


def _is_retriable_poll_error(exc: BaseException) -> bool:
    return _is_retriable_error(exc) or isinstance(
        exc,
        (
            httpcore.TimeoutException,
            httpx.TimeoutException,
            httpcore.NetworkError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    )


def _poll_retry_strategy(
    max_retries: Optional[int],
    retry_interval: Callable[[], float],
    max_wait_time: Optional[int],
    deadline: Optional[float],
    cancellation_event: Optional[Event] = None,
):
    """Retry status requests without sleeping beyond the job wait deadline.

    `retry_interval` is read per attempt because the polling cadence can be
    adaptive: a transient failure is retried at whatever interval the wait has
    backed off to, not at a cadence fixed when the wait began.

    `max_retries` bounds one run of consecutive failures, not the whole wait:
    `poll_job` resumes polling afterwards when it still has deadline left, so a
    patch of gateway errors no longer ends an hour-long wait.
    """

    warned = False

    def wait_for(_: RetryCallState) -> float:
        wait = float(retry_interval())
        if deadline is not None:
            wait = min(wait, max(deadline - time.monotonic(), 0.0))
        return wait

    def should_stop(retry_state: RetryCallState) -> bool:
        if max_retries is not None and retry_state.attempt_number >= max_retries:
            return True
        if (
            max_wait_time is not None
            and retry_state.seconds_since_start is not None
            and retry_state.seconds_since_start >= max_wait_time
        ):
            return True
        return deadline is not None and time.monotonic() >= deadline

    def after_retry(retry_state: RetryCallState) -> None:
        nonlocal warned
        assert retry_state.outcome is not None
        error = retry_state.outcome.exception()
        log = logger.debug if warned else logger.warning
        log(f"Polling attempt {retry_state.attempt_number} failed with error: {error}")
        warned = True

    def sleep(seconds: float) -> None:
        if _wait_for_poll(seconds, cancellation_event):
            raise CancelledError

    return Retrying(
        retry=retry_if_exception(_is_retriable_poll_error),
        wait=wait_for,
        after=after_retry,
        stop=should_stop,
        sleep=sleep,
        reraise=True,
    ).wraps


def get_job_data(
    nixtla_client: "NixtlaClient",
    client: httpx.Client,
    endpoint: str,
    job_id: str,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    return nixtla_client._get_request(
        client, f"{endpoint}/jobs/{job_id}", timeout=timeout
    )


def submit_job(
    nixtla_client: "NixtlaClient",
    client: httpx.Client,
    endpoint: str,
    payload: dict[str, Any],
    multithreaded_compress: bool = True,
    *,
    cancellation_event: Optional[Event] = None,
) -> str:
    # Retrying an ambiguous submission failure can create duplicate jobs.
    def submit() -> dict[str, Any]:
        if cancellation_event is not None and cancellation_event.is_set():
            raise CancelledError
        return nixtla_client._make_request(
            client, f"{endpoint}/async", payload, multithreaded_compress
        )

    body = _submit_retry_strategy(
        **nixtla_client._retry_settings, cancellation_event=cancellation_event
    )(submit)()
    job_id = body.get("job_id") if isinstance(body, dict) else None
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(
            f"Unexpected response when submitting the {endpoint} job: {body}"
        )
    logger.info(f"Submitted {endpoint} job {job_id}.")
    return job_id


def poll_job(
    nixtla_client: "NixtlaClient",
    client: httpx.Client,
    endpoint: str,
    job_id: str,
    poll_interval: Optional[float],
    poll_timeout: Optional[float],
    *,
    task: Optional[str] = None,
    cancellation_event: Optional[Event] = None,
) -> dict[str, Any]:
    """Return the successful status envelope without cancelling on timeout.

    The caller owns cancellation so `Job.wait(cancel_on_timeout=False)`
    stays resumable. Every task polls the same way: `poll_interval` between
    status checks -- fixed when given, adaptive when `None` -- transient
    failures retried on the client's retry budget, and nothing outliving
    `poll_timeout`. `task` only labels the errors raised from here, and
    defaults to the name the endpoint already carries.
    """
    _validate_poll_settings(poll_interval, poll_timeout, allow_unbounded=True)
    task = task or _task_name(endpoint)
    deadline = None if poll_timeout is None else time.monotonic() + poll_timeout
    # The cadence is shared between the sleep after a non-terminal status
    # and the sleep between retries of a failed status check, so a retry
    # never polls faster than the wait has settled to.
    intervals = _poll_intervals(poll_interval)
    current_interval = next(intervals)
    retry_floor = max(
        nixtla_client._retry_settings["retry_interval"], _MIN_POLL_RETRY_INTERVAL
    )
    # The last transient failure, if any, so that a wait spent retrying a
    # flapping server times out saying so rather than reporting only that
    # the job never finished.
    last_poll_error: Optional[BaseException] = None

    def check_wait() -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            raise CancelledError
        if deadline is not None and time.monotonic() >= deadline:
            assert poll_timeout is not None
            raise JobTimeoutError(
                job_id=job_id, poll_timeout=poll_timeout
            ) from last_poll_error

    def get_status() -> dict[str, Any]:
        check_wait()
        timeout = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        return get_job_data(nixtla_client, client, endpoint, job_id, timeout=timeout)

    def _sleep_between_polls(interval: float) -> float:
        """Wait out `interval`, clamped to the deadline, and return the next one."""
        sleep_for = interval
        if deadline is not None:
            sleep_for = min(sleep_for, max(deadline - time.monotonic(), 0.0))
        if _wait_for_poll(sleep_for, cancellation_event):
            raise CancelledError
        return next(intervals)

    # One budget for every task: a run of transient status failures is
    # bounded by the client's retry settings, and never outlives the
    # caller's poll deadline. Each poll cycle gets a fresh budget, and
    # retries keep whatever cadence the wait has settled to.
    get_status_with_retries = _poll_retry_strategy(
        max_retries=nixtla_client._retry_settings["max_retries"],
        retry_interval=lambda: current_interval,
        max_wait_time=nixtla_client._retry_settings["max_wait_time"],
        deadline=deadline,
        cancellation_event=cancellation_event,
    )(get_status)
    while True:
        try:
            job_data = get_status_with_retries()
        except Exception as exc:
            if not _is_retriable_poll_error(exc):
                raise
            last_poll_error = exc
            # The inner budget bounds one run of failures; the wait itself
            # is bounded by `poll_timeout`. A patch of gateway errors should
            # not end an hour-long wait -- which, in a partitioned fan-out,
            # would also cancel the job and its siblings -- so keep polling
            # while there is deadline left. Without a deadline there is
            # nothing left to bound the retrying, so the error propagates.
            check_wait()
            if deadline is None:
                raise
            current_interval = _sleep_between_polls(max(current_interval, retry_floor))
            continue
        check_wait()
        raw_status = job_data.get("status") if isinstance(job_data, dict) else None
        try:
            status = JobStatus(raw_status)
        except ValueError:
            raise JobError(
                job_id=job_id,
                task=task,
                error=f"unexpected job status {raw_status!r}: {job_data}",
            ) from None
        if status == JobStatus.SUCCEEDED:
            return job_data
        if status == JobStatus.CANCELLED:
            raise JobCancelledError(job_id=job_id)
        if status == JobStatus.FAILED:
            raise JobError(
                job_id=job_id,
                task=task,
                status=status.value,
                error=job_data.get("error"),
            )
        current_interval = _sleep_between_polls(current_interval)


def run_async_job(
    nixtla_client: "NixtlaClient",
    client: httpx.Client,
    endpoint: str,
    payload: dict[str, Any],
    poll_interval: Optional[float],
    poll_timeout: Optional[float],
    multithreaded_compress: bool = True,
    job_timeout_seconds: Optional[int] = None,
    *,
    task: Optional[str] = None,
    cancellation_event: Optional[Event] = None,
) -> dict[str, Any]:
    """Submit a job and return its result, cancelling abandoned jobs.

    Each partition's wait timeout starts after successful submission,
    excluding submission retries. If polling stops before a terminal
    state is known, request cancellation without masking the original error.
    """
    _validate_poll_settings(poll_interval, poll_timeout, allow_unbounded=True)
    task = task or _task_name(endpoint)
    if cancellation_event is not None and cancellation_event.is_set():
        raise CancelledError
    payload = _with_job_options(payload, job_timeout_seconds)
    job_id = submit_job(
        nixtla_client,
        client,
        endpoint,
        payload,
        multithreaded_compress,
        cancellation_event=cancellation_event,
    )
    try:
        job_data = poll_job(
            nixtla_client,
            client,
            endpoint,
            job_id,
            poll_interval,
            poll_timeout,
            task=task,
            cancellation_event=cancellation_event,
        )
    except JobCancelledError:
        raise  # The server has already cancelled the job.
    except JobError as exc:
        if exc.status not in ("failed", "cancelled", "succeeded"):
            cancel_job_best_effort(client, job_id, "invalid job status")
        raise
    except BaseException:
        cancel_job_best_effort(client, job_id, "abandoned wait")
        raise
    result = job_data.get("result")
    if not isinstance(result, dict):
        # Success is terminal even if the result is malformed.
        raise JobError(
            job_id=job_id,
            task=task,
            status="succeeded",
            error="job succeeded but returned no result",
        )
    return result


def cancel_job(client: httpx.Client, job_id: str) -> None:
    resp = client.post(f"v2/async/jobs/{job_id}/cancel")
    if resp.status_code not in (
        HTTPStatus.OK,
        HTTPStatus.ACCEPTED,
        HTTPStatus.NO_CONTENT,
    ):
        try:
            body = resp.json()
        except Exception:
            body = f"Could not parse JSON: {resp.content}"
        raise ApiError(status_code=resp.status_code, body=body)


def cancel_job_best_effort(client: httpx.Client, job_id: str, reason: str) -> bool:
    """Request cancellation, swallowing failures. Return True if accepted.

    Unknown or terminal jobs need no cleanup. Return False for them so
    `Job` does not incorrectly cache their status as cancelled.
    """
    try:
        cancel_job(client, job_id)
    except Exception as exc:
        if isinstance(exc, ApiError) and exc.status_code in (
            HTTPStatus.NOT_FOUND,
            HTTPStatus.CONFLICT,
        ):
            return False
        logger.warning("Failed to cancel job %s (%s)", job_id, reason, exc_info=True)
        return False
    return True


def submit_and_wrap_job(
    nixtla_client: "NixtlaClient",
    endpoint: str,
    payload: dict[str, Any],
    job_timeout_seconds: Optional[int],
    parse_result: Callable[..., Any],
    *,
    task: str,
) -> Job:
    _validate_job_timeout_seconds(job_timeout_seconds)
    payload = _with_job_options(payload, job_timeout_seconds)
    with nixtla_client._make_client(**nixtla_client._client_kwargs) as client:
        job_id = submit_job(nixtla_client, client, endpoint, payload)

    def get_result(job_data: dict[str, Any], *_poll_settings: Any) -> Any:
        # A JSON result is inline in the status response, so there is nothing to wait
        # for: the poll settings `Job` passes are accepted and ignored.
        result = job_data.get("result")
        if not isinstance(result, dict):
            # Same guard as `run_async_job`: a malformed success must not surface
            # as a `TypeError` from inside `parse_result`.
            raise JobError(
                job_id=job_id,
                task=task,
                status="succeeded",
                error="job succeeded but returned no result",
            )
        return parse_result(result)

    return Job(
        client=nixtla_client,
        job_id=job_id,
        endpoint=endpoint,
        get_result=get_result,
        task=task,
    )


def submit_binary_job(
    nixtla_client: "NixtlaClient",
    client: httpx.Client,
    endpoint: str,
    metadata: str,
    body: bytes,
) -> str:
    """Submit a job whose request body is opaque bytes rather than a JSON payload.

    The body is sent as-is: it is an already-deflated zip, so the zstd compression
    `_make_request` applies to large JSON payloads would only cost CPU. `content-type` is
    overridden per-request because the client-level default is `application/json`.
    """
    headers = {
        "content-type": _STEP_CONTENT_TYPE,
        _STEP_METADATA_HEADER: metadata,
    }
    resp = client.post(url=f"{endpoint}/async", content=body, headers=headers)
    resp_body = nixtla_client._parse_json_response(
        resp, expected_status=(HTTPStatus.OK, HTTPStatus.ACCEPTED)
    )
    # Same envelope unwrap `_make_request` applies, so a wrapped body doesn't become a KeyError.
    if "data" in resp_body:
        resp_body = resp_body["data"]
    if "job_id" not in resp_body:
        raise ApiError(
            status_code=resp.status_code,
            body=f"Response has no job_id: {resp_body}",
        )
    return resp_body["job_id"]


def get_job_result_bytes(
    client: httpx.Client,
    endpoint: str,
    job_id: str,
) -> tuple[httpx.Headers, bytes]:
    """Fetch a binary job's result from its dedicated endpoint, once.

    Binary results cannot be inlined into the JSON status response, so they are served
    separately. A succeeded job's result is not served the instant its status says so:
    until it is ready the endpoint answers with one of `_RESULT_NOT_READY_CODES`, which
    this method raises as an `ApiError` like any other non-200. Call
    `_wait_for_job_result_bytes` rather than this directly -- it recognises those codes
    as "still being assembled" and keeps polling, instead of reporting a failure.

    The status check stays exact: anything other than 200 raises rather than returning a
    body, so a "not ready" JSON payload is never mistaken for the zip.
    """
    resp = client.get(f"{endpoint}/jobs/{job_id}/result")
    if resp.status_code != HTTPStatus.OK:
        try:
            body = resp.json()
        except Exception:
            body = f"Could not parse JSON: {resp.content}"
        raise ApiError(status_code=resp.status_code, body=body)
    return resp.headers, resp.content


def wait_for_job_result_bytes(
    nixtla_client: "NixtlaClient",
    client: httpx.Client,
    endpoint: str,
    job_id: str,
    poll_interval: Optional[float],
    poll_timeout: Optional[float],
) -> tuple[httpx.Headers, bytes]:
    """Poll a binary job's result endpoint until the payload is served.

    A succeeded job's result is not necessarily available the instant its status says so, and
    the endpoint answers `_RESULT_NOT_READY_CODES` until it is. That is a polling state, not a
    failure, so it gets its own loop here rather than going through `_retry_strategy`: waiting
    is not an error to log as one, and it should not spend the budget reserved for transient
    network failures.

    The two are bounded differently, as they are in `poll_job`. Waiting for a result that is
    still being assembled is bounded only by `poll_timeout`, which `Job.wait(poll_timeout=None)`
    may leave unset. A run of transient network failures is bounded by the client's
    `max_retries` regardless, so an endpoint that never answers cannot spin here forever, and
    its sleeps never fall below the client's `retry_interval` even when `poll_interval` is 0.
    Any answer at all, "not ready" included, clears that budget.
    """
    deadline = None if poll_timeout is None else time.monotonic() + poll_timeout
    intervals = _poll_intervals(poll_interval)
    max_retries = nixtla_client._retry_settings["max_retries"]
    retry_interval = nixtla_client._retry_settings["retry_interval"]
    announced = False
    failures = 0
    while True:
        try:
            return get_job_result_bytes(client, endpoint, job_id)
        except Exception as e:
            not_ready = (
                isinstance(e, ApiError) and e.status_code in _RESULT_NOT_READY_CODES
            )
            if not not_ready and not _is_retriable_poll_error(e):
                raise
            sleep_for = next(intervals)
            if not_ready:
                failures = 0
            else:
                failures += 1
                if failures >= max_retries:
                    raise
                # Never retry a failing endpoint faster than the client
                # would retry any other request.
                sleep_for = max(sleep_for, retry_interval)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    assert poll_timeout is not None
                    raise JobTimeoutError(
                        job_id=job_id, poll_timeout=poll_timeout
                    ) from e
                sleep_for = min(sleep_for, remaining)
            if not announced:
                # Once per wait, not once per attempt: `poll_interval` may be small.
                logger.info("Waiting for the result of job %s...", job_id)
                announced = True
            time.sleep(sleep_for)


def submit_and_wrap_binary_job(
    nixtla_client: "NixtlaClient",
    endpoint: str,
    metadata: str,
    body: bytes,
    *,
    task: str,
) -> Job:
    """Binary counterpart of `submit_and_wrap_job`.

    `job_options` is already folded into `metadata` by the caller, because for these tasks
    the request metadata travels in a header rather than in the body.
    """

    def get_result(
        job_data: dict[str, Any],
        poll_interval: Optional[float],
        poll_timeout: Optional[float],
    ) -> StepResult:
        # The status response leaves `result` null for these tasks; the payload is served
        # from the job's own result endpoint, which may not have it the instant the status
        # says succeeded.
        with nixtla_client._make_client(**nixtla_client._client_kwargs) as client:
            headers, content = wait_for_job_result_bytes(
                nixtla_client, client, endpoint, job_id, poll_interval, poll_timeout
            )
        return _build_step_result(headers, content)

    job_id: str
    with nixtla_client._make_client(**nixtla_client._client_kwargs) as client:
        job_id = _submit_retry_strategy(
            **nixtla_client._retry_settings, cancellation_event=None
        )(partial(submit_binary_job, nixtla_client))(
            client=client, endpoint=endpoint, metadata=metadata, body=body
        )
    return Job(
        client=nixtla_client,
        job_id=job_id,
        endpoint=endpoint,
        get_result=get_result,
        task=task,
    )
