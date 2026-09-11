from enum import Enum
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .nixtla_client import NixtlaClient


# Defaults every async task polls with: a status check every 15 seconds, giving up
# after an hour. `simulate()` and `explain()` expose the same two numbers.
_DEFAULT_POLL_INTERVAL = 15.0
_DEFAULT_POLL_TIMEOUT = 3600.0


def _validate_poll_settings(
    poll_interval: float, poll_timeout: Optional[float], *, allow_unbounded: bool = False
) -> None:
    if (
        isinstance(poll_interval, bool)
        or not isinstance(poll_interval, Real)
        or not math.isfinite(poll_interval)
        or poll_interval < 0
    ):
        raise ValueError("`poll_interval` must be a finite, non-negative number.")
    if poll_timeout is None and allow_unbounded:
        return
    if (
        isinstance(poll_timeout, bool)
        or not isinstance(poll_timeout, Real)
        or not math.isfinite(poll_timeout)
        or poll_timeout <= 0
    ):
        raise ValueError("`poll_timeout` must be a finite, positive number.")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


def _terminal_status(status: Optional[str]) -> Optional[JobStatus]:
    """The `JobStatus` `status` names, or `None` when it is unknown or not terminal."""
    try:
        parsed = JobStatus(status)
    except ValueError:
        return None
    return parsed if parsed.is_terminal else None


class AsyncJobError(RuntimeError):
    """Raised when a job fails or returns an invalid response.

    Attributes:
        job_id (str): Identifier of the job on the server.
        error (Any): Original error reported by the server, or a description
            of an invalid response.
        task (str, optional): Task the job ran (e.g. `"forecast"`,
            `"simulate"`, `"explain"`). Every job submitted through this
            client carries one; `None` only when a caller builds the error
            itself without naming a task.
        status (str, optional): Known terminal status, such as `"failed"` or
            `"succeeded"`. None when the terminal state is unknown.
    """

    def __init__(
        self,
        *,
        job_id: str,
        error: Any = None,
        task: Optional[str] = None,
        status: Optional[str] = None,
    ):
        self.job_id = job_id
        self.error = error
        self.task = task
        self.status = status
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.task is not None:
            detail = self.error if self.error else "no error message was reported"
            status = self.status or "returned an invalid response"
            return f"{self.task} job '{self.job_id}' {status}: {detail}"
        return f"job_id: {self.job_id}, error: {self.error}"


class AsyncJobTimeoutError(Exception):
    """Raised when polling a server-side async job exceeds `poll_timeout`."""

    def __init__(self, *, job_id: str, poll_timeout: float):
        self.job_id = job_id
        self.poll_timeout = poll_timeout

    def __str__(self) -> str:
        return (
            f"job_id: {self.job_id} did not finish within "
            f"poll_timeout={self.poll_timeout}s"
        )


class AsyncJobCancelledError(Exception):
    """Raised when a server-side async job reaches the 'cancelled' terminal state."""

    def __init__(self, *, job_id: str):
        self.job_id = job_id

    def __str__(self) -> str:
        return f"job_id: {self.job_id} was cancelled"


class Job:
    """Handle to a server-side async job submitted via `submit_forecast_job`,
    `submit_finetune_job`, `submit_cross_validation_job`,
    `submit_anomaly_detection_job`, `submit_simulate_job`,
    `submit_explain_job`, or `submit_execute_step_job`.

    `status` queries the server for the job's current status; call `wait()`
    to block until it reaches a terminal state and get its result, or
    `cancel()` to request that the server stop it.

    Can also be used as a context manager: if an exception propagates out of
    the `with` block before the job reaches a terminal state, cancellation is
    requested automatically as best-effort cleanup.
    """

    def __init__(
        self,
        *,
        client: "NixtlaClient",
        job_id: str,
        endpoint: str,
        get_result: Callable[[dict[str, Any], float, Optional[float]], Any],
        task: Optional[str] = None,
    ):
        """
        Args:
            get_result: Builds the job's result, called as
                `get_result(job_data, poll_interval, poll_timeout)`. Tasks whose result is JSON
                read it out of the job-status response's `result` field and ignore the poll
                settings; tasks whose result is binary (`execute_step` returns a zip) leave
                `result` null there and use them to poll their own result endpoint.
            task: Name of the task the job runs (`"forecast"`, `"simulate"`, ...),
                used to label `AsyncJobError` messages.
        """
        self.job_id = job_id
        self.task = task
        self.result: Any = None
        self._status: Optional[JobStatus] = None
        self._client = client
        self._endpoint = endpoint
        self._get_result = get_result

    @property
    def status(self) -> JobStatus:
        """Current job status: `JobStatus.PENDING`, `RUNNING`, `SUCCEEDED`,
        `FAILED`, or `CANCELLED` (each compares equal to its lowercase string,
        e.g. `job.status == "succeeded"`).

        If a terminal status isn't already known (from `wait()` succeeding,
        `cancel()` being called, or a prior live check), this queries the
        server for it. Once a terminal status is observed it's cached, since
        a finished job's status can't change again.
        """
        if self._status is not None:
            return self._status
        with self._client._make_client(**self._client._client_kwargs) as http_client:
            job_data = self._client._get_job_data(
                http_client, self._endpoint, self.job_id
            )
        status = JobStatus(job_data.get("status"))
        if status.is_terminal:
            self._status = status
        return status

    def wait(
        self,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        poll_timeout: Optional[float] = _DEFAULT_POLL_TIMEOUT,
        cancel_on_timeout: bool = True,
    ) -> Any:
        """Poll the job until it reaches a terminal state and return its result.

        Args:
            poll_interval (float): Seconds to wait between job-status polls.
                Must be finite and non-negative. Defaults to 15.
            poll_timeout (float, optional): Maximum seconds to wait for the
                job to reach a terminal state before raising
                `AsyncJobTimeoutError`. Must be finite and positive, or `None`
                to poll until the server reports a terminal status. Defaults
                to 3600.
            cancel_on_timeout (bool): Whether to request cancellation of the
                job when `poll_timeout` elapses. Defaults to True, so that a
                job you have given up on stops consuming server-side compute.
                Set to False to poll in short increments -- calling `wait()`
                again to resume -- which requires the job to still be running.
                Cancellation is best-effort. Unknown or already terminal jobs
                need no cleanup; other cancellation failures are logged as
                warnings. `AsyncJobTimeoutError` is raised regardless.

        Returns:
            The job's parsed result (a DataFrame for forecast/cross_validation/
            anomaly-detection/simulate/explain jobs, a fine-tuned model id
            string for finetune jobs, a `StepResult` for execute_step jobs).

        Raises:
            ValueError: If the polling interval or timeout is invalid.
            AsyncJobError: If the job fails server-side.
            AsyncJobCancelledError: If the job reaches the `"cancelled"`
                terminal state (e.g. after a successful `cancel()`).
            AsyncJobTimeoutError: If `poll_timeout` elapses before the job
                reaches a terminal state. `poll_timeout` only bounds the
                client's polling, so with `cancel_on_timeout=False` the job
                keeps running server-side until its own deadline.

        Note:
            A task whose result is fetched separately (`execute_step`) polls
            for it after the status turns terminal, using these same settings
            again -- so the total wait can reach twice `poll_timeout`.
        """
        _validate_poll_settings(poll_interval, poll_timeout, allow_unbounded=True)
        with self._client._make_client(**self._client._client_kwargs) as http_client:
            try:
                job_data = self._client._poll_job(
                    http_client,
                    self._endpoint,
                    self.job_id,
                    poll_interval,
                    poll_timeout,
                    task=self.task,
                )
            except AsyncJobTimeoutError:
                if cancel_on_timeout:
                    self._cancel_best_effort("client poll timeout")
                raise
            except AsyncJobCancelledError:
                # The server already reported the terminal state; re-querying
                # `status` cannot tell us anything new.
                self._status = JobStatus.CANCELLED
                raise
            except AsyncJobError as exc:
                # A terminal status the server reported is already final; an
                # error without one leaves the job's state unknown.
                status = _terminal_status(exc.status)
                if status is not None:
                    self._status = status
                raise
        # `_poll_job` returns only on success; every other terminal state raises.
        self._status = JobStatus.SUCCEEDED
        self.result = self._get_result(job_data, poll_interval, poll_timeout)
        return self.result

    def cancel(self) -> None:
        """Request cancellation of the job."""
        with self._client._make_client(**self._client._client_kwargs) as http_client:
            self._client._cancel_job(http_client, self.job_id)
        self._status = JobStatus.CANCELLED

    def _cancel_best_effort(self, reason: str) -> None:
        """Request cancellation without letting a failure mask the exception
        that is already propagating.

        `_status` is only marked terminal when the server accepted the request,
        so after a failed cancel `status` re-queries instead of reporting an
        optimistic `"cancelled"`.
        """
        with self._client._make_client(**self._client._client_kwargs) as http_client:
            if self._client._cancel_job_best_effort(http_client, self.job_id, reason):
                self._status = JobStatus.CANCELLED

    def __enter__(self) -> "Job":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            return
        if self._status is not None and self._status.is_terminal:
            return
        if isinstance(exc_val, (AsyncJobCancelledError, AsyncJobTimeoutError)):
            # `wait()` already settled both: a cancelled job is terminal, and a
            # timed-out one was either cancelled there or deliberately left
            # running by `cancel_on_timeout=False`.
            return
        if (
            isinstance(exc_val, AsyncJobError)
            and _terminal_status(exc_val.status) is not None
        ):
            # The server reached a terminal state on its own; there is nothing
            # left to cancel.
            return
        self._cancel_best_effort("exception cleanup")
