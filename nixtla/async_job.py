from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .nixtla_client import NixtlaClient


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class AsyncJobError(Exception):
    """Raised when a server-side async job (forecast/finetune/cross_validation) fails."""

    def __init__(self, *, job_id: str, error: Any):
        self.job_id = job_id
        self.error = error

    def __str__(self) -> str:
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
    `submit_finetune_job`, `submit_cross_validation_job`, or
    `submit_execute_step_job`.

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
        get_result: Callable[[dict[str, Any]], Any],
    ):
        """
        Args:
            get_result: Builds the job's result from the terminal job-status response.
                Tasks whose result is JSON read it out of that response's `result` field;
                tasks whose result is binary (`execute_step` returns a zip) leave `result`
                null there and fetch the payload from their own endpoint instead.
        """
        self.job_id = job_id
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
        poll_interval: float = 15,
        poll_timeout: float = 3600,
        cancel_on_timeout: bool = False,
    ) -> Any:
        """Poll the job until it reaches a terminal state and return its result.

        Args:
            poll_interval (float): Seconds to wait between job-status polls.
                Defaults to 15.
            poll_timeout (float): Maximum seconds to wait for the job to
                reach a terminal state before raising `AsyncJobTimeoutError`.
                Defaults to 3600.
            cancel_on_timeout (bool): Whether to request cancellation of the
                job when `poll_timeout` elapses. Defaults to False, so that
                polling in short increments (calling `wait()` again to resume)
                leaves the job running. Set to True to stop the job from
                consuming server-side compute once you have given up on it.
                Cancellation is best-effort: if the request fails it is logged
                as a warning and `AsyncJobTimeoutError` is raised regardless.

        Returns:
            The job's parsed result (a DataFrame for forecast/cross_validation
            jobs, a fine-tuned model id string for finetune jobs, a `StepResult`
            for execute_step jobs).

        Raises:
            AsyncJobError: If the job fails server-side.
            AsyncJobCancelledError: If the job reaches the `"cancelled"`
                terminal state (e.g. after a successful `cancel()`).
            AsyncJobTimeoutError: If `poll_timeout` elapses before the job
                reaches a terminal state. `poll_timeout` only bounds the
                client's polling, so unless `cancel_on_timeout` is set the job
                keeps running server-side until its own deadline.

        Note:
            `poll_timeout` bounds the status polling only. A task whose result
            is fetched separately (`execute_step`) then spends up to the
            client's `max_wait_time` retrieving it, on top of `poll_timeout`.
        """
        with self._client._make_client(**self._client._client_kwargs) as http_client:
            try:
                job_data = self._client._poll_job(
                    http_client,
                    self._endpoint,
                    self.job_id,
                    poll_interval,
                    poll_timeout,
                )
            except AsyncJobTimeoutError:
                if cancel_on_timeout:
                    self._cancel_best_effort("client poll timeout")
                raise
        # `_poll_job` returns only on success; every other terminal state raises.
        self._status = JobStatus.SUCCEEDED
        self.result = self._get_result(job_data)
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
        self._cancel_best_effort("exception cleanup")
