import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .nixtla_client import NixtlaClient

logger = logging.getLogger(__name__)


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
    `submit_finetune_job`, or `submit_cross_validation_job`.

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
        parse_result: Callable[..., Any],
    ):
        self.job_id = job_id
        self.result: Any = None
        self._status: Optional[JobStatus] = None
        self._client = client
        self._endpoint = endpoint
        self._parse_result = parse_result

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

    def wait(self, poll_interval: float = 15, poll_timeout: float = 3600) -> Any:
        """Poll the job until it reaches a terminal state and return its result.

        Args:
            poll_interval (float): Seconds to wait between job-status polls.
                Defaults to 15.
            poll_timeout (float): Maximum seconds to wait for the job to
                reach a terminal state before raising `AsyncJobTimeoutError`.
                Defaults to 3600.

        Returns:
            The job's parsed result (a DataFrame for forecast/cross_validation
            jobs, a fine-tuned model id string for finetune jobs).

        Raises:
            AsyncJobError: If the job fails server-side.
            AsyncJobCancelledError: If the job reaches the `"cancelled"`
                terminal state (e.g. after a successful `cancel()`).
            AsyncJobTimeoutError: If `poll_timeout` elapses before the job
                reaches a terminal state.
        """
        with self._client._make_client(**self._client._client_kwargs) as http_client:
            job_data = self._client._poll_job(
                http_client, self._endpoint, self.job_id, poll_interval, poll_timeout
            )
        self._status = JobStatus(job_data["status"])
        self.result = self._parse_result(job_data["result"])
        return self.result

    def cancel(self) -> None:
        """Request cancellation of the job."""
        with self._client._make_client(**self._client._client_kwargs) as http_client:
            self._client._cancel_job(http_client, self.job_id)
        self._status = JobStatus.CANCELLED

    def __enter__(self) -> "Job":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            return
        if self._status is not None and self._status.is_terminal:
            return
        try:
            self.cancel()
        except Exception:
            logger.warning(
                "Failed to cancel job %s during exception cleanup",
                self.job_id,
                exc_info=True,
            )
