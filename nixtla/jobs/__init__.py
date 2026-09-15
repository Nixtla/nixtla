"""Server-side jobs: the `client.jobs` namespace and the `Job` handle it returns.

`Jobs` is the namespace itself, reached as `client.jobs` rather than constructed;
`Job` is the handle each of its methods returns. The submit/poll/cancel plumbing
behind both lives in `_transport` and is not part of the public surface.
"""

from ._job import (
    Job,
    JobCancelledError,
    JobError,
    JobStatus,
    JobSummary,
    JobTimeoutError,
)
from ._namespace import Jobs

__all__ = [
    "Job",
    "JobCancelledError",
    "JobError",
    "JobStatus",
    "JobSummary",
    "JobTimeoutError",
    "Jobs",
]
