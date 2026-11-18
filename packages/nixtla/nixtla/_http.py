"""HTTP primitives shared by the client and the async-job transport.

This module sits below both `nixtla_client` and `jobs._transport` so that the
transport can raise and classify API errors without importing the client back.
It has no internal dependencies of its own.
"""

from http import HTTPStatus
import logging
import math
from typing import Any, Optional

import httpcore
import httpx

# The SDK's log channel. Deliberately a fixed name rather than `__name__`: it is
# what callers configure, so splitting the implementation across modules must
# not split the logger they filter on.
logger = logging.getLogger("nixtla.nixtla_client")

# What a binary job's result endpoint answers while the payload has not been served yet. 202 is
# what it returns today; 409 is kept for compatibility, and where it instead means the job ended
# without a result, waiting is merely wasteful rather than wrong.
_RESULT_NOT_READY_CODES = (HTTPStatus.ACCEPTED, HTTPStatus.CONFLICT)


class ApiError(Exception):
    status_code: Optional[int]
    body: Any
    retry_after: Optional[float]

    def __init__(
        self,
        *,
        status_code: Optional[int] = None,
        body: Optional[Any] = None,
        retry_after: Optional[float] = None,
    ):
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after

    def __str__(self) -> str:
        return f"status_code: {self.status_code}, body: {self.body}"


def _is_retriable_error(exc: BaseException) -> bool:
    retriable_exceptions = (
        ConnectionResetError,
        httpcore.ConnectError,
        httpcore.RemoteProtocolError,
        httpx.ConnectTimeout,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        httpx.ReadTimeout,
        httpx.PoolTimeout,
        httpx.WriteError,
        httpx.WriteTimeout,
    )
    retriable_codes = [
        HTTPStatus.REQUEST_TIMEOUT,
        HTTPStatus.CONFLICT,
        HTTPStatus.TOO_MANY_REQUESTS,
        HTTPStatus.BAD_GATEWAY,
        HTTPStatus.SERVICE_UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT,
    ]
    return isinstance(exc, retriable_exceptions) or (
        isinstance(exc, ApiError) and exc.status_code in retriable_codes
    )


def _parse_retry_after(headers: Any) -> Optional[float]:
    """Return the `Retry-After` delay in seconds, if the header is present."""
    try:
        raw = headers.get("retry-after")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return max(seconds, 0.0) if math.isfinite(seconds) else None
