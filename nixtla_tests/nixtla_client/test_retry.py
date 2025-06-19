import httpx
import pytest
import time

from itertools import product
from nixtla.nixtla_client import (
    ApiError,
)
from nixtla_tests.helpers.checks import check_retry_behavior


def raise_api_error_with_text(*args, **kwargs):
    raise ApiError(
        status_code=503,
        body="""
        <html><head>
        <meta http-equiv="content-type" content="text/html;charset=utf-8">
        <title>503 Server Error</title>
        </head>
        <body text=#000000 bgcolor=#ffffff>
        <h1>Error: Server Error</h1>
        <h2>The service you requested is not available at this time.<p>Service error -27.</h2>
        <h2></h2>
        </body></html>
        """,
    )


def raise_api_error_with_json(*args, **kwargs):
    raise ApiError(
        status_code=422,
        body=dict(detail="Please use numbers"),
    )


def raise_read_timeout_error(*args, **kwargs):
    sleep_seconds = 5
    print(f"raising ReadTimeout error after {sleep_seconds} seconds")
    time.sleep(sleep_seconds)
    raise httpx.ReadTimeout("Timed out")


def raise_http_error(*args, **kwargs):
    print("raising HTTP error")
    raise ApiError(status_code=503, body="HTTP error")


@pytest.mark.parametrize(
    "side_effect,side_effect_exception,should_retry",
    [
        (raise_api_error_with_text, ApiError, True),
        (raise_api_error_with_json, ApiError, False),
    ],
)
def test_retry_behavior(
    air_passengers_df, side_effect, side_effect_exception, should_retry
):
    check_retry_behavior(
        df=air_passengers_df,
        side_effect=side_effect,
        side_effect_exception=side_effect_exception,
        should_retry=should_retry,
    )


combs = [
    (2, 5, 30),
    (10, 1, 5),
]
side_effect_settings = [
    (raise_read_timeout_error, httpx.ReadTimeout),
    (raise_http_error, ApiError),
]


@pytest.mark.parametrize(
    "retry_settings,side_effect_settings", product(combs, side_effect_settings)
)
def test_retry_behavior_set2(air_passengers_df, retry_settings, side_effect_settings):
    max_retries, retry_interval, max_wait_time = retry_settings
    side_effect, side_effect_exception = side_effect_settings
    check_retry_behavior(
        df=air_passengers_df,
        side_effect=side_effect,
        side_effect_exception=side_effect_exception,
        max_retries=max_retries,
        retry_interval=retry_interval,
        max_wait_time=max_wait_time,
    )
