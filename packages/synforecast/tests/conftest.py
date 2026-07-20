"""Shared fixtures for the synforecast test suite."""

import pytest


@pytest.fixture(params=["pandas", "polars"])
def engine(request: pytest.FixtureRequest) -> str:
    """Dataframe engine to run generator API tests against."""
    return request.param
