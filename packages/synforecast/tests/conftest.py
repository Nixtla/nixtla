"""Shared fixtures for the synforecast test suite."""

import logging

import pytest


@pytest.fixture(params=["pandas", "polars"])
def engine(request: pytest.FixtureRequest) -> str:
    """Dataframe engine to run generator API tests against."""
    return request.param


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture(autouse=True)
def fail_on_unexpected_ar1_fallback(
    request: pytest.FixtureRequest,
) -> "logging.Handler":
    """Fail any test that silently triggers SynAugment's AR(1) substitution.

    A fallback means a fitted generator produced invalid parameters or crashed
    — historically the symptom of fitter/generator contract bugs (GBM 'S0',
    GARCH 'mean'). Tests that exercise the fallback on purpose opt out with
    ``@pytest.mark.allow_ar1_fallback``.
    """
    logger = logging.getLogger("synforecast.dataset")
    handler = _RecordingHandler()
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)
    if request.node.get_closest_marker("allow_ar1_fallback"):
        return
    fallbacks = [r for r in handler.records if "substituted AR(1)" in r.getMessage()]
    if fallbacks:
        pytest.fail(
            "unexpected SynAugment AR(1) fallback during this test: "
            f"{fallbacks[0].getMessage()}"
        )
