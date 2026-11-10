"""Deprecation warnings for the 1.0 anomaly-detection rename.

In 1.0, batch `detect_anomalies()` is removed and the name is reused for
online detection (today's `detect_anomalies_online()`). Both warnings must
fire before the method does any work, so these tests pass a `df` that fails
local validation and never reaches the network.
"""

import contextlib
import warnings

import pandas as pd
import pytest

from nixtla import NixtlaClient


@pytest.fixture
def offline_client():
    return NixtlaClient(api_key="dummy", base_url="https://example.invalid")


def _future_warnings(fn, **kwargs):
    """Capture FutureWarnings raised before the call does any real work."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.suppress(Exception):
            fn(**kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]


def test_detect_anomalies_warns_about_removal(offline_client):
    messages = _future_warnings(offline_client.detect_anomalies, df=pd.DataFrame())
    assert len(messages) == 1
    assert "REMOVED in nixtla 1.0" in messages[0]
    assert "detect_anomalies_online()" in messages[0]


def test_detect_anomalies_online_warns_about_rename(offline_client):
    messages = _future_warnings(
        offline_client.detect_anomalies_online,
        df=pd.DataFrame(),
        h=1,
        detection_size=1,
    )
    assert len(messages) == 1
    assert "renamed to detect_anomalies()" in messages[0]
    assert "jobs.detect_anomalies()" in messages[0]


def test_jobs_detect_anomalies_does_not_warn(offline_client):
    messages = _future_warnings(
        offline_client.jobs.detect_anomalies,
        df=pd.DataFrame(),
        h=1,
        detection_size=1,
    )
    assert messages == []


def test_snowflake_detect_anomalies_warns_about_switch_to_online():
    """The Snowflake path switches to online detection in 1.0.

    The warning has to fire client-side: the UDTF backing this procedure runs
    inside the warehouse, where a warning would reach nobody.
    """
    from nixtla.snowflake import detect_anomalies

    messages = _future_warnings(detect_anomalies, session=None, table="db.schema.t")
    assert len(messages) == 1
    assert "switch to online detection" in messages[0]
    assert "detection_size" in messages[0]
    assert "Re-deploy" in messages[0]
