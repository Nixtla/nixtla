import inspect
from typing import get_args
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nixtla import NixtlaClient


def _client_with_forecast_response():
    client = NixtlaClient(api_key="test", max_retries=1)
    client._get_model_params = MagicMock(return_value=(28, 7))
    client._make_client = MagicMock()
    response = {
        "mean": [10.0, 11.0],
        "intervals": None,
        "weights_x": None,
        "feature_contributions": [[1.0, 2.0], [9.0, 9.0]],
    }
    client._make_request_with_retries = MagicMock(return_value=response)
    return client


def _dataframes():
    df = pd.DataFrame(
        {
            "unique_id": "series-0",
            "ds": pd.date_range("2024-01-01", periods=40, freq="D"),
            "y": range(40),
            "price": range(40),
        }
    )
    X_df = pd.DataFrame(
        {
            "unique_id": "series-0",
            "ds": pd.date_range("2024-02-10", periods=2, freq="D"),
            "price": [40, 41],
        }
    )
    return df, X_df


@pytest.mark.parametrize(
    "contribution_type",
    ["shapley", "intervention", "granger", "transfer_entropy"],
)
def test_forecast_sends_feature_contribution_type(contribution_type):
    client = _client_with_forecast_response()
    df, X_df = _dataframes()

    client.forecast(
        df=df,
        X_df=X_df,
        h=2,
        freq="D",
        feature_contributions=True,
        feature_contributions_type=contribution_type,
    )

    _, endpoint, payload = client._make_request_with_retries.call_args.args
    assert endpoint == "v2/forecast"
    assert payload["feature_contributions"] is True
    assert payload["feature_contributions_type"] == contribution_type
    assert client.feature_contributions.columns.tolist() == [
        "unique_id",
        "ds",
        "TimeGPT",
        "price",
        "base_value",
    ]
    assert client.feature_contributions["price"].tolist() == [1.0, 2.0]
    assert client.feature_contributions["base_value"].tolist() == [9.0, 9.0]


def test_forecast_rejects_feature_contributions_with_wrong_row_count():
    client = _client_with_forecast_response()
    client._make_request_with_retries = MagicMock(
        return_value={
            "mean": [10.0, 11.0],
            "intervals": None,
            "weights_x": None,
            "feature_contributions": [[1.0, 2.0]],
        }
    )
    df, X_df = _dataframes()

    with pytest.raises(RuntimeError, match="expected 2"):
        client.forecast(
            df=df,
            X_df=X_df,
            h=2,
            freq="D",
            feature_contributions=True,
        )


def test_forecast_rejects_unknown_feature_contribution_type():
    client = _client_with_forecast_response()
    df, X_df = _dataframes()

    with pytest.raises(ValueError, match="feature_contributions_type"):
        client.forecast(
            df=df,
            X_df=X_df,
            h=2,
            freq="D",
            feature_contributions=True,
            feature_contributions_type="unknown",
        )

    client._make_request_with_retries.assert_not_called()


def test_forecast_omits_contribution_type_when_contributions_are_disabled():
    client = _client_with_forecast_response()
    df, X_df = _dataframes()

    client.forecast(df=df, X_df=X_df, h=2, freq="D")

    payload = client._make_request_with_retries.call_args.args[2]
    assert payload["feature_contributions"] is False
    assert "feature_contributions_type" not in payload


def test_forecast_appends_feature_contribution_type_to_public_signature():
    parameters = list(inspect.signature(NixtlaClient.forecast).parameters)

    assert parameters[-3:] == [
        "model_parameters",
        "multivariate",
        "feature_contributions_type",
    ]


def test_accepted_contribution_types_track_the_type_alias():
    import nixtla.nixtla_client as client_module

    assert client_module._FEATURE_CONTRIBUTIONS_TYPES == get_args(
        client_module._FeatureContributionsType
    )
    assert client_module._EXPLAIN_METHODS == get_args(client_module._ExplainMethod)

    client = _client_with_forecast_response()
    df, X_df = _dataframes()
    with pytest.raises(ValueError) as excinfo:
        client.forecast(
            df=df,
            X_df=X_df,
            h=2,
            freq="D",
            feature_contributions=True,
            feature_contributions_type="nope",
        )
    for accepted in client_module._FEATURE_CONTRIBUTIONS_TYPES:
        assert repr(accepted) in str(excinfo.value)
