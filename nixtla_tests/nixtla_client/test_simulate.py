import inspect
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

import nixtla.nixtla_client as client_module
from nixtla import NixtlaClient

_RUN_LIVE_ENDPOINT_TESTS = os.getenv("NIXTLA_RUN_SIMULATE_EXPLAIN_TESTS") == "1"


def _client_with_response(response):
    client = NixtlaClient(api_key="test", max_retries=1)
    client._make_client = MagicMock()
    request = MagicMock(
        side_effect=lambda _http, endpoint, payload: (
            response(endpoint, payload) if callable(response) else response
        )
    )
    client._make_request_with_retries = request
    return client, request


def _series_df(n_series=1, n=4):
    return pd.DataFrame(
        {
            "unique_id": np.repeat([f"id-{i}" for i in range(n_series)], n),
            "ds": list(pd.date_range("2024-01-01", periods=n, freq="D"))
            * n_series,
            "y": np.arange(n_series * n, dtype=float),
        }
    )


def _simulate_response(endpoint, payload):
    assert endpoint == "v2/simulate"
    n_series = len(payload["series"]["sizes"])
    n_values = payload["n_paths"] * n_series * payload["h"]
    return {
        "samples": np.arange(n_values, dtype=float).tolist(),
        "n_paths": payload["n_paths"],
        "h": payload["h"],
        "sizes": [payload["h"]] * n_series,
        "coupled": payload["multivariate"] and n_series > 1,
    }


def test_simulate_public_contract_has_no_configurable_algorithm():
    signature = inspect.signature(NixtlaClient.simulate)
    assert "method" not in signature.parameters


def test_simulate_builds_sample_major_pandas_output_and_payload():
    client, request = _client_with_response(_simulate_response)
    df = _series_df(n_series=2, n=3)

    result = client.simulate(
        df=df,
        h=2,
        freq="D",
        n_paths=2,
        quantiles=[0.1, 0.5, 0.9],
        seed=7,
        finetuned_model_id="ft-123",
        clean_ex_first=False,
        model="timegpt-1",
        multivariate=True,
    )

    assert result.columns.tolist() == [
        "unique_id",
        "ds",
        "sample_id",
        "TimeGPT",
        "coupled",
    ]
    assert result["sample_id"].tolist() == [0] * 4 + [1] * 4
    assert result["unique_id"].tolist() == [
        "id-0",
        "id-0",
        "id-1",
        "id-1",
    ] * 2
    assert result["TimeGPT"].tolist() == list(np.arange(8, dtype=float))
    assert result["coupled"].tolist() == [True] * 8
    assert result.groupby(["sample_id", "unique_id"], observed=True).size().eq(2).all()

    _, endpoint, payload = request.call_args.args
    assert endpoint == "v2/simulate"
    assert "method" not in payload
    assert payload["model"] == "timegpt-1"
    assert payload["h"] == 2
    assert payload["n_paths"] == 2
    assert payload["quantiles"] == [0.1, 0.5, 0.9]
    assert payload["seed"] == 7
    assert payload["finetuned_model_id"] == "ft-123"
    assert payload["clean_ex_first"] is False
    assert payload["multivariate"] is True
    assert request.call_count == 1


def test_simulate_polars_output_and_implicit_single_series_id():
    client, _ = _client_with_response(_simulate_response)
    df = pl.DataFrame(
        {
            "ds": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True
            ),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = client.simulate(df=df, h=2, freq="1d", n_paths=2)

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["ds", "sample_id", "TimeGPT", "coupled"]
    assert result["sample_id"].to_list() == [0, 0, 1, 1]
    assert result["coupled"].to_list() == [False] * 4


def test_simulate_preserves_future_historical_and_categorical_feature_order():
    client, request = _client_with_response(_simulate_response)
    df = _series_df(n=4).assign(
        price=[10.0, 11.0, 12.0, 13.0],
        event=["none", "sale", "none", "sale"],
        trend=[0.0, 1.0, 2.0, 3.0],
        segment=["a", "a", "b", "b"],
    )
    X_df = pd.DataFrame(
        {
            "unique_id": ["id-0", "id-0"],
            "ds": pd.date_range("2024-01-05", periods=2, freq="D"),
            "price": [14.0, 15.0],
            "event": ["none", "sale"],
        }
    )

    client.simulate(
        df=df,
        X_df=X_df,
        h=2,
        freq="D",
        n_paths=1,
        hist_exog_list=["trend", "segment"],
        categorical_exog_list=["event", "segment"],
    )

    payload = request.call_args.args[2]
    series = payload["series"]
    assert [
        row.tolist() if isinstance(row, np.ndarray) else row for row in series["X"]
    ] == [
        [10.0, 11.0, 12.0, 13.0],
        ["none", "sale", "none", "sale"],
        [0.0, 1.0, 2.0, 3.0],
        ["a", "a", "b", "b"],
    ]
    assert [
        row.tolist() if isinstance(row, np.ndarray) else row
        for row in series["X_future"]
    ] == [[14.0, 15.0], ["none", "sale"]]
    assert series["categorical_exog"] == [1, 3]


def test_simulate_leaves_missing_seed_unset_for_server():
    client, request = _client_with_response(_simulate_response)

    client.simulate(_series_df(), h=2, freq="D", n_paths=1)

    payload = request.call_args.args[2]
    assert payload["seed"] is None
    assert "method" not in payload


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"h": 0}, "positive integer"),
        ({"h": 1.5}, "positive integer"),
        ({"n_paths": 0}, "between 1"),
        ({"n_paths": 10_001}, "between 1"),
        ({"n_paths": 1.5}, "integer"),
        ({"quantiles": [0.5]}, "between 2"),
        ({"quantiles": [0.1, 0.1]}, "strictly increasing"),
        ({"quantiles": [0.9, 0.1]}, "strictly increasing"),
        ({"quantiles": [0.0, 0.5]}, "strictly inside"),
        ({"quantiles": [0.5, 1.0]}, "strictly inside"),
        ({"quantiles": [0.1, np.nan]}, "strictly inside"),
        ({"seed": 1.5}, "integer"),
        ({"seed": -(2**63) - 1}, "must be between"),
        ({"seed": 2**64}, "must be between"),
    ],
)
def test_simulate_rejects_invalid_options_before_request(kwargs, match):
    client, request = _client_with_response(_simulate_response)
    params = {"df": _series_df(), "h": 2, "freq": "D", "n_paths": 1}
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        client.simulate(**params)

    request.assert_not_called()


def test_simulate_enforces_output_limit_before_request(monkeypatch):
    monkeypatch.setattr(client_module, "_MAX_SIMULATE_OUTPUT_VALUES", 3)
    client, request = _client_with_response(_simulate_response)

    with pytest.raises(ValueError, match="exceeds the maximum"):
        client.simulate(_series_df(), h=2, freq="D", n_paths=2)

    request.assert_not_called()


def test_simulate_enforces_quantile_cell_limit_before_request(monkeypatch):
    # The path count alone stays within its own limit; only the quantile grid
    # pushes the request over the service's cell limit. One series, h=2 and
    # n_paths=2 with three quantiles is 1 * 2 * (2 + 3) = 10 cells.
    monkeypatch.setattr(client_module, "_MAX_SIMULATE_CELLS", 9)
    client, request = _client_with_response(_simulate_response)

    with pytest.raises(ValueError, match=r"len\(quantiles\)"):
        client.simulate(
            _series_df(),
            h=2,
            freq="D",
            n_paths=2,
            quantiles=[0.1, 0.5, 0.9],
        )

    request.assert_not_called()


def test_simulate_quantile_cell_limit_ignored_without_quantiles(monkeypatch):
    # Without an explicit grid the width depends on the model's native
    # quantiles, so the client cannot compute the cell count and must not guess.
    monkeypatch.setattr(client_module, "_MAX_SIMULATE_CELLS", 1)
    client, request = _client_with_response(_simulate_response)

    client.simulate(_series_df(), h=2, freq="D", n_paths=2)

    request.assert_called_once()


@pytest.mark.parametrize(
    "response,match",
    [
        (
            {
                "samples": [1.0, 2.0],
                "n_paths": 2,
                "h": 2,
                "sizes": [2],
                "coupled": False,
            },
            "metadata",
        ),
        (
            {
                "samples": [1.0, 2.0],
                "n_paths": 1,
                "h": 2,
                "sizes": [1],
                "coupled": False,
            },
            "series sizes",
        ),
        (
            {
                "samples": [1.0],
                "n_paths": 1,
                "h": 2,
                "sizes": [2],
                "coupled": False,
            },
            "expected 2",
        ),
    ],
)
def test_simulate_rejects_malformed_responses(response, match):
    client, _ = _client_with_response(response)

    with pytest.raises(RuntimeError, match=match):
        client.simulate(_series_df(), h=2, freq="D", n_paths=1)


def test_simulate_rejects_distributed_or_unknown_dataframe_types():
    client, request = _client_with_response(_simulate_response)

    with pytest.raises(ValueError, match="pandas and polars"):
        client.simulate([1.0, 2.0], h=1, freq="D")

    request.assert_not_called()


@pytest.mark.integration
@pytest.mark.skipif(
    not _RUN_LIVE_ENDPOINT_TESTS,
    reason="Set NIXTLA_RUN_SIMULATE_EXPLAIN_TESTS=1 after deploying the endpoints.",
)
def test_simulate_live_endpoint_is_reproducible(nixtla_test_client):
    n = 120
    df = pd.DataFrame(
        {
            "unique_id": "series-0",
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": np.sin(np.arange(n) / 7) + np.arange(n) / 100,
        }
    )
    kwargs = {
        "df": df,
        "h": 4,
        "freq": "D",
        "n_paths": 3,
        "seed": 42,
        "model": "timegpt-2.1",
    }

    first = nixtla_test_client.simulate(**kwargs)
    second = nixtla_test_client.simulate(**kwargs)

    assert len(first) == 12
    assert first["sample_id"].nunique() == 3
    assert first["coupled"].eq(False).all()
    pd.testing.assert_frame_equal(first, second)
