import logging
from unittest.mock import MagicMock

import numpy as np
import orjson
import pandas as pd
import polars as pl
import pytest

from nixtla import NixtlaClient


def _client_with_response(response):
    client = NixtlaClient(api_key="test", max_retries=1)
    client._make_client = MagicMock()
    request = MagicMock(
        side_effect=lambda _http, task, payload, **_kwargs: (
            response(task, payload) if callable(response) else response
        )
    )
    client._run_async_job = request
    return client, request


def _explain_response(task, payload):
    assert task == "explain"
    n_features = len(payload["series"]["X"])
    weights = np.arange(n_features, 0, -1, dtype=float)
    weights /= weights.sum()
    return {
        "weights": weights.tolist(),
        "feature_names": None,
        "method": payload["method"],
    }


def _explain_df():
    return pd.DataFrame(
        {
            "unique_id": ["b", "a", "b", "a", "b", "a"],
            "ds": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "y": [20.0, 2.0, 10.0, 1.0, 30.0, 3.0],
            "driver": [200.0, 20.0, 100.0, 10.0, 300.0, 30.0],
            "noise": [5.0, 2.0, 4.0, 1.0, 6.0, 3.0],
        }
    )


def test_explain_preserves_feature_order_and_sorts_observations():
    client, request = _client_with_response(_explain_response)

    result = client.explain(
        _explain_df(),
        features=["noise", "driver"],
        method="transfer_entropy",
    )

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "feature": ["noise", "driver"],
                "weight": [2 / 3, 1 / 3],
                "method": ["transfer_entropy", "transfer_entropy"],
            }
        ),
    )
    _, task, payload = request.call_args.args
    assert task == "explain"
    assert "model" not in payload
    assert payload["method"] == "transfer_entropy"
    assert payload["series"]["sizes"].tolist() == [3, 3]
    assert payload["series"]["y"].tolist() == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    assert [row.tolist() for row in payload["series"]["X"]] == [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    ]
    assert all(isinstance(row, np.ndarray) for row in payload["series"]["X"])


def test_explain_uses_all_non_key_columns_by_default():
    client, request = _client_with_response(_explain_response)

    result = client.explain(_explain_df())

    assert result["feature"].tolist() == ["driver", "noise"]
    assert request.call_args.args[2]["method"] == "granger"


def test_explain_string_categorical_payload_uses_original_position():
    client, request = _client_with_response(_explain_response)
    df = _explain_df().assign(
        segment=["enterprise", "small", "enterprise", "small", "enterprise", "small"]
    )

    result = client.explain(
        df,
        features=["driver", "segment", "noise"],
        categorical_exog_list=["segment"],
    )

    series = request.call_args.args[2]["series"]
    assert series["categorical_exog"] == [1]
    assert series["X"][1] == [
        "small",
        "small",
        "small",
        "enterprise",
        "enterprise",
        "enterprise",
    ]
    assert result["feature"].tolist() == ["driver", "segment", "noise"]


def test_explain_polars_output_and_implicit_single_series_id():
    client, _ = _client_with_response(_explain_response)
    df = pl.DataFrame(
        {
            "ds": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True),
            "y": [1.0, 2.0, 3.0, 4.0],
            "driver": [4.0, 3.0, 2.0, 1.0],
        }
    )

    result = client.explain(df, freq="1d")

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["feature", "weight", "method"]
    assert result.to_dict(as_series=False) == {
        "feature": ["driver"],
        "weight": [1.0],
        "method": ["granger"],
    }


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"features": []}, "at least one"),
        ({"features": ["driver", "driver"]}, "duplicates"),
        ({"features": ["missing"]}, "not found"),
        ({"features": ["y"]}, "cannot be explanation features"),
        (
            {
                "features": ["driver"],
                "categorical_exog_list": ["noise"],
            },
            "categorical feature",
        ),
    ],
)
def test_explain_rejects_invalid_options_before_request(kwargs, match):
    client, request = _client_with_response(_explain_response)

    with pytest.raises(ValueError, match=match):
        client.explain(_explain_df(), **kwargs)

    request.assert_not_called()


def test_explain_rejects_gapped_timestamps_before_request():
    client, request = _client_with_response(_explain_response)
    gapped = _explain_df()
    gapped = gapped[gapped["ds"] != "2024-01-02"]

    with pytest.raises(ValueError, match="missing or duplicate timestamps"):
        client.explain(gapped, features=["driver"], freq="D")

    request.assert_not_called()


def test_explain_rejects_duplicate_timestamps_before_request():
    client, request = _client_with_response(_explain_response)
    df = _explain_df()
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="missing or duplicate timestamps"):
        client.explain(duplicated, features=["driver"], freq="D")

    request.assert_not_called()


@pytest.mark.parametrize("freq", ["D", pd.offsets.Day()])
def test_explain_rejects_balanced_gap_and_duplicate_before_request(freq):
    client, request = _client_with_response(_explain_response)
    invalid = pd.DataFrame(
        {
            "unique_id": "series-0",
            "ds": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-04"]
            ),
            "y": [1.0, 2.0, 3.0, 4.0],
            "driver": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="missing or duplicate timestamps"):
        client.explain(invalid, features=["driver"], freq=freq)

    request.assert_not_called()


def test_explain_names_undeclared_non_numeric_features():
    client, request = _client_with_response(_explain_response)
    df = _explain_df().assign(label=["x", "y", "x", "y", "x", "y"])

    with pytest.raises(ValueError, match=r"not numeric: \['label'\]"):
        client.explain(df, features=["driver", "label"])

    request.assert_not_called()


def test_explain_accepts_non_numeric_feature_declared_as_categorical():
    client, request = _client_with_response(_explain_response)
    df = _explain_df().assign(label=["x", "y", "x", "y", "x", "y"])

    result = client.explain(
        df, features=["driver", "label"], categorical_exog_list=["label"]
    )

    assert result["feature"].tolist() == ["driver", "label"]
    request.assert_called_once()


def test_explain_serializes_missing_pandas_categorical_values_as_json_null(caplog):
    client, request = _client_with_response(_explain_response)
    df = _explain_df()
    df["label"] = pd.Series(
        ["x", pd.NA, "x", "y", "x", "y"], dtype="string"
    )

    with caplog.at_level(logging.WARNING, logger="nixtla.nixtla_client"):
        client.explain(
            df,
            features=["label"],
            categorical_exog_list=["label"],
        )

    payload = request.call_args.args[2]
    assert payload["series"]["X"][0] == ["y", None, "y", "x", "x", "x"]
    assert "missing values: ['label']" in caplog.text
    orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)


@pytest.mark.parametrize(
    "constructor,freq", [(pd.DataFrame, "D"), (pl.from_pandas, "1d")]
)
def test_explain_treats_boolean_features_the_same_on_both_backends(constructor, freq):
    client, request = _client_with_response(_explain_response)
    df = constructor(_explain_df().assign(flag=[True, False, True, False, True, False]))

    with pytest.raises(ValueError, match=r"not numeric: \['flag'\]"):
        client.explain(df, features=["driver", "flag"], freq=freq)
    request.assert_not_called()

    result = client.explain(
        df, features=["driver", "flag"], categorical_exog_list=["flag"], freq=freq
    )
    assert result["feature"].to_list() == ["driver", "flag"]


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_explain_serializes_pandas_nullable_feature_dtypes(dtype):
    client, request = _client_with_response(_explain_response)
    df = _explain_df()
    df["nullable"] = pd.array([1, 2, 3, 4, 5, 6], dtype=dtype)

    client.explain(df, features=["driver", "nullable"])

    values = request.call_args.args[2]["series"]["X"][1]
    assert isinstance(values, np.ndarray)
    assert values.dtype != object
    orjson.dumps(values, option=orjson.OPT_SERIALIZE_NUMPY)


def test_explain_default_features_reject_non_numeric_columns():
    client, request = _client_with_response(_explain_response)
    df = _explain_df().assign(label=["x", "y", "x", "y", "x", "y"])

    with pytest.raises(ValueError, match=r"not numeric: \['label'\]"):
        client.explain(df)

    request.assert_not_called()


def test_explain_rejects_non_numeric_weights():
    client, _ = _client_with_response(
        {"weights": ["a"], "feature_names": None, "method": "granger"}
    )

    with pytest.raises(RuntimeError, match="non-numeric weights"):
        client.explain(_explain_df(), features=["driver"])


@pytest.mark.parametrize(
    "response,match",
    [
        (
            {
                "weights": [1.0],
                "feature_names": None,
                "method": "granger",
            },
            "expected 2",
        ),
        (
            {
                "weights": [0.5, 0.5],
                "feature_names": None,
                "method": "transfer_entropy",
            },
            "metadata",
        ),
    ],
)
def test_explain_rejects_malformed_responses(response, match):
    client, _ = _client_with_response(response)

    with pytest.raises(RuntimeError, match=match):
        client.explain(_explain_df(), features=["driver", "noise"])


def test_explain_rejects_distributed_or_unknown_dataframe_types():
    client, request = _client_with_response(_explain_response)

    with pytest.raises(ValueError, match="pandas and polars"):
        client.explain([1.0, 2.0])

    request.assert_not_called()


@pytest.mark.parametrize(
    "make_driver",
    [
        pytest.param(
            lambda: pd.Series([np.nan, 20.0, 100.0, 10.0, 300.0, 30.0]),
            id="pandas-nan",
        ),
        pytest.param(
            lambda: pd.Series([pd.NA, 20, 100, 10, 300, 30], dtype="Int64"),
            id="pandas-nullable-int",
        ),
    ],
)
def test_explain_warns_and_ships_nan_for_pandas_missing_values(make_driver, caplog):
    client, request = _client_with_response(_explain_response)
    df = _explain_df()
    df["driver"] = make_driver()

    with caplog.at_level(logging.WARNING, logger="nixtla.nixtla_client"):
        result = client.explain(df)

    assert "missing values: ['driver']" in caplog.text
    assert result["feature"].tolist() == ["driver", "noise"]
    payload = request.call_args.args[2]
    driver_values = np.asarray(payload["series"]["X"][0], dtype=np.float64)
    assert np.isnan(driver_values).sum() == 1


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([None, 20, 100, 10, 300, 30], id="int-null"),
        pytest.param([float("nan"), 20.0, 100.0, 10.0, 300.0, 30.0], id="nan"),
    ],
)
def test_explain_warns_and_ships_nan_for_polars_missing_values(values, caplog):
    client, request = _client_with_response(_explain_response)
    df = pl.from_pandas(_explain_df()).with_columns(pl.Series("driver", values))

    with caplog.at_level(logging.WARNING, logger="nixtla.nixtla_client"):
        result = client.explain(df, freq="1d")

    assert "missing values: ['driver']" in caplog.text
    assert result["feature"].to_list() == ["driver", "noise"]
    payload = request.call_args.args[2]
    driver_values = np.asarray(payload["series"]["X"][0], dtype=np.float64)
    assert np.isnan(driver_values).sum() == 1


@pytest.mark.integration
@pytest.mark.parametrize("method", ["granger", "transfer_entropy"])
def test_explain_live_endpoint_returns_normalized_weights(nixtla_test_client, method):
    n = 160
    rng = np.random.default_rng(42)
    driver = rng.normal(size=n)
    target = np.zeros(n)
    target[1:] = 0.9 * driver[:-1] + 0.1 * rng.normal(size=n - 1)
    df = pd.DataFrame(
        {
            "unique_id": "series-0",
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": target,
            "driver": driver,
            "noise": rng.normal(size=n),
        }
    )

    result = nixtla_test_client.explain(df, method=method)

    assert result["feature"].tolist() == ["driver", "noise"]
    assert result["method"].eq(method).all()
    assert result["weight"].ge(0).all()
    assert np.isclose(result["weight"].sum(), 1.0)
    # `y` is planted to follow the lagged driver, so the driver must be
    # weighted above the pure-noise feature for either method.
    weights = result.set_index("feature")["weight"]
    assert weights["driver"] > weights["noise"]


class _OversizedBody:
    def __len__(self):
        return 201 * 2**20


@pytest.mark.parametrize(
    "endpoint,payload,expected,not_expected",
    [
        ("v2/explain", {"series": {}}, "cannot be partitioned", "num_partitions"),
        ("v2/forecast", {"series": {}}, "num_partitions", "cannot be partitioned"),
        (
            "v2/simulate",
            {"series": {}, "multivariate": False},
            "num_partitions",
            "cannot be partitioned",
        ),
        (
            "v2/simulate",
            {"series": {}, "multivariate": True},
            "cannot be partitioned",
            "num_partitions",
        ),
    ],
)
def test_oversized_payload_message_is_actionable_per_endpoint(
    monkeypatch, endpoint, payload, expected, not_expected
):
    import nixtla.nixtla_client as client_module

    monkeypatch.setattr(client_module.orjson, "dumps", lambda *a, **k: _OversizedBody())
    client = NixtlaClient(api_key="test", max_retries=1)

    with pytest.raises(ValueError) as excinfo:
        client._make_request(
            client=MagicMock(),
            endpoint=endpoint,
            payload=payload,
            multithreaded_compress=False,
        )

    message = str(excinfo.value)
    assert expected in message
    assert not_expected not in message
