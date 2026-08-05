import inspect
import logging
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nixtla import NixtlaClient

_RUN_LIVE_ENDPOINT_TESTS = os.getenv("NIXTLA_RUN_SIMULATE_EXPLAIN_TESTS") == "1"


def _client_with_response(response, model_params=(28, 7)):
    client = NixtlaClient(api_key="test", max_retries=1)
    client._make_client = MagicMock()
    client._get_model_params = MagicMock(return_value=model_params)

    # Mirrors NixtlaClient._make_request_with_retries: the single-request path
    # calls it positionally, the partitioned path by keyword.
    def respond(client=None, endpoint=None, payload=None, multithreaded_compress=True):
        return response(endpoint, payload) if callable(response) else response

    request = MagicMock(side_effect=respond)
    client._make_request_with_retries = request
    return client, request


def _payload_of(call):
    if len(call.args) > 2:
        return call.args[2]
    return call.kwargs["payload"]


def _series_df(n_series=1, n=4):
    return pd.DataFrame(
        {
            "unique_id": np.repeat([f"id-{i}" for i in range(n_series)], n),
            "ds": list(pd.date_range("2024-01-01", periods=n, freq="D")) * n_series,
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
    assert (
        result["unique_id"].tolist()
        == [
            "id-0",
            "id-0",
            "id-1",
            "id-1",
        ]
        * 2
    )
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
            "ds": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True),
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


def test_simulate_sorts_categorical_history_for_unsorted_multi_series_input():
    client, request = _client_with_response(_simulate_response)
    df = pd.DataFrame(
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
            "trend": [200.0, 20.0, 100.0, 10.0, 300.0, 30.0],
            "segment": ["B2", "A2", "B1", "A1", "B3", "A3"],
        }
    )

    client.simulate(
        df=df,
        h=1,
        freq="D",
        n_paths=1,
        hist_exog_list=["trend", "segment"],
        categorical_exog_list=["segment"],
    )

    series = request.call_args.args[2]["series"]
    assert series["sizes"].tolist() == [3, 3]
    assert series["y"].tolist() == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    numeric, categorical = series["X"]
    assert numeric.tolist() == [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    assert categorical == ["A1", "A2", "A3", "B1", "B2", "B3"]
    assert series["categorical_exog"] == [1]


def test_simulate_polars_categorical_and_future_exog():
    client, request = _client_with_response(_simulate_response)
    df = pl.DataFrame(
        {
            "unique_id": ["id-0"] * 4,
            "ds": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True),
            "y": [1.0, 2.0, 3.0, 4.0],
            "price": [10.0, 11.0, 12.0, 13.0],
            "event": ["none", "sale", "none", "sale"],
        }
    )
    X_df = pl.DataFrame(
        {
            "unique_id": ["id-0", "id-0"],
            "ds": pl.date_range(pl.date(2024, 1, 5), pl.date(2024, 1, 6), eager=True),
            "price": [14.0, 15.0],
            "event": ["none", "sale"],
        }
    )

    result = client.simulate(
        df=df,
        X_df=X_df,
        h=2,
        freq="1d",
        n_paths=2,
        categorical_exog_list=["event"],
    )

    assert isinstance(result, pl.DataFrame)
    series = request.call_args.args[2]["series"]
    assert series["categorical_exog"] == [1]
    assert [
        row.tolist() if isinstance(row, np.ndarray) else row for row in series["X"]
    ] == [
        [10.0, 11.0, 12.0, 13.0],
        ["none", "sale", "none", "sale"],
    ]
    assert [
        row.tolist() if isinstance(row, np.ndarray) else row
        for row in series["X_future"]
    ] == [[14.0, 15.0], ["none", "sale"]]


def test_simulate_mirrors_server_coupled_false_for_multivariate_request():
    client, request = _client_with_response(
        lambda _endpoint, payload: {
            "samples": [1.0] * (payload["n_paths"] * 2 * payload["h"]),
            "n_paths": payload["n_paths"],
            "h": payload["h"],
            "sizes": [payload["h"]] * 2,
            "coupled": False,
        }
    )

    result = client.simulate(
        _series_df(n_series=2, n=4), h=2, freq="D", n_paths=1, multivariate=True
    )

    assert request.call_args.args[2]["multivariate"] is True
    assert result["coupled"].tolist() == [False] * 4


def test_simulate_warns_when_horizon_exceeds_model_horizon(caplog):
    client, _ = _client_with_response(_simulate_response, model_params=(28, 2))

    with caplog.at_level(logging.WARNING, logger="nixtla.nixtla_client"):
        client.simulate(_series_df(n=8), h=5, freq="D", n_paths=1)

    assert any("exceeds the model horizon" in r.message for r in caplog.records)


def test_simulate_does_not_warn_within_model_horizon(caplog):
    client, _ = _client_with_response(_simulate_response, model_params=(28, 7))

    with caplog.at_level(logging.WARNING, logger="nixtla.nixtla_client"):
        client.simulate(_series_df(n=8), h=5, freq="D", n_paths=1)

    assert not any("exceeds the model horizon" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "invalid_part,match",
    [
        ("ids", "same values of `unique_id`"),
        ("times", "exactly one row for every future"),
    ],
)
def test_simulate_rejects_misaligned_future_keys(invalid_part, match):
    client, request = _client_with_response(_simulate_response)
    df = _series_df(n_series=2, n=3).assign(x=np.arange(6, dtype=float))
    X_df = pd.DataFrame(
        {
            "unique_id": ["id-0", "id-0", "id-1", "id-1"],
            "ds": list(pd.date_range("2024-01-04", periods=2, freq="D")) * 2,
            "x": [10.0, 11.0, 20.0, 21.0],
        }
    )
    if invalid_part == "ids":
        X_df["unique_id"] = ["other-0", "other-0", "other-1", "other-1"]
    else:
        X_df["ds"] = list(pd.date_range("2024-01-03", periods=2, freq="D")) * 2

    with pytest.raises(ValueError, match=match):
        client.simulate(df=df, X_df=X_df, h=2, freq="D", n_paths=1)

    request.assert_not_called()


def test_simulate_aligns_future_exog_when_category_orders_differ():
    # `df` and `X_df` declare opposite category orders, so each is sorted into a
    # different series order. `X_future` is positional, so without realignment
    # id-0 would receive id-1's future values.
    client, request = _client_with_response(_simulate_response)
    df = _series_df(n_series=2, n=3).assign(x=np.arange(6, dtype=float))
    df["unique_id"] = pd.Categorical(df["unique_id"], categories=["id-0", "id-1"])
    X_df = pd.DataFrame(
        {
            "unique_id": pd.Categorical(
                ["id-1", "id-0", "id-1", "id-0"],
                categories=["id-1", "id-0"],
            ),
            "ds": pd.to_datetime(
                ["2024-01-05", "2024-01-04", "2024-01-04", "2024-01-05"]
            ),
            "x": [21.0, 10.0, 20.0, 11.0],
        }
    )

    client.simulate(df=df, X_df=X_df, h=2, freq="D", n_paths=1)

    payload = request.call_args.args[2]
    # history is ordered id-0 then id-1, so the future rows must be too
    assert [list(row) for row in payload["series"]["X_future"]] == [
        [10.0, 11.0, 20.0, 21.0]
    ]


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
        ({"n_paths": 1.5}, "integer"),
        ({"n_paths": 0}, "between 1 and 10,000"),
        ({"n_paths": -1}, "between 1 and 10,000"),
        ({"n_paths": 10_001}, "between 1 and 10,000"),
        ({"quantiles": [[0.1, 0.5]]}, "one-dimensional"),
        ({"quantiles": ["a", "b"]}, "list of numbers"),
        ({"quantiles": []}, "between 2 and 200 values"),
        ({"quantiles": [0.5]}, "between 2 and 200 values"),
        ({"quantiles": np.linspace(0.001, 0.999, 201).tolist()}, "between 2 and 200"),
        ({"quantiles": [0.0, 0.5]}, r"strictly inside \(0, 1\)"),
        ({"quantiles": [0.5, 1.0]}, r"strictly inside \(0, 1\)"),
        ({"quantiles": [0.1, np.nan]}, r"strictly inside \(0, 1\)"),
        ({"quantiles": [0.9, 0.1]}, "strictly increasing"),
        ({"quantiles": [0.5, 0.5]}, "strictly increasing"),
        ({"seed": 1.5}, "integer"),
        ({"seed": -(2**63) - 1}, "must be between"),
        ({"seed": 2**64}, "must be between"),
        ({"num_partitions": 0}, "positive integer"),
        ({"num_partitions": 1.5}, "positive integer"),
        (
            {"num_partitions": 2, "multivariate": True},
            "cannot be combined with `multivariate=True`",
        ),
    ],
)
def test_simulate_rejects_invalid_options_before_request(kwargs, match):
    client, request = _client_with_response(_simulate_response)
    params = {"df": _series_df(), "h": 2, "freq": "D", "n_paths": 1}
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        client.simulate(**params)

    request.assert_not_called()


@pytest.mark.parametrize("seed", [-(2**63), 2**64 - 1])
def test_simulate_accepts_seed_boundaries(seed):
    client, request = _client_with_response(_simulate_response)

    client.simulate(_series_df(), h=2, freq="D", n_paths=1, seed=seed)

    assert request.call_args.args[2]["seed"] == seed


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
        (
            {
                "samples": ["a", "b"],
                "n_paths": 1,
                "h": 2,
                "sizes": [2],
                "coupled": False,
            },
            "non-numeric sizes or samples",
        ),
        (
            {
                "samples": [1.0, 2.0],
                "n_paths": 1,
                "h": 2,
                "sizes": [2],
                "coupled": "false",
            },
            "non-boolean `coupled`",
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


def test_simulate_coerces_null_samples_to_float_nan():
    client, _ = _client_with_response(
        {
            "samples": [None, None],
            "n_paths": 1,
            "h": 2,
            "sizes": [2],
            "coupled": False,
        }
    )

    result = client.simulate(_series_df(), h=2, freq="D", n_paths=1)

    assert result["TimeGPT"].dtype == np.float64
    assert result["TimeGPT"].isna().all()


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


def _addressable_simulate_response(n_hist):
    """Server stub whose values encode (sample_id, global series, step).

    Each partition only sees its own slice, so it recovers the global series
    index from the `y` block. Any mis-ordering in the merge is then visible in
    the returned values rather than only in the row count.
    """

    def respond(endpoint, payload):
        assert endpoint == "v2/simulate"
        sizes = payload["series"]["sizes"]
        n_series = len(sizes)
        first_global = int(np.asarray(payload["series"]["y"])[0] // n_hist)
        values = [
            sample * 10_000 + (first_global + series) * 100 + step
            for sample in range(payload["n_paths"])
            for series in range(n_series)
            for step in range(payload["h"])
        ]
        return {
            "samples": values,
            "n_paths": payload["n_paths"],
            "h": payload["h"],
            "sizes": [payload["h"]] * n_series,
            "coupled": False,
        }

    return respond


@pytest.mark.parametrize("num_partitions", [2, 3, 7, 10])
def test_simulate_partitioned_output_matches_single_request(num_partitions):
    n_series, n_hist, h, n_paths = 7, 6, 3, 4
    df = _series_df(n_series=n_series, n=n_hist)

    single, _ = _client_with_response(_addressable_simulate_response(n_hist))
    expected = single.simulate(df, h=h, freq="D", n_paths=n_paths)

    client, request = _client_with_response(_addressable_simulate_response(n_hist))
    result = client.simulate(
        df, h=h, freq="D", n_paths=n_paths, num_partitions=num_partitions
    )

    pd.testing.assert_frame_equal(result, expected)
    assert request.call_count == min(num_partitions, n_series)


def test_simulate_partitioning_preserves_exogenous_features():
    n_series, n_hist, h = 5, 4, 2
    df = _series_df(n_series=n_series, n=n_hist).assign(
        price=np.arange(100, 100 + n_series * n_hist, dtype=float),
        event=[f"e{i}" for i in range(n_series * n_hist)],
    )
    X_df = pd.DataFrame(
        {
            "unique_id": np.repeat([f"id-{i}" for i in range(n_series)], h),
            "ds": list(pd.date_range("2024-01-05", periods=h, freq="D")) * n_series,
            "price": np.arange(900, 900 + n_series * h, dtype=float),
            "event": [f"f{i}" for i in range(n_series * h)],
        }
    )
    kwargs = dict(
        df=df, X_df=X_df, h=h, freq="D", n_paths=2, categorical_exog_list=["event"]
    )

    single, single_request = _client_with_response(_simulate_response)
    single.simulate(**kwargs)
    whole = _payload_of(single_request.call_args)["series"]

    client, request = _client_with_response(_simulate_response)
    client.simulate(**kwargs, num_partitions=3)
    parts = [_payload_of(call)["series"] for call in request.call_args_list]

    def rows(series, key):
        return [
            row.tolist() if isinstance(row, np.ndarray) else list(row)
            for row in (series[key] or [])
        ]

    for key in ("X", "X_future"):
        rejoined = [
            sum((rows(part, key)[i] for part in parts), [])
            for i in range(len(rows(whole, key)))
        ]
        assert rejoined == rows(whole, key), key
    # Categorical indices address feature rows, so they are identical per partition.
    for part in parts:
        assert part["categorical_exog"] == whole["categorical_exog"]


def test_simulate_partitioning_rejects_mismatched_partition_response():
    def bad(endpoint, payload):
        response = _simulate_response(endpoint, payload)
        response["samples"] = response["samples"][:-1]
        return response

    client, _ = _client_with_response(bad)

    with pytest.raises(RuntimeError, match="expected"):
        client.simulate(
            _series_df(n_series=4), h=2, freq="D", n_paths=2, num_partitions=2
        )
