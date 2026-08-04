from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from nixtla import NixtlaClient


def _df(n_series=2, n=6):
    return pd.DataFrame(
        {
            "unique_id": np.repeat([f"id-{i}" for i in range(n_series)], n),
            "ds": list(pd.date_range("2024-01-01", periods=n, freq="D")) * n_series,
            "y": np.arange(n_series * n, dtype=float),
            "x": np.arange(n_series * n, dtype=float),
        }
    )


def _X_df(h=2, n=6, category_order=None):
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=n)
    times = pd.date_range(start, periods=h, freq="D")
    X_df = pd.DataFrame(
        {
            "unique_id": np.repeat(["id-0", "id-1"], h),
            "ds": list(times) * 2,
            "x": [10.0, 11.0, 20.0, 21.0],
        }
    )
    if category_order is not None:
        # reverse the row order too, so the frame is not already sorted
        X_df = X_df.iloc[::-1].reset_index(drop=True)
        X_df["unique_id"] = pd.Categorical(X_df["unique_id"], categories=category_order)
    return X_df


def _forecast_client(h=2, n_series=2):
    client = NixtlaClient(api_key="test", max_retries=1)
    client._get_model_params = MagicMock(return_value=(28, 7))
    client._make_client = MagicMock()
    client._make_request_with_retries = MagicMock(
        return_value={
            "mean": [0.0] * (h * n_series),
            "intervals": None,
            "weights_x": None,
        }
    )
    return client


def _simulate_client(h=2, n_series=2, n_paths=1):
    client = NixtlaClient(api_key="test", max_retries=1)
    client._get_model_params = MagicMock(return_value=(28, 7))
    client._make_client = MagicMock()
    client._make_request_with_retries = MagicMock(
        return_value={
            "samples": [0.0] * (n_paths * n_series * h),
            "sizes": [h] * n_series,
            "n_paths": n_paths,
            "h": h,
            "coupled": False,
        }
    )
    return client


def _sent_X_future(client):
    payload = client._make_request_with_retries.call_args.args[2]
    return [[float(v) for v in row] for row in payload["series"]["X_future"]]


# `df` and `X_df` are processed independently, so pandas categoricals with
# opposite category orders sort into different series orders. `X_future` is
# positional, so without realignment id-0 receives id-1's future values.
_EXPECTED = [[10.0, 11.0, 20.0, 21.0]]


def test_forecast_aligns_future_exog_when_category_orders_differ():
    client = _forecast_client()
    df = _df()
    df["unique_id"] = pd.Categorical(df["unique_id"], categories=["id-0", "id-1"])

    client.forecast(df=df, X_df=_X_df(category_order=["id-1", "id-0"]), h=2, freq="D")

    assert _sent_X_future(client) == _EXPECTED


def test_simulate_aligns_future_exog_when_category_orders_differ():
    client = _simulate_client()
    df = _df()
    df["unique_id"] = pd.Categorical(df["unique_id"], categories=["id-0", "id-1"])

    client.simulate(
        df=df, X_df=_X_df(category_order=["id-1", "id-0"]), h=2, freq="D", n_paths=1
    )

    assert _sent_X_future(client) == _EXPECTED


def test_forecast_leaves_matching_order_untouched():
    client = _forecast_client()

    client.forecast(df=_df(), X_df=_X_df(), h=2, freq="D")

    assert _sent_X_future(client) == _EXPECTED


@pytest.mark.parametrize("method", ["forecast", "simulate"])
def test_rejects_future_exog_with_mismatched_ids(method):
    client = _forecast_client() if method == "forecast" else _simulate_client()
    X_df = _X_df()
    X_df["unique_id"] = np.repeat(["id-0", "other"], 2)

    with pytest.raises(ValueError, match="same values of `unique_id`"):
        if method == "forecast":
            client.forecast(df=_df(), X_df=X_df, h=2, freq="D")
        else:
            client.simulate(df=_df(), X_df=X_df, h=2, freq="D", n_paths=1)

    client._make_request_with_retries.assert_not_called()
