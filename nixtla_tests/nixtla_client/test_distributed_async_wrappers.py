from unittest.mock import MagicMock

import pandas as pd

from nixtla.nixtla_client import _cross_validation_wrapper, _forecast_wrapper


def _small_df(n=20):
    return pd.DataFrame(
        {
            "unique_id": "id_0",
            "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
            "y": range(n),
        }
    )


def test_forecast_wrapper_forwards_async_kwargs():
    mock_client = MagicMock()

    _forecast_wrapper(
        df=_small_df(),
        client=mock_client,
        h=5,
        freq="D",
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        level=None,
        quantiles=None,
        finetune_steps=0,
        finetune_depth=1,
        finetune_loss="default",
        finetuned_model_id=None,
        clean_ex_first=True,
        hist_exog_list=None,
        categorical_exog_list=None,
        validate_api_key=False,
        add_history=False,
        date_features=False,
        date_features_to_one_hot=False,
        model="timegpt-2.1",
        num_partitions=None,
        feature_contributions=False,
        model_parameters=None,
        multivariate=False,
        _is_async_job=True,
        _poll_interval=7,
        _poll_timeout=42,
    )

    mock_client.forecast.assert_called_once()
    kwargs = mock_client.forecast.call_args.kwargs
    assert kwargs["_is_async_job"] is True
    assert kwargs["_poll_interval"] == 7
    assert kwargs["_poll_timeout"] == 42


def test_forecast_wrapper_defaults_are_sync():
    mock_client = MagicMock()

    _forecast_wrapper(
        df=_small_df(),
        client=mock_client,
        h=5,
        freq="D",
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        level=None,
        quantiles=None,
        finetune_steps=0,
        finetune_depth=1,
        finetune_loss="default",
        finetuned_model_id=None,
        clean_ex_first=True,
        hist_exog_list=None,
        categorical_exog_list=None,
        validate_api_key=False,
        add_history=False,
        date_features=False,
        date_features_to_one_hot=False,
        model="timegpt-2.1",
        num_partitions=None,
        feature_contributions=False,
        model_parameters=None,
        multivariate=False,
    )

    kwargs = mock_client.forecast.call_args.kwargs
    assert kwargs["_is_async_job"] is False


def test_cross_validation_wrapper_forwards_async_kwargs():
    mock_client = MagicMock()

    _cross_validation_wrapper(
        df=_small_df(),
        client=mock_client,
        h=5,
        freq="D",
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        level=None,
        quantiles=None,
        validate_api_key=False,
        n_windows=1,
        step_size=None,
        finetune_steps=0,
        finetune_depth=1,
        finetune_loss="default",
        finetuned_model_id=None,
        refit=True,
        clean_ex_first=True,
        hist_exog_list=None,
        date_features=False,
        date_features_to_one_hot=False,
        model="timegpt-2.1",
        num_partitions=None,
        model_parameters=None,
        multivariate=False,
        categorical_exog_list=None,
        _is_async_job=True,
        _poll_interval=7,
        _poll_timeout=42,
    )

    mock_client.cross_validation.assert_called_once()
    kwargs = mock_client.cross_validation.call_args.kwargs
    assert kwargs["_is_async_job"] is True
    assert kwargs["_poll_interval"] == 7
    assert kwargs["_poll_timeout"] == 42
