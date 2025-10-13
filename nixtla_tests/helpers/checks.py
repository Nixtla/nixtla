import fugue
import fugue.api as fa
import numpy as np
import pandas as pd
import pytest
import time

from nixtla.nixtla_client import NixtlaClient
from typing import Callable

# setting used for distributed related tests
ATOL = 1e-3


# test num partitions
# we need to be sure that we can recover the same results
# using a for loop
# A: be aware that num partitons can produce different results
# when used finetune_steps
def check_num_partitions_same_results(method: Callable, num_partitions: int, **kwargs):
    res_partitioned = method(**kwargs, num_partitions=num_partitions)
    res_no_partitioned = method(**kwargs, num_partitions=1)
    sort_by = ["unique_id", "ds"]
    if "cutoff" in res_partitioned:
        sort_by.extend(["cutoff"])
    pd.testing.assert_frame_equal(
        res_partitioned.sort_values(sort_by).reset_index(drop=True),
        res_no_partitioned.sort_values(sort_by).reset_index(drop=True),
        rtol=1e-2,
        atol=1e-2,
    )


def check_retry_behavior(
    df,
    side_effect,
    side_effect_exception,
    max_retries=5,
    retry_interval=5,
    max_wait_time=40,
    should_retry=True,
    sleep_seconds=5,
):
    mock_nixtla_client = NixtlaClient(
        max_retries=max_retries,
        retry_interval=retry_interval,
        max_wait_time=max_wait_time,
    )
    mock_nixtla_client._make_request = side_effect
    init_time = time.time()
    with pytest.raises(side_effect_exception):
        mock_nixtla_client.forecast(
            df=df, h=12, time_col="timestamp", target_col="value"
        )
    total_mock_time = time.time() - init_time
    if should_retry:
        approx_expected_time = min((max_retries - 1) * retry_interval, max_wait_time)
        upper_expected_time = min(max_retries * retry_interval, max_wait_time)
        assert total_mock_time >= approx_expected_time, "It is not retrying as expected"
        # preprocessing time before the first api call should be less than 60 seconds
        assert (
            total_mock_time - upper_expected_time - (max_retries - 1) * sleep_seconds
            <= sleep_seconds
        )
    else:
        assert total_mock_time <= max_wait_time


# test we recover the same <mean> forecasts
# with and without restricting input
# (add_history)
def check_equal_fcsts_add_history(nixtla_client, **kwargs):
    fcst_no_rest_df = nixtla_client.forecast(**kwargs, add_history=True)
    fcst_no_rest_df = (
        fcst_no_rest_df.groupby("unique_id", observed=True)
        .tail(kwargs["h"])
        .reset_index(drop=True)
    )
    fcst_rest_df = nixtla_client.forecast(**kwargs)
    pd.testing.assert_frame_equal(
        fcst_no_rest_df,
        fcst_rest_df,
        atol=1e-4,
        rtol=1e-3,
    )
    return fcst_rest_df


def check_quantiles(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    id_col: str = "id_col",
    time_col: str = "time_col",
):
    test_qls = list(np.arange(0.1, 1, 0.1))
    exp_q_cols = [f"TimeGPT-q-{int(q * 100)}" for q in test_qls]

    def test_method_qls(method, **kwargs):
        df_qls = method(
            df=df, h=12, id_col=id_col, time_col=time_col, quantiles=test_qls, **kwargs
        )
        df_qls = fa.as_pandas(df_qls)
        assert all(col in df_qls.columns for col in exp_q_cols)
        # test monotonicity of quantiles
        df_qls.apply(lambda x: x.is_monotonic_increasing, axis=1).sum() == len(
            exp_q_cols
        )

    test_method_qls(nixtla_client.forecast)
    test_method_qls(nixtla_client.forecast, add_history=True)
    test_method_qls(nixtla_client.cross_validation)


def check_cv_same_results_num_partitions(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    horizon: int = 12,
    id_col: str = "unique_id",
    time_col: str = "ds",
    **fcst_kwargs,
):
    fcst_df = nixtla_client.cross_validation(
        df=df,
        h=horizon,
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs,
    )
    fcst_df = fa.as_pandas(fcst_df)
    fcst_df_2 = nixtla_client.cross_validation(
        df=df,
        h=horizon,
        num_partitions=2,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs,
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)
    pd.testing.assert_frame_equal(
        fcst_df.sort_values([id_col, time_col]).reset_index(drop=True),
        fcst_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        atol=ATOL,
    )


def check_forecast_diff_results_diff_models(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    horizon: int = 12,
    id_col: str = "unique_id",
    time_col: str = "ds",
    **fcst_kwargs,
):
    fcst_df = nixtla_client.forecast(
        df=df,
        h=horizon,
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        model="timegpt-1",
        **fcst_kwargs,
    )
    fcst_df = fa.as_pandas(fcst_df)
    fcst_df_2 = nixtla_client.forecast(
        df=df,
        h=horizon,
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        model="timegpt-1-long-horizon",
        **fcst_kwargs,
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)

    with pytest.raises(
        AssertionError, match=r'\(column name="TimeGPT"\) are different'
    ):
        pd.testing.assert_frame_equal(
            fcst_df.sort_values([id_col, time_col]).reset_index(drop=True),
            fcst_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        )


def check_forecast(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    horizon: int = 12,
    id_col: str = "unique_id",
    time_col: str = "ds",
    n_series_to_check: int = 4,
    **fcst_kwargs,
):
    fcst_df = nixtla_client.forecast(
        df=df,
        h=horizon,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs,
    )
    fcst_df = fa.as_pandas(fcst_df)
    assert n_series_to_check * 12 == len(fcst_df)
    cols = fcst_df.columns.to_list()
    exp_cols = [id_col, time_col, "TimeGPT"]
    if "level" in fcst_kwargs:
        level = sorted(fcst_kwargs["level"])
        exp_cols.extend([f"TimeGPT-lo-{lv}" for lv in reversed(level)])
        exp_cols.extend([f"TimeGPT-hi-{lv}" for lv in level])
    assert cols == exp_cols


def check_forecast_same_results_num_partitions(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    horizon: int = 12,
    id_col: str = "unique_id",
    time_col: str = "ds",
    **fcst_kwargs,
):
    fcst_df = nixtla_client.forecast(
        df=df,
        h=horizon,
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs,
    )
    fcst_df = fa.as_pandas(fcst_df)
    fcst_df_2 = nixtla_client.forecast(
        df=df,
        h=horizon,
        num_partitions=2,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs,
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)
    pd.testing.assert_frame_equal(
        fcst_df.sort_values([id_col, time_col]).reset_index(drop=True),
        fcst_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        atol=ATOL,
    )


def check_forecast_dataframe(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    n_series_to_check: int = 4,
):
    check_cv_same_results_num_partitions(nixtla_client, df, n_windows=2, step_size=1)
    check_cv_same_results_num_partitions(
        nixtla_client, df, n_windows=3, step_size=None, horizon=1
    )
    check_cv_same_results_num_partitions(
        nixtla_client, df, model="timegpt-1-long-horizon", horizon=1
    )
    check_forecast_diff_results_diff_models(nixtla_client, df)
    check_forecast(nixtla_client, df, num_partitions=1)
    check_forecast(
        nixtla_client,
        df,
        level=[90, 80],
        num_partitions=1,
        n_series_to_check=n_series_to_check,
    )
    check_forecast_same_results_num_partitions(nixtla_client, df)


def check_forecast_dataframe_diff_cols(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    id_col: str = "id_col",
    time_col: str = "time_col",
    target_col: str = "target_col",
):
    check_forecast(
        nixtla_client,
        df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        num_partitions=1,
    )
    check_forecast(
        nixtla_client,
        df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        level=[90, 80],
        num_partitions=1,
    )
    check_forecast_same_results_num_partitions(
        nixtla_client, df, id_col=id_col, time_col=time_col, target_col=target_col
    )


def check_anomalies(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    **anomalies_kwargs,
):
    anomalies_df = nixtla_client.detect_anomalies(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **anomalies_kwargs,
    )
    anomalies_df = fa.as_pandas(anomalies_df)
    assert (fa.as_pandas(df)[id_col].unique() == anomalies_df[id_col].unique()).all()
    cols = anomalies_df.columns.to_list()
    level = anomalies_kwargs.get("level", 99)
    exp_cols = [
        id_col,
        time_col,
        target_col,
        "TimeGPT",
        "anomaly",
        f"TimeGPT-lo-{level}",
        f"TimeGPT-hi-{level}",
    ]
    assert cols == exp_cols


def check_anomalies_same_results_num_partitions(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    **anomalies_kwargs,
):
    anomalies_df = nixtla_client.detect_anomalies(
        df=df,
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **anomalies_kwargs,
    )
    anomalies_df = fa.as_pandas(anomalies_df)
    anomalies_df_2 = nixtla_client.detect_anomalies(
        df=df,
        num_partitions=2,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **anomalies_kwargs,
    )
    anomalies_df_2 = fa.as_pandas(anomalies_df_2)
    pd.testing.assert_frame_equal(
        anomalies_df.sort_values([id_col, time_col]).reset_index(drop=True),
        anomalies_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        atol=ATOL,
    )


def check_anomalies_dataframe(nixtla_client: NixtlaClient, df: fugue.AnyDataFrame):
    check_anomalies(nixtla_client, df, num_partitions=1)
    check_anomalies(nixtla_client, df, level=90, num_partitions=1)
    check_anomalies_same_results_num_partitions(nixtla_client, df)


def check_online_anomalies(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    level=99,
    **reatlime_anomalies_kwargs,
):
    anomalies_df = nixtla_client.detect_anomalies_online(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **reatlime_anomalies_kwargs,
    )
    anomalies_df = fa.as_pandas(anomalies_df)
    assert (fa.as_pandas(df)[id_col].unique() == anomalies_df[id_col].unique()).all()
    cols = anomalies_df.columns.to_list()
    exp_cols = [
        id_col,
        time_col,
        target_col,
        "TimeGPT",
        "anomaly",
        "anomaly_score",
        f"TimeGPT-lo-{level}",
        f"TimeGPT-hi-{level}",
    ]
    assert cols == exp_cols


def check_anomalies_online_same_results_num_partitions(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    **reatlime_anomalies_kwargs,
):
    anomalies_df = nixtla_client.detect_anomalies_online(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        num_partitions=1,
        **reatlime_anomalies_kwargs,
    )
    anomalies_df = fa.as_pandas(anomalies_df)
    anomalies_df_2 = nixtla_client.detect_anomalies_online(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        num_partitions=2,
        **reatlime_anomalies_kwargs,
    )
    anomalies_df_2 = fa.as_pandas(anomalies_df_2)
    pd.testing.assert_frame_equal(
        anomalies_df.sort_values([id_col, time_col]).reset_index(drop=True),
        anomalies_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        atol=ATOL,
    )


def check_anomalies_online_dataframe(
    nixtla_client: NixtlaClient, df: fugue.AnyDataFrame
):
    check_online_anomalies(
        nixtla_client,
        df,
        h=20,
        detection_size=5,
        threshold_method="univariate",
        level=99,
        num_partitions=1,
    )
    check_anomalies_online_same_results_num_partitions(
        nixtla_client,
        df,
        h=20,
        detection_size=5,
        threshold_method="univariate",
        level=99,
    )


def check_anomalies_dataframe_diff_cols(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    id_col: str = "id_col",
    time_col: str = "time_col",
    target_col: str = "target_col",
):
    check_anomalies(
        nixtla_client,
        df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        num_partitions=1,
    )
    check_anomalies(
        nixtla_client,
        df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        level=90,
        num_partitions=1,
    )
    check_anomalies_same_results_num_partitions(
        nixtla_client, df, id_col=id_col, time_col=time_col, target_col=target_col
    )


def check_forecast_x(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    X_df: fugue.AnyDataFrame,
    horizon: int = 24,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    **fcst_kwargs,
):
    fcst_df = nixtla_client.forecast(
        df=df,
        X_df=X_df,
        h=horizon,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **fcst_kwargs,
    )
    fcst_df = fa.as_pandas(fcst_df)
    n_series = fa.as_pandas(X_df)[id_col].nunique()
    assert n_series * horizon == len(fcst_df)

    cols = fcst_df.columns.to_list()
    exp_cols = [id_col, time_col, "TimeGPT"]
    if "level" in fcst_kwargs:
        level = sorted(fcst_kwargs["level"])
        exp_cols.extend([f"TimeGPT-lo-{lv}" for lv in reversed(level)])
        exp_cols.extend([f"TimeGPT-hi-{lv}" for lv in level])
    assert cols == exp_cols

    fcst_df_2 = nixtla_client.forecast(
        df=df,
        h=horizon,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **fcst_kwargs,
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)
    equal_arrays = np.array_equal(
        fcst_df.sort_values([id_col, time_col])["TimeGPT"].values,
        fcst_df_2.sort_values([id_col, time_col])["TimeGPT"].values,
    )
    assert not equal_arrays, "Forecasts with and without ex vars are equal"


def check_forecast_x_same_results_num_partitions(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    X_df: fugue.AnyDataFrame,
    horizon: int = 24,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    **fcst_kwargs,
):
    fcst_df = nixtla_client.forecast(
        df=df,
        X_df=X_df,
        h=horizon,
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **fcst_kwargs,
    )
    fcst_df = fa.as_pandas(fcst_df)
    fcst_df_2 = nixtla_client.forecast(
        df=df,
        h=horizon,
        num_partitions=2,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **fcst_kwargs,
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)
    equal_arrays = np.array_equal(
        fcst_df.sort_values([id_col, time_col])["TimeGPT"].values,
        fcst_df_2.sort_values([id_col, time_col])["TimeGPT"].values,
    )
    assert not equal_arrays, "Forecasts with and without ex vars are equal"


def check_forecast_x_dataframe(
    nixtla_client: NixtlaClient, df: fugue.AnyDataFrame, X_df: fugue.AnyDataFrame
):
    check_forecast_x(nixtla_client, df, X_df, num_partitions=1)
    check_forecast_x(nixtla_client, df, X_df, level=[90, 80], num_partitions=1)
    check_forecast_x_same_results_num_partitions(nixtla_client, df, X_df)


def check_forecast_x_dataframe_diff_cols(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    X_df: fugue.AnyDataFrame,
    id_col: str = "id_col",
    time_col: str = "time_col",
    target_col: str = "target_col",
):
    check_forecast_x(
        nixtla_client,
        df,
        X_df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        num_partitions=1,
    )
    check_forecast_x(
        nixtla_client,
        df,
        X_df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        level=[90, 80],
        num_partitions=1,
    )
    check_forecast_x_same_results_num_partitions(
        nixtla_client, df, X_df, id_col=id_col, time_col=time_col, target_col=target_col
    )


def check_finetuned_model(
    nixtla_client: NixtlaClient,
    df: fugue.AnyDataFrame,
    model_id2: str,
):
    # fine-tuning on distributed fails
    with pytest.raises(
        ValueError, match="Can only fine-tune on pandas or polars dataframes."
    ):
        nixtla_client.finetune(df=df)

    # forecast
    local_fcst = nixtla_client.forecast(
        df=fa.as_pandas(df), h=5, finetuned_model_id=model_id2,
    )
    distr_fcst = (
        fa.as_pandas(nixtla_client.forecast(df=df, h=5, finetuned_model_id=model_id2))
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        local_fcst,
        distr_fcst,
        check_dtype=False,
        atol=1e-4,
        rtol=1e-2,
    )

    # cross-validation
    local_cv = nixtla_client.cross_validation(
        df=fa.as_pandas(df), n_windows=2, h=5, finetuned_model_id=model_id2
    )
    distr_cv = (
        fa.as_pandas(
            nixtla_client.cross_validation(
                df=df, n_windows=2, h=5, finetuned_model_id=model_id2
            )
        )
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        local_cv,
        distr_cv[local_cv.columns],
        check_dtype=False,
        atol=1e-4,
        rtol=1e-2,
    )

    # anomaly detection
    local_anomaly = nixtla_client.detect_anomalies(
        df=fa.as_pandas(df), finetuned_model_id=model_id2
    )
    distr_anomaly = (
        fa.as_pandas(
            nixtla_client.detect_anomalies(df=df, finetuned_model_id=model_id2)
        )
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        local_anomaly,
        distr_anomaly[local_anomaly.columns],
        check_dtype=False,
        atol=1e-3,
        rtol=1e-2,
    )
