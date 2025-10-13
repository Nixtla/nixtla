import platform
import sys

import pytest

from nixtla_tests.helpers.checks import (
    check_anomalies_dataframe,
    check_anomalies_dataframe_diff_cols,
    check_anomalies_online_dataframe,
    check_forecast_dataframe,
    check_forecast_dataframe_diff_cols,
    check_forecast_x_dataframe,
    check_forecast_x_dataframe_diff_cols,
    check_quantiles,
)

pytestmark = [
    pytest.mark.distributed_run,
    pytest.mark.ray_run,
]


def test_quantiles(nixtla_test_client, ray_df):
    check_quantiles(nixtla_test_client, ray_df, id_col="unique_id", time_col="ds")


def test_forecast(nixtla_test_client, ray_df, ray_diff_cols_df, distributed_n_series):
    check_forecast_dataframe(
        nixtla_test_client, ray_df, n_series_to_check=distributed_n_series
    )
    check_forecast_dataframe_diff_cols(nixtla_test_client, ray_diff_cols_df)


def test_anomalies(nixtla_test_client, ray_df, ray_diff_cols_df):
    check_anomalies_dataframe(nixtla_test_client, ray_df)
    check_anomalies_dataframe_diff_cols(nixtla_test_client, ray_diff_cols_df)


def test_anomalies_online(nixtla_test_client, ray_df):
    check_anomalies_online_dataframe(nixtla_test_client, ray_df)

@pytest.mark.xfail(
    reason=(
        "triad.collections.schema.SchemaError: Schema can't be empty"
        "error triggered https://github.com/Nixtla/nixtla/blob/b56a89bf6b80b137c57f3511eef3ed8857705a59/nixtla/nixtla_client.py#L1383"
    )
)
def test_forecast_x_dataframe(
    nixtla_test_client,
    ray_df_x,
    ray_future_ex_vars_df,
    ray_df_x_diff_cols,
    ray_future_ex_vars_df_diff_cols,
):
    check_forecast_x_dataframe(nixtla_test_client, ray_df_x, ray_future_ex_vars_df)
    check_forecast_x_dataframe_diff_cols(
        nixtla_test_client,
        ray_df_x_diff_cols,
        ray_future_ex_vars_df_diff_cols,
    )
