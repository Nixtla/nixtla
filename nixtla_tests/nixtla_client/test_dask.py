from nixtla_tests.helpers.checks import check_anomalies_dataframe
from nixtla_tests.helpers.checks import check_anomalies_online_dataframe
from nixtla_tests.helpers.checks import check_anomalies_dataframe_diff_cols
from nixtla_tests.helpers.checks import check_forecast_dataframe
from nixtla_tests.helpers.checks import check_forecast_dataframe_diff_cols
from nixtla_tests.helpers.checks import check_forecast_x_dataframe
from nixtla_tests.helpers.checks import check_forecast_x_dataframe_diff_cols
from nixtla_tests.helpers.checks import check_quantiles


def test_quantiles(nixtla_test_client, dask_df):
    check_quantiles(nixtla_test_client, dask_df, id_col="unique_id", time_col="ds")


def test_forecast(nixtla_test_client, dask_df, dask_diff_cols_df, distributed_n_series):
    check_forecast_dataframe(
        nixtla_test_client, dask_df, n_series_to_check=distributed_n_series
    )
    check_forecast_dataframe_diff_cols(nixtla_test_client, dask_diff_cols_df)


def test_anomalies(nixtla_test_client, dask_df, dask_diff_cols_df):
    check_anomalies_dataframe(nixtla_test_client, dask_df)
    check_anomalies_dataframe_diff_cols(nixtla_test_client, dask_diff_cols_df)


def test_anomalies_online(nixtla_test_client, dask_df):
    check_anomalies_online_dataframe(nixtla_test_client, dask_df)


def test_forecast_x_dataframe(
    nixtla_test_client,
    dask_df_x,
    dask_future_ex_vars_df,
    dask_df_x_diff_cols,
    dask_future_ex_vars_df_diff_cols,
):
    check_forecast_x_dataframe(nixtla_test_client, dask_df_x, dask_future_ex_vars_df)
    check_forecast_x_dataframe_diff_cols(
        nixtla_test_client,
        dask_df_x_diff_cols,
        dask_future_ex_vars_df_diff_cols,
    )
