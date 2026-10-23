import numpy as np
import pandas as pd
import polars as pl
import pytest

import nixtla.nixtla_client as client_module


def test_forecast_feature_contributions_with_categorical_features(
    nixtla_test_client, air_passengers_with_cat_exog
):
    data = air_passengers_with_cat_exog
    fcst = nixtla_test_client.forecast(
        data.df,
        h=data.h,
        X_df=data.X_df,
        categorical_exog_list=data.cat_cols,
        feature_contributions=True,
    )
    shap_df = nixtla_test_client.feature_contributions
    # Categorical column must appear as a labelled SHAP column.
    for col in data.cat_cols:
        assert (
            col in shap_df.columns
        ), f"'{col}' missing from feature_contributions columns"
    # SHAP values addtivity
    np.testing.assert_allclose(
        fcst["TimeGPT"].values,
        shap_df.iloc[:, 3:].sum(axis=1).values,
        rtol=1e-3,
    )


def test_forecast_with_categorical_features(
    nixtla_test_client, air_passengers_with_cat_exog
):
    data = air_passengers_with_cat_exog
    fcst = nixtla_test_client.forecast(
        data.df,
        h=data.h,
        X_df=data.X_df,
        categorical_exog_list=data.cat_cols,
    )
    assert len(fcst) == data.h
    assert fcst["TimeGPT"].notna().all()


def test_forecast_with_categorical_features_multiple_series(
    nixtla_test_client, multi_series_with_cat_exog
):
    data = multi_series_with_cat_exog
    fcst = nixtla_test_client.forecast(
        data.df,
        h=data.h,
        freq=data.freq,
        X_df=data.X_df,
        categorical_exog_list=data.cat_cols,
    )
    assert len(fcst) == data.n_ids * data.h
    assert fcst["TimeGPT"].notna().all()


def test_forecast_with_numerical_and_categorical_features(
    nixtla_test_client, air_passengers_with_num_and_cat_exog
):
    data = air_passengers_with_num_and_cat_exog
    fcst = nixtla_test_client.forecast(
        data.df,
        h=data.h,
        X_df=data.X_df,
        categorical_exog_list=data.cat_cols,
    )
    assert len(fcst) == data.h
    assert fcst["TimeGPT"].notna().all()


def test_forecast_with_hist_categorical_features(
    nixtla_test_client, air_passengers_with_cat_exog
):
    data = air_passengers_with_cat_exog
    fcst = nixtla_test_client.forecast(
        data.df,
        h=data.h,
        hist_exog_list=data.cat_cols,
        categorical_exog_list=data.cat_cols,
    )
    assert len(fcst) == data.h
    assert fcst["TimeGPT"].notna().all()


def test_forecast_with_hist_cat_and_futr_num_exog(
    nixtla_test_client, air_passengers_with_futr_num_and_hist_cat_exog
):
    """Regression test: hist categoricals in hist_exog_list + X_df provided must not KeyError."""
    data = air_passengers_with_futr_num_and_hist_cat_exog
    fcst = nixtla_test_client.forecast(
        data.df,
        h=data.h,
        X_df=data.X_df,
        hist_exog_list=data.cat_cols,
        categorical_exog_list=data.cat_cols,
    )
    assert len(fcst) == data.h
    assert fcst["TimeGPT"].notna().all()


def test_cv_with_categorical_features(nixtla_test_client, air_passengers_with_cat_exog):
    data = air_passengers_with_cat_exog
    cv = nixtla_test_client.cross_validation(
        data.df,
        h=data.h,
        categorical_exog_list=data.cat_cols,
    )
    assert len(cv) == data.h
    assert cv["TimeGPT"].notna().all()


def test_cv_with_numerical_and_categorical_features(
    nixtla_test_client, air_passengers_with_num_and_cat_exog
):
    data = air_passengers_with_num_and_cat_exog
    cv = nixtla_test_client.cross_validation(
        data.df,
        h=data.h,
        hist_exog_list=["num_feat"],
        categorical_exog_list=data.cat_cols,
    )
    assert len(cv) == data.h
    assert cv["TimeGPT"].notna().all()


def test_cv_with_categorical_features_multiple_series(
    nixtla_test_client, multi_series_with_cat_exog
):
    data = multi_series_with_cat_exog
    cv = nixtla_test_client.cross_validation(
        data.df,
        h=data.h,
        freq=data.freq,
        categorical_exog_list=data.cat_cols,
    )
    assert len(cv) == data.n_ids * data.h
    assert cv["TimeGPT"].notna().all()


def test_cv_with_hist_categorical_features(
    nixtla_test_client, air_passengers_with_cat_exog
):
    data = air_passengers_with_cat_exog
    cv = nixtla_test_client.cross_validation(
        data.df,
        h=data.h,
        hist_exog_list=data.cat_cols,
        categorical_exog_list=data.cat_cols,
    )
    assert len(cv) == data.h
    assert cv["TimeGPT"].notna().all()


def test_detect_anomalies_with_categorical_features(
    nixtla_test_client, air_passengers_with_cat_exog
):
    data = air_passengers_with_cat_exog
    anomalies = nixtla_test_client.detect_anomalies(
        data.df,
        categorical_exog_list=data.cat_cols,
    )
    assert len(anomalies) > 0
    assert "anomaly" in anomalies.columns


def test_detect_anomalies_with_numerical_and_categorical_features(
    nixtla_test_client, air_passengers_with_num_and_cat_exog
):
    data = air_passengers_with_num_and_cat_exog
    anomalies = nixtla_test_client.detect_anomalies(
        data.df,
        categorical_exog_list=data.cat_cols,
    )
    assert len(anomalies) > 0
    assert "anomaly" in anomalies.columns


@pytest.mark.parametrize("frame", ["pandas", "polars"])
def test_extract_categorical_exog_reads_future_values_for_both_backends(frame):
    df_data = {
        "unique_id": ["id-0"] * 3,
        "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "y": [1.0, 2.0, 3.0],
        "event": ["none", "sale", "none"],
    }
    X_df_data = {
        "unique_id": ["id-0", "id-0"],
        "ds": pd.to_datetime(["2024-01-04", "2024-01-05"]),
        "event": ["sale", "none"],
    }
    if frame == "pandas":
        df, X_df = pd.DataFrame(df_data), pd.DataFrame(X_df_data)
    else:
        df, X_df = pl.DataFrame(df_data), pl.DataFrame(X_df_data)

    (
        out_df,
        out_X_df,
        df_cat_vals,
        futr_cat_cols,
        hist_cat_cols,
        X_df_cat_future,
    ) = client_module._extract_categorical_exog(
        df=df,
        categorical_exog_list=["event"],
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        X_df=X_df,
    )

    assert futr_cat_cols == ["event"]
    assert hist_cat_cols == []
    assert X_df_cat_future["event"].to_list() == ["sale", "none"]
    assert df_cat_vals["event"].tolist() == ["none", "sale", "none"]
    assert "event" not in out_df.columns
    assert "event" not in out_X_df.columns
