import numpy as np


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
        assert col in shap_df.columns, f"'{col}' missing from feature_contributions columns"
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


def test_cv_with_categorical_features(
    nixtla_test_client, air_passengers_with_cat_exog
):
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
