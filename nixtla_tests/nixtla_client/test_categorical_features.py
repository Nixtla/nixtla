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
