import pytest

import numpy as np
import pandas as pd

@pytest.mark.parametrize(
    "df_converter, freq",
    [
        pytest.param(
            lambda series, with_gaps: with_gaps,
            '5min',
            id="gaps"
        ),
        pytest.param(
            lambda series, with_gaps: pd.concat([series, series]),
            '5min',
            id="duplicates"
        ),
        pytest.param(
            lambda series, with_gaps: series,
            '1min',
            id="wrong_freq"
        ),
    ]
)
def test_forecast_with_error(series_with_gaps, nixtla_test_client, df_converter, freq):
    series, with_gaps = series_with_gaps

    with pytest.raises(ValueError, match='missing or duplicate timestamps, or the timestamps do not match'):
        nixtla_test_client.forecast(df=df_converter(series, with_gaps), h=1, freq=freq)

def test_cv_forecast_consistency(nixtla_test_client, cv_series_with_features):
    series_with_features, train, valid, x_cols, h, freq = cv_series_with_features
    for hist_exog_list in [None, [], [x_cols[2], x_cols[1]], x_cols]:
        cv_res = nixtla_test_client.cross_validation(
            series_with_features,
            n_windows=1,
            h=h,
            freq=freq,
            hist_exog_list=hist_exog_list,
        )
        fcst_res = nixtla_test_client.forecast(
            train,
            h=h,
            freq=freq,
            hist_exog_list=hist_exog_list,
            X_df=valid,
        )
        np.testing.assert_allclose(
            cv_res['TimeGPT'], fcst_res['TimeGPT'], atol=1e-4, rtol=1e-3
        )

def test_forecast_different_hist_exog_gives_different_results(nixtla_test_client, cv_series_with_features):
    _, train, valid, x_cols, h, freq = cv_series_with_features
    for X_df in (None, valid):
        res1 = nixtla_test_client.forecast(train, h=h, X_df=X_df, freq=freq, hist_exog_list=x_cols[:2])
        res2 = nixtla_test_client.forecast(train, h=h, X_df=X_df, freq=freq, hist_exog_list=x_cols[2:])
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                res1['TimeGPT'],
                res2['TimeGPT'],
                atol=1e-4,
                rtol=1e-3,
            )

def test_custom_business_hours(nixtla_test_client, business_hours_series, custom_business_hours):
    nixtla_test_client.detect_anomalies(df=business_hours_series, freq=custom_business_hours, level=90)
    nixtla_test_client.cross_validation(df=business_hours_series, freq=custom_business_hours, h=7)
    fcst = nixtla_test_client.forecast(df=business_hours_series, freq=custom_business_hours, h=7)
    assert sorted(fcst['ds'].dt.hour.unique().tolist()) == list(range(9, 16))
    assert [(model, freq.lower()) for (model, freq) in nixtla_test_client._model_params.keys()] == [('timegpt-1', 'cbh')]


def test_integer_freq(nixtla_test_client, integer_freq_series):
    nixtla_test_client.detect_anomalies(df=integer_freq_series, level=90, freq=1)
    nixtla_test_client.cross_validation(df=integer_freq_series, h=7, freq=1)
    fcst = nixtla_test_client.forecast(df=integer_freq_series, h=7, freq=1)
    train_ends = integer_freq_series.groupby('unique_id', observed=True)['ds'].max()
    fcst_ends = fcst.groupby('unique_id', observed=True)['ds'].max()
    pd.testing.assert_series_equal(fcst_ends, train_ends + 7)
    assert list(nixtla_test_client._model_params.keys()) == [('timegpt-1', 'MS')]