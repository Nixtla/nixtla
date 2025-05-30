import httpx
import pytest

import numpy as np
import pandas as pd
import zstandard as zstd

from contextlib import contextmanager
from nixtla_tests.helpers.checks import check_num_partitions_same_results
from nixtla_tests.helpers.checks import check_equal_fcsts_add_history
from utilsforecast.data import generate_series

CAPTURED_REQUEST = None

class CapturingClient(httpx.Client):
    def post(self, *args, **kwargs):
        request = self.build_request('POST', *args, **kwargs)
        global CAPTURED_REQUEST
        CAPTURED_REQUEST = {
            'headers': dict(request.headers),
            'content': request.content,
            'method': request.method,
            'url': str(request.url)
        }
        return super().post(*args, **kwargs)

@contextmanager
def capture_request():
    original_client = httpx.Client
    httpx.Client = CapturingClient
    try:
        yield
    finally:
        httpx.Client = original_client

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

def test_forecast_date_features_multiple_series_and_different_ends(nixtla_test_client, two_short_series):
    h = 12
    fcst_test_series = nixtla_test_client.forecast(two_short_series, h=h, date_features=['dayofweek'])
    uids = two_short_series['unique_id']
    for uid in uids:
        expected = pd.date_range(
            periods=h + 1,
            start=two_short_series.query('unique_id == @uid')['ds'].max()
        )[1:].tolist()
        actual = fcst_test_series.query('unique_id == @uid')['ds'].tolist()
        assert actual == expected

def test_compression(nixtla_test_client, series_1MB_payload):
    with capture_request():
        nixtla_test_client.forecast(df=series_1MB_payload, freq='D', h=1, hist_exog_list=['static_0', 'static_1'])
        assert CAPTURED_REQUEST['headers']['content-encoding'] == 'zstd'
        content = CAPTURED_REQUEST['content']
        assert len(content) < 2**20
        assert len(zstd.ZstdDecompressor().decompress(content)) > 2**20

def test_cv_refit_equivalence(nixtla_test_client, air_passengers_df):
    cv_kwargs = dict(
        df=air_passengers_df,
        n_windows=2,
        h=12,
        freq='MS',
        time_col='timestamp',
        target_col='value',
        finetune_steps=2,
    )
    res_refit = nixtla_test_client.cross_validation(refit=True, **cv_kwargs)
    res_no_refit = nixtla_test_client.cross_validation(refit=False, **cv_kwargs)
    np.testing.assert_allclose(res_refit['value'], res_no_refit['value'])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            res_refit['TimeGPT'],
            res_no_refit['TimeGPT'],
            atol=1e-4,
            rtol=1e-3,
        )

def test_forecast_quantiles_error(nixtla_test_client, air_passengers_df):
    with pytest.raises(Exception) as excinfo:
        nixtla_test_client.forecast(
            df=air_passengers_df, 
            h=12, 
            time_col='timestamp', 
            target_col='value', 
            level=[80], 
            quantiles=[0.2, 0.3]
        )
    assert 'not both' in str(excinfo.value)

@pytest.mark.parametrize("method,kwargs", [
    ("forecast", {}),
    ("forecast", {"add_history": True}),
    ("cross_validation", {}),
])
def test_forecast_quantiles_output(nixtla_test_client, air_passengers_df, method, kwargs):
    test_qls = list(np.arange(0.1, 1, 0.1))
    exp_q_cols = [f"TimeGPT-q-{int(100 * q)}" for q in test_qls]

    args = {
        'df': air_passengers_df,
        'h': 12,
        'time_col': 'timestamp',
        'target_col': 'value',
        'quantiles': test_qls,
        **kwargs
    }
    if method == "cross_validation":
        func = nixtla_test_client.cross_validation
    elif method == "forecast":
        func = nixtla_test_client.forecast

    df_qls = func(**args)

    assert all(col in df_qls.columns for col in exp_q_cols)
    assert not any('-lo-' in col for col in df_qls.columns)
    # test monotonicity of quantiles
    for c1, c2 in zip(exp_q_cols[:-1], exp_q_cols[1:]):
        assert df_qls[c1].lt(df_qls[c2]).all()

@pytest.mark.parametrize(
        "freq",
        ["D", "W-THU", "Q-DEC", "15T"]
)
@pytest.mark.parametrize(
    "method_name,method_kwargs,exog",
    [
        ("detect_anomalies", {"level": 98}, False),
        ("cross_validation", {"h": 7, "n_windows": 2}, False),
        ("forecast", {"h": 7, "add_history": True}, False),
        ("detect_anomalies", {"level": 98}, True),
        ("cross_validation", {"h": 7, "n_windows": 2}, False),
        ("forecast", {"h": 7, "add_history": True}, False),
    ]
)
def test_num_partitions_same_results_parametrized(nixtla_test_client, method_name, method_kwargs, freq, exog):
    mathod_mapper = {
        "detect_anomalies": nixtla_test_client.detect_anomalies,
        "cross_validation": nixtla_test_client.cross_validation,
        "forecast": nixtla_test_client.forecast,
    }
    method = mathod_mapper[method_name]


    df_freq = generate_series(
        10,
        min_length=500 if freq != '15T' else 1_200,
        max_length=550 if freq != '15T' else 2_000,
    )
    df_freq['ds'] = df_freq.groupby('unique_id', observed=True)['ds'].transform(
        lambda x: pd.date_range(periods=len(x), freq=freq, end='2023-01-01')
    )
    if exog:
        df_freq["exog_1"] = 1

    kwargs = {
        "method": method,
        "num_partitions": 2,
        "df": df_freq,
        **method_kwargs,
    }

    check_num_partitions_same_results(**kwargs)

@pytest.mark.parametrize(
    "freq,h",
    [
        ('D', 7),
        ('W-THU', 52),
        ('Q-DEC', 8),
        ('15T', 4 * 24 * 7),
    ]
)
def test_forecast_models_different_results(nixtla_test_client, freq, h):
    df_freq = generate_series(
        10,
        min_length=500 if freq != '15T' else 1_200,
        max_length=550 if freq != '15T' else 2_000,
    )
    df_freq['ds'] = df_freq.groupby('unique_id', observed=True)['ds'].transform(
        lambda x: pd.date_range(periods=len(x), freq=freq, end='2023-01-01')
    )
    kwargs = dict(df=df_freq, h=h)
    fcst_1_df = check_equal_fcsts_add_history(nixtla_test_client, **{**kwargs, 'model': 'timegpt-1'})
    fcst_2_df = check_equal_fcsts_add_history(nixtla_test_client, **{**kwargs, 'model': 'timegpt-1-long-horizon'})
    with pytest.raises(AssertionError, match=r'\(column name="TimeGPT"\) are different'):
        pd.testing.assert_frame_equal(fcst_1_df, fcst_2_df)