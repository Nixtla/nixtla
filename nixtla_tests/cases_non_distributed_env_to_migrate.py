import os
import logging
import time
import uuid
import warnings
import numpy as np
import pandas as pd
from itertools import product
from typing import Callable

from dotenv import load_dotenv
from fastcore.test import test_eq, test_fail
import httpx
import zstandard as zstd

import utilsforecast.processing as ufp
from utilsforecast.data import generate_series
from utilsforecast.evaluation import evaluate
from utilsforecast.feature_engineering import fourier, time_features
from utilsforecast.losses import rmse

from nixtla.nixtla_client import (
    _maybe_add_date_features,
    ApiError,
    NixtlaClient,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

common_kwargs = {
    "freq": "D",
    "id_col": 'unique_id',
    "time_col": 'ds'
}


## Client

## Wrappers

## Tests
# add this back to avoid undefined error
nixtla_client = NixtlaClient()

#| hide
# custom client
custom_client = NixtlaClient(
    base_url=os.environ['NIXTLA_BASE_URL_CUSTOM'],
    api_key=os.environ['NIXTLA_API_KEY_CUSTOM'],
)
assert custom_client.validate_api_key()


#| hide
# finetuning
h = 5
series = generate_series(10, equal_ends=True)
train_end = series['ds'].max() - h * pd.offsets.Day()
train_mask = series['ds'] <= train_end
train = series[train_mask]
valid = series[~train_mask]
model_id1 = str(uuid.uuid4())


### Data Quality


#| hide
df = pd.read_csv(
    'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv',
    parse_dates=['timestamp'],
)
df.head()

#| hide
# test add date features
df_ = df.rename(columns={'timestamp': 'ds', 'value': 'y'})
df_.insert(0, 'unique_id', 'AirPassengers')


#| hide
# future and historic exogs
df = generate_series(n_series=2, min_length=5, max_length=20)
train, future = time_features(df, freq='D', features=['year', 'month'], h=5)


#| hide
# Test real-time anomaly detection
detection_size = 5
n_series = 2
size = 100


### Distributed

#| hide
#| distributed
import fugue
import fugue.api as fa

#| hide
#| distributed
ATOL = 1e-3

def test_forecast(
    df: fugue.AnyDataFrame, 
    horizon: int = 12,
    id_col: str = 'unique_id',
    time_col: str = 'ds',
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
    test_eq(n_series * 12, len(fcst_df))
    cols = fcst_df.columns.to_list()
    exp_cols = [id_col, time_col, 'TimeGPT']
    if 'level' in fcst_kwargs:
        level = sorted(fcst_kwargs['level'])
        exp_cols.extend([f'TimeGPT-lo-{lv}' for lv in reversed(level)])
        exp_cols.extend([f'TimeGPT-hi-{lv}' for lv in level])
    test_eq(cols, exp_cols)

def test_forecast_diff_results_diff_models(
    df: fugue.AnyDataFrame, 
    horizon: int = 12, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    **fcst_kwargs,
):
    fcst_df = nixtla_client.forecast(
        df=df, 
        h=horizon, 
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        model='timegpt-1',
        **fcst_kwargs
    )
    fcst_df = fa.as_pandas(fcst_df)
    fcst_df_2 = nixtla_client.forecast(
        df=df, 
        h=horizon, 
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        model='timegpt-1-long-horizon',
        **fcst_kwargs
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)
    test_fail(
        lambda: pd.testing.assert_frame_equal(
            fcst_df.sort_values([id_col, time_col]).reset_index(drop=True),
            fcst_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        ),
        contains='(column name="TimeGPT") are different',
    )

def test_forecast_same_results_num_partitions(
    df: fugue.AnyDataFrame, 
    horizon: int = 12, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    **fcst_kwargs,
):
    fcst_df = nixtla_client.forecast(
        df=df, 
        h=horizon, 
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs
    )
    fcst_df = fa.as_pandas(fcst_df)
    fcst_df_2 = nixtla_client.forecast(
        df=df, 
        h=horizon, 
        num_partitions=2,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)
    pd.testing.assert_frame_equal(
        fcst_df.sort_values([id_col, time_col]).reset_index(drop=True),
        fcst_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        atol=ATOL,
    )

def test_cv_same_results_num_partitions(
    df: fugue.AnyDataFrame, 
    horizon: int = 12, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    **fcst_kwargs,
):
    fcst_df = nixtla_client.cross_validation(
        df=df, 
        h=horizon, 
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs
    )
    fcst_df = fa.as_pandas(fcst_df)
    fcst_df_2 = nixtla_client.cross_validation(
        df=df, 
        h=horizon, 
        num_partitions=2,
        id_col=id_col,
        time_col=time_col,
        **fcst_kwargs
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)
    pd.testing.assert_frame_equal(
        fcst_df.sort_values([id_col, time_col]).reset_index(drop=True),
        fcst_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        atol=ATOL,
    )

def test_forecast_dataframe(df: fugue.AnyDataFrame):
    test_cv_same_results_num_partitions(df, n_windows=2, step_size=1)
    test_cv_same_results_num_partitions(df, n_windows=3, step_size=None, horizon=1)
    test_cv_same_results_num_partitions(df, model='timegpt-1-long-horizon', horizon=1)
    test_forecast_diff_results_diff_models(df)
    test_forecast(df, num_partitions=1)
    test_forecast(df, level=[90, 80], num_partitions=1)
    test_forecast_same_results_num_partitions(df)

def test_forecast_dataframe_diff_cols(
    df: fugue.AnyDataFrame,
    id_col: str = 'id_col',
    time_col: str = 'time_col',
    target_col: str = 'target_col',
):
    test_forecast(df, id_col=id_col, time_col=time_col, target_col=target_col, num_partitions=1)
    test_forecast(
        df, id_col=id_col, time_col=time_col, target_col=target_col, level=[90, 80], num_partitions=1
    )
    test_forecast_same_results_num_partitions(
        df, id_col=id_col, time_col=time_col, target_col=target_col
    )

def test_forecast_x(
    df: fugue.AnyDataFrame, 
    X_df: fugue.AnyDataFrame,
    horizon: int = 24,
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
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
    test_eq(n_series * horizon, len(fcst_df))
    cols = fcst_df.columns.to_list()
    exp_cols = [id_col, time_col, 'TimeGPT']
    if 'level' in fcst_kwargs:
        level = sorted(fcst_kwargs['level'])
        exp_cols.extend([f'TimeGPT-lo-{lv}' for lv in reversed(level)])
        exp_cols.extend([f'TimeGPT-hi-{lv}' for lv in level])
    test_eq(cols, exp_cols)
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
        fcst_df.sort_values([id_col, time_col])['TimeGPT'].values,
        fcst_df_2.sort_values([id_col, time_col])['TimeGPT'].values,
    )
    assert not equal_arrays, 'Forecasts with and without ex vars are equal'

def test_forecast_x_same_results_num_partitions(
    df: fugue.AnyDataFrame, 
    X_df: fugue.AnyDataFrame,
    horizon: int = 24, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
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
        **fcst_kwargs
    )
    fcst_df = fa.as_pandas(fcst_df)
    fcst_df_2 = nixtla_client.forecast(
        df=df,
        h=horizon,
        num_partitions=2,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **fcst_kwargs
    )
    fcst_df_2 = fa.as_pandas(fcst_df_2)
    equal_arrays = np.array_equal(
        fcst_df.sort_values([id_col, time_col])['TimeGPT'].values,
        fcst_df_2.sort_values([id_col, time_col])['TimeGPT'].values,
    )
    assert not equal_arrays, 'Forecasts with and without ex vars are equal'

def test_forecast_x_dataframe(df: fugue.AnyDataFrame, X_df: fugue.AnyDataFrame):
    test_forecast_x(df, X_df, num_partitions=1)
    test_forecast_x(df, X_df, level=[90, 80], num_partitions=1)
    test_forecast_x_same_results_num_partitions(df, X_df)

def test_forecast_x_dataframe_diff_cols(
    df: fugue.AnyDataFrame,
    X_df: fugue.AnyDataFrame,
    id_col: str = 'id_col',
    time_col: str = 'time_col',
    target_col: str = 'target_col'
):
    test_forecast_x(
        df, X_df, id_col=id_col, time_col=time_col, target_col=target_col, num_partitions=1
    )
    test_forecast_x(
        df, X_df, id_col=id_col, time_col=time_col, target_col=target_col, level=[90, 80], num_partitions=1
    )
    test_forecast_x_same_results_num_partitions(
        df, X_df, id_col=id_col, time_col=time_col, target_col=target_col
    )

def test_anomalies(
    df: fugue.AnyDataFrame, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
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
    test_eq(fa.as_pandas(df)[id_col].unique(), anomalies_df[id_col].unique())
    cols = anomalies_df.columns.to_list()
    level = anomalies_kwargs.get('level', 99)
    exp_cols = [
        id_col,
        time_col,
        target_col,
        'TimeGPT',
        'anomaly',
        f'TimeGPT-lo-{level}',
        f'TimeGPT-hi-{level}',
    ]
    test_eq(cols, exp_cols)

def test_anomalies_same_results_num_partitions(
    df: fugue.AnyDataFrame, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
    **anomalies_kwargs,
):
    anomalies_df = nixtla_client.detect_anomalies(
        df=df, 
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **anomalies_kwargs
    )
    anomalies_df = fa.as_pandas(anomalies_df)
    anomalies_df_2 = nixtla_client.detect_anomalies(
        df=df, 
        num_partitions=2,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **anomalies_kwargs
    )
    anomalies_df_2 = fa.as_pandas(anomalies_df_2)
    pd.testing.assert_frame_equal(
        anomalies_df.sort_values([id_col, time_col]).reset_index(drop=True),
        anomalies_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        atol=ATOL,
    )

def test_online_anomalies(
    df: fugue.AnyDataFrame, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
    level=99,
    **reatlime_anomalies_kwargs
):
    anomalies_df = nixtla_client.detect_anomalies_online(
        df=df, 
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        **reatlime_anomalies_kwargs,
    )
    anomalies_df = fa.as_pandas(anomalies_df)
    test_eq(fa.as_pandas(df)[id_col].unique(), anomalies_df[id_col].unique())
    cols = anomalies_df.columns.to_list()
    level = anomalies_kwargs.get('level', 99)
    exp_cols = [
        id_col,
        time_col,
        target_col,
        'TimeGPT',
        'anomaly',
        'anomaly_score',
        f'TimeGPT-lo-{level}',
        f'TimeGPT-hi-{level}',
    ]
    test_eq(cols, exp_cols)

def test_anomalies_online_same_results_num_partitions(
    df: fugue.AnyDataFrame, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
    **reatlime_anomalies_kwargs
):
    anomalies_df = nixtla_client.detect_anomalies_online(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        num_partitions=1,
        **reatlime_anomalies_kwargs
    )
    anomalies_df = fa.as_pandas(anomalies_df)
    anomalies_df_2 = nixtla_client.detect_anomalies_online(
        df=df, 
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        num_partitions=2,
        **reatlime_anomalies_kwargs
    )
    anomalies_df_2 = fa.as_pandas(anomalies_df_2)
    pd.testing.assert_frame_equal(
        anomalies_df.sort_values([id_col, time_col]).reset_index(drop=True),
        anomalies_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        atol=ATOL,
    )

def test_anomalies_diff_results_diff_models(
    df: fugue.AnyDataFrame, 
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
    **anomalies_kwargs,
):
    anomalies_df = nixtla_client.detect_anomalies(
        df=df, 
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        model='timegpt-1',
        **anomalies_kwargs
    )
    anomalies_df = fa.as_pandas(anomalies_df)
    anomalies_df_2 = nixtla_client.detect_anomalies(
        df=df, 
        num_partitions=1,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        model='timegpt-1-long-horizon',
        **anomalies_kwargs
    )
    anomalies_df_2 = fa.as_pandas(anomalies_df_2)
    test_fail(
        lambda: pd.testing.assert_frame_equal(
            anomalies_df.sort_values([id_col, time_col]).reset_index(drop=True),
            anomalies_df_2.sort_values([id_col, time_col]).reset_index(drop=True),
        ),
        contains='(column name="TimeGPT") are different',
    )

def test_anomalies_dataframe(df: fugue.AnyDataFrame):
    test_anomalies(df, num_partitions=1)
    test_anomalies(df, level=90, num_partitions=1)
    test_anomalies_same_results_num_partitions(df)

def test_anomalies_online_dataframe(df: fugue.AnyDataFrame):
    test_online_anomalies(df, h=20, detection_size=5, threshold_method='univariate', level=99, num_partitions=1)
    test_anomalies_online_same_results_num_partitions(df, h=20, detection_size=5, threshold_method='univariate', level=99)

def test_anomalies_dataframe_diff_cols(
    df: fugue.AnyDataFrame,
    id_col: str = 'id_col',
    time_col: str = 'time_col',
    target_col: str = 'target_col',
):
    test_anomalies(df, id_col=id_col, time_col=time_col, target_col=target_col, num_partitions=1)
    test_anomalies(df, id_col=id_col, time_col=time_col, target_col=target_col, level=90, num_partitions=1)
    test_anomalies_same_results_num_partitions(df, id_col=id_col, time_col=time_col, target_col=target_col)
    # @A: document behavior with exogenous variables in distributed environments.  
    #test_anomalies_same_results_num_partitions(df, id_col=id_col, time_col=time_col, date_features=True, clean_ex_first=False)

def test_quantiles(df: fugue.AnyDataFrame, id_col: str = 'id_col', time_col: str = 'time_col'):
    test_qls = list(np.arange(0.1, 1, 0.1))
    exp_q_cols = [f"TimeGPT-q-{int(q * 100)}" for q in test_qls]
    def test_method_qls(method, **kwargs):
        df_qls = method(
            df=df,
            h=12,
            id_col=id_col,
            time_col=time_col,
            quantiles=test_qls,
            **kwargs
        )
        df_qls = fa.as_pandas(df_qls)
        assert all(col in df_qls.columns for col in exp_q_cols)
        # test monotonicity of quantiles
        df_qls.apply(lambda x: x.is_monotonic_increasing, axis=1).sum() == len(exp_q_cols)
    test_method_qls(nixtla_client.forecast)
    test_method_qls(nixtla_client.forecast, add_history=True)
    test_method_qls(nixtla_client.cross_validation)


def test_finetuned_model(df):
    # fine-tuning on distributed fails
    test_fail(
        lambda: custom_client.finetune(df=df),
        contains='Can only fine-tune on pandas or polars dataframes.'
    )
    
    # forecast
    local_fcst = custom_client.forecast(
        df=fa.as_pandas(df), h=5, finetuned_model_id=model_id2
    )
    distr_fcst = fa.as_pandas(
        custom_client.forecast(df=df, h=5, finetuned_model_id=model_id2)
    ).sort_values(['unique_id', 'ds']).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        local_fcst, 
        distr_fcst,
        check_dtype=False,
        atol=1e-4,
        rtol=1e-2,
    )

    # cross-validation
    local_cv = custom_client.cross_validation(
        df=fa.as_pandas(df), n_windows=2, h=5, finetuned_model_id=model_id2
    )
    distr_cv = fa.as_pandas(
        custom_client.cross_validation(df=df, n_windows=2, h=5, finetuned_model_id=model_id2)
    ).sort_values(['unique_id', 'ds']).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        local_cv,
        distr_cv[local_cv.columns],
        check_dtype=False,
        atol=1e-4,
        rtol=1e-2,
    )

    # anomaly detection
    local_anomaly = custom_client.detect_anomalies(
        df=fa.as_pandas(df), finetuned_model_id=model_id2
    )
    distr_anomaly = fa.as_pandas(
        custom_client.detect_anomalies(df=df, finetuned_model_id=model_id2)
    ).sort_values(['unique_id', 'ds']).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        local_anomaly, 
        distr_anomaly[local_anomaly.columns],
        check_dtype=False,
        atol=1e-3,
        rtol=1e-2,
    )
