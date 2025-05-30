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

#| hide
load_dotenv(override=True)

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
date_features = ['year', 'month']
df_date_features, future_df = _maybe_add_date_features(
    df=df_,
    X_df=None,
    h=12, 
    freq='MS', 
    features=date_features,
    one_hot=False,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
)
assert all(col in df_date_features for col in date_features)
assert all(col in future_df for col in date_features)

#| hide
# Test shap values are returned and sum to predictions
h=12
fcst_df = nixtla_client.forecast(df=df_date_features, h=h, X_df=future_df, feature_contributions=True)
shap_values = nixtla_client.feature_contributions
assert len(shap_values) == len(fcst_df)
np.testing.assert_allclose(fcst_df["TimeGPT"].values, shap_values.iloc[:, 3:].sum(axis=1).values)

fcst_hist_df = nixtla_client.forecast(df=df_date_features, h=h, X_df=future_df, add_history=True, feature_contributions=True)
shap_values_hist = nixtla_client.feature_contributions
assert len(shap_values_hist) == len(fcst_hist_df)
np.testing.assert_allclose(fcst_hist_df["TimeGPT"].values, shap_values_hist.iloc[:, 3:].sum(axis=1).values, atol=1e-4)

# test num partitions
_ = nixtla_client.forecast(df=df_date_features, h=h, X_df=future_df, add_history=True, feature_contributions=True, num_partitions=2)
pd.testing.assert_frame_equal(nixtla_client.feature_contributions, shap_values_hist, atol=1e-4, rtol=1e-3)

#| hide
# cross validation tests
df_copy = df_.copy()
pd.testing.assert_frame_equal(
    df_copy,
    df_,
)
df_test = df_.groupby('unique_id').tail(12)
df_train = df_.drop(df_test.index)
hyps = [
    # finetune steps is unstable due
    # to numerical reasons
    # dict(finetune_steps=2),
    dict(),
    dict(clean_ex_first=False),
    dict(date_features=['month']),
    dict(level=[80, 90]),
    #dict(level=[80, 90], finetune_steps=2),
]

#| hide
# test exogenous variables cv
df_ex_ = df_.copy()
df_ex_['exogenous_var'] = df_ex_['y'] + np.random.normal(size=len(df_ex_))
x_df_test = df_test.drop(columns='y').merge(df_ex_.drop(columns='y'))
for hyp in hyps:
    logger.info(f'Hyperparameters: {hyp}')
    logger.info('\n\nPerforming forecast\n')
    fcst_test = nixtla_client.forecast(
        df_train.merge(df_ex_.drop(columns='y')), h=12, X_df=x_df_test, **hyp
    )
    fcst_test = df_test[['unique_id', 'ds', 'y']].merge(fcst_test)
    fcst_test = fcst_test.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    logger.info('\n\nPerforming Cross validation\n')
    fcst_cv = nixtla_client.cross_validation(df_ex_, h=12, **hyp)
    fcst_cv = fcst_cv.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    logger.info('\n\nVerify difference\n')
    pd.testing.assert_frame_equal(
        fcst_test,
        fcst_cv.drop(columns='cutoff'),
        atol=1e-4,
        rtol=1e-3,
    )

#| hide
# test finetune cv
finetune_cv = nixtla_client.cross_validation(
            df=df_,
            h=12,
            n_windows=1,
            finetune_steps=1
        )
test_eq(finetune_cv is not None, True)

#| hide
for hyp in hyps:
    fcst_test = nixtla_client.forecast(df_train, h=12, **hyp)
    fcst_test = df_test[['unique_id', 'ds', 'y']].merge(fcst_test)
    fcst_test = fcst_test.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    fcst_cv = nixtla_client.cross_validation(df_, h=12, **hyp)
    fcst_cv = fcst_cv.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        fcst_test,
        fcst_cv.drop(columns='cutoff'),
        rtol=1e-2,
    )

#| hide
for hyp in hyps:
    fcst_test = nixtla_client.forecast(df_train, h=12, **hyp)
    fcst_test.insert(2, 'y', df_test['y'].values)
    fcst_test = fcst_test.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    fcst_cv = nixtla_client.cross_validation(df_, h=12, **hyp)
    fcst_cv = fcst_cv.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        fcst_test,
        fcst_cv.drop(columns='cutoff'),
        rtol=1e-2,
    )


#| hide
# test pass dataframe with index
df_ds_index = df_.set_index('ds')[['unique_id', 'y']]
df_ds_index.index = pd.DatetimeIndex(df_ds_index.index)
fcst_inferred_df_index = nixtla_client.forecast(df_ds_index, h=10)
anom_inferred_df_index = nixtla_client.detect_anomalies(df_ds_index)
fcst_inferred_df = nixtla_client.forecast(df_[['ds', 'unique_id', 'y']], h=10)
anom_inferred_df = nixtla_client.detect_anomalies(df_[['ds', 'unique_id', 'y']])
pd.testing.assert_frame_equal(fcst_inferred_df_index, fcst_inferred_df, atol=1e-4, rtol=1e-3)
pd.testing.assert_frame_equal(anom_inferred_df_index, anom_inferred_df, atol=1e-4, rtol=1e-3)
df_ds_index = df_ds_index.groupby('unique_id').tail(80)
for freq in ['Y', 'W-MON', 'Q-DEC', 'H']:
    df_ds_index.index = np.concatenate(
        df_ds_index['unique_id'].nunique() * [pd.date_range(end='2023-01-01', periods=80, freq=freq)]
    )
    df_ds_index.index.name = 'ds'
    fcst_inferred_df_index = nixtla_client.forecast(df_ds_index, h=10)
    df_test = df_ds_index.reset_index()
    fcst_inferred_df = nixtla_client.forecast(df_test, h=10)
    pd.testing.assert_frame_equal(fcst_inferred_df_index, fcst_inferred_df, atol=1e-4, rtol=1e-3)

#| hide
# test add date features with exogenous variables 
# and multiple series
date_features = ['year', 'month']
df_actual_future = df_.tail(12)[['unique_id', 'ds']]
df_date_features, future_df = _maybe_add_date_features(
    df=df_,
    X_df=df_actual_future,
    h=24, 
    freq='H', 
    features=date_features,
    one_hot=False,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
)
assert all(col in df_date_features for col in date_features)
assert all(col in future_df for col in date_features)
pd.testing.assert_frame_equal(
    df_date_features[df_.columns],
    df_,
)
pd.testing.assert_frame_equal(
    future_df[df_actual_future.columns],
    df_actual_future,
)

#| hide
# test add date features one hot with exogenous variables 
# and multiple series
date_features = ['month', 'day']
df_date_features, future_df = _maybe_add_date_features(
    df=df_,
    X_df=df_actual_future,
    h=24, 
    freq='H', 
    features=date_features,
    one_hot=date_features,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
)
pd.testing.assert_frame_equal(
    df_date_features[df_.columns],
    df_,
)
pd.testing.assert_frame_equal(
    future_df[df_actual_future.columns],
    df_actual_future.reset_index(drop=True),
)

#| hide
# test warning horizon too long
nixtla_client.forecast(df=df.tail(3), h=100, time_col='timestamp', target_col='value')

#| hide 
# test short horizon with add_history
test_fail(
    lambda: nixtla_client.forecast(df=df.tail(3), h=12, time_col='timestamp', target_col='value', add_history=True),
    contains='make sure'
)

#| hide 
# test short horizon with finetunning
test_fail(
    lambda: nixtla_client.forecast(df=df.tail(3), h=12, time_col='timestamp', target_col='value', finetune_steps=10, finetune_loss='mae'),
    contains='make sure'
)

#| hide
# test using index as time_col
# same results
df_test = df.copy()
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
df_test.set_index(df_test["timestamp"], inplace=True)
df_test.drop(columns="timestamp", inplace=True)

# Using user_provided time_col and freq
timegpt_anomalies_df_1 = nixtla_client.detect_anomalies(df, time_col='timestamp', target_col='value', freq= 'M')
# Infer time_col and freq from index
timegpt_anomalies_df_2 = nixtla_client.detect_anomalies(df_test, time_col='timestamp', target_col='value')

pd.testing.assert_frame_equal(
    timegpt_anomalies_df_1,
    timegpt_anomalies_df_2,
    atol=1e-4,
    rtol=1e-3,
)

#| hide
# Test large requests raise error and suggest partition number
df = generate_series(20_000, min_length=1_000, max_length=1_000, freq='min')
test_fail(
    lambda: nixtla_client.forecast(df=df, h=1, freq='min', finetune_steps=2),
    contains="num_partitions"
)

#| hide
# future and historic exogs
df = generate_series(n_series=2, min_length=5, max_length=20)
train, future = time_features(df, freq='D', features=['year', 'month'], h=5)

# features in df but not in X_df
missing_exogenous = train.columns.drop(['unique_id', 'ds', 'y']).tolist()
expected_warning = (
    f'`df` contains the following exogenous features: {missing_exogenous}, '
    'but `X_df` was not provided and they were not declared in `hist_exog_list`. '
    'They will be ignored.'
)
with warnings.catch_warnings(record=True) as w:
    forecasts = nixtla_client.forecast(train, h=5)
    assert any(expected_warning in str(warning.message) for warning in w)

# features in df not set as historic nor in X_df
expected_warning = (
    f"`df` contains the following exogenous features: ['month'], "
    'but they were not found in `X_df` nor declared in `hist_exog_list`. '
    'They will be ignored.'
)
with warnings.catch_warnings(record=True) as w:
    forecasts = nixtla_client.forecast(train, h=5, X_df=future[['unique_id', 'ds', 'year']])
    assert any(expected_warning in str(warning.message) for warning in w)

# features in X_df not in df
test_fail(
    lambda: nixtla_client.forecast(
        train[['unique_id', 'ds', 'y']],
        h=5,
        X_df=future,
    ),
    contains='features are present in `X_df` but not in `df`'
)

# test setting one as historic and other as future
nixtla_client.forecast(train, h=5, X_df=future[['unique_id', 'ds', 'year']], hist_exog_list=['month'])
test_eq(nixtla_client.weights_x['features'].tolist(), ['year', 'month'])

#| hide
# Test real-time anomaly detection
detection_size = 5
n_series = 2
size = 100

ds = pd.date_range(start='2023-01-01', periods=size, freq='W')
x = np.arange(size)
y = 10 * np.sin(0.1 * x) + 12
y = np.tile(y, n_series)
y[size - 5] = 30
y[2*size - 1] = 30

df = pd.DataFrame({
    'unique_id': np.repeat(np.arange(1, n_series + 1), size),
    'ds': np.tile(ds, n_series),
    'y': y
})

anomaly_df = nixtla_client.detect_anomalies_online(
    df, 
    h=20, 
    detection_size=detection_size, 
    threshold_method="univariate", 
    freq='W-SUN', 
    level=99,
)
assert len(anomaly_df) == n_series * detection_size
assert len(anomaly_df.columns) == 8 # [unique_id, ds, TimeGPT, y, anomaly, anomaly_score, hi, lo]
assert anomaly_df['anomaly'].sum() == 2 
assert anomaly_df['anomaly'].iloc[0] and anomaly_df['anomaly'].iloc[-1]

multi_anomaly_df = nixtla_client.detect_anomalies_online(
    df, 
    h=20, 
    detection_size=detection_size, 
    threshold_method="multivariate", 
    freq='W-SUN', 
    level=99,
)

assert len(multi_anomaly_df) == n_series * detection_size
assert len(multi_anomaly_df.columns) == 7 # [unique_id, ds, TimeGPT, y, anomaly, anomaly_score, accumulated_anomaly_score]
assert multi_anomaly_df['anomaly'].sum() == 4
assert (multi_anomaly_df['anomaly'].iloc[0] and 
        multi_anomaly_df['anomaly'].iloc[4] and
        multi_anomaly_df['anomaly'].iloc[5] and
        multi_anomaly_df['anomaly'].iloc[9])


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
