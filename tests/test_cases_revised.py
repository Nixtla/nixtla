import os
import re
import logging
import time
import uuid
import warnings
import numpy as np
import pandas as pd
from contextlib import contextmanager
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

from nixtla.date_features import SpecialDates

from nixtla.nixtla_client import (
    _model_in_list,
    _audit_duplicate_rows,
    _audit_missing_dates,
    _audit_categorical_variables,
    _audit_leading_zeros,
    _audit_negative_values,
    _maybe_add_date_features,
    AuditDataSeverity,
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

## Utilities

#| hide

assert _model_in_list("a", ("a", "b"))
assert not _model_in_list("a", ("b", "c"))
assert _model_in_list("axb", ("x", re.compile("a.*b")))
assert _model_in_list("axb", ("x", re.compile("^a.*b$")))
assert _model_in_list("a-b", ("x", re.compile("^a-.*b$")))
assert _model_in_list("a-dfdfb", ("x", re.compile("^a-.*b$")))
assert not _model_in_list("abc", ("x", re.compile("ab"), re.compile("abcd")))

### Audit Data

#### Audit Duplicate Rows

#| hide
df = pd.DataFrame(
    {
        'unique_id': [1, 2, 3, 4],
        'ds': ['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01'],
        'y': [1, 2, 3, 4],
    }
)
audit, duplicates = _audit_duplicate_rows(df)
test_eq(audit, AuditDataSeverity.PASS)

# Test with duplicates
df_duplicates = pd.DataFrame(
    {
        'unique_id': [1, 1, 1],
        'ds': ['2020-01-01', '2020-01-01', '2020-01-02'],
        'y': [1, 2, 3],
    }
)
audit, duplicates = _audit_duplicate_rows(df_duplicates)
test_eq(audit, AuditDataSeverity.FAIL)
test_eq(len(duplicates), 2)

#### Audit Missing Dates

#| hide

df_complete = pd.DataFrame(
    {
        'unique_id': [1, 1, 1, 2, 2, 2],
        'ds': ['2020-01-01', '2020-01-02', '2020-01-03',
               '2020-01-01', '2020-01-02', '2020-01-03'],
        'y': [1, 2, 3, 4, 5, 6],
    }
)

# Test with complete data
audit, missing = _audit_missing_dates(df_complete, freq='D')
test_eq(audit, AuditDataSeverity.PASS)
test_eq(len(missing), 0)

# Test with missing dates for multiple IDs
df_missing = pd.DataFrame(
    {
        'unique_id': [1, 1, 2, 2],
        'ds': ['2020-01-01', '2020-01-03', '2020-01-01', '2020-01-03'],
        'y': [1, 3, 4, 6],
    }
)
audit, missing = _audit_missing_dates(df_missing, freq='D')
test_eq(audit, AuditDataSeverity.FAIL)
test_eq(len(missing), 2)  # One missing date per unique_id

#### Audit Categorical Variables 

#| hide

# Test with no categorical variables
df_no_cat = pd.DataFrame({
    'unique_id': [1, 2, 3],
    'ds': pd.date_range('2023-01-01', periods=3),
    'y': [1.0, 2.0, 3.0]
})
audit_, cat_df = _audit_categorical_variables(df_no_cat)
test_eq(audit_, AuditDataSeverity.PASS)
test_eq(len(cat_df), 0)

# Test with categorical variables
df_with_cat = pd.DataFrame({
    'unique_id': ['A', 'B', 'C'],
    'ds': pd.date_range('2023-01-01', periods=3),
    'y': [1.0, 2.0, 3.0],
    'cat_col': ['X', 'Y', 'Z']
})
audit, cat_df = _audit_categorical_variables(df_with_cat)
test_eq(audit, AuditDataSeverity.FAIL)
test_eq(cat_df.shape[1], 1)  # Should include only 'cat_col'

# Test with categorical dtype
df_with_cat_dtype = pd.DataFrame({
    'unique_id': [1, 2, 3],
    'ds': pd.date_range('2023-01-01', periods=3),
    'y': [1.0, 2.0, 3.0],
    'cat_col': pd.Categorical(['X', 'Y', 'Z'])
})
audit, cat_df = _audit_categorical_variables(df_with_cat_dtype)
test_eq(audit, AuditDataSeverity.FAIL)
test_eq(cat_df.shape[1], 1)  # Should include only 'cat_col'

#### Audit Leading Zeros 

#| hide
df = pd.DataFrame({
    'unique_id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'D', 'D', 'D'],
    'ds': pd.date_range('2025-01-01', periods=12),
    'y': [0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 0]
})
audit, leading_zeros_df = _audit_leading_zeros(df)
test_eq(audit, AuditDataSeverity.CASE_SPECIFIC)
test_eq(len(leading_zeros_df), 3)
# note that zero time series are not flagged 

#### Audit Negative Values

#| hide 
df = pd.DataFrame({
    'unique_id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
    'ds': pd.date_range('2025-01-01', periods=8),
    'y': [0, -1, 2, -1, -2, 0, 1, 2]
})

audit, negative_values_df = _audit_negative_values(df)
test_eq(audit, AuditDataSeverity.CASE_SPECIFIC)
test_eq(len(negative_values_df), 3)

## Client

## Wrappers

## Tests

#| hide
@contextmanager
def delete_env_var(key):
    original_value = os.environ.get(key)
    rm = False
    if key in os.environ:
        del os.environ[key]
        rm = True
    try:
        yield
    finally:
        if rm:
            os.environ[key] = original_value
# test api_key fail
with delete_env_var('NIXTLA_API_KEY'), delete_env_var('TIMEGPT_TOKEN'):
    test_fail(
        lambda: NixtlaClient(),
        contains='NIXTLA_API_KEY',
    )

#| hide
nixtla_client = NixtlaClient()

#| hide
assert nixtla_client.validate_api_key()

#| hide
# custom client
custom_client = NixtlaClient(
    base_url=os.environ['NIXTLA_BASE_URL_CUSTOM'],
    api_key=os.environ['NIXTLA_API_KEY_CUSTOM'],
)
assert custom_client.validate_api_key()

#| hide
# usage endpoint
usage = custom_client.usage()
assert sorted(usage.keys()) == ['minute', 'month']

#| hide
# finetuning
h = 5
series = generate_series(10, equal_ends=True)
train_end = series['ds'].max() - h * pd.offsets.Day()
train_mask = series['ds'] <= train_end
train = series[train_mask]
valid = series[~train_mask]
model_id1 = str(uuid.uuid4())
finetune_resp = custom_client.finetune(train, output_model_id=model_id1)
assert finetune_resp == model_id1
model_id2 = custom_client.finetune(train, finetuned_model_id=model_id1)

#| hide
# forecast with fine-tuned models
fcst_base = custom_client.forecast(train, h=h)
fcst1 = custom_client.forecast(train, h=h, finetuned_model_id=model_id1)
fcst2 = custom_client.forecast(train, h=h, finetuned_model_id=model_id2)
all_fcsts = fcst_base.assign(ten_rounds=fcst1['TimeGPT'], twenty_rounds=fcst2['TimeGPT'])
fcst_rmse = evaluate(
    all_fcsts.merge(valid),
    metrics=[rmse],
    agg_fn='mean',
).loc[0]
# error was reduced over 40% by finetuning
assert 1 - fcst_rmse['ten_rounds'] / fcst_rmse['TimeGPT'] > 0.4
# error was reduced over 30% by further finetuning
assert 1 - fcst_rmse['twenty_rounds'] / fcst_rmse['ten_rounds'] > 0.3

# non-existent model returns 404
try:
    custom_client.forecast(train, h=5, finetuned_model_id='unexisting')
except ApiError as e:
    assert e.status_code == 404

#| hide
# cv with fine-tuned model
cv_base = custom_client.cross_validation(series, n_windows=2, h=h)
cv_finetune = custom_client.cross_validation(series, n_windows=2, h=h, finetuned_model_id=model_id1)
all_fcsts = fcst_base.assign(ten_rounds=fcst1['TimeGPT'], twenty_rounds=fcst2['TimeGPT'])
cv_rmse = evaluate(
    cv_base.merge(
        cv_finetune,
        on=['unique_id', 'ds', 'cutoff', 'y'],
        suffixes=('_base', '_finetune')
    ).drop(columns='cutoff'),
    metrics=[rmse],
    agg_fn='mean',
).loc[0]
# error was reduced over 40% by finetuning
assert 1 - cv_rmse['TimeGPT_finetune'] / cv_rmse['TimeGPT_base'] > 0.4

#| hide
# delete finetuned model
custom_client.delete_finetuned_model(model_id1)

### Data Quality

#### All Pass

#| hide

# Audit Data | All pass

df_ok = pd.DataFrame({
    'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2', 'id2'],
    'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
    'y': [1, 2, 3, 4, 5, 6]
})
all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
    df=df_ok,
    **common_kwargs
)
assert all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 0

#### Duplicate rows

#| hide

# Audit Data
df_with_duplicates = pd.DataFrame({
    'unique_id': ['id1', 'id1', 'id1', 'id2'],
    'ds': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'y': [1, 2, 3, 4]
})

all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
    df=df_with_duplicates,
    **common_kwargs
)
assert not all_pass
assert len(case_specific_dfs) == 0
assert len(fail_dfs) == 2
assert 'D001' in fail_dfs
assert len(fail_dfs['D001']) == 2  # The two duplicate rows should be returned
assert 'D002' in fail_dfs
assert fail_dfs["D002"] is None ## D002 can not be run with duplicates


# Clean Data
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=df_with_duplicates,
    fail_dict=fail_dfs, case_specific_dict=case_specific_dfs,
    agg_dict={'y': 'sum'}, **common_kwargs
)
assert all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 0
assert len(cleaned_df) == 3  # Two duplicates rows consolidated into one.


#| hide

# Clean Data | Raises ValueError
all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
    df=df_with_duplicates,
    **common_kwargs
)

test_fail(
    lambda: custom_client.clean_data(
        df=df_with_duplicates,
        fail_dict=fail_dfs,
        case_specific_dict=case_specific_dfs,
        **common_kwargs
    ),
    contains="agg_dict must be provided to resolve D001 failure.",
)

#### Missing dates

#| hide

# Audit Data
df_with_missing_dates = pd.DataFrame({
    'unique_id': ['id1', 'id1', 'id2', 'id2'],
    'ds': ['2023-01-01', '2023-01-03', '2023-01-01', '2023-01-03'],
    'y': [1, 3, 4, 6]
})

all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
    df=df_with_missing_dates,
    **common_kwargs
)
assert not all_pass
assert len(case_specific_dfs) == 0
assert len(fail_dfs) == 1
assert 'D002' in fail_dfs
assert len(fail_dfs['D002']) == 2  # Two missing dates should returned

# Clean Data
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=df_with_missing_dates,
    fail_dict=fail_dfs, case_specific_dict=case_specific_dfs,
    agg_dict={'y': 'sum'},
    **common_kwargs
)
assert all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 0
assert len(cleaned_df) == 6  # Two missing rows added.
assert pd.to_datetime("2023-01-02") in cleaned_df['ds'].values

#### Duplicate rows and missing dates

#| hide

# Audit Data
# Global end on 2023-01-03 which is missing for id1
df_with_duplicates = pd.DataFrame({
    'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2'],
    'ds': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'],
    'y': [1, 2, 3, 4, 5]
})

all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
    df=df_with_duplicates,
    **common_kwargs
)
assert not all_pass
assert len(case_specific_dfs) == 0
assert len(fail_dfs) == 2
assert 'D001' in fail_dfs
assert len(fail_dfs['D001']) == 2  # The two duplicate rows should be returned
assert 'D002' in fail_dfs
assert fail_dfs["D002"] is None ## D002 can not be run with duplicates


# Clean Data (pass 1 will clear the duplicates)
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=df_with_duplicates,
    fail_dict=fail_dfs, case_specific_dict=case_specific_dfs,
    agg_dict={'y': 'sum'}, **common_kwargs
)
assert not all_pass
assert len(fail_dfs) == 1
# Since duplicates have been removed, D002 has been run now.
assert 'D002' in fail_dfs
assert len(fail_dfs["D002"]) == 1
assert len(case_specific_dfs) == 0
assert len(cleaned_df) == 4  # Two duplicates rows consolidated into one.


# Clean Data (pass 2 will clear the missing dates)
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=cleaned_df,
    fail_dict=fail_dfs, case_specific_dict=case_specific_dfs,
    **common_kwargs
)
assert all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 0
# Two duplicates rows consolidated into one plus one missing row added.
assert len(cleaned_df) == 5

#### Categorical Columns

#| hide

# Audit Data
df_with_cat_columns = pd.DataFrame({
    'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2', 'id2'],
    'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
    'y': [1, 2, 3, 4, 5, 6],
    'cat_col1': ['A', 'B', 'C', 'D', 'E', 'F'],
    'cat_col2': pd.Categorical(['X', 'Y', 'Z', 'X', 'Y', 'Z'])
})

all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
    df=df_with_cat_columns,
    **common_kwargs
)

assert not all_pass
assert len(case_specific_dfs) == 0
assert len(fail_dfs) == 1
assert 'F001' in fail_dfs
assert fail_dfs['F001'].shape[1] == 2  # Should return both categorical columns

#### Negative Values 

# | hide 

# Audit Data
df_negative_vals = pd.DataFrame({
    'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2', 'id2'],
    'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
    'y': [-1, 0, 1, 2, -3, -4]
})

all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
    df=df_negative_vals,
    **common_kwargs
)

assert not all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 1
assert 'V001' in case_specific_dfs
assert case_specific_dfs['V001'].shape[0] == 3  # should return all negative values

# Clean Data (without cleaning the case specific issue)
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=df_negative_vals.copy(),
    fail_dict=fail_dfs, 
    case_specific_dict=case_specific_dfs,
    # clean_case_specific=False, # Default
    **common_kwargs
)

assert not all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 1
assert 'V001' in case_specific_dfs
assert case_specific_dfs['V001'].shape[0] == 3  # should return all negative values

# Clean Data (cleaning the case specific issue)
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=df_negative_vals,
    fail_dict=fail_dfs, 
    case_specific_dict=case_specific_dfs,
    clean_case_specific=True,
    **common_kwargs
)

assert not all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 1
assert 'V002' in case_specific_dfs
assert case_specific_dfs['V002'].shape[0] == 1 # should return leading zeros 

# Clean Data, second pass (removes leading zeros)
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=cleaned_df,
    fail_dict=fail_dfs, 
    case_specific_dict=case_specific_dfs,
    clean_case_specific=True,
    **common_kwargs
)

assert all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 0

#### Leading zeros 

#| hide

# Audit Data
df_leading_zeros = pd.DataFrame({
    'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2', 'id2', 'id3', 'id3', 'id3'],
    'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
    'y': [0, 1, 2, 0, 1, 2, 0, 0, 0]
})

all_pass, fail_dfs, case_specific_dfs = custom_client.audit_data(
    df=df_leading_zeros,
    **common_kwargs
)

assert not all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 1
assert 'V002' in case_specific_dfs
assert case_specific_dfs['V002'].shape[0] == 2  # should return ids with leading zeros


# Clean Data (without cleaning the case specific issue)
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=df_leading_zeros,
    fail_dict=fail_dfs,
    case_specific_dict=case_specific_dfs,
    # clean_case_specific=False,  # Default
    **common_kwargs
)

assert not all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 1
assert 'V002' in case_specific_dfs
assert case_specific_dfs['V002'].shape[0] == 2  # should return ids with leading zeros

# Clean Data (clean case specific)
cleaned_df, all_pass, fail_dfs, case_specific_dfs = custom_client.clean_data(
    df=df_leading_zeros,
    fail_dict=fail_dfs,
    case_specific_dict=case_specific_dfs,
    clean_case_specific=True,
    **common_kwargs
)

assert all_pass
assert len(fail_dfs) == 0
assert len(case_specific_dfs) == 0
assert len(cleaned_df) == 7  # all leading zeros removed, zero series unchanged

### Anomaly Detection

#| hide
# anomaly detection with fine-tuned model
train_anomalies = train.copy()
anomaly_date = train_end - 2 * pd.offsets.Day()
train_anomalies.loc[train['ds'] == anomaly_date, 'y'] *= 2
anomaly_base = custom_client.detect_anomalies(train_anomalies)
anomaly_finetune = custom_client.detect_anomalies(train_anomalies, finetuned_model_id=model_id2)
detected_anomalies_base = anomaly_base.set_index('ds').loc[anomaly_date, 'anomaly'].sum()
detected_anomalies_finetune = anomaly_finetune.set_index('ds').loc[anomaly_date, 'anomaly'].sum()
assert detected_anomalies_base < detected_anomalies_finetune

#| hide
# list finetuned models
models = custom_client.finetuned_models()
ids = {m.id for m in models}
assert model_id1 not in ids and model_id2 in ids

#| hide
# get single model
single_model = custom_client.finetuned_model(model_id2)
assert single_model.id == model_id2
assert single_model.base_model_id == model_id1

# non existing not found returns an error
test_fail(lambda: custom_client.finetuned_model('hi'), contains='Model not found')    

#| hide
# test compression
captured_request = None

class CapturingClient(httpx.Client):  
    def post(self, *args, **kwargs):
        request = self.build_request('POST', *args, **kwargs)
        global captured_request
        captured_request = {
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

# this produces a 1MB payload
series = generate_series(250, n_static_features=2)
with capture_request():
    nixtla_client.forecast(df=series, freq='D', h=1, hist_exog_list=['static_0', 'static_1'])

assert captured_request['headers']['content-encoding'] == 'zstd'
content = captured_request['content']
assert len(content) < 2**20
assert len(zstd.ZstdDecompressor().decompress(content)) > 2**20

#| hide
# missing times
series = generate_series(2, min_length=100, freq='5min')
with_gaps = series.sample(frac=0.5, random_state=0)
expected_msg = 'missing or duplicate timestamps, or the timestamps do not match'
# gaps
test_fail(
    lambda: nixtla_client.forecast(df=with_gaps, h=1, freq='5min'),
    contains=expected_msg,
)
# duplicates
test_fail(
    lambda: nixtla_client.forecast(df=pd.concat([series, series]), h=1, freq='5min'),
    contains=expected_msg,
)
# wrong freq
test_fail(
    lambda: nixtla_client.forecast(df=series, h=1, freq='1min'),
    contains=expected_msg,
)

#| hide
# historic exog in cv
freq = 'D'
h = 5
series = generate_series(2, freq=freq)
series_with_features, _ = fourier(series, freq=freq, season_length=7, k=2)
splits = ufp.backtest_splits(
    df=series_with_features,
    n_windows=1,
    h=h,
    id_col='unique_id',
    time_col='ds',
    freq=freq,
)
_, train, valid = next(splits)
x_cols = train.columns.drop(['unique_id', 'ds', 'y']).tolist()
for hist_exog_list in [None, [], [x_cols[2], x_cols[1]], x_cols]:
    cv_res = nixtla_client.cross_validation(
        series_with_features,
        n_windows=1,
        h=h,
        freq=freq,
        hist_exog_list=hist_exog_list,
    )
    fcst_res = nixtla_client.forecast(
        train,
        h=h,
        freq=freq,
        hist_exog_list=hist_exog_list,
        X_df=valid,
    )
    np.testing.assert_allclose(
        cv_res['TimeGPT'], fcst_res['TimeGPT'], atol=1e-4, rtol=1e-3
    )

#| hide
# different hist exog, different results
for X_df in (None, valid):
    res1 = nixtla_client.forecast(train, h=h, X_df=X_df, freq=freq, hist_exog_list=x_cols[:2])
    res2 = nixtla_client.forecast(train, h=h, X_df=X_df, freq=freq, hist_exog_list=x_cols[2:])
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        res1['TimeGPT'],
        res2['TimeGPT'],
        atol=1e-4,
        rtol=1e-3,
    )

#| hide
# custom freq
client = NixtlaClient()
custom_business_hours = pd.tseries.offsets.CustomBusinessHour(
    start='09:00',
    end='16:00',
    holidays=[
        '2022-12-25',  # Christmas
        '2022-01-01',   # New Year's Day
    ]
)
series = pd.DataFrame({
    'unique_id': 1,
    'ds': pd.date_range(start='2000-01-03 09', freq=custom_business_hours, periods=200),
    'y': np.arange(200) % 7,
})
series = pd.concat([series.assign(unique_id=i) for i in range(10)]).reset_index(drop=True)
client.detect_anomalies(df=series, freq=custom_business_hours, level=90)
client.cross_validation(df=series, freq=custom_business_hours, h=7)
fcst = client.forecast(df=series, freq=custom_business_hours, h=7)
assert sorted(fcst['ds'].dt.hour.unique().tolist()) == list(range(9, 16))
assert [(model, freq.lower()) for (model, freq) in client._model_params.keys()] == [('timegpt-1', 'cbh')]

#| hide
# integer freq
client = NixtlaClient()
series = generate_series(5, freq='H', min_length=200)
series['ds'] = series.groupby('unique_id', observed=True)['ds'].cumcount()
client.detect_anomalies(df=series, level=90, freq=1)
client.cross_validation(df=series, h=7, freq=1)
fcst = client.forecast(df=series, h=7, freq=1)
train_ends = series.groupby('unique_id', observed=True)['ds'].max()
fcst_ends = fcst.groupby('unique_id', observed=True)['ds'].max()
pd.testing.assert_series_equal(fcst_ends, train_ends + 7)
assert list(client._model_params.keys()) == [('timegpt-1', 'MS')]

#| hide
test_fail(
    lambda: NixtlaClient(api_key='transphobic').forecast(df=pd.DataFrame(), h=None, validate_api_key=True),
    contains='nixtla'
)

#| hide
# test input_size
test_eq(
    nixtla_client._get_model_params(model='timegpt-1', freq='D'),
    (28, 7),
)
test_eq(
    custom_client._get_model_params(model='timegpt-1', freq='D'),
    (28, 7),
)

#| hide
df = pd.read_csv(
    'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv',
    parse_dates=['timestamp'],
)
df.head()

#| hide
# test date_features with multiple series
# and different ends
test_series = generate_series(n_series=2, min_length=5, max_length=20)
h = 12
fcst_test_series = nixtla_client.forecast(test_series, h=12, date_features=['dayofweek'])
uids = test_series['unique_id']
for uid in uids:
    test_eq(
        fcst_test_series.query('unique_id == @uid')['ds'].tolist(),
        pd.date_range(periods=h + 1, start=test_series.query('unique_id == @uid')['ds'].max())[1:].tolist(),
    )

#| hide
# cv refit
cv_kwargs = dict(
    df=df,
    n_windows=2,
    h=12,
    freq='MS',
    time_col='timestamp',
    target_col='value',
    finetune_steps=2,
)
res_refit = nixtla_client.cross_validation(refit=True, **cv_kwargs)
res_no_refit = nixtla_client.cross_validation(refit=False, **cv_kwargs)
np.testing.assert_allclose(res_refit['value'], res_no_refit['value'])
np.testing.assert_raises(
    AssertionError,
    np.testing.assert_allclose,
    res_refit['TimeGPT'],
    res_no_refit['TimeGPT'],
    atol=1e-4,
    rtol=1e-3,
)

#| hide
# test quantiles
test_fail(
    lambda: nixtla_client.forecast(
        df=df, 
        h=12, 
        time_col='timestamp', 
        target_col='value', 
        level=[80], 
        quantiles=[0.2, 0.3]
    ),
    contains='not both'
)
test_qls = list(np.arange(0.1, 1, 0.1))
exp_q_cols = [f"TimeGPT-q-{int(100 * q)}" for q in test_qls]
def test_method_qls(method, **kwargs):
    df_qls = method(
        df=df, 
        h=12, 
        time_col='timestamp', 
        target_col='value', 
        quantiles=test_qls,
        **kwargs
    )
    assert all(col in df_qls.columns for col in exp_q_cols)
    assert not any('-lo-' in col for col in df_qls.columns)
    # test monotonicity of quantiles
    for c1, c2 in zip(exp_q_cols[:-1], exp_q_cols[1:]):
        assert df_qls[c1].lt(df_qls[c2]).all()
test_method_qls(nixtla_client.forecast)
test_method_qls(nixtla_client.forecast, add_history=True)
test_method_qls(nixtla_client.cross_validation)

#| hide
# test num partitions
# we need to be sure that we can recover the same results
# using a for loop
# A: be aware that num partitons can produce different results
# when used finetune_steps
def test_num_partitions_same_results(method: Callable, num_partitions: int, **kwargs):
    res_partitioned = method(**kwargs, num_partitions=num_partitions)
    res_no_partitioned = method(**kwargs, num_partitions=1)
    sort_by = ['unique_id', 'ds']
    if 'cutoff' in res_partitioned:
        sort_by.extend(['cutoff'])
    pd.testing.assert_frame_equal(
        res_partitioned.sort_values(sort_by).reset_index(drop=True), 
        res_no_partitioned.sort_values(sort_by).reset_index(drop=True),
        rtol=1e-2,
        atol=1e-2,
    )

freqs = {'D': 7, 'W-THU': 52, 'Q-DEC': 8, '15T': 4 * 24 * 7}
for freq, h in freqs.items():
    df_freq = generate_series(
        10, 
        min_length=500 if freq != '15T' else 1_200, 
        max_length=550 if freq != '15T' else 2_000,
    )
    #df_freq['y'] = df_freq['y'].astype(np.float32)
    df_freq['ds'] = df_freq.groupby('unique_id', observed=True)['ds'].transform(
        lambda x: pd.date_range(periods=len(x), freq=freq, end='2023-01-01')
    )
    min_size = df_freq.groupby('unique_id', observed=True).size().min()
    test_num_partitions_same_results(
        nixtla_client.detect_anomalies,
        level=98,
        df=df_freq,
        num_partitions=2,
    )
    test_num_partitions_same_results(
        nixtla_client.cross_validation,
        h=7,
        n_windows=2,
        df=df_freq,
        num_partitions=2,
    )
    test_num_partitions_same_results(
        nixtla_client.forecast,
        df=df_freq,
        h=7,
        add_history=True,
        num_partitions=2,
    )
    df_freq["exog_1"] = 1
    test_num_partitions_same_results(
        nixtla_client.detect_anomalies,
        level=98,
        df=df_freq,
        num_partitions=2,
    )
    test_num_partitions_same_results(
        nixtla_client.cross_validation,
        h=7,
        n_windows=2,
        df=df_freq,
        num_partitions=2,
    )
    test_num_partitions_same_results(
        nixtla_client.forecast,
        df=df_freq,
        h=7,
        add_history=True,
        num_partitions=2,
    )

#| hide
def test_retry_behavior(side_effect, max_retries=5, retry_interval=5, max_wait_time=40, should_retry=True, sleep_seconds=5):
    mock_nixtla_client = NixtlaClient(
        max_retries=max_retries, 
        retry_interval=retry_interval, 
        max_wait_time=max_wait_time,
    )
    mock_nixtla_client._make_request = side_effect
    init_time = time.time()
    test_fail(
        lambda: mock_nixtla_client.forecast(df=df, h=12, time_col='timestamp', target_col='value'),
    )
    total_mock_time = time.time() - init_time
    if should_retry:
        approx_expected_time = min((max_retries - 1) * retry_interval, max_wait_time)
        upper_expected_time = min(max_retries * retry_interval, max_wait_time)
        assert total_mock_time >= approx_expected_time, "It is not retrying as expected"
        # preprocessing time before the first api call should be less than 60 seconds
        assert total_mock_time - upper_expected_time - (max_retries - 1) * sleep_seconds <= sleep_seconds
    else:
        assert total_mock_time <= max_wait_time 

#| hide
# we want the api to retry in these cases
def raise_api_error_with_text(*args, **kwargs):
    raise ApiError(
        status_code=503, 
        body="""
        <html><head>
        <meta http-equiv="content-type" content="text/html;charset=utf-8">
        <title>503 Server Error</title>
        </head>
        <body text=#000000 bgcolor=#ffffff>
        <h1>Error: Server Error</h1>
        <h2>The service you requested is not available at this time.<p>Service error -27.</h2>
        <h2></h2>
        </body></html>
        """)
test_retry_behavior(raise_api_error_with_text)

#| hide
# we want the api to not retry in these cases
# here A is assuming that the endpoint responds always
# with a json
def raise_api_error_with_json(*args, **kwargs):
    raise ApiError(
        status_code=422, 
        body=dict(detail='Please use numbers'),
    )
test_retry_behavior(raise_api_error_with_json, should_retry=False)

#| hide
# test resilience of api calls
def raise_read_timeout_error(*args, **kwargs):
    sleep_seconds = 5
    print(f'raising ReadTimeout error after {sleep_seconds} seconds')
    time.sleep(sleep_seconds)
    raise httpx.ReadTimeout('Timed out')

def raise_http_error(*args, **kwargs):
    print('raising HTTP error')
    raise ApiError(status_code=503, body='HTTP error')
    
combs = [
    (2, 5, 30),
    (10, 1, 5),
]
side_effects = [raise_read_timeout_error, raise_http_error]

for (max_retries, retry_interval, max_wait_time), side_effect in product(combs, side_effects):
    test_retry_behavior(
        max_retries=max_retries, 
        retry_interval=retry_interval, 
        max_wait_time=max_wait_time, 
        side_effect=side_effect,
    )

#| hide
nixtla_client.plot(df, time_col='timestamp', target_col='value', engine='plotly')

#| hide
# test we recover the same <mean> forecasts
# with and without restricting input
# (add_history)
def test_equal_fcsts_add_history(**kwargs):
    fcst_no_rest_df = nixtla_client.forecast(**kwargs, add_history=True)
    fcst_no_rest_df = fcst_no_rest_df.groupby('unique_id', observed=True).tail(kwargs['h']).reset_index(drop=True)
    fcst_rest_df = nixtla_client.forecast(**kwargs)
    pd.testing.assert_frame_equal(
        fcst_no_rest_df,
        fcst_rest_df,
        atol=1e-4,
        rtol=1e-3,
    )
    return fcst_rest_df

freqs = {'D': 7, 'W-THU': 52, 'Q-DEC': 8, '15T': 4 * 24 * 7}
for freq, h in freqs.items():
    df_freq = generate_series(
        10, 
        min_length=500 if freq != '15T' else 1_200, 
        max_length=550 if freq != '15T' else 2_000,
    )
    df_freq['ds'] = df_freq.groupby('unique_id', observed=True)['ds'].transform(
        lambda x: pd.date_range(periods=len(x), freq=freq, end='2023-01-01')
    )
    kwargs = dict(
        df=df_freq,
        h=h,
    )
    fcst_1_df = test_equal_fcsts_add_history(**{**kwargs, 'model': 'timegpt-1'})
    fcst_2_df = test_equal_fcsts_add_history(**{**kwargs, 'model': 'timegpt-1-long-horizon'})
    test_fail(
        lambda: pd.testing.assert_frame_equal(fcst_1_df, fcst_2_df),
        contains='(column name="TimeGPT") are different',
    )
    # add test num_partitions    

#| hide
# test different results for different models
fcst_kwargs = dict(
    df=df,
    h=12,
    level=[90, 95],
    add_history=True,
    time_col='timestamp',
    target_col='value',
)
fcst_kwargs['model'] = 'timegpt-1'
fcst_timegpt_1 = nixtla_client.forecast(**fcst_kwargs)
fcst_kwargs['model'] = 'timegpt-1-long-horizon'
fcst_timegpt_long = nixtla_client.forecast(**fcst_kwargs)
test_fail(
    lambda: pd.testing.assert_frame_equal(fcst_timegpt_1[['TimeGPT']], fcst_timegpt_long[['TimeGPT']]),
    contains='(column name="TimeGPT") are different'
)

#| hide
# test different results for different models
# cross validation
cv_kwargs = dict(
    df=df,
    h=12,
    level=[90, 95],
    time_col='timestamp',
    target_col='value',
)
cv_kwargs['model'] = 'timegpt-1'
cv_timegpt_1 = nixtla_client.cross_validation(**cv_kwargs)
cv_kwargs['model'] = 'timegpt-1-long-horizon'
cv_timegpt_long = nixtla_client.cross_validation(**cv_kwargs)
test_fail(
    lambda: pd.testing.assert_frame_equal(cv_timegpt_1[['TimeGPT']], cv_timegpt_long[['TimeGPT']]),
    contains='(column name="TimeGPT") are different'
)

#| hide
# test different results for different models
# anomalies
anomalies_kwargs = dict(
    df=df,
    level=99,
    time_col='timestamp',
    target_col='value',
)
anomalies_kwargs['model'] = 'timegpt-1'
anomalies_timegpt_1 = nixtla_client.detect_anomalies(**anomalies_kwargs)
anomalies_kwargs['model'] = 'timegpt-1-long-horizon'
anomalies_timegpt_long = nixtla_client.detect_anomalies(**anomalies_kwargs)
test_fail(
    lambda: pd.testing.assert_frame_equal(anomalies_timegpt_1[['TimeGPT']], anomalies_timegpt_long[['TimeGPT']]),
    contains='(column name="TimeGPT") are different'
)

#| hide
# test unsupported model
fcst_kwargs['model'] = 'a-model'
test_fail(
    lambda: nixtla_client.forecast(**fcst_kwargs),
    contains='unsupported model',
)

#| hide
# test unsupported model
anomalies_kwargs['model'] = 'my-awesome-model'
test_fail(
    lambda: nixtla_client.detect_anomalies(**anomalies_kwargs),
    contains='unsupported model',
)

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
# test add callables
date_features = [SpecialDates({'first_dates': ['2021-01-1'], 'second_dates': ['2021-01-01']})]
df_daily = df_.copy()
df_daily['ds'] = pd.date_range(end='2021-01-01', periods=len(df_daily))
df_date_features, future_df = _maybe_add_date_features(
    df=df_,
    X_df=None,
    h=12, 
    freq='D', 
    features=date_features,
    one_hot=False,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
)
assert all(col in df_date_features for col in ['first_dates', 'second_dates'])
assert all(col in future_df for col in ['first_dates', 'second_dates'])

#| hide
# test add date features one hot encoded
date_features = ['year', 'month']
date_features_to_one_hot = ['month']
df_date_features, future_df = _maybe_add_date_features(
    df=df_,
    X_df=None,
    h=12, 
    freq='D', 
    features=date_features,
    one_hot=date_features_to_one_hot,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
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
import dask.dataframe as dd
import fugue
import fugue.api as fa
import ray
from dask.distributed import Client
from pyspark.sql import SparkSession
from ray.cluster_utils import Cluster

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

#| hide
#| distributed
n_series = 4
horizon = 7

series = generate_series(n_series, min_length=100)
series['unique_id'] = series['unique_id'].astype(str)

series_diff_cols = series.copy()
renamer = {'unique_id': 'id_col', 'ds': 'time_col', 'y': 'target_col'}
series_diff_cols = series_diff_cols.rename(columns=renamer)

# data for exogenous tests
df_x = pd.read_csv(
    'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-with-ex-vars.csv',
    parse_dates=['ds'],
)
df_x = df_x.rename(columns=str.lower)
future_ex_vars_df = pd.read_csv(
    'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-future-ex-vars.csv',
    parse_dates=['ds'],
)
future_ex_vars_df = future_ex_vars_df.rename(columns=str.lower)

#### Spark

#| hide
#| distributed
spark = SparkSession.builder.getOrCreate()
spark_df = spark.createDataFrame(series).repartition(2)
spark_diff_cols_df = spark.createDataFrame(series_diff_cols).repartition(2)

test_quantiles(spark_df, id_col="unique_id", time_col="ds")

test_forecast_dataframe(spark_df)
test_forecast_dataframe_diff_cols(spark_diff_cols_df)
test_anomalies_dataframe(spark_df)
test_anomalies_online_dataframe(spark_df)
test_anomalies_dataframe_diff_cols(spark_diff_cols_df)
# test exogenous variables
spark_df_x = spark.createDataFrame(df_x).repartition(2)
spark_future_ex_vars_df = spark.createDataFrame(future_ex_vars_df).repartition(2)
test_forecast_x_dataframe(spark_df_x, spark_future_ex_vars_df)
# test x different cols
spark_df_x_diff_cols = spark.createDataFrame(df_x.rename(columns=renamer)).repartition(2)
spark_future_ex_vars_df_diff_cols = spark.createDataFrame(
    future_ex_vars_df.rename(columns=renamer)
).repartition(2)
test_forecast_x_dataframe_diff_cols(spark_df_x_diff_cols, spark_future_ex_vars_df_diff_cols)

# test finetuning
test_finetuned_model(spark_df)

spark.stop()

#### Dask

#| hide
#| distributed
client = Client()
dask_df = dd.from_pandas(series, npartitions=2)
dask_diff_cols_df = dd.from_pandas(series_diff_cols, npartitions=2)


test_quantiles(dask_df, id_col="unique_id", time_col="ds")


test_forecast_dataframe(dask_df)
test_forecast_dataframe_diff_cols(dask_diff_cols_df)
test_anomalies_dataframe(dask_df)
test_anomalies_online_dataframe(dask_df)
test_anomalies_dataframe_diff_cols(dask_diff_cols_df)

# test exogenous variables
dask_df_x = dd.from_pandas(df_x, npartitions=2)
dask_future_ex_vars_df = dd.from_pandas(future_ex_vars_df, npartitions=2)
test_forecast_x_dataframe(dask_df_x, dask_future_ex_vars_df)

# test x different cols
dask_df_x_diff_cols = dd.from_pandas(df_x.rename(columns=renamer), npartitions=2)
dask_future_ex_vars_df_diff_cols = dd.from_pandas(future_ex_vars_df.rename(columns=renamer), npartitions=2)
test_forecast_x_dataframe_diff_cols(dask_df_x_diff_cols, dask_future_ex_vars_df_diff_cols)

# test finetuning
test_finetuned_model(dask_df)

client.close()

#### Ray

#| hide
#| distributed
ray_cluster = Cluster(
    initialize_head=True,
    head_node_args={"num_cpus": 2}
)
ray.init(address=ray_cluster.address, ignore_reinit_error=True)
# add mock node to simulate a cluster
mock_node = ray_cluster.add_node(num_cpus=2)

ray_df = ray.data.from_pandas(series)
ray_diff_cols_df = ray.data.from_pandas(series_diff_cols)

test_quantiles(ray_df, id_col="unique_id", time_col="ds")

test_forecast_dataframe(ray_df)
test_forecast_dataframe_diff_cols(ray_diff_cols_df)
test_anomalies_dataframe(ray_df)
test_anomalies_online_dataframe(ray_df)
test_anomalies_dataframe_diff_cols(ray_diff_cols_df)

# test exogenous variables
ray_df_x = ray.data.from_pandas(df_x)
ray_future_ex_vars_df = ray.data.from_pandas(future_ex_vars_df)
test_forecast_x_dataframe(ray_df_x, ray_future_ex_vars_df)

# test x different cols
ray_df_x_diff_cols = ray.data.from_pandas(df_x.rename(columns=renamer))
ray_future_ex_vars_df_diff_cols = ray.data.from_pandas(future_ex_vars_df.rename(columns=renamer))
test_forecast_x_dataframe_diff_cols(ray_df_x_diff_cols, ray_future_ex_vars_df_diff_cols)

# test finetuning
test_finetuned_model(ray_df)

ray.shutdown()

#| hide
# cleanup
custom_client.delete_finetuned_model(model_id2)