import os
import numpy as np
import pandas as pd
import pytest
import utilsforecast.processing as ufp

from nixtla.nixtla_client import NixtlaClient
from utilsforecast.data import generate_series
from utilsforecast.feature_engineering import fourier
from types import SimpleNamespace
from nixtla_tests.helpers.states import model_ids_object

# note that scope="session" will result in failed test
@pytest.fixture(scope="class")
def nixtla_test_client():
    client = NixtlaClient()
    yield client

    try:
        client.delete_finetuned_model(model_ids_object.model_id1)
    except:
        print("model_id1 not found, skipping deletion.")

    try:
        client.delete_finetuned_model(model_ids_object.model_id2)
    except:
        print("model_id2 not found, skipping deletion.")


@pytest.fixture(scope="class")
def custom_client():
    client = NixtlaClient(
        base_url=os.environ['NIXTLA_BASE_URL_CUSTOM'],
        api_key=os.environ['NIXTLA_API_KEY_CUSTOM'],
    )
    yield client

    try:
        client.delete_finetuned_model(model_ids_object.model_id1)
    except:
        print("model_id1 not found, skipping deletion.")

    try:
        client.delete_finetuned_model(model_ids_object.model_id2)
    except:
        print("model_id2 not found, skipping deletion.")

@pytest.fixture
def series_with_gaps():
    series = generate_series(2, min_length=100, freq='5min')
    with_gaps = series.sample(frac=0.5, random_state=0)
    return series, with_gaps

@pytest.fixture
def df_no_duplicates():
    return pd.DataFrame(
        {
            'unique_id': [1, 2, 3, 4],
            'ds': ['2020-01-01'] * 4,
            'y': [1, 2, 3, 4],
        }
    )

@pytest.fixture
def df_with_duplicates():
    return pd.DataFrame(
        {
            'unique_id': [1, 1, 1],
            'ds': ['2020-01-01', '2020-01-01', '2020-01-02'],
            'y': [1, 2, 3],
        }
    )

@pytest.fixture
def df_complete():
    return pd.DataFrame({
        'unique_id': [1, 1, 1, 2, 2, 2],
        'ds': ['2020-01-01', '2020-01-02', '2020-01-03',
               '2020-01-01', '2020-01-02', '2020-01-03'],
        'y': [1, 2, 3, 4, 5, 6],
    })

@pytest.fixture
def df_missing():
    return pd.DataFrame({
        'unique_id': [1, 1, 2, 2],
        'ds': ['2020-01-01', '2020-01-03', '2020-01-01', '2020-01-03'],
        'y': [1, 3, 4, 6],
    })

@pytest.fixture
def df_no_cat():
    return pd.DataFrame({
        'unique_id': [1, 2, 3],
        'ds': pd.date_range('2023-01-01', periods=3),
        'y': [1.0, 2.0, 3.0]
    })

@pytest.fixture
def df_with_cat():
    return pd.DataFrame({
        'unique_id': ['A', 'B', 'C'],
        'ds': pd.date_range('2023-01-01', periods=3),
        'y': [1.0, 2.0, 3.0],
        'cat_col': ['X', 'Y', 'Z']
    })

@pytest.fixture
def df_with_cat_dtype():
    return pd.DataFrame({
        'unique_id': [1, 2, 3],
        'ds': pd.date_range('2023-01-01', periods=3),
        'y': [1.0, 2.0, 3.0],
        'cat_col': pd.Categorical(['X', 'Y', 'Z'])
    })

@pytest.fixture
def df_leading_zeros():
    return pd.DataFrame({
        'unique_id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'D', 'D', 'D'],
        'ds': pd.date_range('2025-01-01', periods=12),
        'y': [0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 0]
    })

@pytest.fixture
def df_negative_values():
    return pd.DataFrame({
        'unique_id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
        'ds': pd.date_range('2025-01-01', periods=8),
        'y': [0, -1, 2, -1, -2, 0, 1, 2]
    })

@pytest.fixture(scope="module")
def ts_data_set1():
    h = 5
    series = generate_series(10, equal_ends=True)
    train_end = series['ds'].max() - h * pd.offsets.Day()
    train_mask = series['ds'] <= train_end
    train = series[train_mask]
    valid = series[~train_mask]

    return SimpleNamespace(
        h=h,
        series=series,
        train=train,
        train_end=train_end,
        valid=valid,
    )

@pytest.fixture(scope="module")
def ts_anomaly_data(ts_data_set1):
    train_anomalies = ts_data_set1.train.copy()
    anomaly_date = ts_data_set1.train_end - 2 * pd.offsets.Day()
    train_anomalies.loc[train_anomalies['ds'] == anomaly_date, 'y'] *= 2

    return SimpleNamespace(
        train_anomalies=train_anomalies,
        anomaly_date=anomaly_date,
    )

@pytest.fixture
def common_kwargs():
    return {
        "freq": "D",
        "id_col": 'unique_id',
        "time_col": 'ds'
    }

@pytest.fixture
def df_ok():
    return pd.DataFrame({
        'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2', 'id2'],
        'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
        'y': [1, 2, 3, 4, 5, 6]
    })

@pytest.fixture
def df_with_duplicates_set2():
    return pd.DataFrame({
        'unique_id': ['id1', 'id1', 'id1', 'id2'],
        'ds': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'y': [1, 2, 3, 4]
    })


@pytest.fixture
def df_with_missing_dates():
    return pd.DataFrame({
        'unique_id': ['id1', 'id1', 'id2', 'id2'],
        'ds': ['2023-01-01', '2023-01-03', '2023-01-01', '2023-01-03'],
        'y': [1, 3, 4, 6]
    })

@pytest.fixture
def df_with_duplicates_and_missing_dates():
    # Global end on 2023-01-03 which is missing for id1
    return pd.DataFrame({
        'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2'],
        'ds': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'],
        'y': [1, 2, 3, 4, 5]
    })

@pytest.fixture
def df_with_cat_columns():
    return pd.DataFrame({
        'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2', 'id2'],
        'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
        'y': [1, 2, 3, 4, 5, 6],
        'cat_col1': ['A', 'B', 'C', 'D', 'E', 'F'],
        'cat_col2': pd.Categorical(['X', 'Y', 'Z', 'X', 'Y', 'Z'])
    })

@pytest.fixture
def df_negative_vals():
    return pd.DataFrame({
        'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2', 'id2'],
        'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
        'y': [-1, 0, 1, 2, -3, -4]
    })

@pytest.fixture
def df_leading_zeros_set2():
    return pd.DataFrame({
        'unique_id': ['id1', 'id1', 'id1', 'id2', 'id2', 'id2', 'id3', 'id3', 'id3'],
        'ds': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
        'y': [0, 1, 2, 0, 1, 2, 0, 0, 0]
    })

@pytest.fixture
def cv_series_with_features():
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
    return series_with_features, train, valid, x_cols, h, freq

@pytest.fixture
def custom_business_hours():
    return pd.tseries.offsets.CustomBusinessHour(
        start='09:00',
        end='16:00',
        holidays=[
            '2022-12-25',  # Christmas
            '2022-01-01',   # New Year's Day
        ]
    )

@pytest.fixture
def business_hours_series(custom_business_hours):
    series = pd.DataFrame({
        'unique_id': 1,
        'ds': pd.date_range(start='2000-01-03 09', freq=custom_business_hours, periods=200),
        'y': np.arange(200) % 7,
    })
    series = pd.concat([series.assign(unique_id=i) for i in range(10)]).reset_index(drop=True)
    return series

@pytest.fixture
def integer_freq_series():
    series = generate_series(5, freq='H', min_length=200)
    series['ds'] = series.groupby('unique_id', observed=True)['ds'].cumcount()
    return series

@pytest.fixture
def two_short_series():
    return generate_series(n_series=2, min_length=5, max_length=20)

@pytest.fixture
def series_1MB_payload():
    series = generate_series(250, n_static_features=2)
    return series

@pytest.fixture(scope="module")
def air_passengers_df():
    return pd.read_csv(
        'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv',
        parse_dates=['timestamp'],
    )