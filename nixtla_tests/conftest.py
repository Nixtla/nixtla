import os
import pandas as pd
import pytest
import uuid

from utilsforecast.data import generate_series
from types import SimpleNamespace

# try this global scope for test, ortherwise consider function scope
@pytest.fixture
def custom_client():
    from nixtla.nixtla_client import NixtlaClient
    return NixtlaClient(
        base_url=os.environ['NIXTLA_BASE_URL_CUSTOM'],
        api_key=os.environ['NIXTLA_API_KEY_CUSTOM'],
    )

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
    model_id1 = str(uuid.uuid4())

    return SimpleNamespace(
        h=h,
        series=series,
        train=train,
        valid=valid,
        model_id1=model_id1,
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
def df_with_duplicates():
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
# Global end on 2023-01-03 which is missing for id1
def df_with_duplicates_and_missing_dates():
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
