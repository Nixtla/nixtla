import pandas as pd
import pytest

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