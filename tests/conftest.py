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
