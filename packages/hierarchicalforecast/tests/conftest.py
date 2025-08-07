import pandas as pd
import pytest


def assert_raises_with_message(func, expected_msg, *args, **kwargs):
    with pytest.raises((AssertionError, ValueError, Exception)) as exc_info:
        func(*args, **kwargs)
    assert expected_msg in str(exc_info.value)

@pytest.fixture(scope="module")
def tourism_df():
    df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
    df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
    df.insert(0, 'Country', 'Australia')
    return df

@pytest.fixture(scope="module")
def hiers_grouped():
    # grouped structure
    hiers_grouped = [['Country'],
                    ['Country', 'State'],
                    ['Country', 'Purpose'],
                    ['Country', 'State', 'Region'],
                    ['Country', 'State', 'Purpose'],
                    ['Country', 'State', 'Region', 'Purpose'],
                    ]
    return hiers_grouped


@pytest.fixture(scope="module")
def hiers_strictly():
    # strictly hierarchical structure
    hiers_strictly = [['Country'],
                    ['Country', 'State'],
                    ['Country', 'State', 'Region']]
    return hiers_strictly
