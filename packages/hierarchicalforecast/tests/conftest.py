import numpy as np
import pandas as pd
import pytest

from hierarchicalforecast.utils import aggregate


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


@pytest.fixture
def strict_hierarchy_data(hiers_strictly):
    """Fixture providing strictly hierarchical data for coherency tests.

    Creates a simple strict hierarchy (Country -> State -> Region) with synthetic data.
    """
    # Create a simple dataset with a strict hierarchy structure
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    data = []

    # Bottom-level structure: Country -> State -> Region
    structure = {
        ("AU", "NSW", "Sydney"): 100,
        ("AU", "NSW", "Newcastle"): 50,
        ("AU", "VIC", "Melbourne"): 80,
        ("AU", "VIC", "Geelong"): 40,
    }

    for (country, state, region), base_val in structure.items():
        for ds in dates:
            data.append({
                "Country": country,
                "State": state,
                "Region": region,
                "ds": ds,
                "y": base_val + np.random.randn() * 10
            })

    df = pd.DataFrame(data)

    # Create aggregated hierarchy using the hiers_strictly fixture
    Y_df, S_df, tags = aggregate(df, hiers_strictly)

    # Prepare train/test split
    Y_df["y_model"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(12).copy()
    ds_h = Y_hat_df["ds"].unique()  # noqa: F841
    Y_train_df = Y_df.query("~(ds in @ds_h)").copy()
    Y_train_df["y_model"] += np.random.uniform(-1, 1, len(Y_train_df))

    return {
        "Y_hat_df": Y_hat_df,
        "Y_train_df": Y_train_df,
        "S_df": S_df,
        "tags": tags,
    }
