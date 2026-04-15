import pandas as pd
import pytest


def assert_raises_with_message(func, expected_msg, *args, **kwargs):
    with pytest.raises((AssertionError, ValueError, Exception)) as exc_info:
        func(*args, **kwargs)
    assert expected_msg in str(exc_info.value)


@pytest.fixture
def grouped_expanding_mean_df():
    rows = []
    for uid, cat, vals in [
        ("a", 0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        ("b", 0, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
        ("c", 1, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
        ("d", 1, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
    ]:
        for i, y in enumerate(vals):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "y": y,
                    "cat_code": cat,
                }
            )
    df = pd.DataFrame(rows)
    df["cat_code"] = df["cat_code"].astype("int32")
    return df
