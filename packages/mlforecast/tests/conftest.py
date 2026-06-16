import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals, generate_daily_series


def assert_raises_with_message(func, expected_msg, *args, **kwargs):
    with pytest.raises((AssertionError, ValueError, Exception)) as exc_info:
        func(*args, **kwargs)
    assert expected_msg in str(exc_info.value)


def make_groupby_df(engine, series_data):
    """Create a test DataFrame with brand column for groupby tests.

    Args:
        engine: "pandas" or "polars"
        series_data: list of (uid, y_values, brand) tuples
    """
    rows = {"unique_id": [], "ds": [], "y": [], "brand": []}
    for uid, y_vals, brand in series_data:
        n = len(y_vals)
        rows["unique_id"].extend([uid] * n)
        rows["ds"].extend(range(1, n + 1))
        rows["y"].extend(y_vals)
        rows["brand"].extend([brand] * n)
    if engine == "polars":
        return pl.DataFrame(rows).with_columns(pl.col("unique_id").cast(pl.Categorical))
    return pd.DataFrame(rows)


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


@pytest.fixture
def setup_forecast_data():
    df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
    ids = df['unique_id'].unique()
    random.seed(0)
    sample_ids = random.choices(ids, k=4)
    sample_df = df[df['unique_id'].isin(sample_ids)]
    horizon = 48
    valid = sample_df.groupby('unique_id').tail(horizon)
    train = sample_df.drop(valid.index)
    return df, train, valid


@pytest.fixture
def weighted_conformal_setup():
    """Small synthetic dataset + fitted MLForecast for weighted conformal tests."""
    series = generate_daily_series(4, min_length=60, max_length=80, seed=0)
    h = 7
    n_windows = 2
    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1, 7],
        date_features=["dayofweek"],
        num_threads=1,
    )
    pi = PredictionIntervals(method="weighted_conformal_error", n_windows=n_windows, h=h)
    fcst.fit(series, prediction_intervals=pi)
    return fcst, series, h, n_windows


@pytest.fixture
def scale_aligned_setup():
    """Source series at normal scale; target series scaled up ×400."""
    rng = np.random.default_rng(42)  # noqa: F841
    n_source = 4
    source_series = generate_daily_series(n_source, min_length=60, max_length=80, seed=0)
    target_series = source_series.copy()
    target_series["y"] = target_series["y"] * 400.0

    h = 7
    n_windows = 2
    fcst = MLForecast(
        models=lgb.LGBMRegressor(random_state=0, verbosity=-1, n_estimators=5),
        freq="D",
        lags=[1, 7],
        date_features=["dayofweek"],
        num_threads=1,
    )
    pi = PredictionIntervals(method="conformal_error", n_windows=n_windows, h=h, scale_estimator="mad")
    fcst.fit(source_series, prediction_intervals=pi)
    return fcst, source_series, target_series, h
