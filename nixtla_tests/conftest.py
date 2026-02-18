import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import utilsforecast.processing as ufp
from dotenv import load_dotenv
from utilsforecast.data import generate_series
from utilsforecast.feature_engineering import fourier, time_features

from nixtla.nixtla_client import NixtlaClient, _maybe_add_date_features
from nixtla_tests.helpers.states import model_ids_object

load_dotenv(override=True)

pytest_plugins = [
    "nixtla_tests.fixtures.dask_fixtures",
    "nixtla_tests.fixtures.spark_fixtures",
    "nixtla_tests.fixtures.ray_fixtures",
]


# note that scope="session" will result in failed test
@pytest.fixture(scope="module")
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
        base_url=os.environ["NIXTLA_BASE_URL_CUSTOM"],
        api_key=os.environ["NIXTLA_API_KEY_CUSTOM"],
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
    series = generate_series(2, min_length=100, freq="5min")
    with_gaps = series.sample(frac=0.5, random_state=0)
    return series, with_gaps


@pytest.fixture
def df_no_duplicates():
    return pd.DataFrame(
        {
            "unique_id": [1, 2, 3, 4],
            "ds": ["2020-01-01"] * 4,
            "y": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def df_with_duplicates():
    return pd.DataFrame(
        {
            "unique_id": [1, 1, 1],
            "ds": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "y": [1, 2, 3],
        }
    )


@pytest.fixture
def df_complete():
    return pd.DataFrame(
        {
            "unique_id": [1, 1, 1, 2, 2, 2],
            "ds": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def df_missing():
    return pd.DataFrame(
        {
            "unique_id": [1, 1, 2, 2],
            "ds": ["2020-01-01", "2020-01-03", "2020-01-01", "2020-01-03"],
            "y": [1, 3, 4, 6],
        }
    )


@pytest.fixture
def df_no_cat():
    return pd.DataFrame(
        {
            "unique_id": [1, 2, 3],
            "ds": pd.date_range("2023-01-01", periods=3),
            "y": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def df_with_cat():
    return pd.DataFrame(
        {
            "unique_id": ["A", "B", "C"],
            "ds": pd.date_range("2023-01-01", periods=3),
            "y": [1.0, 2.0, 3.0],
            "cat_col": ["X", "Y", "Z"],
        }
    )


@pytest.fixture
def df_with_cat_dtype():
    return pd.DataFrame(
        {
            "unique_id": [1, 2, 3],
            "ds": pd.date_range("2023-01-01", periods=3),
            "y": [1.0, 2.0, 3.0],
            "cat_col": pd.Categorical(["X", "Y", "Z"]),
        }
    )


@pytest.fixture
def df_leading_zeros():
    return pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "B", "B", "C", "C", "C", "C", "D", "D", "D"],
            "ds": pd.date_range("2025-01-01", periods=12),
            "y": [0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 0],
        }
    )


@pytest.fixture
def df_negative_values():
    return pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "B", "B", "C", "C", "C"],
            "ds": pd.date_range("2025-01-01", periods=8),
            "y": [0, -1, 2, -1, -2, 0, 1, 2],
        }
    )


@pytest.fixture(scope="module")
def ts_data_set1():
    h = 5
    freq = "D"
    series = generate_series(n_series=10, freq=freq, equal_ends=True)
    train_end = series["ds"].max() - h * pd.offsets.Day()
    train_mask = series["ds"] <= train_end
    train = series[train_mask]
    valid = series[~train_mask]

    return SimpleNamespace(
        h=h,
        series=series,
        train=train,
        train_end=train_end,
        valid=valid,
        freq=freq,
    )


@pytest.fixture(scope="module")
def ts_anomaly_data(ts_data_set1):
    train_anomalies = ts_data_set1.train.copy()
    anomaly_date = ts_data_set1.train_end - 2 * pd.offsets.Day()
    train_anomalies.loc[train_anomalies["ds"] == anomaly_date, "y"] *= 2

    return SimpleNamespace(
        train_anomalies=train_anomalies,
        anomaly_date=anomaly_date,
    )


@pytest.fixture
def common_kwargs():
    return {"freq": "D", "id_col": "unique_id", "time_col": "ds"}


@pytest.fixture
def df_ok():
    return pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2", "id2", "id2"],
            "ds": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
            ],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def df_with_duplicates_set2():
    return pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2"],
            "ds": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "y": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def df_with_missing_dates():
    return pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id2", "id2"],
            "ds": ["2023-01-01", "2023-01-03", "2023-01-01", "2023-01-03"],
            "y": [1, 3, 4, 6],
        }
    )


@pytest.fixture
def df_with_duplicates_and_missing_dates():
    # Global end on 2023-01-03 which is missing for id1
    return pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2", "id2"],
            "ds": [
                "2023-01-01",
                "2023-01-01",
                "2023-01-02",
                "2023-01-02",
                "2023-01-03",
            ],
            "y": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def df_with_cat_columns():
    return pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2", "id2", "id2"],
            "ds": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
            ],
            "y": [1, 2, 3, 4, 5, 6],
            "cat_col1": ["A", "B", "C", "D", "E", "F"],
            "cat_col2": pd.Categorical(["X", "Y", "Z", "X", "Y", "Z"]),
        }
    )


@pytest.fixture
def df_negative_vals():
    return pd.DataFrame(
        {
            "unique_id": ["id1", "id1", "id1", "id2", "id2", "id2"],
            "ds": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
            ],
            "y": [-1, 0, 1, 2, -3, -4],
        }
    )


@pytest.fixture
def df_leading_zeros_set2():
    return pd.DataFrame(
        {
            "unique_id": [
                "id1",
                "id1",
                "id1",
                "id2",
                "id2",
                "id2",
                "id3",
                "id3",
                "id3",
            ],
            "ds": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
            ],
            "y": [0, 1, 2, 0, 1, 2, 0, 0, 0],
        }
    )


@pytest.fixture
def cv_series_with_features():
    freq = "D"
    h = 5
    series = generate_series(2, freq=freq)
    series_with_features, _ = fourier(series, freq=freq, season_length=7, k=2)
    splits = ufp.backtest_splits(
        df=series_with_features,
        n_windows=1,
        h=h,
        id_col="unique_id",
        time_col="ds",
        freq=freq,
    )
    _, train, valid = next(splits)
    x_cols = train.columns.drop(["unique_id", "ds", "y"]).tolist()
    return series_with_features, train, valid, x_cols, h, freq


@pytest.fixture
def custom_business_hours():
    return pd.tseries.offsets.CustomBusinessHour(
        start="09:00",
        end="16:00",
        holidays=[
            "2022-12-25",  # Christmas
            "2022-01-01",  # New Year's Day
        ],
    )


@pytest.fixture
def business_hours_series(custom_business_hours):
    series = pd.DataFrame(
        {
            "unique_id": 1,
            "ds": pd.date_range(
                start="2000-01-03 09", freq=custom_business_hours, periods=200
            ),
            "y": np.arange(200) % 7,
        }
    )
    series = pd.concat([series.assign(unique_id=i) for i in range(10)]).reset_index(
        drop=True
    )
    return series


@pytest.fixture
def integer_freq_series():
    series = generate_series(5, freq="H", min_length=200)
    series["ds"] = series.groupby("unique_id", observed=True)["ds"].cumcount()
    return series


@pytest.fixture
def two_short_series():
    return generate_series(n_series=2, min_length=5, max_length=20)


@pytest.fixture
def two_short_series_with_time_features_train_future(two_short_series):
    train, future = time_features(
        two_short_series, freq="D", features=["year", "month"], h=5
    )
    return train, future


@pytest.fixture
def series_1MB_payload():
    series = generate_series(250, n_static_features=2)
    return series


@pytest.fixture(scope="module")
def air_passengers_df():
    return pd.read_csv(
        "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv",
        parse_dates=["timestamp"],
    )


@pytest.fixture(scope="module")
def air_passengers_renamed_df(air_passengers_df):
    df_copy = deepcopy(air_passengers_df)
    df_copy.rename(columns={"timestamp": "ds", "value": "y"}, inplace=True)
    df_copy.insert(0, "unique_id", "AirPassengers")
    return df_copy


@pytest.fixture(scope="module")
def air_passengers_with_nans(air_passengers_renamed_df):
    df = deepcopy(air_passengers_renamed_df)
    rng = np.random.default_rng(42)
    nan_idx = rng.choice(len(df), size=10, replace=False)
    df.loc[df.index[nan_idx], "y"] = np.nan
    return df


@pytest.fixture(scope="module")
def multi_series_with_nans(ts_data_set1):
    df = deepcopy(ts_data_set1.train)
    rng = np.random.default_rng(42)
    nan_idx = rng.choice(len(df), size=20, replace=False)
    df.loc[df.index[nan_idx], "y"] = np.nan
    return SimpleNamespace(
        df=df,
        h=ts_data_set1.h,
        freq=ts_data_set1.freq,
        n_ids=df["unique_id"].nunique(),
    )


@pytest.fixture(scope="module")
def air_passengers_renamed_df_with_index(air_passengers_renamed_df):
    df_copy = deepcopy(air_passengers_renamed_df)
    df_ds_index = df_copy.set_index("ds")[["unique_id", "y"]]
    df_ds_index.index = pd.DatetimeIndex(df_ds_index.index)
    return df_ds_index


@pytest.fixture
def df_freq_generator():
    def _df_freq(n_series, min_length, max_length, freq):
        df_freq = generate_series(
            n_series,
            min_length=min_length if freq != "15T" else 1_200,
            max_length=max_length if freq != "15T" else 2_000,
        )
        return df_freq

    return _df_freq


@pytest.fixture(scope="module")
def date_features_result(air_passengers_renamed_df):
    date_features = ["year", "month"]
    df_date_features, future_df = _maybe_add_date_features(
        df=air_passengers_renamed_df,
        X_df=None,
        h=12,
        freq="MS",
        features=date_features,
        one_hot=False,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    return df_date_features, future_df, date_features


HYPER_PARAMS_TEST = [
    # finetune steps is unstable due
    # to numerical reasons
    # dict(finetune_steps=2),
    dict(),
    dict(clean_ex_first=False),
    dict(date_features=["month"]),
    dict(level=[80, 90]),
    # dict(level=[80, 90], finetune_steps=2),
]


@pytest.fixture(scope="module")
def train_test_split(air_passengers_renamed_df):
    df_ = deepcopy(air_passengers_renamed_df)
    df_test = df_.groupby("unique_id").tail(12)
    df_train = df_.drop(df_test.index)
    return df_train, df_test


@pytest.fixture(scope="module")
def exog_data(air_passengers_renamed_df, train_test_split):
    df_ = deepcopy(air_passengers_renamed_df)
    df_ex_ = df_.copy()
    df_ex_["exogenous_var"] = df_ex_["y"] + np.random.normal(size=len(df_ex_))
    df_train, df_test = train_test_split
    x_df_test = df_test.drop(columns="y").merge(df_ex_.drop(columns="y"))
    return df_ex_, df_train, df_test, x_df_test


@pytest.fixture(scope="module")
def large_series():
    return generate_series(20_000, min_length=1_000, max_length=1_000, freq="min")


@pytest.fixture(scope="module")
def anomaly_online_df():
    detection_size = 5
    n_series = 2
    size = 100
    ds = pd.date_range(start="2023-01-01", periods=size, freq="W")
    x = np.arange(size)
    y = 10 * np.sin(0.1 * x) + 12
    y = np.tile(y, n_series)
    y[size - 5] = 30
    y[2 * size - 1] = 30
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(1, n_series + 1), size),
            "ds": np.tile(ds, n_series),
            "y": y,
        }
    )
    return df, n_series, detection_size


@pytest.fixture(scope="module")
def distributed_n_series():
    return 4


@pytest.fixture(scope="module")
def renamer():
    return {
        "unique_id": "id_col",
        "ds": "time_col",
        "y": "target_col",
    }


@pytest.fixture(scope="module")
def distributed_series(distributed_n_series):
    series = generate_series(distributed_n_series, min_length=100)
    series["unique_id"] = series["unique_id"].astype(str)
    return series


@pytest.fixture(scope="module")
def distributed_df_x():
    df_x = pd.read_csv(
        "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-with-ex-vars.csv",
        parse_dates=["ds"],
    ).rename(columns=str.lower)
    return df_x


@pytest.fixture(scope="module")
def distributed_future_ex_vars_df():
    future_ex_vars_df = pd.read_csv(
        "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-future-ex-vars.csv",
        parse_dates=["ds"],
    ).rename(columns=str.lower)
    return future_ex_vars_df
