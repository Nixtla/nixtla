import copy
import datetime
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import utilsforecast.processing as ufp

from mlforecast.callbacks import SaveFeatures
from mlforecast.core import (
    TimeSeries,
    _build_function_transform_name,
    _build_lag_transform_name,
    _name_models,
)
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.utils import generate_daily_series, generate_prices_for_series


@pytest.fixture
def series():
    series = generate_daily_series(20, n_static_features=2)
    return series


@pytest.fixture
def serie(series):
    uids = series["unique_id"].unique()
    serie = series[series["unique_id"].eq(uids[0])]
    return serie


def _expanding_mean(x):
    return pd.Series(x).expanding().mean().to_numpy()


def _rolling_mean(x, window_size):
    return pd.Series(x).rolling(window_size).mean().to_numpy()

def test_build_function_transform_name():
    assert _build_function_transform_name(_expanding_mean, 1) == "_expanding_mean_lag1"
    assert _build_function_transform_name(_rolling_mean, 2, 7) == "_rolling_mean_lag2_window_size7"
    assert _build_lag_transform_name(ExpandingMean(), 1) == "expanding_mean_lag1"
    assert _build_lag_transform_name(RollingMean(7), 2) == "rolling_mean_lag2_window_size7"

# one duplicate
def test_name_models_with_duplicates():
    names = ["a", "b", "a", "c"]
    expected = ["a", "b", "a2", "c"]
    actual = _name_models(names)
    assert actual == expected

# no duplicates
def test_name_models_without_duplicates():
    names = ["a", "b", "c"]
    actual = _name_models(names)
    assert actual == names
    with pytest.raises(ValueError) as exec:
        TimeSeries(freq="D", lags=list(range(2))),
    assert "lags must be positive integers" in str(exec.value)

    with pytest.raises(ValueError) as exec:
        TimeSeries(freq="D", lag_transforms={0: 1}),

    assert "keys of lag_transforms must be positive integers" in str(exec.value)

@pytest.fixture
def x():
    n = 7 * 14
    x = pd.DataFrame({
            "id": np.repeat(0, n),
            "ds": np.arange(n),
            "y": np.arange(7)[[x % 7 for x in np.arange(n)]],
        })
    x["y"] = x["ds"] * 0.1 + x["y"]
    return x

# differences
def test_target_transform_differences(x):
    ts = TimeSeries(freq=1, target_transforms=[Differences([1, 7])])
    ts._fit(x.iloc[:-14], id_col="id", time_col="ds", target_col="y")
    ts.as_numpy = False
    np.testing.assert_allclose(
        x["y"].diff(1).diff(7).values[:-14],
        ts.ga.data,
    )
    ts.y_pred = np.zeros(14)

    xx = ts.predict({"A": A()}, 14)
    np.testing.assert_allclose(xx["A"], x["y"].tail(14).values)



class A:
    def fit(self, X): # noqa
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])




# tfms namer
def namer(cls, lag, *args): # noqa
    return f"hello_from_{type(cls).__name__.lower()}"

@pytest.fixture
def ts():
    ts = TimeSeries(
        freq=1,
        lag_transforms={1: [RollingMean(7), ExpandingMean()]},
        lag_transforms_namer=namer,
    )
    return ts

def test_tfms_namer(x, ts):
    transformed = ts.fit_transform(x, id_col="id", time_col="ds", target_col="y")
    assert transformed.columns.tolist() == ["id", "ds", "y", "hello_from_rollingmean", "hello_from_expandingmean"]
    with pytest.raises(ValueError) as exec:
        TimeSeries(freq=1, date_features=[lambda: 1])

    assert "Can't use a lambda" in str(exec.value)


def month_start_or_end(dates):
    return dates.is_month_start | dates.is_month_end

@pytest.fixture
def flow_config():
    flow_config = dict(
        freq="W-THU",
        lags=[7],
        lag_transforms={1: [ExpandingMean(), RollingMean(7)]},
        date_features=["dayofweek", "week", month_start_or_end],
    )
    return flow_config

def test_ts_flow_config(flow_config):
    ts = TimeSeries(**flow_config)
    assert TimeSeries(freq=ts.freq).freq == TimeSeries(freq="W-THU").freq
    assert ts.freq == pd.tseries.frequencies.to_offset(flow_config["freq"])
    assert ts.date_features == flow_config["date_features"]
    assert list(ts.transforms.keys()) == ["lag7", "expanding_mean_lag1", "rolling_mean_lag1_window_size7"]

# int y is converted to float32
def test_y_fp32(serie):
    serie2 = serie.copy()
    serie2["y"] = serie2["y"].astype(int)
    ts = TimeSeries(num_threads=1, freq="D")
    ts._fit(serie2, id_col="unique_id", time_col="ds", target_col="y")
    assert ts.ga.data.dtype == np.float32


# _compute_transforms
def _shift_array(x, n):
    return np.hstack([np.full(n, np.nan), x[:-n]])

def test_compute_transforms(flow_config, serie):
    y = serie.y.values
    lag_1 = _shift_array(y, 1)

    for num_threads in (1, 2):
        ts = TimeSeries(**flow_config)
        ts._fit(serie, id_col="unique_id", time_col="ds", target_col="y")
        transforms = ts._compute_transforms(ts.transforms, updates_only=False)

        np.testing.assert_allclose(transforms["lag7"], _shift_array(y, 7))
        np.testing.assert_allclose(
            transforms["expanding_mean_lag1"], _expanding_mean(lag_1)
        )
        np.testing.assert_allclose(
            transforms["rolling_mean_lag1_window_size7"], _rolling_mean(lag_1, 7)
        )

# update_y
def test_update_y(serie):
    ts = TimeSeries(freq="D", lags=[1])
    ts._fit(serie, id_col="unique_id", time_col="ds", target_col="y")

    max_size = np.diff(ts.ga.indptr)
    ts._update_y([1])
    ts._update_y([2])

    assert np.diff(ts.ga.indptr) == max_size + 2
    assert ts.ga.data[-2:].tolist() == [1, 2]

# _update_features
def test_update_features(flow_config, serie):
    ts = TimeSeries(**flow_config)
    ts.fit_transform(serie, id_col="unique_id", time_col="ds", target_col="y")
    ts._predict_setup()
    updates = ts._update_features()

    last_date = serie["ds"].max()
    first_prediction_date = last_date + pd.offsets.Day()

    y = serie.y.values


    # these have an offset becase we can now "see" our last y value
    expected = pd.DataFrame(
        {
            "unique_id": ts.uids,
            "lag7": _shift_array(y, 6)[-1],
            "expanding_mean_lag1": _expanding_mean(y)[-1],
            "rolling_mean_lag1_window_size7": _rolling_mean(y, 7)[-1],
            "dayofweek": np.uint8([getattr(first_prediction_date, "dayofweek")]),
            "week": np.uint8([first_prediction_date.isocalendar()[1]]),
            "month_start_or_end": month_start_or_end(first_prediction_date),
        }
    )
    statics = serie.tail(1).drop(columns=["ds", "y"])
    pd.testing.assert_frame_equal(updates, statics.merge(expected))


    assert ts.curr_dates[0] == first_prediction_date

# _get_predictions
def test_get_predictions(serie):
    ts = TimeSeries(freq="D", lags=[1])
    ts._fit(serie, id_col="unique_id", time_col="ds", target_col="y")
    ts._predict_setup()
    ts._update_features()
    ts._update_y([1.0])
    preds = ts._get_predictions()

    last_ds = serie["ds"].max()
    expected = pd.DataFrame(
        {
            "unique_id": serie["unique_id"][[0]],
            "ds": [last_ds + pd.offsets.Day()],
            "y_pred": [1.0],
        }
    )
    pd.testing.assert_frame_equal(preds, expected)

@pytest.fixture
def flow_config2():
    flow_config = dict(
        freq="D",
        lags=[7, 14],
        lag_transforms={
            2: [
                RollingMean(7),
                RollingMean(14),
            ]
        },
        date_features=["dayofweek", "month", "year"],
        num_threads=2,
    )
    return flow_config

def test_ts_fit_transform(flow_config2, series):
    ts = TimeSeries(**flow_config2)
    _ = ts.fit_transform(series, id_col="unique_id", time_col="ds", target_col="y")
    np.testing.assert_equal(
        ts.ga.data,
        series.groupby("unique_id", observed=True).tail(ts.keep_last_n)["y"],
    )
    assert ts.uids.tolist() == series["unique_id"].unique().tolist()
    np.testing.assert_array_equal(ts.last_dates, series.groupby("unique_id", observed=True)["ds"].max().values)
    pd.testing.assert_frame_equal(
        ts.static_features_,
        series.groupby("unique_id", observed=True)
        .tail(1)
        .drop(columns=["ds", "y"])
        .reset_index(drop=True),
    )
    ts.fit_transform(
        series,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=["static_0"],
    )

    pd.testing.assert_frame_equal(
        ts.static_features_,
        series.groupby("unique_id", observed=True)
        .tail(1)[["unique_id", "static_0"]]
        .reset_index(drop=True),
    )

def test_keep_last_n(flow_config2, series):
    keep_last_n = 15

    ts = TimeSeries(**flow_config2)
    df = ts.fit_transform(
        series, id_col="unique_id", time_col="ds", target_col="y", keep_last_n=keep_last_n
    )
    ts._predict_setup()

    expected_lags = ["lag7", "lag14"]
    expected_transforms = [
        "rolling_mean_lag2_window_size7",
        "rolling_mean_lag2_window_size14",
    ]
    expected_date_features = ["dayofweek", "month", "year"]

    assert ts.features == expected_lags + expected_transforms + expected_date_features
    assert ts.static_features_.columns.tolist() + ts.features == df.columns.drop(["ds", "y"]).tolist()
    # we dropped 2 rows because of the lag 2 and 13 more to have the window of size 14
    assert df.shape[0] == series.shape[0] - (2 + 13) * ts.ga.n_groups
    assert ts.ga.data.size == ts.ga.n_groups * keep_last_n


    series_with_nulls = series.copy()
    series_with_nulls.loc[1, "y"] = np.nan
    with pytest.raises(Exception) as exec:
        ts.fit_transform(series_with_nulls, id_col="unique_id", time_col="ds", target_col="y")
    assert "y column contains null values" in str(exec.value)

# unsorted df
def test_unsorted_df(flow_config2, series):
    ts = TimeSeries(**flow_config2)
    df = ts.fit_transform(series, id_col="unique_id", time_col="ds", target_col="y")
    unordered_series = series.sample(frac=1.0)
    assert not unordered_series.set_index("ds", append=True).index.is_monotonic_increasing
    df2 = ts.fit_transform(
        unordered_series, id_col="unique_id", time_col="ds", target_col="y"
    )
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        df2.sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )

# existing features arent recomputed
def test_existing_features():
    df_with_features = pd.DataFrame(
        {
            "unique_id": [1, 1, 1],
            "ds": pd.date_range("2000-01-01", freq="D", periods=3),
            "y": [10.0, 11.0, 12.0],
            "lag1": [1, 1, 1],
            "month": [12, 12, 12],
        }
    )
    ts = TimeSeries(freq="D", lags=[1, 2], date_features=["year", "month"])
    transformed = ts.fit_transform(
        df_with_features, id_col="unique_id", time_col="ds", target_col="y", dropna=False
    )
    pd.testing.assert_series_equal(transformed["lag1"], df_with_features["lag1"])
    pd.testing.assert_series_equal(transformed["month"], df_with_features["month"])
    np.testing.assert_array_equal(transformed["year"], 3 * [2000])
    np.testing.assert_array_equal(transformed["lag2"].values, [np.nan, np.nan, 10.0])


# non-standard df
def test_non_standard_df(flow_config2, series):
    ts = TimeSeries(**flow_config2)
    df = ts.fit_transform(series, id_col="unique_id", time_col="ds", target_col="y")
    non_std_series = series.reset_index().rename(
        columns={"unique_id": "some_id", "ds": "timestamp", "y": "value"}
    )
    non_std_res = ts.fit_transform(
        non_std_series,
        id_col="some_id",
        time_col="timestamp",
        target_col="value",
        static_features=[],
    )
    non_std_res = non_std_res.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df.reset_index(),
        non_std_res.rename(
            columns={"timestamp": "ds", "value": "y", "some_id": "unique_id"}
        ),
    )


# integer timestamps
def identity(x):
    return x

def test_integer_timestamps(flow_config2, series):
    flow_config_int_ds = copy.deepcopy(flow_config2)
    flow_config_int_ds["date_features"] = [identity]
    flow_config_int_ds["freq"] = 1
    ts = TimeSeries(**flow_config_int_ds)
    int_ds_series = series.copy()
    int_ds_series["ds"] = int_ds_series["ds"].astype("int64")
    int_ds_res = ts.fit_transform(
        int_ds_series, id_col="unique_id", time_col="ds", target_col="y"
    )
    int_ds_res["ds"] = pd.to_datetime(int_ds_res["ds"])
    int_ds_res["identity"] = pd.to_datetime(int_ds_res["ds"])

    df = TimeSeries(**flow_config2).fit_transform(series, id_col="unique_id", time_col="ds", target_col="y")

    df2 = df.drop(columns=flow_config2["date_features"])
    df2["identity"] = df2["ds"]
    pd.testing.assert_frame_equal(df2, int_ds_res)



class DummyModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag7"].values

def test_ts_predict(flow_config2, series):
    horizon = 7
    model = DummyModel()
    ts = TimeSeries(**flow_config2)
    ts.fit_transform(series, id_col="unique_id", time_col="ds", target_col="y")
    predictions = ts.predict({"DummyModel": model}, horizon)

    grouped_series = series.groupby("unique_id", observed=True)
    expected_preds = grouped_series["y"].tail(7)  # the model predicts the lag-7
    last_dates = grouped_series["ds"].max()
    expected_dsmin = last_dates + pd.offsets.Day()
    expected_dsmax = last_dates + horizon * pd.offsets.Day()
    grouped_preds = predictions.groupby("unique_id", observed=True)

    np.testing.assert_allclose(predictions["DummyModel"], expected_preds)
    pd.testing.assert_series_equal(grouped_preds["ds"].min(), expected_dsmin)
    pd.testing.assert_series_equal(grouped_preds["ds"].max(), expected_dsmax)

def test_ts_with_diff_conf(flow_config2, series):
    horizon = 7

    flow_config_int_ds = copy.deepcopy(flow_config2)
    flow_config_int_ds["date_features"] = [identity]
    flow_config_int_ds["freq"] = 1
    model = DummyModel()
    ts = TimeSeries(**flow_config2)
    ts.fit_transform(series, id_col="unique_id", time_col="ds", target_col="y")
    predictions = ts.predict({"DummyModel": model}, horizon=horizon)
    ts = TimeSeries(**flow_config_int_ds)

    int_ds_series = series.copy()
    int_ds_series['ds'] = int_ds_series['ds'].astype('int64')

    ts.fit_transform(int_ds_series, id_col="unique_id", time_col="ds", target_col="y")
    int_ds_predictions = ts.predict({"DummyModel": model}, horizon=horizon)
    pd.testing.assert_frame_equal(
        predictions.drop(columns="ds"), int_ds_predictions.drop(columns="ds")
    )


class PredictPrice:
    def predict(self, X):
        return X["price"]

def test_dynamic_features(flow_config2):
    series = generate_daily_series(20, n_static_features=2, equal_ends=True)
    dynamic_series = series.rename(columns={"static_1": "product_id"})
    prices_catalog = generate_prices_for_series(dynamic_series)
    series_with_prices = dynamic_series.merge(prices_catalog, how="left")

    model = PredictPrice()
    ts = TimeSeries(**flow_config2)
    ts.fit_transform(
        series_with_prices,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=["static_0", "product_id"],
    )
    predictions = ts.predict({"PredictPrice": model}, horizon=1, X_df=prices_catalog)
    pd.testing.assert_frame_equal(
        predictions.rename(columns={"PredictPrice": "price"}),
        prices_catalog.merge(predictions[["unique_id", "ds"]])[
            ["unique_id", "ds", "price"]
        ],
    )

    # predicting for a subset
    sample_ids = ["id_00", "id_16"]
    sample_preds = ts.predict({"price": model}, 1, X_df=prices_catalog, ids=sample_ids)
    pd.testing.assert_frame_equal(
        sample_preds,
        prices_catalog.merge(
            predictions[predictions["unique_id"].isin(sample_ids)][["unique_id", "ds"]]
        )[["unique_id", "ds", "price"]],
    )
    with pytest.raises(Exception) as exec:
        ts.predict({"y": model}, 1, ids=["bonjour"])
    assert "{'bonjour'}" in str(exec.value)



class SeasonalNaiveModel:
    def predict(self, X):
        return X["lag7"]


class NaiveModel:
    def predict(self, X: pd.DataFrame):
        return X["lag1"]

def test_ts_update(series):
    two_series = series[series["unique_id"].isin(["id_00", "id_19"])].copy()
    two_series["unique_id"] = pd.Categorical(two_series["unique_id"], ["id_00", "id_19"])
    ts = TimeSeries(freq="D", lags=[1], date_features=["dayofweek"])
    ts.fit_transform(
        two_series,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    last_vals_two_series = two_series.groupby("unique_id", observed=True).tail(1)
    last_val_id0 = last_vals_two_series[lambda x: x["unique_id"].eq("id_00")].copy()
    new_values = last_val_id0.copy()
    new_values["ds"] += pd.offsets.Day()
    new_serie = pd.DataFrame(
        {
            "unique_id": ["new_idx", "new_idx"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "y": [5.0, 6.0],
            "static_0": [0, 0],
            "static_1": [1, 1],
        }
    )
    new_values = pd.concat([new_values, new_serie])
    ts.update(new_values)
    preds = ts.predict({"Naive": NaiveModel()}, 1)
    expected_id0 = last_val_id0.copy()
    expected_id0["ds"] += pd.offsets.Day()
    expected_id1 = last_vals_two_series[lambda x: x["unique_id"].eq("id_19")].copy()
    last_val_new_serie = new_serie.tail(1)[["unique_id", "ds", "y"]]
    expected = pd.concat([expected_id0, expected_id1, last_val_new_serie])
    expected = expected[["unique_id", "ds", "y"]]
    expected = expected.rename(columns={"y": "Naive"}).reset_index(drop=True)
    expected["unique_id"] = pd.Categorical(
        expected["unique_id"], categories=["id_00", "id_19", "new_idx"]
    )
    expected["ds"] += pd.offsets.Day()
    pd.testing.assert_frame_equal(preds, expected)
    pd.testing.assert_frame_equal(
        ts.static_features_,
        (
            pd.concat([last_vals_two_series, new_serie.tail(1)])[
                ["unique_id", "static_0", "static_1"]
            ]
            .astype(ts.static_features_.dtypes)
            .reset_index(drop=True)
        ),
    )
    # with target transforms
    ts = TimeSeries(
        freq="D",
        lags=[7],
        target_transforms=[Differences([1, 2]), LocalStandardScaler()],
    )
    ts.fit_transform(two_series, id_col="unique_id", time_col="ds", target_col="y")
    new_values = two_series.groupby("unique_id", observed=True).tail(7).copy()
    new_values["ds"] += 7 * pd.offsets.Day()
    orig_last7 = ts.ga.take_from_groups(slice(-7, None)).data
    ts.update(new_values)
    preds = ts.predict({"SeasonalNaive": SeasonalNaiveModel()}, 7)
    np.testing.assert_allclose(
        new_values["y"].values,
        preds["SeasonalNaive"].values,
    )
    last7 = ts.ga.take_from_groups(slice(-7, None)).data
    assert 0 < np.abs(last7 / orig_last7 - 1).mean() < 0.5

def test_ts_polars():
    two_series = generate_daily_series(2, n_static_features=2, engine="polars")
    ts = TimeSeries(freq="1d", lags=[1], date_features=["weekday"])
    ts.fit_transform(
        two_series,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    last_vals_two_series = two_series.join(
        two_series.group_by("unique_id").agg(pl.col("ds").max()), on=["unique_id", "ds"]
    )
    last_val_id0 = last_vals_two_series.filter(pl.col("unique_id") == "id_0")
    new_values = last_val_id0.with_columns(
        pl.col("unique_id").cast(pl.Categorical),
        pl.col("ds").dt.offset_by("1d"),
        pl.col("static_0").cast(pl.Int64),
        pl.col("static_1").cast(pl.Int64),
    )
    new_serie = pl.DataFrame(
        {
            "unique_id": ["new_idx", "new_idx"],
            "ds": [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)],
            "y": [5.0, 6.0],
            "static_0": [0, 0],
            "static_1": [1, 1],
        }
    ).with_columns(
        pl.col("ds").dt.cast_time_unit("ns"),
        pl.col("unique_id").cast(pl.Categorical),
    )
    new_values = pl.concat([new_values, new_serie])
    ts.update(new_values)
    preds = ts.predict({"Naive": NaiveModel()}, 1)
    expected_id0 = last_val_id0.with_columns(pl.col("ds").dt.offset_by("1d"))
    expected_id1 = last_vals_two_series.filter(pl.col("unique_id") == "id_1")
    last_val_new_serie = new_serie.tail(1)
    expected = pl.concat([expected_id0, expected_id1])
    expected = ufp.vertical_concat([expected, last_val_new_serie])
    pd.testing.assert_series_equal(
        expected["unique_id"].cat.get_categories().to_pandas(),
        pd.Series(["id_0", "id_1", "new_idx"], name="unique_id"),
    )
    expected = expected[["unique_id", "ds", "y"]]
    expected = ufp.rename(expected, {"y": "Naive"})
    expected = expected.with_columns(pl.col("ds").dt.offset_by("1d"))
    pd.testing.assert_frame_equal(preds.to_pandas(), expected.to_pandas())
    pd.testing.assert_frame_equal(
        ts.static_features_.to_pandas(),
        (
            ufp.vertical_concat([last_vals_two_series, new_serie.tail(1)])[
                ["unique_id", "static_0", "static_1"]
            ]
            .to_pandas()
            .astype(ts.static_features_.to_pandas().dtypes)
            .reset_index(drop=True)
        ),
    )
    # with target transforms
    ts = TimeSeries(
        freq="1d",
        lags=[7],
        target_transforms=[Differences([1, 2]), LocalStandardScaler()],
    )
    ts.fit_transform(two_series, id_col="unique_id", time_col="ds", target_col="y")
    new_values = two_series.group_by("unique_id").tail(7)
    new_values = new_values.with_columns(pl.col("ds").dt.offset_by("7d"))
    orig_last7 = ts.ga.take_from_groups(slice(-7, None)).data
    ts.update(new_values)
    preds = ts.predict({"SeasonalNaive": SeasonalNaiveModel()}, 7)
    np.testing.assert_allclose(
        new_values["y"].to_numpy(),
        preds["SeasonalNaive"].to_numpy(),
    )
    last7 = ts.ga.take_from_groups(slice(-7, None)).data
    assert 0 < np.abs(last7 / orig_last7 - 1).mean() < 0.5


# target_transform with keep_last_n
def test_target_transform_with_keep_last_n(series):
    ts = TimeSeries(freq="D", lags=[1], target_transforms=[LocalStandardScaler()])
    ts.fit_transform(
        series, id_col="unique_id", time_col="ds", target_col="y", keep_last_n=10
    )
    preds = ts.predict({"y": NaiveModel()}, 1)
    expected = (
        series.groupby("unique_id", observed=True)
        .tail(1)[["unique_id", "ds", "y"]]
        .reset_index(drop=True)
    )
    expected["ds"] += pd.offsets.Day()
    pd.testing.assert_frame_equal(preds, expected)

# raise error when omitting the static_features argument and passing them as dynamic in predict
def test_omit_static_features(series):
    valid = series.groupby("unique_id", observed=True).tail(10)
    train = series.drop(valid.index)
    ts = TimeSeries(freq="D", lags=[1], target_transforms=[LocalStandardScaler()])
    ts.fit_transform(
        train, id_col="unique_id", time_col="ds", target_col="y", keep_last_n=10
    )
    with pytest.raises(Exception) as exec:
        ts.predict({"y": NaiveModel()}, 1, X_df=valid.drop(columns=["y"]))

    assert "['static_0', 'static_1']" in str(exec.value)


def test_pd_vs_pl():
    series_pl = generate_daily_series(
        5, static_as_categorical=False, n_static_features=5, engine="polars"
    )
    series_pd = generate_daily_series(
        5, static_as_categorical=False, n_static_features=5, engine="pandas"
    )
    series_pl = series_pl.with_columns(pl.col("unique_id").cast(str))
    series_pd["unique_id"] = series_pd["unique_id"].astype(str)

    cfg = dict(
        lags=[1, 2, 3, 4],
        lag_transforms={
            1: [ExpandingMean(), RollingMean(7), RollingMean(14)],
            2: [ExpandingMean(), RollingMean(7), RollingMean(14)],
            3: [ExpandingMean(), RollingMean(7), RollingMean(14)],
            4: [ExpandingMean(), RollingMean(7), RollingMean(14)],
        },
        date_features=["day", "month", "quarter", "year"],
        target_transforms=[Differences([1])],
    )
    feats_pl = SaveFeatures()
    ts_pl = TimeSeries(freq="1d", **cfg)
    prep_pl = ts_pl.fit_transform(series_pl, "unique_id", "ds", "y")
    fcst_pl = ts_pl.predict({"y": NaiveModel()}, 2, before_predict_callback=feats_pl)

    feats_pd = SaveFeatures()
    ts_pd = TimeSeries(freq="1D", **cfg)
    prep_pd = ts_pd.fit_transform(series_pd, "unique_id", "ds", "y")
    fcst_pd = ts_pd.predict({"y": NaiveModel()}, 2, before_predict_callback=feats_pd)

    prep_pd = prep_pd.reset_index(drop=True)
    prep_pl = prep_pl.to_pandas()
    fcst_pl = fcst_pl.to_pandas()
    # date features have different dtypes
    pd.testing.assert_frame_equal(prep_pl, prep_pd, check_dtype=False)
    pd.testing.assert_frame_equal(
        feats_pl.get_features(with_step=True).to_pandas(),
        feats_pd.get_features(with_step=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(fcst_pl, fcst_pd)

# dropped series
@pytest.mark.parametrize("ordered", "[True, False]")
def test_dropped_series(ordered):
    series = generate_daily_series(10, min_length=5, max_length=20)
    if not ordered:
        series = series.sample(frac=1.0, random_state=40)
    ts = TimeSeries(freq="D", lags=[10])
    with warnings.catch_warnings(record=True):
        prep = ts.fit_transform(series, "unique_id", "ds", "y")
    dropped = ts.uids[ts._dropped_series].tolist()
    assert not prep["unique_id"].isin(dropped).any()
    assert set(prep["unique_id"].unique().tolist() + dropped) == set(
        series["unique_id"].unique()
    )

# short series exception
def test_short_series_exception():
    series = generate_daily_series(2, min_length=5, max_length=15)
    ts = TimeSeries(freq="D", lags=[1], target_transforms=[Differences([20])])
    with pytest.raises(Exception) as exec:
        ts.fit_transform(series, "unique_id", "ds", "y"),
    assert "are too short for the 'Differences' transformation" in str(exec.value)



# test predict
class Lag1PlusOneModel:
    def predict(self, X):
        return X["lag1"] + 1

def test_lag_predict(series):
    valid = series.groupby('unique_id', observed=True).tail(10)
    train = series.drop(valid.index)

    ts = TimeSeries(freq="D", lags=[1])
    for max_horizon in [None, 2]:
        if max_horizon is None:
            mod1 = Lag1PlusOneModel()
            mod2 = Lag1PlusOneModel()
        else:
            mod1 = [Lag1PlusOneModel() for _ in range(max_horizon)]
            mod2 = [Lag1PlusOneModel() for _ in range(max_horizon)]
        ts.fit_transform(train, "unique_id", "ds", "y", max_horizon=max_horizon)
        # each model gets the correct historic values
        preds = ts.predict(models={"mod1": mod1, "mod2": mod2}, horizon=2)
        np.testing.assert_allclose(preds["mod1"], preds["mod2"])
        # idempotency
        preds2 = ts.predict(models={"mod1": mod1, "mod2": mod2}, horizon=2)
        np.testing.assert_allclose(preds2["mod1"], preds2["mod2"])
        pd.testing.assert_frame_equal(preds, preds2)

# save & load
def test_save_and_load():
    series = generate_daily_series(2, n_static_features=2)
    ts = TimeSeries(
        freq="D",
        lags=[1, 2],
        date_features=["dayofweek"],
        lag_transforms={1: [RollingMean(1)]},
        target_transforms=[Differences([20])],
    )
    ts.fit_transform(series, "unique_id", "ds", "y")
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = Path(tmpdir) / "hi"
        ts.save(fname)
        ts2 = TimeSeries.load(fname)
    preds = ts.predict({"model": NaiveModel()}, 10)
    preds2 = ts2.predict({"model": NaiveModel()}, 10)
    pd.testing.assert_frame_equal(preds, preds2)

# automatically set keep_last_n for built-in lag transforms
def test_keep_last_n_for_built_in_lag_transforms(series):
    ts = TimeSeries(
        freq="D",
        lags=[1, 2],
        date_features=["dayofweek"],
        lag_transforms={
            1: [RollingMean(1), RollingMean(4)],
        },
    )
    ts.fit_transform(series, "unique_id", "ds", "y", keep_last_n=20)
    assert ts.keep_last_n == 20
    ts.fit_transform(series, "unique_id", "ds", "y")
    assert ts.keep_last_n == 4
    # we can't infer it for functions
    ts = TimeSeries(
        freq="D",
        lags=[1, 2],
        date_features=["dayofweek"],
        lag_transforms={
            1: [RollingMean(1), RollingMean(4)],
            5: [ExpandingMean()],
        },
    )
    ts.fit_transform(series, "unique_id", "ds", "y", keep_last_n=20)
    assert ts.keep_last_n == 20
    ts.fit_transform(series, "unique_id", "ds", "y")
    assert ts.keep_last_n == 4

# no target nulls when dropna=False
def test_target_nulls(series):
    ts = TimeSeries(
        freq="D",
        lags=[1, 2],
        target_transforms=[Differences([5])],
    )
    prep = ts.fit_transform(series, "unique_id", "ds", "y", dropna=False)
    assert not prep["y"].isnull().any()
