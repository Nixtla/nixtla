import copy
import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
import utilsforecast.processing as ufp
from sklearn.linear_model import LinearRegression

import mlforecast.date_features as dtf
from mlforecast import MLForecast
from mlforecast.callbacks import SaveFeatures
from mlforecast.utils import generate_daily_series


def _pandas_reference(dates: pd.Series, name: str) -> np.ndarray:
    if name == "week_of_year":
        return dates.dt.isocalendar().week.to_numpy()
    attr = {"day_of_week": "dayofweek", "day_of_year": "dayofyear"}.get(name, name)
    return getattr(dates.dt, attr).to_numpy()


def test_available_lists_every_exported_feature():
    features = dtf.available()
    exported = set(dtf.__all__) - {"available"}
    assert [f.name for f in features] == [
        name for name in dtf.__all__ if name in exported
    ]
    for feature in features:
        assert getattr(dtf, feature.name) is feature


@pytest.mark.parametrize(
    "dates",
    [
        # spans 1900 and 2100 (not leap years), 2000 (leap) and 53-week ISO years
        pd.Series(pd.date_range("1899-12-01", "2101-02-01", freq="D")),
        pd.Series(
            pd.date_range("1999-12-01", "2031-01-01", freq="7h", tz="America/New_York")
        ),
        pd.Series(
            pd.date_range("1999-12-01", "2031-01-01", freq="7h", tz="Asia/Tokyo")
        ),
    ],
    ids=["naive", "tz-behind-utc", "tz-ahead-of-utc"],
)
@pytest.mark.parametrize(
    "container", ["pandas_series", "pandas_index", "polars", "polars_date"]
)
def test_compute_matches_pandas_attributes(dates, container):
    if container == "pandas_index":
        inp = pd.DatetimeIndex(dates)
    elif container == "polars":
        inp = pl.from_pandas(dates)
    elif container == "polars_date":
        if dates.dt.tz is not None or (dates.dt.hour != 0).any():
            pytest.skip("polars Date has no time or time zone")
        inp = pl.from_pandas(dates).cast(pl.Date)
    else:
        inp = dates
    for feature in dtf.available():
        if container == "polars_date" and feature in (dtf.hour, dtf.minute, dtf.second):
            continue
        vals = feature.compute(inp)
        assert vals.dtype == feature.dtype
        np.testing.assert_array_equal(
            vals, _pandas_reference(dates, feature.name), err_msg=feature.name
        )


def test_same_features_for_pandas_and_polars():
    series = generate_daily_series(3, min_length=400, max_length=500)
    pd_res = MLForecast(models=[], freq="D", date_features=dtf.available()).preprocess(
        series
    )
    pl_res = MLForecast(models=[], freq="1d", date_features=dtf.available()).preprocess(
        pl.from_pandas(series)
    )
    for feature in dtf.available():
        pd_vals = pd_res[feature.name].to_numpy()
        pl_vals = pl_res[feature.name].to_numpy()
        assert pd_vals.dtype == pl_vals.dtype == feature.dtype
        np.testing.assert_array_equal(pd_vals, pl_vals, err_msg=feature.name)
        np.testing.assert_array_equal(
            pd_vals, feature.compute(series["ds"]), err_msg=feature.name
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("same_end", [True, False])
def test_predict_features_match_future_dates(engine, same_end):
    series = generate_daily_series(4, min_length=50, max_length=80, equal_ends=same_end)
    freq = "D"
    if engine == "polars":
        series = pl.from_pandas(series)
        freq = "1d"
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=freq,
        lags=[1],
        date_features=dtf.available(),
    )
    fcst.fit(series)
    save_feats = SaveFeatures()
    h = 5
    preds = fcst.predict(h, before_predict_callback=save_feats)
    features = save_feats.get_features()
    # saved features are stacked by step, predictions by series
    step_major = np.arange(len(preds)).reshape(-1, h).T.ravel()
    future_dates = ufp.take_rows(preds, step_major)["ds"]
    for feature in dtf.available():
        np.testing.assert_array_equal(
            np.asarray(features[feature.name]),
            feature.compute(future_dates),
            err_msg=feature.name,
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_dummies(engine):
    series = generate_daily_series(2, min_length=40, max_length=40)
    freq = "D"
    if engine == "polars":
        series = pl.from_pandas(series)
        freq = "1d"
    fcst = MLForecast(
        models=[LinearRegression()],
        freq=freq,
        lags=[1],
        date_features=[dtf.day_of_week, dtf.days_in_month, dtf.year, dtf.is_month_end],
        date_features_as_dummies=True,
    )
    res = fcst.preprocess(series)
    dummy_cols = [f"day_of_week_{i}" for i in range(7)]
    days_cols = [f"days_in_month_{d}" for d in range(28, 32)]
    assert fcst.ts.features == [
        "lag1",
        *dummy_cols,
        *days_cols,
        "year",
        "is_month_end",
    ]
    np.testing.assert_array_equal(
        np.asarray(res[days_cols]).argmax(axis=1) + 28,
        dtf.days_in_month.compute(res["ds"]),
    )
    assert "day_of_week" not in res.columns
    dummies = np.asarray(res[dummy_cols])
    assert dummies.dtype == np.uint8
    np.testing.assert_array_equal(
        dummies.argmax(axis=1), dtf.day_of_week.compute(res["ds"])
    )
    np.testing.assert_array_equal(dummies.sum(axis=1), 1)
    fcst.fit(series)
    save_feats = SaveFeatures()
    fcst.predict(3, before_predict_callback=save_feats)
    assert list(save_feats.get_features().columns) == fcst.ts.features


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_null_dates_raise(engine):
    dates = pd.Series(pd.to_datetime(["2020-01-01", None, "2020-03-01"]))
    if engine == "polars":
        dates = pl.from_pandas(dates)
    with pytest.raises(ValueError, match="'month', found 1 null dates"):
        dtf.month.compute(dates)


def test_pickling_keeps_identity():
    assert pickle.loads(pickle.dumps(dtf.month)) is dtf.month
    assert copy.deepcopy(dtf.day_of_week) is dtf.day_of_week
    series = generate_daily_series(2, min_length=40, max_length=40)
    fcst = MLForecast(
        models=[LinearRegression()],
        freq="D",
        lags=[1],
        date_features=[dtf.day_of_week, dtf.month],
    ).fit(series)
    loaded = pickle.loads(pickle.dumps(fcst))
    assert loaded.ts.date_features == [dtf.day_of_week, dtf.month]
    pd.testing.assert_frame_equal(loaded.predict(3), fcst.predict(3))
