"""Runtime benchmarks for the pandas pipeline, run by the CodSpeed job.

Not named `test_*.py` so the ordinary pytest run skips it; CodSpeed passes the
path explicitly. Correctness tests belong in a `test_*.py` module.

    pytest tests/bench_pipeline.py --codspeed
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from utilsforecast.losses import smape

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMax, RollingMean, RollingMin
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.utils import generate_daily_series


class SeasonalNaive(BaseEstimator):
    def fit(self, X, y=None):  # noqa: ARG002
        return self

    def predict(self, X, y=None):  # noqa: ARG002
        return X["lag7"]


N_STATIC = 10


def _panel(min_length, max_length, n_series=1_000):
    return generate_daily_series(
        n_series=n_series,
        min_length=min_length,
        max_length=max_length,
        n_static_features=N_STATIC,
        static_as_categorical=False,
        equal_ends=True,
    )


def _with_exog(series):
    series = series.copy()
    n_exog = 10
    exog_names = [f"exog_{i}" for i in range(n_exog)]
    series[exog_names] = np.random.random((series.shape[0], n_exog))
    return series


@pytest.fixture(scope="module")
def series():
    return _panel(500, 2_000)


# the per-group work in `GroupedArray` is driven by the number of series, not the
# number of rows, so it needs a panel with many short ones
@pytest.fixture(scope="module")
def many_series():
    return _panel(30, 60, n_series=50_000)


@pytest.fixture(scope="module")
def series_with_exog(series):
    return _with_exog(series)


# `predict` is flat in history length but the `fit` it needs isn't, and that fit is
# untimed setup, so the predict benchmarks get a shorter panel.
@pytest.fixture(scope="module")
def short_series():
    return _panel(100, 400)


@pytest.fixture(scope="module")
def short_series_with_exog(short_series):
    return _with_exog(short_series)


@pytest.fixture
def fcst():
    return MLForecast(
        models={
            "lr": LinearRegression(),
            "seas_naive": SeasonalNaive(),
        },
        freq="D",
        lags=[1, 7, 14, 28],
        lag_transforms={
            1: [RollingMean(7)],
            7: [RollingMean(7), RollingMin(7), RollingMax(7)],
            14: [RollingMean(7), RollingMin(7), RollingMax(7)],
            28: [RollingMean(7), RollingMin(7), RollingMax(7)],
        },
        date_features=["dayofweek", "month", "year", "day"],
        target_transforms=[Differences([1, 7]), LocalStandardScaler()],
    )


# `fcst` needs 42 samples per serie (differences plus the lag 28 windows),
# which would drop most of `many_series`
@pytest.fixture
def short_history_fcst():
    return MLForecast(
        models={"lr": LinearRegression()},
        freq="D",
        lags=[1, 7],
        lag_transforms={1: [RollingMean(7)]},
        date_features=["dayofweek", "month", "year", "day"],
        target_transforms=[Differences([1]), LocalStandardScaler()],
    )


@pytest.fixture
def statics():
    return [f"static_{i}" for i in range(N_STATIC)]


@pytest.fixture
def exogs(series_with_exog, statics):
    return series_with_exog.columns.drop(["unique_id", "ds", "y"] + statics).tolist()


@pytest.mark.parametrize("use_exog", [True, False])
def test_preprocess(benchmark, fcst, series, use_exog, series_with_exog, statics):
    if use_exog:
        series = series_with_exog
    benchmark(fcst.preprocess, series, static_features=statics)


@pytest.mark.parametrize("max_horizon", [None, 14])
def test_preprocess_many_series(
    benchmark, short_history_fcst, many_series, statics, max_horizon
):
    benchmark(
        short_history_fcst.preprocess,
        many_series,
        static_features=statics,
        max_horizon=max_horizon,
    )


@pytest.mark.parametrize("use_exog", [True, False])
@pytest.mark.parametrize("keep_last_n", [None, 50])
def test_predict(
    benchmark,
    fcst,
    short_series,
    use_exog,
    short_series_with_exog,
    exogs,
    statics,
    keep_last_n,
):
    horizon = 14
    series = short_series_with_exog if use_exog else short_series
    valid = series.groupby("unique_id").tail(horizon)
    train = series.drop(valid.index)
    pred_kwargs = {}
    if use_exog:
        pred_kwargs["X_df"] = valid[["unique_id", "ds"] + exogs]
    fcst.fit(train, static_features=statics, keep_last_n=keep_last_n)
    preds = benchmark(fcst.predict, horizon, **pred_kwargs)
    full_preds = preds.merge(valid[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    models = fcst.models.keys()
    evaluation = smape(full_preds, models=models)
    summary = evaluation[models].mean(axis=0)
    assert summary["lr"] < summary["seas_naive"]
