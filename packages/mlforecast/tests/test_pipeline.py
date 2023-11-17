import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from utilsforecast.losses import smape

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingMax, RollingMin
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.utils import generate_daily_series


class SeasonalNaive(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return X["lag7"]


@pytest.fixture(scope="module")
def series():
    n_series = 2_000
    n_static = 10
    return generate_daily_series(
        n_series=n_series,
        min_length=500,
        max_length=2_000,
        n_static_features=n_static,
        static_as_categorical=False,
        equal_ends=True,
    )


@pytest.fixture(scope="module")
def series_with_exog(series):
    series = series.copy()
    n_exog = 10
    exog_names = [f"exog_{i}" for i in range(n_exog)]
    series[exog_names] = np.random.random((series.shape[0], n_exog))
    return series


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
            1 : [RollingMean(7)],
            7 : [RollingMean(7), RollingMin(7), RollingMax(7)],
            14: [RollingMean(7), RollingMin(7), RollingMax(7)],
            28: [RollingMean(7), RollingMin(7), RollingMax(7)],
        },
        date_features=["dayofweek", "month", "year", "day"],
        target_transforms=[Differences([1, 7]), LocalStandardScaler()],
    )


@pytest.fixture
def statics(series):
    return series.columns.drop(["unique_id", "ds", "y"]).tolist()


@pytest.fixture
def exogs(series_with_exog, statics):
    return series_with_exog.columns.drop(["unique_id", "ds", "y"] + statics).tolist()


@pytest.mark.parametrize("use_exog", [True, False])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_preprocess(benchmark, fcst: MLForecast, series, use_exog, series_with_exog, statics, num_threads):
    if use_exog:
        series = series_with_exog
    fcst.ts.num_threads = num_threads
    benchmark(fcst.preprocess, series, static_features=statics)


@pytest.mark.parametrize("use_exog", [True, False])
@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("keep_last_n", [None, 50])
def test_predict(benchmark, fcst: MLForecast, series, use_exog, series_with_exog, exogs, statics, keep_last_n, num_threads):
    horizon = 14
    if use_exog:
        series = series_with_exog
    valid = series.groupby("unique_id").tail(horizon)
    train = series.drop(valid.index)
    pred_kwargs = {}
    if use_exog:
        pred_kwargs["X_df"] = valid[["unique_id", "ds"] + exogs]
    fcst.ts.num_threads = num_threads
    fcst.fit(train, static_features=statics, keep_last_n=keep_last_n)
    preds = benchmark(fcst.predict, horizon, **pred_kwargs)
    full_preds = preds.merge(valid[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    models = fcst.models.keys()
    evaluation = smape(full_preds, models=models)
    summary = evaluation[models].mean(axis=0)
    assert summary["lr"] < summary["seas_naive"]
