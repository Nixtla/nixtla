import sys
import warnings

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.dask.lgb import DaskLGBMForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.utils import generate_daily_series

warnings.simplefilter("ignore", FutureWarning)


def _reset_index_partition(partition: pd.DataFrame) -> pd.DataFrame:
    return partition.reset_index()


def _make_partitioned_series(df: pd.DataFrame, npartitions: int = 4) -> dd.DataFrame:
    partitioned = dd.from_pandas(df.set_index("unique_id"), npartitions=npartitions)
    partitioned = partitioned.map_partitions(_reset_index_partition)
    partitioned["unique_id"] = partitioned["unique_id"].astype(str)
    return partitioned


@pytest.fixture(scope="module")
def partitioned_series():
    series = generate_daily_series(
        100, equal_ends=True, min_length=500, max_length=1_000
    )
    return _make_partitioned_series(series)


@pytest.fixture
def small_ordered_series():
    series = generate_daily_series(5, min_length=60, max_length=60)
    return series.sort_values(["unique_id", "ds"]).reset_index(drop=True)


class _RecordingLocalModel:
    def __init__(self, sample_weight):
        if sample_weight is None:
            self.sample_weight_ = None
            self.weight_mean_ = 0.0
        else:
            self.sample_weight_ = np.asarray(sample_weight, dtype=float)
            self.weight_mean_ = float(self.sample_weight_.mean())

    def predict(self, X):
        length = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(length, self.weight_mean_, dtype=float)


class _RecordingDaskRegressor(BaseEstimator):
    def fit(self, X, y, sample_weight=None):  # noqa: ARG002, D401, N803
        if sample_weight is None:
            weights = None
        else:
            if hasattr(sample_weight, "compute"):
                sample_weight = sample_weight.compute()
            weights = (
                sample_weight.to_numpy()
                if hasattr(sample_weight, "to_numpy")
                else np.asarray(sample_weight, dtype=float)
            )
        self.model_ = _RecordingLocalModel(weights)
        return self

@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_dask_distributed_forecast(partitioned_series):
    # test existing features provide the same result
    fcst = DistributedMLForecast(
        models=[DaskLGBMForecast(verbosity=-1, random_state=0)],
        freq="D",
        lags=[1, 2, 3, 4, 5, 6, 7],
        lag_transforms={
            1: [RollingMean(7), RollingMean(30), ExpandingMean()],
            335: [RollingMean(30)],
        },
        date_features=["dayofweek"],
    )
    # df with features
    training_df_featured = fcst.preprocess(
        partitioned_series, static_features=[], dropna=False
    )
    fcst.fit(training_df_featured, static_features=[], dropna=False)
    preds1 = fcst.predict(10).compute()

    # df without features
    fcst.preprocess(partitioned_series, static_features=[], dropna=False)
    preds2 = fcst.predict(10).compute()
    pd.testing.assert_frame_equal(preds1, preds2)


@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_dask_distributed_weight_col_affects_predictions(small_ordered_series):
    def _fit_and_forecast(weights):
        weighted = small_ordered_series.copy()
        weighted["weight"] = weights
        partitioned = _make_partitioned_series(weighted, npartitions=2)
        fcst = DistributedMLForecast(
            models={"stub": _RecordingDaskRegressor()},
            freq="D",
            lags=[1],
            date_features=["dayofweek"],
        )
        fcst.fit(
            partitioned,
            static_features=[],
            dropna=False,
            weight_col="weight",
        )
        return fcst.predict(5).compute()

    uniform_weights = np.ones(len(small_ordered_series))
    skewed_weights = np.arange(len(small_ordered_series), dtype=float)

    preds_uniform = _fit_and_forecast(uniform_weights)
    preds_skewed = _fit_and_forecast(skewed_weights)

    assert not np.allclose(preds_uniform["stub"], preds_skewed["stub"])


@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_dask_distributed_forecast_with_x_df():
    """predict() with X_df as a Dask DataFrame must give the same result as pandas X_df."""
    h = 7
    series = generate_daily_series(20, equal_ends=True, min_length=50, max_length=100)
    # add a simple exogenous feature
    rng = np.random.default_rng(42)
    series["price"] = rng.random(len(series))
    partitioned = _make_partitioned_series(series, npartitions=4)

    fcst = DistributedMLForecast(
        models=[DaskLGBMForecast(verbosity=-1, random_state=0)],
        freq="D",
        lags=[1, 2, 7],
        date_features=["dayofweek"],
    )
    fcst.fit(partitioned, static_features=[])

    # build future exog: one row per (unique_id, future date)
    last_dates = (
        series.groupby("unique_id")["ds"].max().reset_index().rename(columns={"ds": "last_ds"})
    )
    future_rows = []
    for _, row in last_dates.iterrows():
        for step in range(1, h + 1):
            future_rows.append(
                {
                    "unique_id": row["unique_id"],
                    "ds": row["last_ds"] + pd.Timedelta(days=step),
                    "price": rng.random(),
                }
            )
    future_pd = pd.DataFrame(future_rows)
    future_pd["unique_id"] = future_pd["unique_id"].astype(str)

    # predict with pandas X_df (baseline)
    preds_pandas = fcst.predict(h, X_df=future_pd).compute()

    # predict with Dask X_df — caller must not need to call .compute() / .toPandas()
    future_dask = dd.from_pandas(future_pd, npartitions=2)
    preds_dask = fcst.predict(h, X_df=future_dask).compute()

    pd.testing.assert_frame_equal(
        preds_pandas.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        preds_dask.sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Distributed tests are not supported on Windows")
@pytest.mark.skipif(sys.version_info <= (3, 9), reason="Distributed tests are not supported on Python < 3.10")
def test_dask_distributed_forecast_with_new_df():
    """predict() with new_df must produce predictions for all series without errors."""
    series = generate_daily_series(5, equal_ends=True, min_length=50, max_length=50)
    partitioned = _make_partitioned_series(series, npartitions=2)

    fcst = DistributedMLForecast(
        models=[DaskLGBMForecast(verbosity=-1, random_state=0)],
        freq="D",
        lags=[1, 2, 7],
    )
    fcst.fit(partitioned, static_features=[])

    # Provide updated observations (same data here) as new_df
    preds = fcst.predict(5, new_df=partitioned).compute()

    assert preds.shape[0] == 5 * 5
    assert set(preds["unique_id"]) == set(series["unique_id"])


