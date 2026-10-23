import coreforecast.lag_transforms as lag_tf
import numpy as np
import pytest
from coreforecast.exponentially_weighted import exponentially_weighted_mean
from coreforecast.grouped_array import GroupedArray

from . import (
    lag_tfms_map,
    min_samples,
    scaler2core,
    scalers,
    season_length,
    window_size,
)

all_lag_tfms = {
    **lag_tfms_map,
    "rolling_quantile": (lag_tf.RollingQuantile, [0.5, window_size, min_samples]),
    "seasonal_rolling_quantile": (
        lag_tf.SeasonalRollingQuantile,
        [0.5, season_length, window_size, min_samples],
    ),
}


@pytest.fixture
def indptr(rng):
    lengths = rng.integers(low=1_000, high=2_000, size=1_000)
    return np.append(0, lengths.cumsum()).astype(np.int32)


@pytest.fixture
def data(rng, indptr):
    return rng.normal(size=indptr[-1])


@pytest.mark.parametrize("scaler", scalers)
def test_scalers(benchmark, data, indptr, scaler):
    ga = GroupedArray(data, indptr)
    scaler = scaler2core[scaler]
    benchmark(scaler.fit, ga)


@pytest.mark.parametrize("tfm", all_lag_tfms.keys())
def test_lag_transforms(benchmark, data, indptr, tfm):
    ga = GroupedArray(data, indptr)
    Transform, args = all_lag_tfms[tfm]
    tfm = Transform(5, *args)
    benchmark(tfm.transform, ga)


def test_ewm(benchmark, data, indptr):
    benchmark(exponentially_weighted_mean, x=data[: indptr[1]], alpha=0.9)
