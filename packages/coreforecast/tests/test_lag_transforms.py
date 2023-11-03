import numpy as np
import pytest
from window_ops.expanding import *
from window_ops.ewm import ewm_mean
from window_ops.rolling import *
from window_ops.shift import shift_array

from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import *

lag = 2
season_length = 7
window_size = 4
min_samples = 2
lengths = np.random.randint(low=100, high=200, size=100)
indptr = np.append(0, lengths.cumsum()).astype(np.int32)
data = 10 * np.random.rand(indptr[-1])


def transform(data, indptr, updates_only, lag, func, *args) -> np.ndarray:
    n_series = len(indptr) - 1
    if updates_only:
        out = np.empty_like(data[:n_series])
        for i in range(n_series):
            lagged = shift_array(data[indptr[i] : indptr[i + 1]], lag)
            out[i] = func(lagged, *args)[-1]
    else:
        out = np.empty_like(data)
        for i in range(n_series):
            lagged = shift_array(data[indptr[i] : indptr[i + 1]], lag)
            out[indptr[i] : indptr[i + 1]] = func(lagged, *args)
    return out


@pytest.fixture
def data():
    return 10 * np.random.rand(indptr[-1])


combs_map = {
    "rolling_mean": (rolling_mean, RollingMean, [window_size, min_samples]),
    "rolling_std": (rolling_std, RollingStd, [window_size, min_samples]),
    "rolling_min": (rolling_min, RollingMin, [window_size, min_samples]),
    "rolling_max": (rolling_max, RollingMax, [window_size, min_samples]),
    "seasonal_rolling_mean": (
        seasonal_rolling_mean,
        SeasonalRollingMean,
        [season_length, window_size, min_samples],
    ),
    "seasonal_rolling_std": (
        seasonal_rolling_std,
        SeasonalRollingStd,
        [season_length, window_size, min_samples],
    ),
    "seasonal_rolling_min": (
        seasonal_rolling_min,
        SeasonalRollingMin,
        [season_length, window_size, min_samples],
    ),
    "seasonal_rolling_max": (
        seasonal_rolling_max,
        SeasonalRollingMax,
        [season_length, window_size, min_samples],
    ),
    "expanding_mean": (expanding_mean, ExpandingMean, []),
    "expanding_std": (expanding_std, ExpandingStd, []),
    "expanding_min": (expanding_min, ExpandingMin, []),
    "expanding_max": (expanding_max, ExpandingMax, []),
    "ewm_mean": (ewm_mean, ExponentiallyWeightedMean, [0.8]),
}


@pytest.mark.parametrize("comb", list(combs_map.keys()))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correctness(data, comb, dtype):
    rtol = 1e-5 if dtype == np.float32 else 1e-7
    if dtype == np.float32:
        if "rolling_std" in comb:
            rtol = 1e-2
        elif "expanding_std" in comb:
            rtol = 5e-5
    data = data.astype(dtype, copy=True)
    ga = GroupedArray(data, indptr)
    wf, cf, args = combs_map[comb]
    # transform
    wres = transform(data, indptr, False, lag, wf, *args)
    cobj = cf(lag, *args)
    cres = cobj.transform(ga)
    np.testing.assert_allclose(wres, cres, rtol=rtol)
    # update
    wres = transform(data, indptr, True, lag - 1, wf, *args)
    cres = cobj.update(ga)
    np.testing.assert_allclose(wres, cres, rtol=rtol)
