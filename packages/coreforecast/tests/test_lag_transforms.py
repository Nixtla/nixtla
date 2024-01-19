import numpy as np
import pandas as pd
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


def pd_rolling_quantile(x, lag, p, window_size, min_samples):
    return (
        pd.Series(x)
        .shift(lag)
        .rolling(window=window_size, min_periods=min_samples)
        .quantile(p)
    )


def pd_seasonal_rolling_quantile(x, lag, p, season_length, window_size, min_samples):
    out = np.empty_like(x)
    x = pd.Series(x).shift(lag).to_numpy()
    for season in range(season_length):
        out[season::season_length] = pd_rolling_quantile(
            x[season::season_length], 0, p, window_size, min_samples
        )
    return out


def pd_expanding_quantile(x, lag, p):
    return pd.Series(x).shift(lag).expanding().quantile(p)


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
    if "expanding_std" in comb:
        rtol *= 100
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


@pytest.mark.parametrize("window_type", ["rolling", "seasonal_rolling", "expanding"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.9, 0.99])
def test_correctness_quantiles(data, dtype, p, window_type):
    rtol = 1e-5 if dtype == np.float32 else 1e-7
    data = data.astype(dtype, copy=True)
    ga = GroupedArray(data, indptr)
    if window_type == "rolling":
        core_cls = RollingQuantile(lag, p, window_size, min_samples)
        pd_fun = pd_rolling_quantile
        pd_kwargs = dict(window_size=window_size, min_samples=min_samples)
    elif window_type == "seasonal_rolling":
        core_cls = SeasonalRollingQuantile(
            lag, p, season_length, window_size, min_samples
        )
        pd_fun = pd_seasonal_rolling_quantile
        pd_kwargs = dict(
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
        )
    else:
        core_cls = ExpandingQuantile(lag, p)
        pd_fun = pd_expanding_quantile
        pd_kwargs = {}
    cres = core_cls.transform(ga)
    core_cls.lag = lag + 1
    cres_upd = core_cls.update(ga)
    pres = np.hstack([pd_fun(ga[i], lag, p, **pd_kwargs) for i in range(len(ga))])
    pres_upd = pres[indptr[1:] - 1]
    np.testing.assert_allclose(cres, pres, rtol=rtol)
    np.testing.assert_allclose(cres_upd, pres_upd, rtol=rtol)
