import coreforecast.lag_transforms as lag_tf
import numpy as np
import pandas as pd
import pytest
from coreforecast.grouped_array import GroupedArray
from window_ops.shift import shift_array

from . import lag_tfms_map, min_samples, season_length, window_size

lag = 2
lengths = np.random.randint(low=100, high=200, size=100)
indptr = np.append(0, lengths.cumsum()).astype(np.int32)


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


@pytest.mark.parametrize("comb", list(lag_tfms_map.keys()))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correctness(data, comb, dtype):
    atol = 1e-4
    rtol = 1e-5 if dtype == np.float32 else 1e-7
    if dtype == np.float32:
        if "rolling_std" in comb:
            rtol = 1e-2
    if "expanding_std" in comb:
        rtol *= 100
    data = data.astype(dtype, copy=True)
    ga = GroupedArray(data, indptr)
    wf, cf, args = lag_tfms_map[comb]
    # transform
    wres = transform(data, indptr, False, lag, wf, *args)
    cobj = cf(lag, *args)
    cres = cobj.transform(ga)
    np.testing.assert_allclose(wres, cres, atol=atol, rtol=rtol)
    # update
    wres = transform(data, indptr, True, lag - 1, wf, *args)
    cres = cobj.update(ga)
    np.testing.assert_allclose(wres, cres, atol=atol, rtol=rtol)
    # stack
    combined = cobj.stack([cobj, cobj])
    if hasattr(cobj, "stats_"):
        assert combined.stats_.shape[0] == 2 * cobj.stats_.shape[0]
    else:
        assert combined is cobj


@pytest.mark.parametrize("window_type", ["rolling", "seasonal_rolling", "expanding"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.9, 0.99])
def test_correctness_quantiles(data, dtype, p, window_type):
    rtol = 1e-5 if dtype == np.float32 else 1e-7
    data = data.astype(dtype, copy=True)
    ga = GroupedArray(data, indptr)
    if window_type == "rolling":
        core_cls = lag_tf.RollingQuantile(lag, p, window_size, min_samples)
        pd_fun = pd_rolling_quantile
        pd_kwargs = dict(window_size=window_size, min_samples=min_samples)
    elif window_type == "seasonal_rolling":
        core_cls = lag_tf.SeasonalRollingQuantile(
            lag, p, season_length, window_size, min_samples
        )
        pd_fun = pd_seasonal_rolling_quantile
        pd_kwargs = dict(
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
        )
    else:
        core_cls = lag_tf.ExpandingQuantile(lag, p)
        pd_fun = pd_expanding_quantile
        pd_kwargs = {}
    cres = core_cls.transform(ga)
    core_cls.lag = lag + 1
    cres_upd = core_cls.update(ga)
    pres = np.hstack([pd_fun(ga[i], lag, p, **pd_kwargs) for i in range(len(ga))])
    pres_upd = pres[indptr[1:] - 1]
    np.testing.assert_allclose(cres, pres, rtol=rtol)
    np.testing.assert_allclose(cres_upd, pres_upd, rtol=rtol)
