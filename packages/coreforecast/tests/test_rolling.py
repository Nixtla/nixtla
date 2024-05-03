import numpy as np
import pytest
import window_ops.rolling as wops_rolling

import coreforecast.rolling as cf_rolling
from .test_lag_transforms import pd_rolling_quantile, pd_seasonal_rolling_quantile

quantile_ops = ["rolling_quantile", "seasonal_rolling_quantile"]
other_ops = [op for op in cf_rolling.__all__ if op not in quantile_ops]


@pytest.mark.parametrize("op", other_ops)
@pytest.mark.parametrize("min_samples", [None, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rolling(op, min_samples, dtype):
    window_size = 10
    season_length = 4
    x = np.random.rand(100).astype(dtype)
    if op.startswith("seasonal"):
        args = (season_length, window_size, min_samples)
    else:
        args = (window_size, min_samples)
    cf_res = getattr(cf_rolling, op)(x, *args)
    wo_res = getattr(wops_rolling, op)(x, *args)
    np.testing.assert_allclose(cf_res, wo_res, rtol=1e-5)


@pytest.mark.parametrize("op", quantile_ops)
@pytest.mark.parametrize("min_samples", [None, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_quantiles(op, min_samples, dtype, p):
    window_size = 10
    season_length = 4
    x = np.random.rand(100).astype(dtype)
    if op == "seasonal_rolling_quantile":
        pd_res = pd_seasonal_rolling_quantile(
            x, 0, p, season_length, window_size, min_samples
        )
        cf_res = cf_rolling.seasonal_rolling_quantile(
            x, p, season_length, window_size, min_samples
        )
    else:
        pd_res = pd_rolling_quantile(x, 0, p, window_size, min_samples)
        cf_res = cf_rolling.rolling_quantile(x, p, window_size, min_samples)
    np.testing.assert_allclose(pd_res, cf_res, atol=1e-5)
