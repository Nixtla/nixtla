import numpy as np
import pytest
import window_ops.expanding as wops_expanding

import coreforecast.expanding as cf_expanding
from .test_lag_transforms import pd_rolling_quantile, pd_seasonal_rolling_quantile

quantile_ops = ["expanding_quantile"]
other_ops = [op for op in cf_expanding.__all__ if op not in quantile_ops]


@pytest.mark.parametrize("op", other_ops)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rolling(op, dtype):
    x = np.random.rand(100).astype(dtype)
    cf_res = getattr(cf_expanding, op)(x)
    wo_res = getattr(wops_expanding, op)(x)
    np.testing.assert_allclose(cf_res, wo_res, rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_quantiles(dtype, p):
    x = np.random.rand(100).astype(dtype)
    pd_res = pd_rolling_quantile(x, 0, p, x.size, 1)
    cf_res = cf_expanding.expanding_quantile(x, p)
    np.testing.assert_allclose(pd_res, cf_res, atol=1e-5)
