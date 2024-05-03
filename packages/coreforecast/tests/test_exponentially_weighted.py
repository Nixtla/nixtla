import numpy as np
import pytest
from window_ops.ewm import ewm_mean

from coreforecast.exponentially_weighted import exponentially_weighted_mean


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exponentially_weighted_mean(alpha, dtype):
    x = np.random.rand(100).astype(dtype)
    cf_res = exponentially_weighted_mean(x, alpha)
    wo_res = ewm_mean(x, alpha)
    np.testing.assert_allclose(cf_res, wo_res, rtol=1e-5)
