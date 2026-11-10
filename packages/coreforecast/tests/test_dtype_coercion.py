"""How the input dtype is chosen.

An explicit float dtype is always honoured. Anything else -- integers, lists,
Series -- carries no dtype the caller asked for, so it widens to float64
instead of silently losing precision in float32.
"""

import numpy as np
import pytest

from coreforecast.differences import diff
from coreforecast.expanding import expanding_mean
from coreforecast.exponentially_weighted import exponentially_weighted_mean
from coreforecast.grouped_array import GroupedArray
from coreforecast.rolling import rolling_mean, seasonal_rolling_mean
from coreforecast.scalers import boxcox, inv_boxcox

free_functions = [
    ("rolling_mean", lambda x: rolling_mean(x, 2, 1)),
    ("seasonal_rolling_mean", lambda x: seasonal_rolling_mean(x, 2, 2, 1)),
    ("expanding_mean", lambda x: expanding_mean(x)),
    ("exponentially_weighted_mean", lambda x: exponentially_weighted_mean(x, 0.5)),
    ("diff", lambda x: diff(x, 1)),
    ("boxcox", lambda x: boxcox(x, 0.5)),
    ("inv_boxcox", lambda x: inv_boxcox(x, 0.5)),
]

values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name,fn", free_functions, ids=[f[0] for f in free_functions]
)
def test_an_explicit_float_dtype_is_preserved(name, fn, dtype):
    # these match an overload exactly, so no conversion happens at all
    assert fn(np.array(values, dtype=dtype)).dtype == dtype


@pytest.mark.parametrize(
    "x",
    [np.array(values, dtype=np.int64), np.array(values, dtype=np.int32), values],
    ids=["int64", "int32", "list"],
)
@pytest.mark.parametrize(
    "name,fn", free_functions, ids=[f[0] for f in free_functions]
)
def test_input_without_a_float_dtype_widens_to_float64(name, fn, x):
    # pybind's conversion pass takes the first registered overload, so this is
    # decided by the order of the init_* calls
    assert fn(x).dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_grouped_array_preserves_an_explicit_float_dtype(dtype):
    ga = GroupedArray(np.array(values, dtype=dtype), np.array([0, 3, 6]))
    assert np.asarray(ga.data).dtype == dtype


@pytest.mark.parametrize(
    "x",
    [np.array(values, dtype=np.int64), np.array(values, dtype=np.int32)],
    ids=["int64", "int32"],
)
def test_grouped_array_widens_like_the_free_functions(x):
    # the factory casts explicitly rather than going through overload
    # resolution, so it has to make the same choice on its own
    ga = GroupedArray(x, np.array([0, 3, 6]))
    assert np.asarray(ga.data).dtype == np.float64


def test_integers_beyond_float32_precision_survive():
    # 2**24 + 1 is the first integer float32 cannot represent; these used to
    # come back as [16777216., 16777220., 16777220.]
    x = np.array([16777217, 16777219, 16777221])
    np.testing.assert_array_equal(rolling_mean(x, 1, 1), x.astype(np.float64))
    np.testing.assert_array_equal(
        np.asarray(GroupedArray(x, np.array([0, 3])).data), x.astype(np.float64)
    )
