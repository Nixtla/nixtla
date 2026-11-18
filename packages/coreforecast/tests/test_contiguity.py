"""Non-contiguous input.

The bound parameters take a raw pointer to the input buffer, so without
c_style a strided view is read as if it were contiguous and silently returns
results for the wrong elements. Every entry point must agree with what it
returns for a contiguous copy of the same values.
"""

import numpy as np
import pytest

from coreforecast.differences import diff, num_diffs, num_seas_diffs
from coreforecast.expanding import (
    expanding_max,
    expanding_mean,
    expanding_min,
    expanding_quantile,
    expanding_std,
)
from coreforecast.exponentially_weighted import exponentially_weighted_mean
from coreforecast.grouped_array import GroupedArray
from coreforecast.rolling import (
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_quantile,
    rolling_std,
    seasonal_rolling_max,
    seasonal_rolling_mean,
    seasonal_rolling_min,
    seasonal_rolling_quantile,
    seasonal_rolling_std,
)
from coreforecast.scalers import boxcox, boxcox_lambda, inv_boxcox
from coreforecast.seasonal import find_season_length

free_functions = [
    ("rolling_mean", lambda x: rolling_mean(x, 3, 1)),
    ("rolling_std", lambda x: rolling_std(x, 3, 2)),
    ("rolling_min", lambda x: rolling_min(x, 3, 1)),
    ("rolling_max", lambda x: rolling_max(x, 3, 1)),
    ("rolling_quantile", lambda x: rolling_quantile(x, 0.5, 3, 1)),
    ("seasonal_rolling_mean", lambda x: seasonal_rolling_mean(x, 2, 2, 1)),
    ("seasonal_rolling_std", lambda x: seasonal_rolling_std(x, 2, 2, 2)),
    ("seasonal_rolling_min", lambda x: seasonal_rolling_min(x, 2, 2, 1)),
    ("seasonal_rolling_max", lambda x: seasonal_rolling_max(x, 2, 2, 1)),
    ("seasonal_rolling_quantile", lambda x: seasonal_rolling_quantile(x, 0.5, 2, 2, 1)),
    ("expanding_mean", expanding_mean),
    ("expanding_std", expanding_std),
    ("expanding_min", expanding_min),
    ("expanding_max", expanding_max),
    ("expanding_quantile", lambda x: expanding_quantile(x, 0.5)),
    ("ewm_mean", lambda x: exponentially_weighted_mean(x, 0.5)),
    ("diff", lambda x: diff(x, 1)),
    ("num_diffs", lambda x: num_diffs(x, 1)),
    ("num_seas_diffs", lambda x: num_seas_diffs(x, 2, 1)),
    ("find_season_length", lambda x: find_season_length(x, 4)),
    ("boxcox_lambda_loglik", lambda x: boxcox_lambda(x, method="loglik")),
    ("boxcox_lambda_guerrero", lambda x: boxcox_lambda(x, method="guerrero", season_length=2)),
    ("boxcox", lambda x: boxcox(x, 0.5)),
    ("inv_boxcox", lambda x: inv_boxcox(x, 0.5)),
]


@pytest.fixture
def buffer(rng):
    return rng.uniform(low=1.0, high=20.0, size=60)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name,fn", free_functions, ids=[f[0] for f in free_functions]
)
def test_free_functions_match_a_contiguous_copy(name, fn, buffer, dtype):
    # every other element of a larger buffer, so reading it as if it were
    # contiguous picks up the wrong half. Slice after the cast: astype would
    # return a contiguous copy.
    view = buffer.astype(dtype)[::2]
    assert not view.flags["C_CONTIGUOUS"]
    strided, contiguous = fn(view), fn(np.ascontiguousarray(view))
    # bit-identical, not merely close: the same values are being read
    np.testing.assert_array_equal(strided, contiguous)
    # putting c_style on the parameter instead of ensuring inside would make a
    # strided float64 array fall through to the float32 overload
    if isinstance(contiguous, np.ndarray):
        assert strided.dtype == contiguous.dtype == dtype


grouped_operations = [
    ("rolling_mean", lambda ga: ga._rolling_mean(0, 3, 1)),
    ("expanding_std", lambda ga: ga._expanding_std(0)[0]),
    ("minmax_stats", lambda ga: ga._minmax_stats()),
    ("diff", lambda ga: ga._diff(1)),
    ("tail", lambda ga: ga._tail(2)),
]


@pytest.mark.parametrize(
    "name,op", grouped_operations, ids=[o[0] for o in grouped_operations]
)
def test_grouped_array_accepts_strided_data(name, op, rng):
    full = rng.uniform(low=1.0, high=20.0, size=48)
    view = full[::2]
    indptr = np.array([0, 8, 16, 24], dtype=np.int32)
    np.testing.assert_allclose(
        op(GroupedArray(view, indptr)),
        op(GroupedArray(np.ascontiguousarray(view), indptr)),
        equal_nan=True,
    )


def test_grouped_array_accepts_a_strided_indptr():
    data = np.arange(12, dtype=np.float64)
    # [0, 4, 8, 12] taken as every other element of a larger array
    indptr = np.arange(0, 14, 2, dtype=np.int64)[::2]
    assert not indptr.flags["C_CONTIGUOUS"]
    ga = GroupedArray(data, indptr)
    np.testing.assert_array_equal(np.asarray(ga.indptr), [0, 4, 8, 12])
    assert len(ga) == 3


def test_take_and_with_data_accept_strided_input():
    data = np.arange(12, dtype=np.float64)
    ga = GroupedArray(data, np.array([0, 4, 8, 12], dtype=np.int32))
    replacement = np.arange(24, dtype=np.float64)[::2]
    np.testing.assert_array_equal(
        ga._with_data(replacement).data, np.ascontiguousarray(replacement)
    )
    indices = np.array([0, 9, 2, 9], dtype=np.int32)[::2]
    np.testing.assert_array_equal(ga._take(indices), ga._take(np.array([0, 2], dtype=np.int32)))
