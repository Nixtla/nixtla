"""The two supported NaN paths.

skipna=False assumes NaNs form a leading run only; skipna=True ignores them
wherever they are. Both must behave the same through the free functions and
through GroupedArray.
"""

import numpy as np
import pandas as pd
import pytest

from coreforecast.differences import diff
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

window_size = 3
min_samples = 2
season_length = 2

# (name, free function, equivalent GroupedArray call)
equivalences = [
    ("rolling_mean", lambda x: rolling_mean(x, window_size, min_samples),
     lambda ga: ga._rolling_mean(0, window_size, min_samples)),
    ("rolling_std", lambda x: rolling_std(x, window_size, min_samples),
     lambda ga: ga._rolling_std(0, window_size, min_samples)),
    ("rolling_min", lambda x: rolling_min(x, window_size, min_samples),
     lambda ga: ga._rolling_min(0, window_size, min_samples)),
    ("rolling_max", lambda x: rolling_max(x, window_size, min_samples),
     lambda ga: ga._rolling_max(0, window_size, min_samples)),
    ("rolling_quantile", lambda x: rolling_quantile(x, 0.5, window_size, min_samples),
     lambda ga: ga._rolling_quantile(0, 0.5, window_size, min_samples)),
    ("seasonal_rolling_mean",
     lambda x: seasonal_rolling_mean(x, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_mean(0, season_length, window_size, min_samples)),
    ("seasonal_rolling_std",
     lambda x: seasonal_rolling_std(x, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_std(0, season_length, window_size, min_samples)),
    ("seasonal_rolling_min",
     lambda x: seasonal_rolling_min(x, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_min(0, season_length, window_size, min_samples)),
    ("seasonal_rolling_max",
     lambda x: seasonal_rolling_max(x, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_max(0, season_length, window_size, min_samples)),
    ("seasonal_rolling_quantile",
     lambda x: seasonal_rolling_quantile(x, 0.5, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_quantile(0, 0.5, season_length, window_size, min_samples)),
    ("expanding_mean", expanding_mean, lambda ga: ga._expanding_mean(0)[0]),
    ("expanding_std", expanding_std, lambda ga: ga._expanding_std(0)[0]),
    ("expanding_min", expanding_min, lambda ga: ga._expanding_min(0)),
    ("expanding_max", expanding_max, lambda ga: ga._expanding_max(0)),
    ("expanding_quantile", lambda x: expanding_quantile(x, 0.5),
     lambda ga: ga._expanding_quantile(0, 0.5)),
    ("ewm_mean", lambda x: exponentially_weighted_mean(x, 0.5),
     lambda ga: ga._exponentially_weighted_mean(0, 0.5)),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_leading", [0, 1, 3])
@pytest.mark.parametrize("name,free_fn,ga_fn", equivalences, ids=[e[0] for e in equivalences])
def test_leading_nans_match_grouped_array(name, free_fn, ga_fn, n_leading, dtype):
    x = np.concatenate(
        [np.full(n_leading, np.nan), np.arange(1, 9)]
    ).astype(dtype)
    ga = GroupedArray(x, np.array([0, x.size], dtype=np.int32))
    np.testing.assert_allclose(free_fn(x), ga_fn(ga), equal_nan=True, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_leading_nans_are_preserved_and_the_rest_is_computed(dtype):
    x = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    res = rolling_mean(x, window_size=3, min_samples=1)
    assert np.isnan(res[:2]).all()
    np.testing.assert_allclose(res[2:], [1.0, 1.5, 2.0, 3.0, 4.0], rtol=1e-6)


@pytest.mark.parametrize(
    "fn", [rolling_mean, rolling_min, rolling_max, expanding_mean, expanding_min]
)
def test_degenerate_inputs(fn):
    kwargs = {} if fn.__name__.startswith("expanding") else {"window_size": 3}
    np.testing.assert_array_equal(fn(np.array([]), **kwargs), np.array([]))
    assert np.isnan(fn(np.full(4, np.nan), **kwargs)).all()


def test_skipna_min_max_expire_stale_front_on_a_nan_step():
    # regression: the monotonic deque's front expires once per *step*, so it must
    # be checked on NaN steps too, otherwise an out-of-window entry is returned
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0])
    # index 3's window is [2, 3, nan]; position 0 has left it
    np.testing.assert_allclose(
        rolling_min(x, 3, 1, skipna=True), [1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 5.0]
    )
    np.testing.assert_allclose(
        rolling_max(x, 3, 1, skipna=True), [1.0, 2.0, 3.0, 3.0, 5.0, 6.0, 7.0]
    )


@pytest.mark.parametrize("stat", ["min", "max"])
def test_skipna_min_max_match_pandas_with_scattered_nans(stat, rng):
    # min_samples=1 is where coreforecast's positional min_samples and pandas'
    # observation-based min_periods agree, so pandas is a valid oracle there
    fn = {"min": rolling_min, "max": rolling_max}[stat]
    for _ in range(200):
        n = int(rng.integers(2, 30))
        w = int(rng.integers(1, 10))
        x = rng.normal(size=n)
        x[rng.random(n) < 0.4] = np.nan
        expected = getattr(pd.Series(x).rolling(w, min_periods=1), stat)().to_numpy()
        np.testing.assert_allclose(
            fn(x, w, 1, skipna=True), expected, equal_nan=True
        )


rolling_fns = [rolling_mean, rolling_std, rolling_min, rolling_max]


@pytest.mark.parametrize("fn", rolling_fns, ids=lambda f: f.__name__)
def test_zero_window_size_is_rejected(fn):
    # zero made the growing loop start at -1 and read and write one element
    # before the buffers
    with pytest.raises(ValueError, match="window_size must be greater than 0"):
        fn(np.arange(5.0), 0, 1)


@pytest.mark.parametrize("fn", rolling_fns, ids=lambda f: f.__name__)
def test_zero_min_samples_is_rejected(fn):
    # same out-of-bounds start, reached through min_samples instead
    with pytest.raises(ValueError, match="min_samples must be greater than 0"):
        fn(np.arange(5.0), 3, 0)


@pytest.mark.parametrize("fn", rolling_fns, ids=lambda f: f.__name__)
def test_negative_counts_are_rejected(fn):
    with pytest.raises(ValueError, match="window_size must be greater than 0"):
        fn(np.arange(5.0), -3, 1)
    with pytest.raises(ValueError, match="min_samples must be greater than 0"):
        fn(np.arange(5.0), 3, -1)


def test_zero_season_length_is_rejected():
    # season_length divides, so zero used to raise SIGFPE and kill the process
    with pytest.raises(ValueError, match="season_length must be greater than 0"):
        seasonal_rolling_mean(np.arange(5.0), 0, 2, 1)


@pytest.mark.parametrize(
    "call",
    [
        lambda ga, lag: ga._lag(lag),
        lambda ga, lag: ga._rolling_mean(lag, 2, 1),
        lambda ga, lag: ga._rolling_mean_update(lag, 2, 1),
        lambda ga, lag: ga._expanding_std(lag),
    ],
    ids=["lag", "transform", "reduce", "transform_and_reduce"],
)
def test_negative_lag_is_rejected(call):
    # a negative lag makes the transform write before the start of the output
    ga = GroupedArray(np.arange(10.0), np.array([0, 5, 10], dtype=np.int32))
    with pytest.raises(ValueError, match="lag must be non-negative"):
        call(ga, -3)


@pytest.mark.parametrize(
    "call",
    [
        lambda ga, d: diff(np.arange(10.0), d),
        lambda ga, d: ga._diff(d),
        lambda ga, d: ga._diffs(np.array([d, 1], dtype=np.int32)),
    ],
    ids=["free_fn", "grouped", "grouped_variable"],
)
def test_negative_d_is_rejected(call):
    # the difference loop starts at d, so a negative one reads and writes one
    # element before both buffers
    ga = GroupedArray(np.arange(10.0), np.array([0, 5, 10], dtype=np.int32))
    with pytest.raises(ValueError, match="d must be non-negative"):
        call(ga, -1)


def test_zero_d_is_a_copy():
    np.testing.assert_array_equal(diff(np.arange(5.0), 0), np.arange(5.0))


@pytest.mark.parametrize(
    "call",
    [lambda ga: ga._expanding_mean(0), lambda ga: ga._expanding_std(0)],
    ids=["expanding_mean", "expanding_std"],
)
@pytest.mark.parametrize(
    "data,indptr",
    [
        ([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0], [0, 2, 6]),
        ([1.0, 2.0, 3.0, 4.0], [0, 0, 4]),
    ],
    ids=["all_nan_group", "empty_group"],
)
def test_skipped_groups_get_nan_stats(call, data, indptr):
    # the kernel never runs for these groups, so their stats row used to be
    # left at whatever the freshly allocated buffer happened to hold
    ga = GroupedArray(np.array(data), np.array(indptr, dtype=np.int32))
    stats = np.asarray(call(ga)[1])
    assert np.isnan(stats[0]).all()
    assert np.isfinite(stats[1]).all()


def test_seasonal_nan_only_latches_within_its_own_season():
    # seasonal ops de-interleave, so a NaN poisons its own subseries only
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    np.testing.assert_allclose(
        seasonal_rolling_mean(x, 2, 2, 1),
        [1.0, 2.0, 2.0, np.nan, 4.0, np.nan, 6.0, np.nan, 8.0, np.nan],
    )


quantile_entry_points = [
    ("rolling_quantile", lambda x, ga, p: rolling_quantile(x, p, 3, 1)),
    ("seasonal_rolling_quantile", lambda x, ga, p: seasonal_rolling_quantile(x, p, 2, 2, 1)),
    ("expanding_quantile", lambda x, ga, p: expanding_quantile(x, p)),
    ("ga_rolling_quantile", lambda x, ga, p: ga._rolling_quantile(0, p, 3, 1)),
    ("ga_rolling_quantile_update", lambda x, ga, p: ga._rolling_quantile_update(0, p, 3, 1)),
    ("ga_expanding_quantile", lambda x, ga, p: ga._expanding_quantile(0, p)),
    ("ga_expanding_quantile_update", lambda x, ga, p: ga._expanding_quantile_update(0, p)),
    ("ga_seasonal_rolling_quantile", lambda x, ga, p: ga._seasonal_rolling_quantile(0, p, 2, 2, 1)),
]


@pytest.mark.parametrize("p", [-2.0, 1.5, np.nan])
@pytest.mark.parametrize(
    "name,fn", quantile_entry_points, ids=[q[0] for q in quantile_entry_points]
)
def test_quantile_level_outside_the_unit_interval_is_rejected(name, fn, p):
    # p indexes into the sorted window, so an out-of-range value read past the
    # end of the buffer; expanding_quantile_update returned uninitialised memory
    x = np.arange(20.0)
    ga = GroupedArray(x, np.array([0, 10, 20], dtype=np.int32))
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        fn(x, ga, p)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_quantile_level_bounds_are_inclusive(p):
    assert np.isfinite(rolling_quantile(np.arange(20.0), p, 3, 1)).any()
