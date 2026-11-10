# test _append_one
import numpy as np
from conftest import assert_raises_with_message

from utilsforecast.data import generate_series
from utilsforecast.grouped_array import GroupedArray, _append_one, _append_several


def test_append_one():
    data = np.arange(5)
    indptr = np.array([0, 2, 5])
    new = np.array([7, 8])
    new_data, new_indptr = _append_one(data, indptr, new)
    np.testing.assert_equal(new_data, np.array([0, 1, 7, 2, 3, 4, 8]))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 3, 7]),
    )

# 2d
def test_append_one_2d():
    data = np.arange(5).reshape(-1, 1)
    indptr = np.array([0, 2, 5])
    new = np.array([7, 8])
    new_data, new_indptr = _append_one(data, indptr, new)
    np.testing.assert_equal(new_data, np.array([0, 1, 7, 2, 3, 4, 8]).reshape(-1, 1))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 3, 7]),
    )

# test append several
def test_append_several():
    data = np.arange(5)
    indptr = np.array([0, 2, 5])
    new_sizes = np.array([0, 2, 1])
    new_values = np.array([6, 7, 5])
    new_groups = np.array([False, True, False])
    new_data, new_indptr = _append_several(data, indptr, new_sizes, new_values, new_groups)
    np.testing.assert_equal(new_data, np.array([0, 1, 6, 7, 2, 3, 4, 5]))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 2, 4, 8]),
    )

# 2d
def test_append_several_2d():
    data = np.arange(5).reshape(-1, 1)
    indptr = np.array([0, 2, 5])
    new_sizes = np.array([0, 2, 1])
    new_values = np.array([6, 7, 5]).reshape(-1, 1)
    new_groups = np.array([False, True, False])
    new_data, new_indptr = _append_several(data, indptr, new_sizes, new_values, new_groups)
    np.testing.assert_equal(new_data, np.array([0, 1, 6, 7, 2, 3, 4, 5]).reshape(-1, 1))
    np.testing.assert_equal(
        new_indptr,
        np.array([0, 2, 4, 8]),
    )


# The `GroupedArray` is used internally for storing the series values and performing transformations.
def test_grouped_array():
    data = np.arange(20, dtype=np.float32).reshape(-1, 2)
    indptr = np.array([0, 2, 10])  # group 1: [0, 1], group 2: [2..9]
    ga = GroupedArray(data, indptr)
    assert len(ga) == 2
    # Iterate through the groups
    ga_iter = iter(ga)
    np.testing.assert_equal(next(ga_iter), np.arange(4).reshape(-1, 2))
    np.testing.assert_equal(next(ga_iter), np.arange(4, 20).reshape(-1, 2))
    # Take the last two observations from each group
    last2_data, last2_indptr = ga.take_from_groups(slice(-2, None))
    np.testing.assert_equal(
        last2_data,
        np.vstack(
            [
                np.arange(4).reshape(-1, 2),
                np.arange(16, 20).reshape(-1, 2),
            ]
        ),
    )
    np.testing.assert_equal(last2_indptr, np.array([0, 2, 4]))

# 1d
def test_grouped_array_1d():
    data = np.arange(20, dtype=np.float32).reshape(-1, 2)
    indptr = np.array([0, 2, 10])
    ga = GroupedArray(data, indptr)
    ga1d = GroupedArray(np.arange(10), indptr)
    last2_data1d, last2_indptr1d = ga1d.take_from_groups(slice(-2, None))
    np.testing.assert_equal(last2_data1d, np.array([0, 1, 8, 9]))
    np.testing.assert_equal(last2_indptr1d, np.array([0, 2, 4]))
    # Take the second observation from each group
    second_data, second_indptr = ga.take_from_groups(1)
    np.testing.assert_equal(second_data, np.array([[2, 3], [6, 7]]))
    np.testing.assert_equal(second_indptr, np.array([0, 1, 2]))

    # 1d
    second_data1d, second_indptr1d = ga1d.take_from_groups(1)
    np.testing.assert_equal(second_data1d, np.array([1, 3]))
    np.testing.assert_equal(second_indptr1d, np.array([0, 1, 2]))
    # Take the last four observations from every group. Note that since group 1 only has two elements, only these are returned.
    last4_data, last4_indptr = ga.take_from_groups(slice(-4, None))
    np.testing.assert_equal(
        last4_data,
        np.vstack(
            [
                np.arange(4).reshape(-1, 2),
                np.arange(12, 20).reshape(-1, 2),
            ]
        ),
    )
    np.testing.assert_equal(last4_indptr, np.array([0, 2, 6]))

    # 1d
    last4_data1d, last4_indptr1d = ga1d.take_from_groups(slice(-4, None))
    np.testing.assert_equal(last4_data1d, np.array([0, 1, 6, 7, 8, 9]))
    np.testing.assert_equal(last4_indptr1d, np.array([0, 2, 6]))
    # Select a specific subset of groups
    indptr = np.array([0, 2, 4, 7, 10])
    ga2 = GroupedArray(data, indptr)
    subset = GroupedArray(*ga2.take([0, 2]))
    np.testing.assert_allclose(subset[0].data, ga2[0].data)
    np.testing.assert_allclose(subset[1].data, ga2[2].data)

    # 1d
    ga2_1d = GroupedArray(np.arange(10), indptr)
    subset1d = GroupedArray(*ga2_1d.take([0, 2]))
    np.testing.assert_allclose(subset1d[0].data, ga2_1d[0].data)
    np.testing.assert_allclose(subset1d[1].data, ga2_1d[2].data)
    # try to append new values that don't match the number of groups
    assert_raises_with_message(lambda: ga.append(np.array([1.0, 2.0, 3.0])), "new must have 2 rows")
    # build from df
    series_pd = generate_series(10, static_as_categorical=False, engine="pandas")
    ga_pd = GroupedArray.from_sorted_df(series_pd, "unique_id", "ds", "y")
    series_pl = generate_series(10, static_as_categorical=False, engine="polars")
    ga_pl = GroupedArray.from_sorted_df(series_pl, "unique_id", "ds", "y")
    np.testing.assert_allclose(ga_pd.data, ga_pl.data)
    np.testing.assert_equal(ga_pd.indptr, ga_pl.indptr)
