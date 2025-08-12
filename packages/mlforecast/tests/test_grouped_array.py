import copy

import numpy as np

from mlforecast.grouped_array import GroupedArray

from .conftest import assert_raises_with_message


def test_grouped_array_append_several():
    data = np.arange(5)
    indptr = np.array([0, 2, 5])
    new_sizes = np.array([0, 2, 1])
    new_values = np.array([6, 7, 5])
    new_groups = np.array([False, True, False])
    new_ga = GroupedArray(data, indptr).append_several(new_sizes, new_values, new_groups)
    np.testing.assert_equal(
        new_ga.data,
        np.array([0, 1, 6, 7, 2, 3, 4, 5]),
    )
    np.testing.assert_equal(
        new_ga.indptr,
        np.array([0, 2, 4, 8]),
    )

# The `GroupedArray` is used internally for storing the series values and performing transformations.
def test_grouped_array():
    data = np.arange(10, dtype=np.float32)
    indptr = np.array([0, 2, 10])  # group 1: [0, 1], group 2: [2..9]
    ga = GroupedArray(data, indptr)
    assert len(ga) == 2
    assert str(ga) == "GroupedArray(ndata=10, n_groups=2)"

# Iterate through the groups
def test_grouped_array_iter():
    data = np.arange(10, dtype=np.float32)
    indptr = np.array([0, 2, 10])  # group 1: [0, 1], group 2: [2..9]
    ga = GroupedArray(data, indptr)
    ga_iter = iter(ga)
    np.testing.assert_equal(next(ga_iter), np.array([0, 1]))
    np.testing.assert_equal(next(ga_iter), np.arange(2, 10))

    # Take the last two observations from every group
    last_2 = ga.take_from_groups(slice(-2, None))
    np.testing.assert_equal(last_2.data, np.array([0, 1, 8, 9]))
    np.testing.assert_equal(last_2.indptr, np.array([0, 2, 4]))

    # Take the last four observations from every group. Note that since group 1 only has two elements, only these are returned.
    last_4 = ga.take_from_groups(slice(-4, None))
    np.testing.assert_equal(last_4.data, np.array([0, 1, 6, 7, 8, 9]))
    np.testing.assert_equal(last_4.indptr, np.array([0, 2, 6]))

    # Select a specific subset of groups
    indptr = np.array([0, 2, 4, 7, 10])
    ga2 = GroupedArray(data, indptr)
    subset = ga2.take([0, 2])
    np.testing.assert_allclose(subset[0].data, ga2[0].data)
    np.testing.assert_allclose(subset[1].data, ga2[2].data)

    # The groups are [0, 1], [2, ..., 9]. expand_target(2) should take rolling pairs of them and fill with nans when there aren't enough
    np.testing.assert_equal(
        ga.expand_target(2),
        np.array(
            [
                [0, 1],
                [1, np.nan],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, np.nan],
            ]
        ),
    )
    # append
    combined = ga.append(np.array([-1, -2]))
    np.testing.assert_equal(
        combined.data,
        np.hstack([ga.data[:2], np.array([-1]), ga.data[2:], np.array([-2])]),
    )
    # try to append new values that don't match the number of groups
    assert_raises_with_message(
        lambda: ga.append(np.array([1.0, 2.0, 3.0])),
        "`new_data` must be of size 2",
    )
    # __setitem__
    new_vals = np.array([10, 11])
    ga[0] = new_vals
    np.testing.assert_equal(ga.data, np.append(new_vals, np.arange(2, 10)))
    ga_copy = copy.copy(ga)
    ga_copy.data[0] = 900
    assert ga.data[0] == 10
    assert ga.indptr is ga_copy.indptr
