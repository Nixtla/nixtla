import copy

import numpy as np
import pytest

from mlforecast.grouped_array import GroupedArray

from .conftest import assert_raises_with_message


def test_grouped_array_append_several():
    data = np.arange(5)
    indptr = np.array([0, 2, 5])
    new_sizes = np.array([0, 2, 1])
    new_values = np.array([6, 7, 5])
    new_groups = np.array([False, True, False])
    new_ga = GroupedArray(data, indptr).append_several(
        new_sizes, new_values, new_groups
    )
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


# references for the randomized checks below: the per-group python loops that
# the vectorized implementations replaced
def _ref_take(ga, idxs):
    ranges = [range(ga.indptr[i], ga.indptr[i + 1]) for i in idxs]
    items = [ga.data[rng] for rng in ranges]
    sizes = np.array([item.size for item in items])
    return GroupedArray(np.hstack(items), np.append(0, sizes.cumsum()))


def _ref_take_from_groups(ga, idx):
    ranges = [range(ga.indptr[i], ga.indptr[i + 1])[idx] for i in range(ga.n_groups)]
    items = [ga.data[rng] for rng in ranges]
    sizes = np.array([item.size for item in items])
    return GroupedArray(np.hstack(items), np.append(0, sizes.cumsum()))


def _ref_expand_target(ga, max_horizon):
    out = np.full_like(ga.data, np.nan, shape=(ga.data.size, max_horizon), order="F")
    for j in range(max_horizon):
        for i in range(ga.n_groups):
            if ga.indptr[i + 1] - ga.indptr[i] > j:
                out[ga.indptr[i] : ga.indptr[i + 1] - j, j] = ga.data[
                    ga.indptr[i] + j : ga.indptr[i + 1]
                ]
    return out


def _ref_append_several(ga, new_sizes, new_values, new_groups):
    new_data = np.empty(ga.data.size + new_values.size, dtype=ga.data.dtype)
    new_indptr = np.empty(new_sizes.size + 1, dtype=ga.indptr.dtype)
    new_indptr[0] = 0
    old_indptr_idx = 0
    new_vals_idx = 0
    for i, is_new in enumerate(new_groups):
        new_size = new_sizes[i]
        if is_new:
            old_size = 0
        else:
            prev_slice = slice(ga.indptr[old_indptr_idx], ga.indptr[old_indptr_idx + 1])
            old_indptr_idx += 1
            old_size = prev_slice.stop - prev_slice.start
            new_size += old_size
            new_data[new_indptr[i] : new_indptr[i] + old_size] = ga.data[prev_slice]
        new_indptr[i + 1] = new_indptr[i] + new_size
        new_data[new_indptr[i] + old_size : new_indptr[i + 1]] = new_values[
            new_vals_idx : new_vals_idx + new_sizes[i]
        ]
        new_vals_idx += new_sizes[i]
    return GroupedArray(new_data, new_indptr)


def _random_ga(rng, dtype, n_groups=200, max_size=60, min_size=1):
    sizes = rng.integers(min_size, max_size + 1, size=n_groups)
    indptr = np.append(0, sizes.cumsum()).astype(np.int32)
    return GroupedArray(rng.normal(size=int(indptr[-1])).astype(dtype), indptr)


def _assert_same_ga(actual, expected):
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    assert actual.data.dtype == expected.data.dtype


# min_size=0 gives empty groups
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("min_size", [1, 0])
def test_take_matches_loop(dtype, min_size):
    rng = np.random.default_rng(0)
    ga = _random_ga(rng, dtype, min_size=min_size)
    for idxs in (
        rng.permutation(ga.n_groups)[:50],
        np.arange(ga.n_groups),
        rng.integers(0, ga.n_groups, size=20),  # repeated groups
        np.array([7]),
    ):
        subset = ga.take(idxs)
        _assert_same_ga(subset, _ref_take(ga, idxs))
        assert subset.indptr.dtype == ga.indptr.dtype
    # the loop can't build these
    empty = ga.take(np.array([], dtype=np.int64))
    assert empty.n_groups == 0
    assert empty.data.size == 0
    with pytest.raises(IndexError):
        ga.take(np.array([-1]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("min_size", [1, 0])
@pytest.mark.parametrize(
    "idx",
    [
        slice(-7, None),
        slice(-100, None),  # more than any group has
        slice(None, 5),
        slice(4, None),
        slice(2, 9),
        slice(-9, -2),
        slice(-2, -9),  # empty
        slice(None, None),
        0,
        -1,
        3,  # out of range for the short groups
        -4,
    ],
)
def test_take_from_groups_matches_loop(dtype, min_size, idx):
    ga = _random_ga(np.random.default_rng(0), dtype, min_size=min_size)
    try:
        expected = _ref_take_from_groups(ga, idx)
    except IndexError:
        with pytest.raises(IndexError):
            ga.take_from_groups(idx)
        return
    subset = ga.take_from_groups(idx)
    _assert_same_ga(subset, expected)
    assert subset.indptr.dtype == ga.indptr.dtype


def test_take_from_groups_rejects_stepped_slices():
    ga = _random_ga(np.random.default_rng(0), np.float32)
    with pytest.raises(NotImplementedError):
        ga.take_from_groups(slice(None, None, 2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("min_size", [1, 0])
@pytest.mark.parametrize(
    "n_groups,max_size,max_horizon",
    [(200, 60, 1), (200, 60, 14), (200, 60, 100), (3, 4, 100)],  # last exceeds the data
)
def test_expand_target_matches_loop(dtype, min_size, n_groups, max_size, max_horizon):
    ga = _random_ga(np.random.default_rng(0), dtype, n_groups, max_size, min_size)
    expanded = ga.expand_target(max_horizon)
    expected = _ref_expand_target(ga, max_horizon)
    np.testing.assert_array_equal(expanded, expected)
    assert expanded.dtype == expected.dtype
    assert expanded.flags.f_contiguous


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("min_size", [1, 0])
def test_append_several_matches_loop(dtype, min_size):
    rng = np.random.default_rng(0)
    ga = _random_ga(rng, dtype, min_size=min_size)
    n_new = 20
    new_groups = np.zeros(ga.n_groups + n_new, dtype=bool)
    new_groups[rng.permutation(new_groups.size)[:n_new]] = True
    new_sizes = rng.integers(0, 5, size=new_groups.size).astype(np.int32)
    new_sizes[new_groups] = np.maximum(new_sizes[new_groups], 1)
    new_values = rng.normal(size=int(new_sizes.sum())).astype(dtype)
    appended = ga.append_several(new_sizes, new_values, new_groups)
    _assert_same_ga(
        appended, _ref_append_several(ga, new_sizes, new_values, new_groups)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_grouped_array(dtype):
    ga = GroupedArray(np.array([], dtype=dtype), np.array([0], dtype=np.int32))
    for subset in (
        ga.take(np.array([], dtype=np.int64)),
        ga.take_from_groups(slice(-7, None)),
        ga.take_from_groups(0),
    ):
        assert subset.n_groups == 0
        assert subset.data.size == 0
        assert subset.data.dtype == dtype
        assert subset.indptr.dtype == np.int32
    assert ga.expand_target(3).shape == (0, 3)
    new_values = np.arange(3, dtype=dtype)
    appended = ga.append_several(np.array([1, 2]), new_values, np.array([True, True]))
    _assert_same_ga(
        appended, GroupedArray(new_values, np.array([0, 1, 3], dtype=np.int32))
    )
