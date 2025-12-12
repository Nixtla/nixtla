---
title: Grouped Array
description: Something abou `Grouped Array`
---

##

::: mlforecast.grouped_array.GroupedArray

```python
import copy

from fastcore.test import test_eq, test_fail
```

```python
# The `GroupedArray` is used internally for storing the series values and performing transformations.
data = np.arange(10, dtype=np.float32)
indptr = np.array([0, 2, 10])  # group 1: [0, 1], group 2: [2..9]
ga = GroupedArray(data, indptr)
test_eq(len(ga), 2)
test_eq(str(ga), 'GroupedArray(ndata=10, n_groups=2)')
```

```python
# Iterate through the groups
ga_iter = iter(ga)
np.testing.assert_equal(next(ga_iter), np.array([0, 1]))
np.testing.assert_equal(next(ga_iter), np.arange(2, 10))
```

```python
# Take the last two observations from every group
last_2 = ga.take_from_groups(slice(-2, None))
np.testing.assert_equal(last_2.data, np.array([0, 1, 8, 9]))
np.testing.assert_equal(last_2.indptr, np.array([0, 2, 4]))
```

```python
# Take the last four observations from every group. Note that since group 1 only has two elements, only these are returned.
last_4 = ga.take_from_groups(slice(-4, None))
np.testing.assert_equal(last_4.data, np.array([0, 1, 6, 7, 8, 9]))
np.testing.assert_equal(last_4.indptr, np.array([0, 2, 6]))
```

```python
# Select a specific subset of groups
indptr = np.array([0, 2, 4, 7, 10])
ga2 = GroupedArray(data, indptr)
subset = ga2.take([0, 2])
np.testing.assert_allclose(subset[0].data, ga2[0].data)
np.testing.assert_allclose(subset[1].data, ga2[2].data)
```

```python
# The groups are [0, 1], [2, ..., 9]. expand_target(2) should take rolling pairs of them and fill with nans when there aren't enough
np.testing.assert_equal(
    ga.expand_target(2),
    np.array([
        [0, 1],
        [1, np.nan],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, np.nan]
    ])
)
```

```python
# append
combined = ga.append(np.array([-1, -2]))
np.testing.assert_equal(
    combined.data,
    np.hstack([ga.data[:2], np.array([-1]), ga.data[2:], np.array([-2])]),
)
# try to append new values that don't match the number of groups
test_fail(lambda: ga.append(np.array([1., 2., 3.])), contains='`new_data` must be of size 2')
```

```python
# __setitem__
new_vals = np.array([10, 11])
ga[0] = new_vals
np.testing.assert_equal(ga.data, np.append(new_vals, np.arange(2, 10)))
```

```python
ga_copy = copy.copy(ga)
ga_copy.data[0] = 900
assert ga.data[0] == 10
assert ga.indptr is ga_copy.indptr
```
