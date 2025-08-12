__all__ = ['GroupedArray']


import concurrent.futures
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray
from utilsforecast.compat import njit

from .compat import shift_array
from .lag_transforms import _BaseLagTransform


@njit(nogil=True)
def _transform_series(data, indptr, updates_only, lag, func, *args) -> np.ndarray:
    """Shifts every group in `data` by `lag` and computes `func(shifted, *args)`.

    If `updates_only=True` only last value of the transformation for each group is returned,
    otherwise the full transformation is returned"""
    n_series = len(indptr) - 1
    if updates_only:
        out = np.empty_like(data[:n_series])
        for i in range(n_series):
            lagged = shift_array(data[indptr[i] : indptr[i + 1]], lag)
            out[i] = func(lagged, *args)[-1]
    else:
        out = np.empty_like(data)
        for i in range(n_series):
            lagged = shift_array(data[indptr[i] : indptr[i + 1]], lag)
            out[indptr[i] : indptr[i + 1]] = func(lagged, *args)
    return out


class GroupedArray:
    """Array made up of different groups. Can be thought of (and iterated) as a list of arrays.

    All the data is stored in a single 1d array `data`.
    The indices for the group boundaries are stored in another 1d array `indptr`."""

    def __init__(self, data: np.ndarray, indptr: np.ndarray):
        self.data = data
        self.indptr = indptr
        self.n_groups = len(indptr) - 1

    def __len__(self) -> int:
        return self.n_groups

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[self.indptr[idx] : self.indptr[idx + 1]]

    def __setitem__(self, idx: int, vals: np.ndarray):
        if self[idx].size != vals.size:
            raise ValueError(f"vals must be of size {self[idx].size}")
        self[idx][:] = vals

    def __copy__(self):
        return GroupedArray(self.data.copy(), self.indptr)

    def take(self, idxs: np.ndarray) -> "GroupedArray":
        idxs = np.asarray(idxs)
        ranges = [range(self.indptr[i], self.indptr[i + 1]) for i in idxs]
        items = [self.data[rng] for rng in ranges]
        sizes = np.array([item.size for item in items])
        data = np.hstack(items)
        indptr = np.append(0, sizes.cumsum())
        return GroupedArray(data, indptr)

    def apply_transforms(
        self,
        transforms: Mapping[str, Union[Tuple[Any, ...], _BaseLagTransform]],
        updates_only: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Apply the transformations using the main process.

        If `updates_only` then only the updates are returned.
        """
        results = {}
        offset = 1 if updates_only else 0
        if any(isinstance(tfm, _BaseLagTransform) for tfm in transforms.values()):
            core_ga = CoreGroupedArray(self.data, self.indptr)
        for tfm_name, tfm in transforms.items():
            if isinstance(tfm, _BaseLagTransform):
                if updates_only:
                    results[tfm_name] = tfm.update(core_ga)
                else:
                    results[tfm_name] = tfm.transform(core_ga)
            else:
                lag, tfm, *args = tfm
                results[tfm_name] = _transform_series(
                    self.data, self.indptr, updates_only, lag - offset, tfm, *args
                )
        return results

    def apply_multithreaded_transforms(
        self,
        transforms: Mapping[str, Union[Tuple[Any, ...], _BaseLagTransform]],
        num_threads: int,
        updates_only: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Apply the transformations using multithreading.

        If `updates_only` then only the updates are returned.
        """
        future_to_result = {}
        results = {}
        offset = 1 if updates_only else 0
        numba_tfms = {}
        core_tfms = {}
        for name, tfm in transforms.items():
            if isinstance(tfm, _BaseLagTransform):
                core_tfms[name] = tfm
            else:
                numba_tfms[name] = tfm
        if numba_tfms:
            with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
                for tfm_name, (lag, tfm, *args) in numba_tfms.items():
                    future = executor.submit(
                        _transform_series,
                        self.data,
                        self.indptr,
                        updates_only,
                        lag - offset,
                        tfm,
                        *args,
                    )
                    future_to_result[future] = tfm_name
                for future in concurrent.futures.as_completed(future_to_result):
                    tfm_name = future_to_result[future]
                    results[tfm_name] = future.result()
        if core_tfms:
            core_ga = CoreGroupedArray(self.data, self.indptr, num_threads)
            for name, tfm in core_tfms.items():
                if updates_only:
                    results[name] = tfm.update(core_ga)
                else:
                    results[name] = tfm.transform(core_ga)
        return results

    def expand_target(self, max_horizon: int) -> np.ndarray:
        out = np.full_like(
            self.data, np.nan, shape=(self.data.size, max_horizon), order="F"
        )
        for j in range(max_horizon):
            for i in range(self.n_groups):
                if self.indptr[i + 1] - self.indptr[i] > j:
                    out[self.indptr[i] : self.indptr[i + 1] - j, j] = self.data[
                        self.indptr[i] + j : self.indptr[i + 1]
                    ]
        return out

    def take_from_groups(self, idx: Union[int, slice]) -> "GroupedArray":
        """Takes `idx` from each group in the array."""
        ranges = [
            range(self.indptr[i], self.indptr[i + 1])[idx] for i in range(self.n_groups)
        ]
        items = [self.data[rng] for rng in ranges]
        sizes = np.array([item.size for item in items])
        data = np.hstack(items)
        indptr = np.append(0, sizes.cumsum())
        return GroupedArray(data, indptr)

    def append(self, new_data: np.ndarray) -> "GroupedArray":
        """Appends each element of `new_data` to each existing group. Returns a copy."""
        if new_data.size != self.n_groups:
            raise ValueError(f"`new_data` must be of size {self.n_groups:,}")
        core_ga = CoreGroupedArray(self.data, self.indptr)
        new_data = new_data.astype(self.data.dtype, copy=False)
        new_indptr = np.arange(self.n_groups + 1, dtype=np.int32)
        new_ga = CoreGroupedArray(new_data, new_indptr)
        combined = core_ga._append(new_ga)
        return GroupedArray(combined.data, combined.indptr)

    def append_several(
        self, new_sizes: np.ndarray, new_values: np.ndarray, new_groups: np.ndarray
    ) -> "GroupedArray":
        new_data = np.empty(self.data.size + new_values.size, dtype=self.data.dtype)
        new_indptr = np.empty(new_sizes.size + 1, dtype=self.indptr.dtype)
        new_indptr[0] = 0
        old_indptr_idx = 0
        new_vals_idx = 0
        for i, is_new in enumerate(new_groups):
            new_size = new_sizes[i]
            if is_new:
                old_size = 0
            else:
                prev_slice = slice(
                    self.indptr[old_indptr_idx], self.indptr[old_indptr_idx + 1]
                )
                old_indptr_idx += 1
                old_size = prev_slice.stop - prev_slice.start
                new_size += old_size
                new_data[new_indptr[i] : new_indptr[i] + old_size] = self.data[
                    prev_slice
                ]
            new_indptr[i + 1] = new_indptr[i] + new_size
            new_data[new_indptr[i] + old_size : new_indptr[i + 1]] = new_values[
                new_vals_idx : new_vals_idx + new_sizes[i]
            ]
            new_vals_idx += new_sizes[i]
        return GroupedArray(new_data, new_indptr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ndata={self.data.size}, n_groups={self.n_groups})"
