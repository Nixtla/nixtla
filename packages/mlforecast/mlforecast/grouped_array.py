__all__ = ["GroupedArray"]


import concurrent.futures
import warnings
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray
from utilsforecast.compat import njit

from .compat import shift_array
from .lag_transforms import _BaseLagTransform


def _shift_py(x: np.ndarray, offset: int) -> np.ndarray:
    n = x.size
    out = np.empty_like(x)
    out[:offset] = np.nan
    out[offset:] = x[: n - offset]
    return out


def _base_transform_series(
    shift_fn, data, indptr, updates_only, lag, func, *args
) -> np.ndarray:
    """Shifts every group in `data` by `lag` and computes `func(shifted, *args)`.

    If `updates_only=True` only last value of the transformation for each group is returned,
    otherwise the full transformation is returned"""
    n_series = len(indptr) - 1
    if updates_only:
        out = np.empty_like(data[:n_series])
        for i in range(n_series):
            lagged = shift_fn(data[indptr[i] : indptr[i + 1]], lag)
            out[i] = func(lagged, *args)[-1]
    else:
        out = np.empty_like(data)
        for i in range(n_series):
            lagged = shift_fn(data[indptr[i] : indptr[i + 1]], lag)
            out[indptr[i] : indptr[i + 1]] = func(lagged, *args)
    return out


_jitted_transform_series = njit(_base_transform_series, nogil=True)


def _transform_series(data, indptr, updates_only, lag, func, *args) -> np.ndarray:
    return _jitted_transform_series(
        shift_array, data, indptr, updates_only, lag, func, *args
    )


def _transform_series_nojit(data, indptr, updates_only, lag, func, *args) -> np.ndarray:
    return _base_transform_series(
        _shift_py, data, indptr, updates_only, lag, func, *args
    )


def _sizes_to_indptr(sizes: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Builds the boundaries of consecutive groups with the given sizes."""
    indptr = np.empty(sizes.size + 1, dtype=dtype)
    indptr[0] = 0
    indptr[1:] = np.cumsum(sizes)
    return indptr


def _gather_idxs(starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Positions of the elements of the groups that start at `starts` and have `sizes` elements.

    They're ordered by group, so indexing with them lays the groups out contiguously."""
    ends = np.cumsum(sizes)
    total = int(ends[-1]) if ends.size else 0
    return np.repeat(starts - (ends - sizes), sizes) + np.arange(
        total, dtype=starts.dtype
    )


def _clip_slice_bound(
    bound: Optional[int], sizes: np.ndarray, default: np.ndarray
) -> np.ndarray:
    """Resolves a slice bound against each group size, as python slicing would."""
    if bound is None:
        return default
    if bound < 0:
        return np.maximum(sizes + bound, 0)
    return np.minimum(sizes, bound)


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
        if (idxs < 0).any():
            raise IndexError("negative group indices are not supported")
        starts = self.indptr[idxs]
        sizes = self.indptr[idxs + 1] - starts
        indptr = _sizes_to_indptr(sizes, self.indptr.dtype)
        return GroupedArray(self.data[_gather_idxs(starts, sizes)], indptr)

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
                if hasattr(tfm, "nopython_signatures"):
                    series_tfm_fn = _transform_series
                else:
                    series_tfm_fn = _transform_series_nojit
                results[tfm_name] = series_tfm_fn(
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
        py_tfms = {}
        for name, tfm in transforms.items():
            if isinstance(tfm, _BaseLagTransform):
                core_tfms[name] = tfm
            else:
                tfm_fn = tfm[1]
                if hasattr(tfm_fn, "nopython_signatures"):
                    numba_tfms[name] = tfm
                else:
                    py_tfms[name] = tfm
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
        if py_tfms:
            warnings.warn("Non-numba transforms are computed sequentially")
            results.update(self.apply_transforms(py_tfms, updates_only))
        return results

    def expand_target(self, max_horizon: int) -> np.ndarray:
        n = self.data.size
        out = np.full_like(self.data, np.nan, shape=(n, max_horizon), order="F")
        # elements from each position to the end of its group, i.e. the horizons it can fill
        remaining = np.repeat(self.indptr[1:], np.diff(self.indptr)) - np.arange(
            n, dtype=self.indptr.dtype
        )
        for j in range(min(max_horizon, n)):
            # shifting the whole array leaks values across groups, the mask restores them
            col = out[:, j]
            col[: n - j] = self.data[j:]
            col[remaining <= j] = np.nan
        return out

    def take_from_groups(self, idx: Union[int, slice]) -> "GroupedArray":
        """Takes `idx` from each group in the array."""
        group_sizes = np.diff(self.indptr)
        if isinstance(idx, slice):
            if idx.step not in (None, 1):
                raise NotImplementedError("stepped slices are not supported")
            starts = _clip_slice_bound(
                idx.start, group_sizes, np.zeros_like(group_sizes)
            )
            stops = _clip_slice_bound(idx.stop, group_sizes, group_sizes)
            sizes = np.maximum(stops - starts, 0)
            data = self.data[_gather_idxs(self.indptr[:-1] + starts, sizes)]
        else:
            positions = np.where(idx < 0, group_sizes + idx, idx)
            if ((positions < 0) | (positions >= group_sizes)).any():
                raise IndexError(f"index {idx} is out of range for some groups")
            data = self.data[self.indptr[:-1] + positions]
            sizes = np.ones_like(group_sizes)
        return GroupedArray(data, _sizes_to_indptr(sizes, self.indptr.dtype))

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
        new_sizes = np.asarray(new_sizes)
        new_groups = np.asarray(new_groups, dtype=bool)
        old_sizes = np.zeros(new_sizes.size, dtype=self.indptr.dtype)
        old_sizes[~new_groups] = np.diff(self.indptr)
        new_indptr = _sizes_to_indptr(old_sizes + new_sizes, self.indptr.dtype)
        new_data = np.empty(self.data.size + new_values.size, dtype=self.data.dtype)
        new_data[_gather_idxs(new_indptr[:-1], old_sizes)] = self.data
        new_data[_gather_idxs(new_indptr[:-1] + old_sizes, new_sizes)] = new_values
        return GroupedArray(new_data, new_indptr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ndata={self.data.size}, n_groups={self.n_groups})"
