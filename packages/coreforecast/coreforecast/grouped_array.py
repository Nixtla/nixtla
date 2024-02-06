import ctypes
from typing import Union

import numpy as np

from ._lib import _LIB, _indptr_dtype, _indptr_t
from .utils import _ensure_float, _data_as_void_ptr, _pyfloat_to_np_c


class GroupedArray:
    """Array of grouped data

    Args:
        data (np.ndarray): 1d array with the values.
        indptr (np.ndarray): 1d array with the group boundaries.
        num_threads (int): Number of threads to use when computing transformations."""

    def __init__(self, data: np.ndarray, indptr: np.ndarray, num_threads: int = 1):
        self.data = np.ascontiguousarray(data, dtype=data.dtype)
        self.data = _ensure_float(self.data)
        if self.data.dtype == np.float32:
            self.prefix = "GroupedArrayFloat32"
        else:
            self.prefix = "GroupedArrayFloat64"
        self.indptr = indptr.astype(_indptr_dtype, copy=False)
        self.num_threads = num_threads
        self._handle = ctypes.c_void_p()
        _LIB[f"{self.prefix}_Create"](
            _data_as_void_ptr(self.data),
            _indptr_t(self.data.size),
            self.indptr.ctypes.data_as(ctypes.POINTER(_indptr_t)),
            _indptr_t(self.indptr.size),
            ctypes.c_int(num_threads),
            ctypes.byref(self._handle),
        )

    def with_data(self, data: np.ndarray) -> "GroupedArray":
        data = data.astype(self.data.dtype, copy=False)
        data = np.ascontiguousarray(data)
        return GroupedArray(data, self.indptr, self.num_threads)

    def __del__(self):
        _LIB[f"{self.prefix}_Delete"](self._handle)

    def __len__(self):
        return self.indptr.size - 1

    def __getitem__(self, i):
        return self.data[self.indptr[i] : self.indptr[i + 1]]

    def _pyfloat_to_c(self, x: float) -> Union[ctypes.c_float, ctypes.c_double]:
        if self.prefix == "GroupedArrayFloat32":
            return ctypes.c_float(x)
        return ctypes.c_double(x)

    def _scaler_fit(self, scaler_type: str) -> np.ndarray:
        stats = np.empty_like(self.data, shape=(len(self), 2))
        _LIB[f"{self.prefix}_{scaler_type}ScalerStats"](
            self._handle,
            _data_as_void_ptr(stats),
        )
        return stats

    def _scaler_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ScalerTransform"](
            self._handle,
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def _scaler_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ScalerInverseTransform"](
            self._handle,
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def index_from_end(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_IndexFromEnd"](
            self._handle,
            ctypes.c_int(k),
            _data_as_void_ptr(out),
        )
        return out

    def _head(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=k * len(self))
        _LIB[f"{self.prefix}_Head"](
            self._handle,
            ctypes.c_int(k),
            _data_as_void_ptr(out),
        )
        return out

    def _tail(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=k * len(self))
        _LIB[f"{self.prefix}_Tail"](
            self._handle,
            ctypes.c_int(k),
            _data_as_void_ptr(out),
        )
        return out

    def _lag_transform(self, lag: int) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_LagTransform"](
            self._handle,
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
        )
        return out

    def _rolling_transform(
        self, stat_name: str, lag: int, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Rolling{stat_name}Transform"](
            self._handle,
            ctypes.c_int(lag),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _rolling_quantile_transform(
        self, lag: int, p: float, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_RollingQuantileTransform"](
            self._handle,
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _rolling_update(
        self, stat_name: str, lag: int, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_Rolling{stat_name}Update"](
            self._handle,
            ctypes.c_int(lag),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _rolling_quantile_update(
        self, lag: int, p: float, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_RollingQuantileUpdate"](
            self._handle,
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _seasonal_rolling_transform(
        self,
        stat_name: str,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_SeasonalRolling{stat_name}Transform"](
            self._handle,
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _seasonal_rolling_update(
        self,
        stat_name: str,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_SeasonalRolling{stat_name}Update"](
            self._handle,
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _seasonal_rolling_quantile_transform(
        self,
        lag: int,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_SeasonalRollingQuantileTransform"](
            self._handle,
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            self._pyfloat_to_c(p),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _seasonal_rolling_quantile_update(
        self,
        lag: int,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_SeasonalRollingQuantileUpdate"](
            self._handle,
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            self._pyfloat_to_c(p),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _expanding_transform_with_aggs(
        self,
        stat_name: str,
        lag: int,
        aggs: np.ndarray,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Expanding{stat_name}Transform"](
            self._handle,
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
            _data_as_void_ptr(aggs),
        )
        return out

    def _expanding_transform(
        self,
        stat_name: str,
        lag: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Expanding{stat_name}Transform"](
            self._handle,
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
        )
        return out

    def _expanding_quantile_transform(self, lag: int, p: float) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ExpandingQuantileTransform"](
            self._handle,
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            _data_as_void_ptr(out),
        )
        return out

    def _expanding_quantile_update(self, lag: int, p: float) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_ExpandingQuantileUpdate"](
            self._handle,
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            _data_as_void_ptr(out),
        )
        return out

    def _exponentially_weighted_transform(
        self,
        stat_name: str,
        lag: int,
        alpha: float,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ExponentiallyWeighted{stat_name}Transform"](
            self._handle,
            ctypes.c_int(lag),
            self._pyfloat_to_c(alpha),
            _data_as_void_ptr(out),
        )
        return out

    def _boxcox_fit(
        self, season_length: int, lower: float, upper: float, method: str
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=(len(self), 2))
        _LIB[f"{self.prefix}_BoxCoxLambda{method}"](
            self._handle,
            ctypes.c_int(season_length),
            _pyfloat_to_np_c(lower, self.data.dtype),
            _pyfloat_to_np_c(upper, self.data.dtype),
            _data_as_void_ptr(out),
        )
        return out

    def _boxcox_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_BoxCoxTransform"](
            self._handle,
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def _boxcox_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_BoxCoxInverseTransform"](
            self._handle,
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def _num_diffs(self, max_d: int = 1) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_NumDiffs"](
            self._handle,
            ctypes.c_int(max_d),
            _data_as_void_ptr(out),
        )
        return out

    def _num_seas_diffs(self, season_length: int, max_d: int = 1) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_NumSeasDiffs"](
            self._handle,
            ctypes.c_int(season_length),
            ctypes.c_int(max_d),
            _data_as_void_ptr(out),
        )
        return out

    def _diff(self, d: int) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Difference"](
            self._handle,
            ctypes.c_int(d),
            _data_as_void_ptr(out),
        )
        return out

    def _inv_diff(self, d: int, tails: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_InvertDifference"](
            self._handle,
            ctypes.c_int(d),
            _data_as_void_ptr(tails),
            _data_as_void_ptr(out),
        )
        return out

    def _conditional_diff(self, d: int, mask: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ConditionalDifference"](
            self._handle,
            ctypes.c_int(d),
            _data_as_void_ptr(mask),
            _data_as_void_ptr(out),
        )
        return out

    def _conditional_inv_diff(
        self, d: int, mask: np.ndarray, tails: np.ndarray
    ) -> np.ndarray:
        mask_with_tails = np.hstack([mask.reshape(-1, 1), tails.reshape(-1, d)])
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ConditionalInvertDifference"](
            self._handle,
            ctypes.c_int(d),
            _data_as_void_ptr(mask_with_tails),
            _data_as_void_ptr(out),
        )
        return out
