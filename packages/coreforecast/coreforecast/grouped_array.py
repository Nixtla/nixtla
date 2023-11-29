import ctypes
import platform
import sys
from typing import Union

import numpy as np

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


if platform.system() in ("Windows", "Microsoft"):
    prefix = "Release"
    extension = "dll"
else:
    prefix = ""
    extension = "so"

_LIB = ctypes.CDLL(
    str(files("coreforecast").joinpath("lib", prefix, f"libcoreforecast.{extension}"))
)


def _data_as_void_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))


class GroupedArray:
    def __init__(self, data: np.ndarray, indptr: np.ndarray, num_threads: int = 1):
        data = np.ascontiguousarray(data, dtype=data.dtype)
        if data.dtype == np.float32:
            self.prefix = "GroupedArrayFloat32"
        elif data.dtype == np.float64:
            self.prefix = "GroupedArrayFloat64"
        else:
            self.prefix = "GroupedArrayFloat32"
            data = data.astype(np.float32)
        self.data = data
        if indptr.dtype != np.int32:
            indptr = indptr.astype(np.int32)
        self.indptr = indptr
        self._handle = ctypes.c_void_p()
        _LIB[f"{self.prefix}_Create"](
            _data_as_void_ptr(data),
            ctypes.c_int32(data.size),
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(indptr.size),
            ctypes.c_int(num_threads),
            ctypes.byref(self._handle),
        )

    def __del__(self):
        _LIB[f"{self.prefix}_Delete"](self._handle)

    def __len__(self):
        return self.indptr.size - 1

    def __getitem__(self, i):
        return self.data[self.indptr[i] : self.indptr[i + 1]]

    def _pyfloat_to_c(self, x: float) -> Union[ctypes.c_float, ctypes.c_double]:
        if self.prefix == "GroupedArrayFloat32":
            out = ctypes.c_float(x)
        else:
            out = ctypes.c_double(x)
        return out

    def scaler_fit(self, scaler_type: str) -> np.ndarray:
        stats = np.empty_like(self.data, shape=(len(self), 2))
        _LIB[f"{self.prefix}_{scaler_type}ScalerStats"](
            self._handle,
            _data_as_void_ptr(stats),
        )
        return stats

    def scaler_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ScalerTransform"](
            self._handle,
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def scaler_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ScalerInverseTransform"](
            self._handle,
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def take_from_groups(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_TakeFromGroups"](
            self._handle,
            ctypes.c_int(k),
            _data_as_void_ptr(out),
        )
        return out

    def lag_transform(self, lag: int) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_LagTransform"](
            self._handle,
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
        )
        return out

    def rolling_transform(
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

    def rolling_quantile_transform(
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

    def rolling_update(
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

    def rolling_quantile_update(
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

    def seasonal_rolling_transform(
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

    def seasonal_rolling_update(
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

    def seasonal_rolling_quantile_transform(
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

    def seasonal_rolling_quantile_update(
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

    def expanding_transform_with_aggs(
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

    def expanding_transform(
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

    def expanding_quantile_transform(self, lag: int, p: float) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ExpandingQuantileTransform"](
            self._handle,
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            _data_as_void_ptr(out),
        )
        return out

    def expanding_quantile_update(self, lag: int, p: float) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_ExpandingQuantileUpdate"](
            self._handle,
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            _data_as_void_ptr(out),
        )
        return out

    def exponentially_weighted_transform(
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
