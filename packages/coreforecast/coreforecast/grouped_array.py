import ctypes
import platform
import sys

import numpy as np

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


DTYPE_FLOAT32 = ctypes.c_int(0)
DTYPE_FLOAT64 = ctypes.c_int(1)

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
    def __init__(self, data: np.ndarray, indptr: np.ndarray):
        if data.dtype == np.float32:
            self.dtype = DTYPE_FLOAT32
        elif data.dtype == np.float64:
            self.dtype = DTYPE_FLOAT64
        else:
            self.dtype = DTYPE_FLOAT32
            data = data.astype(np.float32)
        self.data = data
        if indptr.dtype != np.int32:
            indptr = indptr.astype(np.int32)
        self.indptr = indptr
        self._handle = ctypes.c_void_p()
        _LIB.GroupedArray_CreateFromArrays(
            _data_as_void_ptr(data),
            ctypes.c_int32(data.size),
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(indptr.size),
            self.dtype,
            ctypes.byref(self._handle),
        )

    def __del__(self):
        _LIB.GroupedArray_Delete(self._handle, self.dtype)

    def __len__(self):
        return self.indptr.size - 1

    def __getitem__(self, i):
        return self.data[self.indptr[i] : self.indptr[i + 1]]

    def scaler_fit(self, stats_fn_name: str) -> np.ndarray:
        stats = np.full((len(self), 2), np.nan, dtype=self.data.dtype)
        stats_fn = _LIB[stats_fn_name]
        stats_fn(
            self._handle,
            self.dtype,
            _data_as_void_ptr(stats),
        )
        return stats

    def scaler_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.full_like(self.data, np.nan)
        _LIB.GroupedArray_ScalerTransform(
            self._handle,
            _data_as_void_ptr(stats),
            self.dtype,
            _data_as_void_ptr(out),
        )
        return out

    def scaler_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB.GroupedArray_ScalerInverseTransform(
            self._handle,
            _data_as_void_ptr(stats),
            self.dtype,
            _data_as_void_ptr(out),
        )
        return out
