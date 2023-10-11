import ctypes
import platform
import sys

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


def _data_as_ptr(arr: np.ndarray, dtype):
    return arr.ctypes.data_as(ctypes.POINTER(dtype))


class GroupedArray:
    def __init__(self, data: np.ndarray, indptr: np.ndarray):
        self.data = data
        self.indptr = indptr
        self._handle = ctypes.c_void_p()
        _LIB.GroupedArray_CreateFromArrays(
            _data_as_ptr(data, ctypes.c_float),
            ctypes.c_int32(data.size),
            _data_as_ptr(indptr, ctypes.c_int32),
            ctypes.c_int32(indptr.size),
            ctypes.byref(self._handle),
        )

    def __del__(self):
        _LIB.GroupedArray_Delete(self._handle)

    def __len__(self):
        return self.indptr.size - 1

    def scaler_fit(self, stats_fn_name: str) -> np.ndarray:
        stats = np.empty((len(self), 2), dtype=np.float64)
        stats_fn = _LIB[stats_fn_name]
        stats_fn(
            self._handle,
            _data_as_ptr(stats, ctypes.c_double),
        )
        return stats

    def scaler_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.full_like(self.data, np.nan)
        _LIB.GroupedArray_ScalerTransform(
            self._handle,
            _data_as_ptr(stats, ctypes.c_double),
            _data_as_ptr(out, ctypes.c_float),
        )
        return out

    def scaler_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB.GroupedArray_ScalerInverseTransform(
            self._handle,
            _data_as_ptr(stats, ctypes.c_double),
            _data_as_ptr(out, ctypes.c_float),
        )
        return out
