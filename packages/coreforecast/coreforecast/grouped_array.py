import ctypes

import numpy as np


_LIB = ctypes.CDLL("build/libcoreforecast.so")


class GroupedArray:
    def __init__(self, data: np.ndarray, indptr: np.ndarray):
        self.data = data
        self.indptr = indptr
        self._handle = ctypes.c_void_p()
        _LIB.GroupedArray_CreateFromArrays(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(data.size),
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(indptr.size),
            ctypes.byref(self._handle),
        )

    def __del__(self):
        _LIB.GroupedArray_Delete(self._handle)

    def __len__(self):
        return self.indptr.size - 1

    def scaler_fit(self, stats_fn_name: str) -> None:
        stats = np.empty((len(self), 2), dtype=np.float64)
        stats_fn = getattr(_LIB, stats_fn_name)
        stats_fn(
            self._handle,
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return stats

    def scaler_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.full_like(self.data, np.nan)
        _LIB.GroupedArray_ScalerTransform(
            self._handle,
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out

    def scaler_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB.GroupedArray_ScalerInverseTransform(
            self._handle,
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out
