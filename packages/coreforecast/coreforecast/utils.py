from typing import Union

import ctypes

import numpy as np


def _data_as_void_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))


def _ensure_float(x: np.ndarray) -> np.ndarray:
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float32)
    return x


def _pyfloat_to_np_c(x: float, t: np.dtype) -> Union[ctypes.c_float, ctypes.c_double]:
    if t == np.float32:
        return ctypes.c_float(x)
    return ctypes.c_double(x)


def _float_arr_to_prefix(x: np.ndarray) -> str:
    if x.dtype == np.float32:
        prefix = "Float32"
    else:
        prefix = "Float64"
    return prefix
