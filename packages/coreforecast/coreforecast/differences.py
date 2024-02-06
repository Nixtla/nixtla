import ctypes

import numpy as np

from ._lib import _LIB, _indptr_t
from .utils import _data_as_void_ptr, _ensure_float, _float_arr_to_prefix

_LIB.Float32_NumDiffs.restype = ctypes.c_int
_LIB.Float64_NumDiffs.restype = ctypes.c_int
_LIB.Float32_NumSeasDiffs.restype = ctypes.c_int
_LIB.Float64_NumSeasDiffs.restype = ctypes.c_int


def num_diffs(x: np.ndarray, max_d: int = 1) -> int:
    """Find the optimal number of differences

    Args:
        x (np.ndarray): Array with the time series.
        max_d (int, optional): Maximum number of differences to consider. Defaults to 1.

    Returns:
        int: Optimal number of differences."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    return getattr(_LIB, f"{prefix}_NumDiffs")(
        _data_as_void_ptr(x),
        _indptr_t(x.size),
        ctypes.c_int(max_d),
    )


def num_seas_diffs(x: np.ndarray, season_length: int, max_d: int = 1) -> int:
    """Find the optimal number of seasonal differences

    Args:
        x (np.ndarray): Array with the time series.
        season_length (int): Length of the seasonal pattern.
        max_d (int, optional): Maximum number of differences to consider. Defaults to 1.

    Returns:
        int: Optimal number of seasonal differences."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    return getattr(_LIB, f"{prefix}_NumSeasDiffs")(
        _data_as_void_ptr(x),
        _indptr_t(x.size),
        ctypes.c_int(season_length),
        ctypes.c_int(max_d),
    )


def diff(x: np.ndarray, d: int) -> np.ndarray:
    """Subtract previous values of the series

    Args:
        x (np.ndarray): Array with the time series.
        d (int): Lag to subtract

    Returns:
        np.ndarray: Differenced time series."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    getattr(_LIB, f"{prefix}_Difference")(
        _data_as_void_ptr(x),
        _indptr_t(x.size),
        ctypes.c_int(d),
        _data_as_void_ptr(out),
    )
    return out
