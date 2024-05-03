__all__ = [
    "expanding_mean",
    "expanding_std",
    "expanding_min",
    "expanding_max",
    "expanding_quantile",
]

import ctypes
from typing import Callable

import numpy as np

from ._lib import _LIB
from .utils import (
    _data_as_void_ptr,
    _ensure_float,
    _float_arr_to_prefix,
    _pyfloat_to_np_c,
)


def _expanding_stat(x: np.ndarray, stat: str) -> np.ndarray:
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    _LIB[f"{prefix}_Expanding{stat}Transform"](
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _data_as_void_ptr(out),
    )
    return out


def _expanding_docstring(*args, **kwargs) -> Callable:
    base_docstring = """Compute the {} of the input array.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array with the expanding statistic
    """

    def docstring_decorator(function: Callable):
        function.__doc__ = base_docstring.format(function.__name__)
        return function

    return docstring_decorator(*args, **kwargs)


@_expanding_docstring
def expanding_mean(x: np.ndarray) -> np.ndarray:
    return _expanding_stat(x, "Mean")


@_expanding_docstring
def expanding_std(x: np.ndarray) -> np.ndarray:
    return _expanding_stat(x, "Std")


@_expanding_docstring
def expanding_min(x: np.ndarray) -> np.ndarray:
    return _expanding_stat(x, "Min")


@_expanding_docstring
def expanding_max(x: np.ndarray) -> np.ndarray:
    return _expanding_stat(x, "Max")


def expanding_quantile(x: np.ndarray, p: float) -> np.ndarray:
    """Compute the expanding_quantile of the input array.

    Args:
        x (np.ndarray): Input array.
        p (float): Quantile to compute.

    Returns:
        np.ndarray: Array with the expanding statistic
    """
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    _LIB[f"{prefix}_ExpandingQuantileTransform"](
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(p, x.dtype),
        _data_as_void_ptr(out),
    )
    return out
