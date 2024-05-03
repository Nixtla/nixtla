__all__ = ["exponentially_weighted_mean"]

import ctypes

import numpy as np

from ._lib import _LIB
from .utils import (
    _ensure_float,
    _float_arr_to_prefix,
    _data_as_void_ptr,
    _pyfloat_to_np_c,
)


def exponentially_weighted_mean(x: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the exponentially weighted mean of the input array.

    Args:
        x (np.ndarray): Input array.
        alpha (float): Weight parameter.

    Returns:
        np.ndarray: Array with the exponentially weighted mean.
    """
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    _LIB[f"{prefix}_ExponentiallyWeightedMeanTransform"](
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(alpha, x.dtype),
        _data_as_void_ptr(out),
    )
    return out
