__all__ = [
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_quantile",
    "seasonal_rolling_mean",
    "seasonal_rolling_std",
    "seasonal_rolling_min",
    "seasonal_rolling_max",
    "seasonal_rolling_quantile",
]

import ctypes
from typing import Callable, Optional

import numpy as np

from ._lib import _LIB
from .utils import (
    _data_as_void_ptr,
    _ensure_float,
    _float_arr_to_prefix,
    _pyfloat_to_np_c,
)


def _rolling_stat(
    x: np.ndarray,
    stat: str,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    if min_samples is None:
        min_samples = window_size
    _LIB[f"{prefix}_Rolling{stat}Transform"](
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        ctypes.c_int(window_size),
        ctypes.c_int(min_samples),
        _data_as_void_ptr(out),
    )
    return out


def _seasonal_rolling_stat(
    x: np.ndarray,
    stat: str,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    if min_samples is None:
        min_samples = window_size
    _LIB[f"{prefix}_SeasonalRolling{stat}Transform"](
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        ctypes.c_int(season_length),
        ctypes.c_int(window_size),
        ctypes.c_int(min_samples),
        _data_as_void_ptr(out),
    )
    return out


def _rolling_docstring(*args, **kwargs) -> Callable:
    base_docstring = """Compute the {} of the input array.

    Args:
        x (np.ndarray): Input array.
        window_size (int): The size of the rolling window.
        min_samples (int, optional): The minimum number of samples required to compute the statistic.
            If None, it is set to `window_size`.

    Returns:
        np.ndarray: Array with the rolling statistic
    """

    def docstring_decorator(function: Callable):
        function.__doc__ = base_docstring.format(function.__name__)
        return function

    return docstring_decorator(*args, **kwargs)


def _seasonal_rolling_docstring(*args, **kwargs) -> Callable:
    base_docstring = """Compute the {} of the input array

    Args:
        x (np.ndarray): Input array.
        season_length (int): The length of the seasonal period.
        window_size (int): The size of the rolling window.
        min_samples (int, optional): The minimum number of samples required to compute the statistic.
            If None, it is set to `window_size`.

    Returns:
        np.ndarray: Array with the seasonal rolling statistic
    """

    def docstring_decorator(function: Callable):
        function.__doc__ = base_docstring.format(function.__name__)
        return function

    return docstring_decorator(*args, **kwargs)


@_rolling_docstring
def rolling_mean(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "Mean", window_size, min_samples)


@_rolling_docstring
def rolling_std(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "Std", window_size, min_samples)


@_rolling_docstring
def rolling_min(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "Min", window_size, min_samples)


@_rolling_docstring
def rolling_max(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "Max", window_size, min_samples)


def rolling_quantile(
    x: np.ndarray, p: float, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    """Compute the rolling_quantile of the input array.

    Args:
        x (np.ndarray): Input array.
        q (float): Quantile to compute.
        window_size (int): The size of the rolling window.
        min_samples (int, optional): The minimum number of samples required to compute the statistic.
            If None, it is set to `window_size`.

    Returns:
        np.ndarray: Array with rolling statistic
    """
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    if min_samples is None:
        min_samples = window_size
    _LIB[f"{prefix}_RollingQuantileTransform"](
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(p, x.dtype),
        ctypes.c_int(window_size),
        ctypes.c_int(min_samples),
        _data_as_void_ptr(out),
    )
    return out


@_seasonal_rolling_docstring
def seasonal_rolling_mean(
    x: np.ndarray,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    return _seasonal_rolling_stat(x, "Mean", season_length, window_size, min_samples)


@_seasonal_rolling_docstring
def seasonal_rolling_std(
    x: np.ndarray,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    return _seasonal_rolling_stat(x, "Std", season_length, window_size, min_samples)


@_seasonal_rolling_docstring
def seasonal_rolling_min(
    x: np.ndarray,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    return _seasonal_rolling_stat(x, "Min", season_length, window_size, min_samples)


@_seasonal_rolling_docstring
def seasonal_rolling_max(
    x: np.ndarray,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    return _seasonal_rolling_stat(x, "Max", season_length, window_size, min_samples)


def seasonal_rolling_quantile(
    x: np.ndarray,
    p: float,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    """Compute the seasonal_rolling_quantile of the input array.

    Args:
        x (np.ndarray): Input array.
        q (float): Quantile to compute.
        season_length (int): The length of the seasonal period.
        window_size (int): The size of the rolling window.
        min_samples (int, optional): The minimum number of samples required to compute the statistic.
            If None, it is set to `window_size`.

    Returns:
        np.ndarray: Array with rolling statistic
    """
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    if min_samples is None:
        min_samples = window_size
    _LIB[f"{prefix}_SeasonalRollingQuantileTransform"](
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        ctypes.c_int(season_length),
        _pyfloat_to_np_c(p, x.dtype),
        ctypes.c_int(window_size),
        ctypes.c_int(min_samples),
        _data_as_void_ptr(out),
    )
    return out
