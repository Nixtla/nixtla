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

from typing import Callable, Optional

import numpy as np

from ._lib import rolling as _rolling


def _rolling_stat(
    x: np.ndarray,
    stat: str,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    if min_samples is None:
        min_samples = window_size
    return getattr(_rolling, f"rolling_{stat}")(x, window_size, min_samples)


def _seasonal_rolling_stat(
    x: np.ndarray,
    stat: str,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    if min_samples is None:
        min_samples = window_size
    return getattr(_rolling, f"seasonal_rolling_{stat}")(
        x, season_length, window_size, min_samples
    )


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
    return _rolling_stat(x, "mean", window_size, min_samples)


@_rolling_docstring
def rolling_std(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "std", window_size, min_samples)


@_rolling_docstring
def rolling_min(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "min", window_size, min_samples)


@_rolling_docstring
def rolling_max(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "max", window_size, min_samples)


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
    if min_samples is None:
        min_samples = window_size
    return _rolling.rolling_quantile(x, window_size, min_samples, p)


@_seasonal_rolling_docstring
def seasonal_rolling_mean(
    x: np.ndarray,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    return _seasonal_rolling_stat(x, "mean", season_length, window_size, min_samples)


@_seasonal_rolling_docstring
def seasonal_rolling_std(
    x: np.ndarray,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    return _seasonal_rolling_stat(x, "std", season_length, window_size, min_samples)


@_seasonal_rolling_docstring
def seasonal_rolling_min(
    x: np.ndarray,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    return _seasonal_rolling_stat(x, "min", season_length, window_size, min_samples)


@_seasonal_rolling_docstring
def seasonal_rolling_max(
    x: np.ndarray,
    season_length: int,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    return _seasonal_rolling_stat(x, "max", season_length, window_size, min_samples)


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
    if min_samples is None:
        min_samples = window_size
    return _rolling.seasonal_rolling_quantile(
        x, season_length, window_size, min_samples, p
    )
