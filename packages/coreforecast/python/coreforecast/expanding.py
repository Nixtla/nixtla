__all__ = [
    "expanding_mean",
    "expanding_std",
    "expanding_min",
    "expanding_max",
    "expanding_quantile",
]

from typing import Callable

import numpy as np

from ._lib import expanding as _expanding


def _expanding_stat(x: np.ndarray, stat: str) -> np.ndarray:
    return getattr(_expanding, f"expanding_{stat}")(x)


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
    return _expanding_stat(x, "mean")


@_expanding_docstring
def expanding_std(x: np.ndarray) -> np.ndarray:
    return _expanding_stat(x, "std")


@_expanding_docstring
def expanding_min(x: np.ndarray) -> np.ndarray:
    return _expanding_stat(x, "min")


@_expanding_docstring
def expanding_max(x: np.ndarray) -> np.ndarray:
    return _expanding_stat(x, "max")


def expanding_quantile(x: np.ndarray, p: float) -> np.ndarray:
    """Compute the expanding_quantile of the input array.

    Args:
        x (np.ndarray): Input array.
        p (float): Quantile to compute.

    Returns:
        np.ndarray: Array with the expanding statistic
    """
    return _expanding.expanding_quantile(x, p)
