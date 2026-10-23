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


def _expanding_stat(x: np.ndarray, stat: str, skipna: bool = False) -> np.ndarray:
    return getattr(_expanding, f"expanding_{stat}")(x, skipna)


def _expanding_docstring(*args, **kwargs) -> Callable:
    base_docstring = """Compute the {} of the input array.

    Args:
        x (np.ndarray): Input array.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), any NaN value causes the result to be NaN,
            maintaining backwards compatibility. When True, NaN values are
            ignored (matching pandas default behavior).

    Returns:
        np.ndarray: Array with the expanding statistic

    Examples:
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        >>> # Default behavior: NaN propagates
        >>> {}(x)
        array([1., 1.5, nan, nan, nan])
        >>> # With skipna=True: NaN values are excluded
        >>> {}(x, skipna=True)
        array([1., 1.5, 1.5, 2.33..., 3.0])
    """

    def docstring_decorator(function: Callable):
        fname = function.__name__
        function.__doc__ = base_docstring.format(fname, fname, fname)
        return function

    return docstring_decorator(*args, **kwargs)


@_expanding_docstring
def expanding_mean(x: np.ndarray, skipna: bool = False) -> np.ndarray:
    return _expanding_stat(x, "mean", skipna)


@_expanding_docstring
def expanding_std(x: np.ndarray, skipna: bool = False) -> np.ndarray:
    return _expanding_stat(x, "std", skipna)


@_expanding_docstring
def expanding_min(x: np.ndarray, skipna: bool = False) -> np.ndarray:
    return _expanding_stat(x, "min", skipna)


@_expanding_docstring
def expanding_max(x: np.ndarray, skipna: bool = False) -> np.ndarray:
    return _expanding_stat(x, "max", skipna)


def expanding_quantile(x: np.ndarray, p: float, skipna: bool = False) -> np.ndarray:
    """Compute the expanding_quantile of the input array.

    Args:
        x (np.ndarray): Input array.
        p (float): Quantile to compute.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), any NaN value causes the result to be NaN,
            maintaining backwards compatibility. When True, NaN values are
            ignored (matching pandas default behavior).

    Returns:
        np.ndarray: Array with the expanding statistic

    Examples:
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        >>> # Default behavior: NaN propagates
        >>> expanding_quantile(x, 0.5)
        array([1., 1.5, nan, nan, nan])
        >>> # With skipna=True: NaN values are excluded
        >>> expanding_quantile(x, 0.5, skipna=True)
        array([1., 1.5, 1.5, 2., 2.5])
    """
    return _expanding.expanding_quantile(x, p, skipna)
