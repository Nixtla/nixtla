import ctypes
from typing import List, Tuple

import numpy as np

from ._lib import _LIB
from .grouped_array import GroupedArray
from .utils import (
    _data_as_void_ptr,
    _ensure_float,
    _float_arr_to_prefix,
    _pyfloat_to_np_c,
)


__all__ = [
    "LocalBoxCoxScaler",
    "LocalMinMaxScaler",
    "LocalRobustScaler",
    "LocalStandardScaler",
    "boxcox",
    "boxcox_lambda",
    "inv_boxcox",
]


_LIB.Float32_BoxCoxLambdaGuerrero.restype = ctypes.c_float
_LIB.Float64_BoxCoxLambdaGuerrero.restype = ctypes.c_double


def boxcox_lambda(
    x: np.ndarray,
    season_length: int,
    lower: float = -0.9,
    upper: float = 2.0,
    method: str = "guerrero",
) -> float:
    """Find optimum lambda for the Box-Cox transformation (supports negative numbers)

    Args:
        x (np.ndarray): Array with data to transform.
        season_length (int): Length of the seasonal period.
        lower (float): Lower bound for the lambda.
        upper (float): Upper bound for the lambda.
        method (str): Method to use. Valid options are 'guerrero'.

    Returns:
        float: Optimum lambda."""
    if method != "guerrero":
        raise NotImplementedError(f"Method {method} not implemented")
    if lower >= upper:
        raise ValueError("lower must be less than upper")
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    fn = f"{prefix}_BoxCoxLambdaGuerrero"
    # _LIB[fn] doesn't get the restype assignment (keeps c_int)
    return getattr(_LIB, fn)(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        ctypes.c_int(season_length),
        _pyfloat_to_np_c(lower, x.dtype),
        _pyfloat_to_np_c(upper, x.dtype),
    )


def boxcox(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Apply the Box-Cox transformation

    Args:
        x (np.ndarray): Array with data to transform.
        lmbda (float): Lambda value to use.

    Returns:
        np.ndarray: Array with the transformed data."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    fn = f"{prefix}_BoxCoxTransform"
    out = np.empty_like(x)
    getattr(_LIB, fn)(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(lmbda, x.dtype),
        _data_as_void_ptr(out),
    )
    return out


def inv_boxcox(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Invert the Box-Cox transformation

    Args:
        x (np.ndarray): Array with data to transform.
        lmbda (float): Lambda value to use.

    Returns:
        np.ndarray: Array with the inverted transformation."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    fn = f"{prefix}_BoxCoxInverseTransform"
    out = np.empty_like(x)
    getattr(_LIB, fn)(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(lmbda, x.dtype),
        _data_as_void_ptr(out),
    )
    return out


class _BaseLocalScaler:
    _scaler_type: str

    def fit(self, ga: GroupedArray) -> "_BaseLocalScaler":
        """Compute the statistics for each group.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            self: The fitted scaler object."""
        self.stats_ = ga._scaler_fit(self._scaler_type)
        return self

    def transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed statistics to apply the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        return ga._scaler_transform(self.stats_)

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """ "Compute the statistics for each group and apply the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        return self.fit(ga).transform(ga)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed statistics to invert the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return ga._scaler_inverse_transform(self.stats_)


class LocalMinMaxScaler(_BaseLocalScaler):
    """Scale each group to the [0, 1] interval"""

    _scaler_type = "MinMax"


class LocalStandardScaler(_BaseLocalScaler):
    """Scale each group to have zero mean and unit variance"""

    _scaler_type = "Standard"


class LocalRobustScaler(_BaseLocalScaler):
    """Scale each group using robust statistics

    Args:
        scale (str): Type of robust scaling to use. Valid options are 'iqr' and 'mad'.
            If 'iqr' will use the inter quartile range as the scale.
            If 'mad' will use median absolute deviation as the scale."""

    def __init__(self, scale: str):
        if scale == "iqr":
            self._scaler_type = "RobustIqr"
        else:
            self._scaler_type = "RobustMad"


class LocalBoxCoxScaler(_BaseLocalScaler):
    """Find the optimum lambda for the Box-Cox transformation by group and apply it

    Args:
        season_length (int): Length of the seasonal period.
        lower (float): Lower bound for the lambda.
        upper (float): Upper bound for the lambda.
        method (str): Method to use. Valid options are 'guerrero'."""

    def __init__(
        self,
        season_length: int,
        lower: float = -0.9,
        upper: float = 2.0,
        method="guerrero",
    ):
        self.season_length = season_length
        self.lower = lower
        self.upper = upper
        if method != "guerrero":
            raise NotImplementedError(f"Method {method} not implemented")
        self.method = method.capitalize()

    def fit(self, ga: GroupedArray) -> "_BaseLocalScaler":
        """Compute the statistics for each group.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            self: The fitted scaler object."""
        self.stats = ga._boxcox_fit(
            self.season_length, self.lower, self.upper, self.method
        )
        return self

    def transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed lambdas to apply the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        return ga._boxcox_transform(self.stats)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed lambdas to invert the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return ga._boxcox_inverse_transform(self.stats)


class AutoDifferences:
    """Find and apply the optimal number of differences to each group.

    Args:
        max_diffs (int): Maximum number of differences to apply."""

    def __init__(self, max_diffs: int):
        if not isinstance(max_diffs, int) or max_diffs <= 0:
            raise ValueError("max_diff must be a positive integer")
        self.max_diffs = max_diffs

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Compute and apply the optimal number of differences for each group

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        self.diffs_ = ga._num_diffs(self.max_diffs)
        self.tails_ = []
        max_d = int(self.diffs_.max())
        transformed = ga.data.copy()
        dtype = ga.data.dtype.type
        for i in range(max_d):
            ga = ga.with_data(transformed)
            self.tails_.append(ga._tail(1))
            mask = np.where(self.diffs_ > i, dtype(1), dtype(0))
            transformed = ga._conditional_diff(1, mask)
        return transformed

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the differences

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        transformed = ga.data.copy()
        n_diffs = len(self.tails_)
        dtype = ga.data.dtype.type
        for i, tails in enumerate(self.tails_[::-1]):
            ga = ga.with_data(transformed)
            mask = np.where(self.diffs_ >= n_diffs - i, dtype(1), dtype(0))
            transformed = ga._conditional_inv_diff(1, mask, tails)
        return transformed


class AutoSeasonalDifferences:
    """Find and apply the optimal number of seasonal differences to each group.

    Args:
        season_length (int): Length of the seasonal period.
        max_diff (int): Maximum number of seasonal differences to apply."""

    def __init__(self, season_length: int, max_diffs: int):
        if not isinstance(max_diffs, int) or max_diffs <= 0:
            raise ValueError("max_diff must be a positive integer")
        self.max_diffs = max_diffs
        self.season_length = season_length

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Compute and apply the optimal number of seasonal differences for each group

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        self.diffs_ = ga._num_seas_diffs(self.season_length, self.max_diffs)
        self.tails_ = []
        max_d = int(self.diffs_.max())
        transformed = ga.data.copy()
        dtype = ga.data.dtype.type
        for i in range(max_d):
            ga = ga.with_data(transformed)
            self.tails_.append(ga._tail(self.season_length))
            mask = np.where(self.diffs_ > i, dtype(1), dtype(0))
            transformed = ga._conditional_diff(self.season_length, mask)
        return transformed

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the seasonal differences

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        transformed = ga.data.copy()
        n_diffs = len(self.tails_)
        dtype = ga.data.dtype.type
        for i, tails in enumerate(self.tails_[::-1]):
            ga = ga.with_data(transformed)
            mask = np.where(self.diffs_ >= (n_diffs - i), dtype(1), dtype(0))
            transformed = ga._conditional_inv_diff(self.season_length, mask, tails)
        return transformed
