import copy
import ctypes
from typing import List, Optional, Sequence

import numpy as np

from ._lib import _LIB, _indptr_dtype
from .grouped_array import GroupedArray
from .utils import (
    _data_as_void_ptr,
    _diffs_to_indptr,
    _ensure_float,
    _float_arr_to_prefix,
    _pyfloat_to_np_c,
)


__all__ = [
    "AutoDifferences",
    "AutoSeasonalDifferences",
    "AutoSeasonalityAndDifferences",
    "Difference",
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
_LIB.Float32_BoxCoxLambdaLogLik.restype = ctypes.c_float
_LIB.Float64_BoxCoxLambdaLogLik.restype = ctypes.c_double


def _boxcox_lambda_checks(
    season_length: Optional[int],
    lower: float,
    upper: float,
    method: str,
) -> None:
    available_methods = ("guerrero", "loglik")
    if method not in available_methods:
        raise NotImplementedError(
            f"Method {method} not implemented. Available methods are {available_methods}"
        )
    if lower >= upper:
        raise ValueError("lower must be less than upper")
    if method == "guerrero" and season_length is None:
        raise ValueError("season_length is required when method='guerrero'")


def boxcox_lambda(
    x: np.ndarray,
    method: str,
    season_length: Optional[int] = None,
    lower: float = -0.9,
    upper: float = 2.0,
) -> float:
    """Find optimum lambda for the Box-Cox transformation

    Args:
        x (np.ndarray): Array with data to transform.
        method (str): Method to use. Valid options are 'guerrero' and 'loglik'.
            'guerrero' minimizes the coefficient of variation for subseries of `x` and supports negative values.
            'loglik' maximizes the log-likelihood function.
        season_length (int, optional): Length of the seasonal period.
            Only required if method='guerrero'.
        lower (float): Lower bound for the lambda.
        upper (float): Upper bound for the lambda.

    Returns:
        float: Optimum lambda."""
    _boxcox_lambda_checks(
        method=method,
        season_length=season_length,
        lower=lower,
        upper=upper,
    )
    if method == "loglik" and any(x <= 0):
        raise ValueError("All values in x must be positive when method='loglik'")
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    if method == "guerrero":
        assert season_length is not None
        # _LIB[fn] doesn't get the restype assignment (keeps c_int)
        res = getattr(_LIB, f"{prefix}_BoxCoxLambdaGuerrero")(
            _data_as_void_ptr(x),
            ctypes.c_int(x.size),
            ctypes.c_int(season_length),
            _pyfloat_to_np_c(lower, x.dtype),
            _pyfloat_to_np_c(upper, x.dtype),
        )
    else:
        res = getattr(_LIB, f"{prefix}_BoxCoxLambdaLogLik")(
            _data_as_void_ptr(x),
            ctypes.c_int(x.size),
            _pyfloat_to_np_c(lower, x.dtype),
            _pyfloat_to_np_c(upper, x.dtype),
        )
    return res


def boxcox(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Apply the Box-Cox transformation

    Args:
        x (np.ndarray): Array with data to transform.
        lmbda (float): Lambda value to use.

    Returns:
        np.ndarray: Array with the transformed data."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    getattr(_LIB, f"{prefix}_BoxCoxTransform")(
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
    out = np.empty_like(x)
    getattr(_LIB, f"{prefix}_BoxCoxInverseTransform")(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(lmbda, x.dtype),
        _data_as_void_ptr(out),
    )
    return out


def _validate_scaler_size(stats: np.ndarray, ga: GroupedArray) -> None:
    if stats.shape[0] != len(ga):
        raise ValueError(
            f"Number of groups in the scaler ({stats.shape[0]:,}) "
            f"doesn't match the ones in data ({len(ga):,})."
        )


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
        _validate_scaler_size(self.stats_, ga)
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
        _validate_scaler_size(self.stats_, ga)
        return ga._scaler_inverse_transform(self.stats_)

    def take(self, idxs: np.ndarray) -> "_BaseLocalScaler":
        out = copy.deepcopy(self)
        out.stats_ = self.stats_[idxs].copy()
        return out

    @staticmethod
    def stack(scalers: Sequence["_BaseLocalScaler"]) -> "_BaseLocalScaler":
        out = copy.deepcopy(scalers[0])
        out.stats_ = np.vstack([sc.stats_ for sc in scalers])
        return out


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
        season_length (int, optional): Length of the seasonal period.
            Only required if method='guerrero'.
        lower (float): Lower bound for the lambda.
        upper (float): Upper bound for the lambda.
        method (str): Method to use. Valid options are 'guerrero' and 'loglik'.
            'guerrero' minimizes the coefficient of variation for subseries of `x` and supports negative values.
            'loglik' maximizes the log-likelihood function."""

    def __init__(
        self,
        method: str,
        season_length: Optional[int] = None,
        lower: float = -0.9,
        upper: float = 2.0,
    ):
        _boxcox_lambda_checks(season_length, lower, upper, method)
        self.method = method
        self.season_length = season_length
        self.lower = lower
        self.upper = upper

    def fit(self, ga: GroupedArray) -> "_BaseLocalScaler":
        """Compute the statistics for each group.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            self: The fitted scaler object."""
        if self.method == "loglik" and any(ga.data < 0):
            raise ValueError("All values in data must be positive for method='loglik'")
        self.stats_ = ga._boxcox_fit(
            method=self.method,
            season_length=self.season_length,
            lower=self.lower,
            upper=self.upper,
        )
        return self

    def transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed lambdas to apply the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        _validate_scaler_size(self.stats_, ga)
        return ga._boxcox_transform(self.stats_)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed lambdas to invert the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        _validate_scaler_size(self.stats_, ga)
        return ga._boxcox_inverse_transform(self.stats_)


class Difference:
    """Subtract a lag to each group

    Args:
        d (int): Lag to subtract."""

    def __init__(self, d: int):
        self.d = d

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Apply the transformation

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        self.tails_ = ga._tail(self.d)
        return ga._diff(self.d)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the transformation

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return ga._inv_diff(self.d, self.tails_)

    def update(self, ga: GroupedArray) -> np.ndarray:
        """Update the last observations from each serie

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the updated data."""
        tails_indptr = np.arange(
            0, ga.indptr.size * self.d, self.d, dtype=_indptr_dtype
        )
        if self.tails_.size != tails_indptr[-1]:
            raise ValueError("Number of tails doesn't match the number of groups")
        tails_ga = GroupedArray(self.tails_, tails_indptr, num_threads=ga.num_threads)
        combined = tails_ga._append(ga)
        self.tails_ = combined._tail(self.d)
        combined_transformed = combined._diff(self.d)
        transformed = combined._with_data(combined_transformed)._tails(ga.indptr)
        return transformed

    def take(self, idxs: np.ndarray) -> "Difference":
        out = Difference(self.d)
        out.tails_ = self.tails_.reshape(-1, self.d)[idxs].ravel()
        return out

    @staticmethod
    def stack(scalers: Sequence["Difference"]) -> "Difference":
        out = Difference(scalers[0].d)
        out.tails_ = np.hstack([sc.tails_ for sc in scalers])
        return out


class AutoDifferences:
    """Find and apply the optimal number of differences to each group.

    Args:
        max_diffs (int): Maximum number of differences to apply."""

    def __init__(self, max_diffs: int):
        if not isinstance(max_diffs, int) or max_diffs <= 0:
            raise ValueError("max_diff must be a positive integer")
        self.max_diffs = max_diffs

    def _transform(self, ga: GroupedArray, season_length: int) -> np.ndarray:
        max_d = int(self.diffs_.max())
        transformed = ga.data.copy()
        self.tails_: List[np.ndarray] = []
        for i in range(max_d):
            ga = ga._with_data(transformed)
            mask = self.diffs_ > i
            ds = np.where(mask, _indptr_dtype(season_length), _indptr_dtype(0))
            tails_indptr = _diffs_to_indptr(ds)
            self.tails_.append(ga._tails(tails_indptr))
            transformed = ga._diffs(ds)
        return transformed

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Compute and apply the optimal number of differences for each group

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        self.diffs_ = ga._num_diffs(self.max_diffs)
        return self._transform(ga, season_length=1)

    def _inverse_transform(self, ga: GroupedArray, season_length: int) -> np.ndarray:
        _validate_scaler_size(self.diffs_, ga)
        transformed = ga.data.copy()
        n_diffs = len(self.tails_)
        for i, tails in enumerate(self.tails_[::-1]):
            ga = ga._with_data(transformed)
            mask = self.diffs_ >= n_diffs - i
            ds = np.where(mask, _indptr_dtype(season_length), _indptr_dtype(0))
            transformed = ga._inv_diffs(ds, tails)
        return transformed

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the differences

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return self._inverse_transform(ga, 1)

    def _update(self, ga: GroupedArray, season_length: int) -> np.ndarray:
        _validate_scaler_size(self.diffs_, ga)
        tail_indptr = np.arange(
            0, season_length * ga.indptr.size, season_length, dtype=_indptr_dtype
        )
        new_tails = []
        transformed = ga.data.copy()
        for i, tail in enumerate(self.tails_):
            ga = ga._with_data(transformed)
            tail_ga = GroupedArray(tail, tail_indptr, num_threads=ga.num_threads)
            combined = tail_ga._append(ga)
            new_tails.append(combined._tail(season_length))
            ds = np.where(
                self.diffs_ > i, _indptr_dtype(season_length), _indptr_dtype(0)
            )
            combined_transformed = combined._diffs(ds)
            transformed = combined._with_data(combined_transformed)._tails(ga.indptr)
        self.tails_ = new_tails
        return transformed

    def update(self, ga: GroupedArray) -> np.ndarray:
        """Update the last observations from each serie

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the updated data."""
        return self._update(ga, season_length=1)

    def _take(self, idxs: np.ndarray, season_length: int) -> "AutoDifferences":
        out = copy.deepcopy(self)
        out.diffs_ = self.diffs_[idxs].copy()
        out.tails_ = []
        for i, tail in enumerate(self.tails_):
            ds = np.where(
                self.diffs_ > i, _indptr_dtype(season_length), _indptr_dtype(0)
            )
            tails_indptr = _diffs_to_indptr(ds)
            out.tails_.append(GroupedArray(tail, tails_indptr)._take(idxs))
        return out

    def take(self, idxs: np.ndarray) -> "AutoDifferences":
        return self._take(idxs, season_length=1)

    @staticmethod
    def stack(scalers: Sequence["AutoDifferences"]) -> "AutoDifferences":
        out = copy.deepcopy(scalers[0])
        out.diffs_ = np.hstack([sc.diffs_ for sc in scalers])
        out.tails_ = [
            np.hstack([sc.tails_[i] for sc in scalers])
            for i in range(len(scalers[0].tails_))
        ]
        return out


class AutoSeasonalDifferences(AutoDifferences):
    """Find and apply the optimal number of seasonal differences to each group.

    Args:
        season_length (int): Length of the seasonal period.
        max_diffs (int): Maximum number of differences to apply.
        n_seasons (int | None): Number of seasons to use to determine the number of differences. Defaults to 10.
            If `None` will use all samples, otherwise `season_length` * `n_seasons` samples will be used for the test.
            Smaller values will be faster but could be less accurate.
    """

    def __init__(
        self,
        season_length: int,
        max_diffs: int,
        n_seasons: Optional[int] = 10,
    ):
        self.season_length = season_length
        self.max_diffs = max_diffs
        if isinstance(n_seasons, int) and n_seasons < 2:
            raise ValueError("n_seasons must be at least 2")
        self.n_seasons = n_seasons

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Compute and apply the optimal number of seasonal differences for each group

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        if self.n_seasons is None:
            tails_ga = ga
        else:
            n_samples = self.season_length * self.n_seasons
            tails_ga = GroupedArray(
                ga._tail(n_samples),
                np.arange(0, (len(ga) + 1) * n_samples, n_samples),
                num_threads=ga.num_threads,
            )
        self.diffs_ = tails_ga._num_seas_diffs(self.season_length, self.max_diffs)
        return self._transform(ga, self.season_length)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the seasonal differences

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return self._inverse_transform(ga, self.season_length)

    def update(self, ga: GroupedArray) -> np.ndarray:
        """Update the last observations from each serie

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the updated data."""
        return self._update(ga, self.season_length)

    def take(self, idxs: np.ndarray) -> "AutoDifferences":
        return self._take(idxs, self.season_length)


class AutoSeasonalityAndDifferences:
    """Find the length of the seasonal period and apply the optimal number of differences to each group.

    Args:
        max_season_length (int): Maximum length of the seasonal period.
        max_diffs (int): Maximum number of differences to apply.
        n_seasons (int | None): Number of seasons to use to determine the number of differences. Defaults to 10.
            If `None` will use all samples, otherwise `max_season_length` * `n_seasons` samples will be used for the test.
            Smaller values will be faster but could be less accurate.
    """

    def __init__(
        self, max_season_length: int, max_diffs: int, n_seasons: Optional[int] = 10
    ):
        self.max_season_length = max_season_length
        if not isinstance(max_diffs, int) or max_diffs < 1:
            raise ValueError("max_diff must be a positive integer")
        self.max_diffs = max_diffs
        if isinstance(n_seasons, int) and n_seasons < 2:
            raise ValueError("n_seasons must be at least 2")
        self.n_seasons = n_seasons

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Compute the optimal length of the seasonal period and apply the optimal number of differences for each group

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        self.diffs_: List[np.ndarray] = []
        self.tails_: List[np.ndarray] = []
        transformed = ga.data.copy()
        for _ in range(self.max_diffs):
            ga = ga._with_data(transformed)
            if self.n_seasons is None:
                tails_ga = ga
            else:
                n_samples = self.max_season_length * self.n_seasons
                tails_ga = GroupedArray(
                    ga._tail(n_samples),
                    np.arange(0, (len(ga) + 1) * n_samples, n_samples),
                    num_threads=ga.num_threads,
                )
            periods = tails_ga._periods(self.max_season_length)
            diffs = periods * tails_ga._num_seas_diffs_periods(1, periods)
            diffs = diffs.astype(_indptr_dtype)
            self.diffs_.append(diffs)
            tails_indptr = _diffs_to_indptr(diffs)
            self.tails_.append(ga._tails(tails_indptr))
            transformed = ga._diffs(diffs)
        return transformed

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the seasonal differences

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        _validate_scaler_size(self.diffs_[0], ga)
        transformed = ga.data.copy()
        for diffs, tails in zip(self.diffs_[::-1], self.tails_[::-1]):
            ga = ga._with_data(transformed)
            transformed = ga._inv_diffs(diffs, tails)
        return transformed

    def update(self, ga: GroupedArray) -> np.ndarray:
        _validate_scaler_size(self.diffs_[0], ga)
        new_tails = []
        transformed = ga.data.copy()
        for i in range(self.max_diffs):
            ga = ga._with_data(transformed)
            tails_indptr = _diffs_to_indptr(self.diffs_[i])
            tails_ga = GroupedArray(
                self.tails_[i], tails_indptr, num_threads=ga.num_threads
            )
            combined = tails_ga._append(ga)
            new_tails.append(combined._tails(tails_indptr))
            combined_transformed = combined._diffs(self.diffs_[i])
            transformed = combined._with_data(combined_transformed)._tails(ga.indptr)
        self.tails_ = new_tails
        return transformed

    def take(self, idxs: np.ndarray) -> "AutoSeasonalityAndDifferences":
        out = AutoSeasonalityAndDifferences(
            self.max_season_length, self.max_diffs, self.n_seasons
        )
        out.diffs_ = [diffs[idxs].copy() for diffs in self.diffs_]
        out.tails_ = []
        for diffs, tails in zip(self.diffs_, self.tails_):
            tails_indptr = _diffs_to_indptr(diffs)
            tails_ga = GroupedArray(tails, tails_indptr)
            out.tails_.append(tails_ga._take(idxs))
        return out

    @staticmethod
    def stack(
        scalers: Sequence["AutoSeasonalityAndDifferences"],
    ) -> "AutoSeasonalityAndDifferences":
        out = AutoSeasonalityAndDifferences(
            scalers[0].max_season_length, scalers[0].max_diffs, scalers[0].n_seasons
        )
        out.diffs_ = [
            np.hstack([sc.diffs_[i] for sc in scalers]) for i in range(out.max_diffs)
        ]
        out.tails_ = [
            np.hstack([sc.tails_[i] for sc in scalers]) for i in range(out.max_diffs)
        ]
        return out
