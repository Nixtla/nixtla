__all__ = ['BaseTargetTransform', 'Differences', 'AutoDifferences', 'AutoSeasonalDifferences', 'AutoSeasonalityAndDifferences',
           'LocalStandardScaler', 'LocalMinMaxScaler', 'LocalRobustScaler', 'LocalBoxCox', 'GlobalSklearnTransformer']


import abc
import copy
from typing import Iterable, List, Optional, Sequence

import coreforecast.scalers as core_scalers
import numpy as np
import utilsforecast.processing as ufp
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray
from sklearn.base import TransformerMixin, clone
from utilsforecast.compat import DataFrame

from .grouped_array import GroupedArray
from .utils import _ShortSeriesException


class BaseTargetTransform(abc.ABC):
    """Base class used for target transformations."""

    def set_column_names(self, id_col: str, time_col: str, target_col: str):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

    def update(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError

    @staticmethod
    def stack(transforms: Sequence["BaseTargetTransform"]) -> "BaseTargetTransform":
        raise NotImplementedError

    @abc.abstractmethod
    def fit_transform(self, df: DataFrame) -> DataFrame: ...

    @abc.abstractmethod
    def inverse_transform(self, df: DataFrame) -> DataFrame: ...


class _BaseGroupedArrayTargetTransform(abc.ABC):
    """Base class used for target transformations that operate on grouped arrays."""

    num_threads: int = 1
    scaler_: core_scalers._BaseLocalScaler

    def set_num_threads(self, num_threads: int) -> None:
        self.num_threads = num_threads

    @abc.abstractmethod
    def update(self, ga: GroupedArray) -> GroupedArray: ...

    @abc.abstractmethod
    def fit_transform(self, ga: GroupedArray) -> GroupedArray: ...

    @abc.abstractmethod
    def inverse_transform(self, ga: GroupedArray) -> GroupedArray: ...

    def inverse_transform_fitted(self, ga: GroupedArray) -> GroupedArray:
        return self.inverse_transform(ga)

    @abc.abstractmethod
    def take(self, idxs: np.ndarray) -> "_BaseGroupedArrayTargetTransform": ...

    @staticmethod
    def stack(
        scalers: Sequence["_BaseGroupedArrayTargetTransform"],
    ) -> "_BaseGroupedArrayTargetTransform":
        first_scaler = scalers[0]
        core_scaler = first_scaler.scaler_
        out = copy.deepcopy(first_scaler)
        out.scaler_ = core_scaler.stack([sc.scaler_ for sc in scalers])
        return out


class Differences(_BaseGroupedArrayTargetTransform):
    """Subtracts previous values of the serie. Can be used to remove trend or seasonalities."""

    store_fitted = False

    def __init__(self, differences: Iterable[int]):
        self.differences = list(differences)

    def fit_transform(self, ga: GroupedArray) -> GroupedArray:
        self.fitted_: List[np.ndarray] = []
        self.fitted_indptr_: Optional[np.ndarray] = None
        original_sizes = np.diff(ga.indptr)
        total_diffs = sum(self.differences)
        small_series = original_sizes < total_diffs
        if small_series.any():
            raise _ShortSeriesException(np.where(small_series)[0])
        self.scalers_ = []
        core_ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        for d in self.differences:
            if self.store_fitted:
                # these are saved in order to be able to perform a correct
                # inverse transform when trying to retrieve the fitted values.
                self.fitted_.append(core_ga.data.copy())
                if self.fitted_indptr_ is None:
                    self.fitted_indptr_ = core_ga.indptr.copy()
            scaler = core_scalers.Difference(d)
            transformed = scaler.fit_transform(core_ga)
            self.scalers_.append(scaler)
            core_ga = core_ga._with_data(transformed)
        return GroupedArray(core_ga.data, ga.indptr)

    def update(self, ga: GroupedArray) -> GroupedArray:
        core_ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        for scaler in self.scalers_:
            transformed = scaler.update(core_ga)
            core_ga = core_ga._with_data(transformed)
        return GroupedArray(transformed, ga.indptr)

    def inverse_transform(self, ga: GroupedArray) -> GroupedArray:
        core_ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        for scaler in self.scalers_[::-1]:
            transformed = scaler.inverse_transform(core_ga)
            core_ga = core_ga._with_data(transformed)
        return GroupedArray(transformed, ga.indptr)

    def inverse_transform_fitted(self, ga: GroupedArray) -> GroupedArray:
        if self.fitted_[0].size < ga.data.size:
            raise ValueError("fitted differences are smaller than provided target.")
        transformed = ga.data
        for d, fitted in zip(reversed(self.differences), reversed(self.fitted_)):
            fitted_ga = CoreGroupedArray(fitted, self.fitted_indptr_)
            adds = fitted_ga._lag(d)
            if adds.size > ga.data.size:
                adds = CoreGroupedArray(adds, self.fitted_indptr_)._tails(ga.indptr)
            transformed = transformed + adds
        return GroupedArray(transformed, ga.indptr)

    def take(self, idxs: np.ndarray) -> "Differences":
        out = Differences(self.differences)
        if self.fitted_indptr_ is None:
            out.fitted_ = []
            out.fitted_indptr_ = None
        else:
            out.fitted_ = [
                np.hstack(
                    [
                        data[self.fitted_indptr_[i] : self.fitted_indptr_[i + 1]]
                        for i in idxs
                    ]
                )
                for data in self.fitted_
            ]
            sizes = np.diff(self.fitted_indptr_)[idxs]
            out.fitted_indptr_ = np.append(0, sizes.cumsum())
        out.scalers_ = [scaler.take(idxs) for scaler in self.scalers_]
        return out

    @staticmethod
    def stack(scalers: Sequence["Differences"]) -> "Differences":  # type: ignore[override]
        first_scaler = scalers[0]
        core_scaler = first_scaler.scalers_[0]
        diffs = first_scaler.differences
        out = Differences(diffs)
        out.fitted_ = []
        if first_scaler.fitted_indptr_ is None:
            out.fitted_indptr_ = None
        else:
            for i in range(len(scalers[0].fitted_)):
                out.fitted_.append(np.hstack([sc.fitted_[i] for sc in scalers]))
            sizes = np.hstack([np.diff(sc.fitted_indptr_) for sc in scalers])
            out.fitted_indptr_ = np.append(0, sizes.cumsum())
        out.scalers_ = [
            core_scaler.stack([sc.scalers_[i] for sc in scalers])
            for i in range(len(diffs))
        ]
        return out


class AutoDifferences(_BaseGroupedArrayTargetTransform):
    """Find and apply the optimal number of differences to each serie.

    Args:
        max_diffs (int): Maximum number of differences to apply.
    """

    def __init__(self, max_diffs: int):
        self.scaler_ = core_scalers.AutoDifferences(max_diffs)

    def fit_transform(self, ga: GroupedArray) -> GroupedArray:
        core_ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        return GroupedArray(self.scaler_.fit_transform(core_ga), ga.indptr)

    def update(self, ga: GroupedArray) -> GroupedArray:
        core_ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        return GroupedArray(self.scaler_.update(core_ga), ga.indptr)

    def inverse_transform(self, ga: GroupedArray) -> GroupedArray:
        core_ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        return GroupedArray(self.scaler_.inverse_transform(core_ga), ga.indptr)

    def inverse_transform_fitted(self, ga: GroupedArray) -> GroupedArray:
        raise NotImplementedError

    def take(self, idxs: np.ndarray) -> "AutoDifferences":
        out = AutoDifferences(self.scaler_.max_diffs)
        out.scaler_ = self.scaler_.take(idxs)
        return out


class AutoSeasonalDifferences(AutoDifferences):
    """Find and apply the optimal number of seasonal differences to each group.

    Args:
        season_length (int): Length of the seasonal period.
        max_diffs (int): Maximum number of differences to apply.
        n_seasons (int, optional): Number of seasons to use to determine the number of differences. Defaults to 10.
            If `None` will use all samples, otherwise `season_length` * `n_seasons samples` will be used for the test.
            Smaller values will be faster but could be less accurate.
    """

    def __init__(
        self, season_length: int, max_diffs: int, n_seasons: Optional[int] = 10
    ):
        self.scaler_ = core_scalers.AutoSeasonalDifferences(
            season_length=season_length,
            max_diffs=max_diffs,
            n_seasons=n_seasons,
        )


class AutoSeasonalityAndDifferences(AutoDifferences):
    """Find the length of the seasonal period and apply the optimal number of differences to each group.

    Args:
        max_season_length (int): Maximum length of the seasonal period.
        max_diffs (int): Maximum number of differences to apply.
        n_seasons (int, optional): Number of seasons to use to determine the number of differences. Defaults to 10.
            If `None` will use all samples, otherwise `max_season_length` * `n_seasons samples` will be used for the test.
            Smaller values will be faster but could be less accurate.
    """

    def __init__(
        self, max_season_length: int, max_diffs: int, n_seasons: Optional[int] = 10
    ):
        self.scaler_ = core_scalers.AutoSeasonalityAndDifferences(
            max_season_length=max_season_length,
            max_diffs=max_diffs,
            n_seasons=n_seasons,
        )


class _BaseLocalScaler(_BaseGroupedArrayTargetTransform):
    scaler_factory: type

    def update(self, ga: GroupedArray) -> GroupedArray:
        ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        return GroupedArray(self.scaler_.transform(ga), ga.indptr)

    def fit_transform(self, ga: GroupedArray) -> GroupedArray:
        self.scaler_ = self.scaler_factory()
        core_ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        transformed = self.scaler_.fit_transform(core_ga)
        return GroupedArray(transformed, ga.indptr)

    def inverse_transform(self, ga: GroupedArray) -> GroupedArray:
        core_ga = CoreGroupedArray(ga.data, ga.indptr, self.num_threads)
        transformed = self.scaler_.inverse_transform(core_ga)
        return GroupedArray(transformed, ga.indptr)

    def take(self, idxs: np.ndarray) -> "_BaseLocalScaler":
        out = copy.deepcopy(self)
        out.scaler_ = self.scaler_.take(idxs)
        return out


class LocalStandardScaler(_BaseLocalScaler):
    """Standardizes each serie by subtracting its mean and dividing by its standard deviation."""

    scaler_factory = core_scalers.LocalStandardScaler


class LocalMinMaxScaler(_BaseLocalScaler):
    """Scales each serie to be in the [0, 1] interval."""

    scaler_factory = core_scalers.LocalMinMaxScaler


class LocalRobustScaler(_BaseLocalScaler):
    """Scaler robust to outliers.

    Args:
        scale (str): Statistic to use for scaling. Can be either 'iqr' (Inter Quartile Range) or 'mad' (Median Asbolute Deviation).
            Defaults to 'iqr'.
    """

    def __init__(self, scale: str):
        self.scaler_factory = lambda: core_scalers.LocalRobustScaler(scale)  # type: ignore


class LocalBoxCox(_BaseLocalScaler):
    """Finds the optimum lambda for each serie and applies the Box-Cox transformation"""

    def __init__(self):
        self.scaler_factory = lambda: core_scalers.LocalBoxCoxScaler(
            method="loglik", lower=0.0
        )


class GlobalSklearnTransformer(BaseTargetTransform):
    """Applies the same scikit-learn transformer to all series."""

    def __init__(self, transformer: TransformerMixin):
        self.transformer = transformer

    def fit_transform(self, df: DataFrame) -> DataFrame:
        df = ufp.copy_if_pandas(df, deep=False)
        self.transformer_ = clone(self.transformer)
        transformed = self.transformer_.fit_transform(df[[self.target_col]].to_numpy())
        return ufp.assign_columns(df, self.target_col, transformed[:, 0])

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        df = ufp.copy_if_pandas(df, deep=False)
        cols_to_transform = [
            c for c in df.columns if c not in (self.id_col, self.time_col)
        ]
        transformed = np.hstack(
            [
                self.transformer_.inverse_transform(df[[col]].to_numpy())
                for col in cols_to_transform
            ]
        )
        return ufp.assign_columns(df, cols_to_transform, transformed)

    def update(self, df: DataFrame) -> DataFrame:
        df = ufp.copy_if_pandas(df, deep=False)
        transformed = self.transformer_.transform(df[[self.target_col]].to_numpy())
        return ufp.assign_columns(df, self.target_col, transformed[:, 0])

    @staticmethod
    def stack(transforms: Sequence["GlobalSklearnTransformer"]) -> "GlobalSklearnTransformer":  # type: ignore[override]
        return transforms[0]
