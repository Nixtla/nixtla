__all__ = [
    "Lag",
    "RollingMean",
    "RollingStd",
    "RollingMin",
    "RollingMax",
    "RollingQuantile",
    "SeasonalRollingMean",
    "SeasonalRollingStd",
    "SeasonalRollingMin",
    "SeasonalRollingMax",
    "SeasonalRollingQuantile",
    "ExpandingMean",
    "ExpandingStd",
    "ExpandingMin",
    "ExpandingMax",
    "ExpandingQuantile",
    "ExponentiallyWeightedMean",
]

import abc
from typing import Callable, Optional

import numpy as np

from .grouped_array import GroupedArray


class BaseLagTransform(abc.ABC):
    @abc.abstractmethod
    def transform(self, ga: GroupedArray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def update(self, ga: GroupedArray) -> np.ndarray:
        ...


class Lag(BaseLagTransform):
    def __init__(self, lag: int):
        self.lag = lag

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.lag_transform(self.lag)

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.take_from_groups(self.lag - 1)


class RollingBase(BaseLagTransform):
    stat_name: str
    lag: int
    window_size: int
    min_samples: int

    def __init__(self, lag: int, window_size: int, min_samples: Optional[int] = None):
        self.lag = lag
        if min_samples is None:
            min_samples = window_size
        if min_samples > window_size:
            min_samples = window_size
        self.window_size = window_size
        self.min_samples = min_samples

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.rolling_transform(
            self.stat_name, self.lag, self.window_size, self.min_samples
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.rolling_update(
            self.stat_name, self.lag - 1, self.window_size, self.min_samples
        )


class RollingMean(RollingBase):
    stat_name = "Mean"


class RollingStd(RollingBase):
    stat_name = "Std"


class RollingMin(RollingBase):
    stat_name = "Min"


class RollingMax(RollingBase):
    stat_name = "Max"


class RollingQuantile(RollingBase):
    def __init__(
        self, lag: int, p: float, window_size: int, min_samples: Optional[int] = None
    ):
        super().__init__(lag=lag, window_size=window_size, min_samples=min_samples)
        self.p = p

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.rolling_quantile_transform(
            self.lag, self.p, self.window_size, self.min_samples
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.rolling_quantile_update(
            self.lag - 1, self.p, self.window_size, self.min_samples
        )


class SeasonalRollingBase(RollingBase):
    season_length: int

    def __init__(
        self,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
    ):
        super().__init__(lag, window_size, min_samples)
        self.season_length = season_length

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.seasonal_rolling_transform(
            self.stat_name,
            self.lag,
            self.season_length,
            self.window_size,
            self.min_samples,
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.seasonal_rolling_update(
            self.stat_name,
            self.lag - 1,
            self.season_length,
            self.window_size,
            self.min_samples,
        )


class SeasonalRollingMean(SeasonalRollingBase):
    stat_name = "Mean"


class SeasonalRollingStd(SeasonalRollingBase):
    stat_name = "Std"


class SeasonalRollingMin(SeasonalRollingBase):
    stat_name = "Min"


class SeasonalRollingMax(SeasonalRollingBase):
    stat_name = "Max"


class SeasonalRollingQuantile(SeasonalRollingBase):
    def __init__(
        self,
        lag: int,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
    ):
        super().__init__(
            lag=lag,
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
        )
        self.p = p

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.seasonal_rolling_quantile_transform(
            lag=self.lag,
            p=self.p,
            season_length=self.season_length,
            window_size=self.window_size,
            min_samples=self.min_samples,
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.seasonal_rolling_quantile_update(
            lag=self.lag - 1,
            p=self.p,
            season_length=self.season_length,
            window_size=self.window_size,
            min_samples=self.min_samples,
        )


class ExpandingBase(BaseLagTransform):
    def __init__(self, lag: int):
        self.lag = lag


class ExpandingMean(ExpandingBase):
    def transform(self, ga: GroupedArray) -> np.ndarray:
        self.n = np.empty_like(ga.data, shape=len(ga))
        out = ga.expanding_transform_with_aggs("Mean", self.lag, self.n)
        self.cumsum = out[ga.indptr[1:] - 1] * self.n
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        self.n += 1
        self.cumsum += ga.take_from_groups(self.lag - 1)
        return self.cumsum / self.n


class ExpandingStd(ExpandingBase):
    def transform(self, ga: GroupedArray) -> np.ndarray:
        self.stats = np.empty_like(ga.data, shape=(len(ga), 3))
        out = ga.expanding_transform_with_aggs("Std", self.lag, self.stats)
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        x = ga.take_from_groups(self.lag - 1)
        self.stats[:, 0] += 1.0
        n = self.stats[:, 0]
        prev_avg = self.stats[:, 1].copy()
        self.stats[:, 1] = prev_avg + (x - prev_avg) / n
        self.stats[:, 2] += (x - prev_avg) * (x - self.stats[:, 1])
        self.stats[:, 2] = np.maximum(self.stats[:, 2], 0.0)
        return np.sqrt(self.stats[:, 2] / (n - 1))


class ExpandingComp(ExpandingBase):
    stat: str
    comp_fn: Callable

    def transform(self, ga: GroupedArray) -> np.ndarray:
        out = ga.expanding_transform(self.stat, self.lag)
        self.stats = out[ga.indptr[1:] - 1]
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        self.stats = self.comp_fn(self.stats, ga.take_from_groups(self.lag - 1))
        return self.stats


class ExpandingMin(ExpandingComp):
    stat = "Min"
    comp_fn = np.minimum


class ExpandingMax(ExpandingComp):
    stat = "Max"
    comp_fn = np.maximum


class ExpandingQuantile(BaseLagTransform):
    def __init__(self, lag: int, p: float):
        self.lag = lag
        self.p = p

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.expanding_quantile_transform(self.lag, self.p)

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.expanding_quantile_update(self.lag - 1, self.p)


class ExponentiallyWeightedMean(BaseLagTransform):
    def __init__(self, lag: int, alpha: float):
        self.lag = lag
        self.alpha = alpha

    def transform(self, ga: GroupedArray) -> np.ndarray:
        out = ga.exponentially_weighted_transform("Mean", self.lag, self.alpha)
        self.ewm = out[ga.indptr[1:] - 1]
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        x = ga.take_from_groups(self.lag - 1)
        self.ewm = self.alpha * x + (1 - self.alpha) * self.ewm
        return self.ewm
