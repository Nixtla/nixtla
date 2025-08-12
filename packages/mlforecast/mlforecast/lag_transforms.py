__all__ = ['RollingMean', 'RollingStd', 'RollingMin', 'RollingMax', 'RollingQuantile', 'SeasonalRollingMean',
           'SeasonalRollingStd', 'SeasonalRollingMin', 'SeasonalRollingMax', 'SeasonalRollingQuantile', 'ExpandingMean',
           'ExpandingStd', 'ExpandingMin', 'ExpandingMax', 'ExpandingQuantile', 'ExponentiallyWeightedMean', 'Offset',
           'Combine']


import copy
import inspect
import re
from typing import Callable, Optional, Sequence

import coreforecast.lag_transforms as core_tfms
import numpy as np
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray
from sklearn.base import BaseEstimator, clone


def _pascal2camel(pascal_str: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_str).lower()


class _BaseLagTransform(BaseEstimator):
    def _get_init_signature(self):
        return {
            k: v
            for k, v in inspect.signature(self.__class__.__init__).parameters.items()
            if k != "self"
        }

    def _set_core_tfm(self, lag: int) -> "_BaseLagTransform":
        init_args = {k: getattr(self, k) for k in self._get_init_signature()}
        self._core_tfm = getattr(core_tfms, self.__class__.__name__)(
            lag=lag, **init_args
        )
        return self

    def _get_name(self, lag: int) -> str:
        init_params = self._get_init_signature()
        result = f"{_pascal2camel(self.__class__.__name__)}_lag{lag}"
        changed_params = [
            f"{name}{getattr(self, name)}"
            for name, arg in init_params.items()
            if arg.default != getattr(self, name)
        ]
        if changed_params:
            result += "_" + "_".join(changed_params)
        return result

    def transform(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.transform(ga)

    def update(self, ga: CoreGroupedArray) -> np.ndarray:
        return self._core_tfm.update(ga)

    def take(self, idxs: np.ndarray) -> "_BaseLagTransform":
        out = copy.deepcopy(self)
        out._core_tfm = self._core_tfm.take(idxs)
        return out

    @staticmethod
    def stack(transforms: Sequence["_BaseLagTransform"]) -> "_BaseLagTransform":
        out = copy.deepcopy(transforms[0])
        out._core_tfm = transforms[0]._core_tfm.stack(
            [tfm._core_tfm for tfm in transforms]
        )
        return out

    @property
    def _lag(self):
        return self._core_tfm.lag - 1

    @property
    def update_samples(self) -> int:
        return -1


class Lag(_BaseLagTransform):

    def __init__(self, lag: int):
        self.lag = lag
        self._core_tfm = core_tfms.Lag(lag=lag)

    def _set_core_tfm(self, _lag: int) -> "Lag":
        return self

    def _get_name(self, lag: int) -> str:
        return f"lag{lag}"

    def __eq__(self, other):
        return isinstance(other, Lag) and self.lag == other.lag

    @property
    def update_samples(self) -> int:
        return self.lag


class _RollingBase(_BaseLagTransform):
    "Rolling statistic"

    def __init__(self, window_size: int, min_samples: Optional[int] = None):
        """
        Args:
            window_size (int): Number of samples in the window.
            min_samples (int, optional): Minimum samples required to output the statistic.
                If `None`, will be set to `window_size`. Defaults to None.
        """
        self.window_size = window_size
        self.min_samples = min_samples

    @property
    def update_samples(self) -> int:
        return self._lag + self.window_size


class RollingMean(_RollingBase): ...


class RollingStd(_RollingBase): ...


class RollingMin(_RollingBase): ...


class RollingMax(_RollingBase): ...


class RollingQuantile(_RollingBase):
    def __init__(self, p: float, window_size: int, min_samples: Optional[int] = None):
        super().__init__(window_size=window_size, min_samples=min_samples)
        self.p = p

    def _set_core_tfm(self, lag: int):
        self._core_tfm = core_tfms.RollingQuantile(
            lag=lag,
            p=self.p,
            window_size=self.window_size,
            min_samples=self.min_samples,
        )
        return self


class _Seasonal_RollingBase(_BaseLagTransform):
    """Rolling statistic over seasonal periods"""

    def __init__(
        self, season_length: int, window_size: int, min_samples: Optional[int] = None
    ):
        """
        Args:
            season_length (int): Periodicity of the seasonal period.
            window_size (int): Number of samples in the window.
            min_samples (int, optional): Minimum samples required to output the statistic.
                If `None`, will be set to `window_size`. Defaults to None.
        """
        self.season_length = season_length
        self.window_size = window_size
        self.min_samples = min_samples

    @property
    def update_samples(self) -> int:
        return self._lag + self.season_length * self.window_size


class SeasonalRollingMean(_Seasonal_RollingBase): ...


class SeasonalRollingStd(_Seasonal_RollingBase): ...


class SeasonalRollingMin(_Seasonal_RollingBase): ...


class SeasonalRollingMax(_Seasonal_RollingBase): ...


class SeasonalRollingQuantile(_Seasonal_RollingBase):
    def __init__(
        self,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
    ):
        super().__init__(
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
        )
        self.p = p


class _ExpandingBase(_BaseLagTransform):
    """Expanding statistic"""

    def __init__(self): ...

    @property
    def update_samples(self) -> int:
        return 1


class ExpandingMean(_ExpandingBase): ...


class ExpandingStd(_ExpandingBase): ...


class ExpandingMin(_ExpandingBase): ...


class ExpandingMax(_ExpandingBase): ...


class ExpandingQuantile(_ExpandingBase):
    def __init__(self, p: float):
        self.p = p

    @property
    def update_samples(self) -> int:
        return -1


class ExponentiallyWeightedMean(_BaseLagTransform):
    """Exponentially weighted average

    Args:
        alpha (float): Smoothing factor.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    @property
    def update_samples(self) -> int:
        return 1


class Offset(_BaseLagTransform):
    """Shift series before computing transformation

    Args:
        tfm (LagTransform): Transformation to be applied
        n (int): Number of positions to shift (lag) series before applying the transformation
    """

    def __init__(self, tfm: _BaseLagTransform, n: int):
        self.tfm = tfm
        self.n = n

    def _get_name(self, lag: int) -> str:
        return self.tfm._get_name(lag + self.n)

    def _set_core_tfm(self, lag: int) -> "Offset":
        self.tfm = clone(self.tfm)._set_core_tfm(lag + self.n)
        self._core_tfm = self.tfm._core_tfm
        return self

    @property
    def update_samples(self) -> int:
        return self.tfm.update_samples + self.n


class Combine(_BaseLagTransform):
    """Combine two lag transformations using an operator

    Args:
        tfm1 (LagTransform): First transformation.
        tfm2 (LagTransform): Second transformation.
        operator (callable): Binary operator that defines how to combine the two transformations.
    """

    def __init__(
        self, tfm1: _BaseLagTransform, tfm2: _BaseLagTransform, operator: Callable
    ):
        self.tfm1 = tfm1
        self.tfm2 = tfm2
        self.operator = operator

    def _set_core_tfm(self, lag: int) -> "Combine":
        self.tfm1 = clone(self.tfm1)._set_core_tfm(lag)
        self.tfm2 = clone(self.tfm2)._set_core_tfm(lag)
        return self

    def _get_name(self, lag: int) -> str:
        lag1 = getattr(self.tfm1, "lag", lag)
        lag2 = getattr(self.tfm2, "lag", lag)
        return f"{self.tfm1._get_name(lag1)}_{self.operator.__name__}_{self.tfm2._get_name(lag2)}"

    def transform(self, ga: CoreGroupedArray) -> np.ndarray:
        return self.operator(self.tfm1.transform(ga), self.tfm2.transform(ga))

    def update(self, ga: CoreGroupedArray) -> np.ndarray:
        return self.operator(self.tfm1.update(ga), self.tfm2.update(ga))

    @property
    def update_samples(self):
        return max(self.tfm1.update_samples, self.tfm2.update_samples)
