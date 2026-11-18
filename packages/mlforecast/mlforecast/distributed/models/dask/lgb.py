__all__ = ["DaskLGBMForecast"]


import lightgbm as lgb

from ...._warnings import _warn


class DaskLGBMForecast(lgb.dask.DaskLGBMRegressor):
    if lgb.__version__ < "3.3.0":
        _warn(
            "It is recommended to install LightGBM version >= 3.3.0, since "
            "the current LightGBM version might be affected by https://github.com/microsoft/LightGBM/issues/4026, "
            "which was fixed in 3.3.0"
        )

    @property
    def model_(self):
        return self.to_local()
