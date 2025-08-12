__all__ = ['DaskLGBMForecast']


import warnings

import lightgbm as lgb


class DaskLGBMForecast(lgb.dask.DaskLGBMRegressor):
    if lgb.__version__ < "3.3.0":
        warnings.warn(
            "It is recommended to install LightGBM version >= 3.3.0, since "
            "the current LightGBM version might be affected by https://github.com/microsoft/LightGBM/issues/4026, "
            "which was fixed in 3.3.0"
        )

    @property
    def model_(self):
        return self.to_local()
