__all__ = ['RayLGBMForecast']


import lightgbm as lgb
from lightgbm_ray import RayLGBMRegressor


class RayLGBMForecast(RayLGBMRegressor):
    @property
    def model_(self):
        return self._lgb_ray_to_local(lgb.LGBMRegressor)
