__all__ = ["RayXGBForecast"]


import xgboost as xgb
from xgboost_ray import RayXGBRegressor


class RayXGBForecast(RayXGBRegressor):
    @property
    def model_(self):
        model_str = self.get_booster().save_raw("ubj")
        local_model = xgb.XGBRegressor()
        local_model.load_model(model_str)
        return local_model
