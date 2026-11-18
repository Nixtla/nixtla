__all__ = ["DaskXGBForecast"]


import xgboost as xgb
import xgboost.dask as dxgb


class DaskXGBForecast(dxgb.DaskXGBRegressor):
    @property
    def model_(self):
        model_str = self.get_booster().save_raw("ubj")
        local_model = xgb.XGBRegressor()
        local_model.load_model(model_str)
        return local_model
