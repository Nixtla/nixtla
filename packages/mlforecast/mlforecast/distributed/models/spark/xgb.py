__all__ = ["SparkXGBForecast"]


import xgboost as xgb

try:
    from xgboost.spark import SparkXGBRegressor  # type: ignore
except ModuleNotFoundError:
    import os

    if os.getenv("IN_TEST", "0") == "1":
        SparkXGBRegressor = object
    else:
        raise


class SparkXGBForecast(SparkXGBRegressor):
    def _pre_fit(self, target_col):
        self.setParams(label_col=target_col)
        return self

    def extract_local_model(self, trained_model):
        model_str = trained_model.get_booster().save_raw("ubj")
        local_model = xgb.XGBRegressor()
        local_model.load_model(model_str)
        return local_model
