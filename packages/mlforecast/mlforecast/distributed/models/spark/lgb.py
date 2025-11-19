__all__ = ["SparkLGBMForecast"]


import lightgbm as lgb

try:
    import pyspark

    spark = (
        pyspark.sql.SparkSession.builder.appName("MyApp")
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.13")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .getOrCreate()
    )
    import synapse.ml.lightgbm.LightGBMRegressor as LightGBMRegressor
except ModuleNotFoundError:
    import os

    if os.getenv("QUARTO_PREVIEW", "0") == "1" or os.getenv("IN_TEST", "0") == "1":
        LightGBMRegressor = object
    else:
        raise


class SparkLGBMForecast(LightGBMRegressor):
    def _pre_fit(self, target_col):
        return self.setLabelCol(target_col)

    def extract_local_model(self, trained_model):
        model_str = trained_model.getNativeModel()
        local_model = lgb.Booster(model_str=model_str)
        return local_model
