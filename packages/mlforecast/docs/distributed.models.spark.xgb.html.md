---
description: spark XGBoost forecaster
output-file: distributed.models.spark.xgb.html
title: SparkXGBForecast
---


Wrapper of `xgboost.spark.SparkXGBRegressor` that adds an
`extract_local_model` method to get a local version of the trained model
and broadcast it to the workers.

::: mlforecast.distributed.models.spark.xgb.SparkXGBForecast
