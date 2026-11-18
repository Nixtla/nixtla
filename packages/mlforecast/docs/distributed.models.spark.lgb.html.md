---
description: spark LightGBM forecaster
output-file: distributed.models.spark.lgb.html
title: SparkLGBMForecast
---


Wrapper of `synapse.ml.lightgbm.LightGBMRegressor` that adds an
`extract_local_model` method to get a local version of the trained model
and broadcast it to the workers.

::: mlforecast.distributed.models.spark.lgb.SparkLGBMForecast
