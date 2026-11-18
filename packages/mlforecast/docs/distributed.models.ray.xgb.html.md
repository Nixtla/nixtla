---
description: ray XGBoost forecaster
output-file: distributed.models.ray.xgb.html
title: RayXGBForecast
---


XGBoost forecaster trained with `ray.train.xgboost.XGBoostTrainer`. Adds a
`model_` attribute that contains the fitted booster as a local
`xgboost.XGBRegressor` and is sent to the workers in the forecasting step.

::: mlforecast.distributed.models.ray.xgb.RayXGBForecast
