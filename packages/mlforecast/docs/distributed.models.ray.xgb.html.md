---
description: ray XGBoost forecaster
output-file: distributed.models.ray.xgb.html
title: RayXGBForecast
---


Wrapper of `xgboost.ray.RayXGBRegressor` that adds a `model_` property
that contains the fitted model and is sent to the workers in the
forecasting step.

::: mlforecast.distributed.models.ray.xgb.RayXGBForecast
