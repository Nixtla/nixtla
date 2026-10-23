---
description: ray LightGBM forecaster
output-file: distributed.models.ray.lgb.html
title: RayLGBMForecast
---


LightGBM forecaster trained with `ray.train.lightgbm.LightGBMTrainer`. Adds a
`model_` attribute that contains the fitted booster as a local
`lightgbm.LGBMRegressor` and is sent to the workers in the forecasting step.

::: mlforecast.distributed.models.ray.lgb.RayLGBMForecast
