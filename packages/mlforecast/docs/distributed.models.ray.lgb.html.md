---
description: ray LightGBM forecaster
output-file: distributed.models.ray.lgb.html
title: RayLGBMForecast
---


Wrapper of `lightgbm.ray.RayLGBMRegressor` that adds a `model_` property
that contains the fitted booster and is sent to the workers to in the
forecasting step.

::: mlforecast.distributed.models.ray.lgb.RayLGBMForecast
