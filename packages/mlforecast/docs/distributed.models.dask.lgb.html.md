---
description: dask LightGBM forecaster
output-file: distributed.models.dask.lgb.html
title: DaskLGBMForecast
---


Wrapper of `lightgbm.dask.DaskLGBMRegressor` that adds a `model_`
property that contains the fitted booster and is sent to the workers to
in the forecasting step.

::: mlforecast.distributed.models.dask.lgb.DaskLGBMForecast
