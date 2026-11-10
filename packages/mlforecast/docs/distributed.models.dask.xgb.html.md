---
description: dask XGBoost forecaster
output-file: distributed.models.dask.xgb.html
title: DaskXGBForecast
---


Wrapper of `xgboost.dask.DaskXGBRegressor` that adds a `model_` property
that contains the fitted model and is sent to the workers in the
forecasting step.

::: mlforecast.distributed.models.dask.xgb.DaskXGBForecast
