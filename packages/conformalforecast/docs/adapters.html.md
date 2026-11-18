---
title: Adapters
description: Uniform interface over statsforecast, mlforecast, neuralforecast and plain callables
---

::: conformalforecast.adapters.CallableAdapter
    options:
      members:
        - horizon
        - n_series
        - predict
        - predict_batch
        - predict_fn
        - supports

::: conformalforecast.adapters.StatsForecastAdapter
    options:
      members:
        - cross_validate
        - cross_validate_quantiles
        - exog_features
        - freq
        - horizon
        - id_col
        - model_name
        - n_series
        - predict
        - predict_batch
        - predict_quantiles
        - refit
        - sf
        - supports
        - target_col
        - time_col
        - train_df

::: conformalforecast.adapters.MLForecastAdapter
    options:
      members:
        - cross_validate
        - exog_features
        - freq
        - horizon
        - id_col
        - mlf
        - model_name
        - n_series
        - predict
        - predict_batch
        - refit
        - supports
        - target_col
        - time_col
        - train_df

::: conformalforecast.adapters.NeuralForecastAdapter
    options:
      members:
        - cross_validate
        - cross_validate_quantiles
        - exog_features
        - freq
        - horizon
        - id_col
        - model_name
        - n_series
        - nf
        - predict
        - predict_batch
        - predict_quantiles
        - refit
        - supports
        - target_col
        - time_col
        - train_df
