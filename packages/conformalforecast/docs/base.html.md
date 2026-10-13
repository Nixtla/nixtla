---
title: Core contracts
description: ForecasterAdapter, ScoreFunction, results and exceptions
---

::: conformalforecast.base.ForecasterAdapter
    options:
      members:
        - predict
        - predict_batch
        - supports

::: conformalforecast.base.ScoreFunction
    options:
      members:
        - kind
        - fit
        - score
        - invert
        - from_kind

::: conformalforecast.base.CalibrationResult
    options:
      members:
        - n_calibration_samples
        - diagnostics

::: conformalforecast.base.PredictionResult
    options:
      members:
        - point
        - interval
        - alpha

## Exceptions

::: conformalforecast.base.ConformalForecastError

::: conformalforecast.base.CalibrationError

::: conformalforecast.base.InsufficientCalibrationSamples

::: conformalforecast.base.UnsupportedCapability
