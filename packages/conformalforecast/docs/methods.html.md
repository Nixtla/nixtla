---
title: Methods
description: Each conformal method is a declarative config whose fit returns a fitted forecaster
---

## Contracts

::: conformalforecast.methods.base.ConformalMethod
    options:
      members:
        - alpha
        - seed
        - score
        - fit
        - calibrate
        - calibrate_cv

::: conformalforecast.methods.base.ConformalForecaster
    options:
      members:
        - interval
        - update
        - predict
        - recalibrate
        - calibration_intervals
        - method_
        - score_fn_
        - n_calibration_samples_
        - calibration_
        - forecaster_
        - predictions_calibration_
        - truths_calibration_
        - quantile_predictions_calibration_

## Offline methods

::: conformalforecast.methods.SplitConformal
    options:
      members:
        - fit
        - _default_score

::: conformalforecast.methods.ConformalizedQuantileRegression
    options:
      members:
        - fit
        - quantile_levels
        - _default_score

::: conformalforecast.methods.FixedOffsetForecaster
    options:
      members:
        - interval
        - score_quantile_

::: conformalforecast.methods.WeightedSplit
    options:
      members:
        - fit
        - _default_score

::: conformalforecast.methods.WeightedSplitForecaster
    options:
      members:
        - interval
        - base_quantile_
        - score_quantile_

::: conformalforecast.methods.transfer.SupportsTransfer
    options:
      members:
        - adapt
        - weighting_
        - scores_
        - X_calibration_
        - source_scales_
        - target_scales_
        - scale_ratio_

## Online methods

::: conformalforecast.methods.AdaptiveConformalInference
    options:
      members:
        - fit
        - _default_score

::: conformalforecast.methods.ACIForecaster
    options:
      members:
        - interval
        - update
        - alpha_t_
        - n_observations_
        - scores_

::: conformalforecast.methods.AggregatedAdaptiveConformalInference
    options:
      members:
        - fit
        - n_experts
        - _default_score

::: conformalforecast.methods.AgACIForecaster
    options:
      members:
        - interval
        - update
        - alpha_t_per_expert_
        - aggregator_lower_
        - aggregator_upper_
        - interval_clip_lower_
        - interval_clip_upper_
        - n_observations_
        - scores_

::: conformalforecast.methods.NonexchangeableConformalPrediction
    options:
      members:
        - fit
        - _default_score

::: conformalforecast.methods.NexCPForecaster
    options:
      members:
        - interval
        - update
        - score_quantile_
        - n_observations_
        - scores_

::: conformalforecast.methods.SequentialPredictiveConformalInference
    options:
      members:
        - fit
        - _default_score

::: conformalforecast.methods.SPCIForecaster
    options:
      members:
        - interval
        - calibration_intervals
        - update
        - quantile_regressors_
        - residuals_
        - X_
        - n_observations_
        - steps_since_refit_
