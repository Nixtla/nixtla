---
title: Weighting
description: Weight functions for covariate-shift-weighted conformal prediction
---

::: conformalforecast.weighting.WeightFunction
    options:
      members:
        - fit_source
        - fit_target
        - calibration_weights
        - test_weights

::: conformalforecast.weighting.DensityRatioEstimator
    options:
      members:
        - fit
        - predict_proba

::: conformalforecast.weighting.DensityRatio
    options:
      members:
        - fit_target
        - calibration_weights
        - test_weights

::: conformalforecast.weighting.UniformWeights
    options:
      members:
        - calibration_weights
        - test_weights

::: conformalforecast.weighting.PrecomputedWeights
    options:
      members:
        - fit_source
        - calibration_weights
        - test_weights
