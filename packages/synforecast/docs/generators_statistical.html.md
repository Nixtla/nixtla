---
title: Statistical Generators
description: RandomWalk, Seasonal, SARIMA, and ETS generators
---

::: synforecast.generators.random_walk.RandomWalkGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.seasonal.SeasonalGenerator
    options:
      members:
        - generate_single_series

::: synforecast.generators.sarima.SARIMAGenerator
    options:
      members:
        - generate_single_series
        - get_model_info

::: synforecast.generators.ets.ETSGenerator
    options:
      members:
        - generate_single_series
        - generate_with_states
        - get_model_info
