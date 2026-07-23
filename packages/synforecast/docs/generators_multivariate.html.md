---
title: Multivariate Generators
description: Copula and VAR generators for correlated time series
---

::: synforecast.generators.copula.CopulaGenerator
    options:
      members:
        - generate_single_series
        - generate

::: synforecast.generators.var.VARGenerator
    options:
      members:
        - generate_single_series
        - generate

::: synforecast.generators.gaussian_process.GaussianProcessGenerator
    options:
      members:
        - generate_single_series
        - get_model_info
