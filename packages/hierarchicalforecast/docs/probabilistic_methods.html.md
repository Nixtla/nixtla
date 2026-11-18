---
output-file: probabilistic_methods.html
title: Probabilistic Methods
---


Here we provide a collection of methods designed to provide
hierarchically coherent probabilistic distributions, which means that
they generate samples of multivariate time series with hierarchical
linear constraints.

We designed these methods to extend the `core.HierarchicalForecast`
capabilities class. Check their [usage example
here](https://nixtlaverse.nixtla.io/hierarchicalforecast/examples/introduction.html).

## 1. Normality

::: hierarchicalforecast.probabilistic_methods.Normality
    handler: python
    options:
      docstring_style: google
      members:
        - get_samples
      heading_level: 3
      show_root_heading: true
      show_source: true

## 2. Bootstrap

::: hierarchicalforecast.probabilistic_methods.Bootstrap
    handler: python
    options:
      docstring_style: google
      members:
        - get_samples
      heading_level: 3
      show_root_heading: true
      show_source: true

## 3. PERMBU

::: hierarchicalforecast.probabilistic_methods.PERMBU
    handler: python
    options:
      docstring_style: google
      members:
        - get_samples
      heading_level: 3
      show_root_heading: true
      show_source: true

## References

- [Rob J. Hyndman and George Athanasopoulos (2018). “Forecasting
    principles and practice, Reconciled distributional
    forecasts”.](https://otexts.com/fpp3/rec-prob.html)
- [Puwasala Gamakumara Ph. D. dissertation. Monash University,
    Econometrics and Business Statistics (2020). “Probabilistic Forecast
    Reconciliation”](https://bridges.monash.edu/articles/thesis/Probabilistic_Forecast_Reconciliation_Theory_and_Applications/11869533)
- [Panagiotelis A., Gamakumara P. Athanasopoulos G., and Hyndman R. J.
    (2022). “Probabilistic forecast reconciliation: Properties,
    evaluation and score optimisation”. European Journal of Operational
    Research.](https://www.sciencedirect.com/science/article/pii/S0377221722006087)
- [Taieb, Souhaib Ben and Taylor, James W and Hyndman, Rob J. (2017).
    Coherent probabilistic forecasts for hierarchical time series.
    International conference on machine learning
    ICML.](https://proceedings.mlr.press/v70/taieb17a.html)
