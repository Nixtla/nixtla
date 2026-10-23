---
output-file: methods.html
title: Reconciliation Methods
---


In hierarchical forecasting, we aim to create forecasts for many time
series concurrently, whilst adhering to pre-specified hierarchical
relationships that exist between the time series. We can enforce this
coherence by performing a post-processing reconciliation step on the
forecasts.

The `HierarchicalForecast` package provides the most comprehensive
collection of Python implementations of hierarchical forecasting
algorithms that follow classic hierarchical reconciliation. All the
methods have a `reconcile` function capable of reconciling base
forecasts using `numpy` arrays.

## Cross-sectional hierarchies

Traditionally, hierarchical forecasting methods reconcile
*cross-sectional* aggregations. For example, we may have forecasts for
individual product demand, but also for the overall product group,
department and store, and we are interested in making sure these
forecasts are coherent with each other. This can be formalized as:

$$\tilde{\textbf{Y}} = SP\hat{\textbf{Y}} \;, $$

where $\hat{\textbf{Y}} \in \mathbb{R}^{m \times p}$ denotes the matrix
of forecasts for all $m$ time series for all $p$ time steps in the
hierarchy, $S \in \lbrace 0, 1 \rbrace^{m \times n}$ is a matrix that
defines the hierarchical relationship between the $n$ bottom-level time
series and the $m^* = m - n$ aggregations,
$P \in \mathbb{R}^{n \times m}$ is a matrix that encapsulates the
contribution of each forecast to the final estimate, and
$\tilde{\textbf{Y}} \in \mathbb{R}^{m \times p}$ is the matrix of
reconciled forecasts. We can use the matrix $P$ to define various
forecast contribution scenarios.

Cross-sectional reconciliation methods aim to find the optimal $P$
matrix.

## Temporal hierarchies

We can also perform *temporal* reconciliation. For example, we may have
forecasts for daily demand, weekly, and monthly, and we are interested
in making sure these forecasts are coherent with each other. We
formalize the temporal hierarchical forecasting problem as:

$$\tilde{\textbf{Y}} = \left( S_{te} P_{te} \hat{\textbf{Y}}^{\intercal} \right)^{\intercal}  \;, $$

where $S_{te} \in \lbrace 0, 1 \rbrace^{p \times k}$ is a matrix that
defines the hierarchical relationship between the $k$ bottom-level time
steps and the $p^* = p - k$ aggregations and
$P_{te} \in \mathbb{R}^{k \times p}$ is a matrix that encapsulates the
contribution of each forecast to the final estimate. We can use the
matrix $P_{te}$ to define various forecast contribution scenarios.

Temporal reconciliation methods aim to find the optimal $P_{te}$ matrix.

## Cross-temporal reconciliation

We can combine cross-sectional and temporal hierarchical forecasting by
performing cross-sectional reconciliation and temporal reconciliation in
a two-step procedure.

## References

-[Hyndman, Rob. Notation for forecast reconciliation.](https://robjhyndman.com/hyndsight/reconciliation-notation.html)

## 1. Bottom-Up

------------------------------------------------------------------------

::: hierarchicalforecast.methods.BottomUp
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

::: hierarchicalforecast.methods.BottomUpSparse
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

## 2. Top-Down

::: hierarchicalforecast.methods.TopDown
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

::: hierarchicalforecast.methods.TopDownSparse
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

```python
cls_top_down(
                S=S, y_hat=S @ y_hat_bottom, y_insample=S @ y_bottom, tags=tags
            )["mean"]
```

```python
cls_top_down = TopDownSparse(method="average_proportions")
test_fail(
    cls_top_down,
    contains="Top-down reconciliation requires strictly hierarchical structures.",
    args=(sparse.csr_matrix(S_non_hier), None, tags_non_hier),
)
```

## 3. Middle-Out

::: hierarchicalforecast.methods.MiddleOut
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

::: hierarchicalforecast.methods.MiddleOutSparse
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

## 4. Min-Trace

::: hierarchicalforecast.methods.MinTrace
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

::: hierarchicalforecast.methods.MinTraceSparse
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

## 5. Optimal Combination

::: hierarchicalforecast.methods.OptimalCombination
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

## 6. Emp. Risk Minimization

::: hierarchicalforecast.methods.ERM
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - fit_predict
        - sample
      heading_level: 3
      show_root_heading: true
      show_source: true

## References

### General Reconciliation

- [Orcutt, G.H., Watts, H.W., & Edwards, J.B.(1968). Data aggregation
    and information loss. The American Economic Review, 58 ,
    773(787).](http://www.jstor.org/stable/1815532)
- [Disaggregation methods to expedite product line forecasting.
    Journal of Forecasting, 9 , 233–254.
    doi:10.1002/for.3980090304](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3980090304).
- [An investigation of aggregate variable time series forecast
    strategies with specific subaggregate time series statistical
    correlation. Computers and Operations Research, 26 , 1133–1149.
    doi:10.1016/S0305-0548(99)00017-9.](https://doi.org/10.1016/S0305-0548(99)00017-9)
- [Hyndman, R.J., & Athanasopoulos, G. (2021). “Forecasting:
    principles and practice, 3rd edition: Chapter 11: Forecasting
    hierarchical and grouped series.”. OTexts: Melbourne, Australia.
    OTexts.com/fpp3 Accessed on July
    2022.](https://otexts.com/fpp3/hierarchical.html)
- [Rob J. Hyndman, Roman A. Ahmed, George Athanasopoulos, Han Lin
    Shang. “Optimal Combination Forecasts for Hierarchical Time Series”
    (2010).](https://robjhyndman.com/papers/Hierarchical6.pdf)
- [Shanika L. Wickramasuriya, George Athanasopoulos and Rob J.
    Hyndman. “Optimal Combination Forecasts for Hierarchical Time
    Series” (2010).](https://robjhyndman.com/papers/MinT.pdf)
- [Ben Taieb, S., & Koo, B. (2019). Regularized regression for
    hierarchical forecasting without unbiasedness conditions. In
    Proceedings of the 25th ACM SIGKDD International Conference on
    Knowledge Discovery & Data Mining KDD ’19 (p. 1337-1347). New York,
    NY, USA: Association for Computing
    Machinery.](https://doi.org/10.1145/3292500.3330976)

### Hierarchical Probabilistic Coherent Predictions

- [Puwasala Gamakumara Ph. D. dissertation. Monash University,
    Econometrics and Business Statistics. “Probabilistic Forecast
    Reconciliation”.](https://bridges.monash.edu/articles/thesis/Probabilistic_Forecast_Reconciliation_Theory_and_Applications/11869533)
- [Taieb, Souhaib Ben and Taylor, James W and Hyndman, Rob J. (2017).
    Coherent probabilistic forecasts for hierarchical time series.
    International conference on machine learning
    ICML.](https://proceedings.mlr.press/v70/taieb17a.html)
