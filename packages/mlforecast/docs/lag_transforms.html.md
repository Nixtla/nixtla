---
description: Built-in lag transformations
output-file: lag_transforms.html
title: Lag transforms
---

##

The `mlforecast.lag_transforms` module provides built-in **lag
transformations**: statistics computed over lagged values of the target that
are used as features by the forecasting model. You pass them to `MLForecast`
through the `lag_transforms` argument, a dict whose keys are the lags to apply
the transformation to and whose values are lists of transformation instances.

```python
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingStd, RollingMean

fcst = MLForecast(
    models=[...],
    freq='D',
    lag_transforms={
        1: [ExpandingStd()],
        7: [RollingMean(window_size=7), RollingMean(window_size=28)],
    },
)
```

The transforms fall into four families, each with several variants:

- **Rolling** — `RollingMean`, `RollingStd`, `RollingMin`, `RollingMax`,
  `RollingQuantile`: fixed-window statistics over the lagged target.
- **Seasonal rolling** — `SeasonalRollingMean`, `SeasonalRollingStd`,
  `SeasonalRollingMin`, `SeasonalRollingMax`, `SeasonalRollingQuantile`:
  rolling statistics computed across same-position observations in successive
  seasons (e.g. last 4 Mondays).
- **Expanding** — `ExpandingMean`, `ExpandingStd`, `ExpandingMin`,
  `ExpandingMax`, `ExpandingQuantile`: statistics over all observations up to
  the lag.
- **Exponentially weighted** — `ExponentiallyWeightedMean`: a weighted mean
  that emphasises recent observations.

Two combinators let you build richer features from these primitives:
**`Offset`** applies a transformation at a shifted lag, and **`Combine`**
joins two transformations with a binary operator (for example a ratio of two
rolling means at different windows).

The basic usage is per-series — each transformation is computed independently
for every series. The next section describes how to instead compute these
statistics **across multiple series at once**.

For a worked walkthrough of all of the above, including the `Combine` /
`Offset` combinators and how to plug in custom numba-based transforms, see
the [Lag transformations](docs/how-to-guides/lag_transforms_guide.html)
how-to guide.

## Pooled mode: `global_` and `groupby`

Every built-in rolling, expanding, seasonal-rolling, and exponentially weighted
transform accepts two pooling parameters that let you compute the statistic
across **multiple series at once**:

- **`global_: bool`** — when `True`, the statistic is computed across **all
  series** aggregated by timestamp. Every series receives the same feature
  value at each timestamp. Requires every series to end at the same timestamp.
- **`groupby: Sequence[str]`** — column names to group by before computing the
  statistic. Columns must be declared as static features when calling
  `fit` / `preprocess`. Series in the same group share the feature value at
  each timestamp; series in different groups get different values.

`global_` and `groupby` are **mutually exclusive** on the same transform.

**RANGE semantics.** Pooled transforms use SQL-style
`RANGE BETWEEN ... PRECEDING` windows over actual timestamps, not row
positions. Series with staggered starts simply do not contribute to the window
until they have observations — no synthetic zeros are injected. Pooled mode
assumes a **continuous, gap-free time grid** within each series; combining
`validate_data=False` with a pooled transform raises a `UserWarning`.

**`min_samples` divergence.** In local (per-series) mode, `min_samples` is
capped at `window_size` by `coreforecast`. In pooled mode, `min_samples`
counts **total non-NaN observations across all series** in the bucket within
the rolling window, with no capping. This makes it useful as a coverage
threshold: `RollingMean(window_size=1, min_samples=2, groupby=["brand"])`
produces a non-null value only at timestamps where at least two series in the
brand contribute observations.

See the [Pooled lag transforms](docs/how-to-guides/pooled_lag_transforms.html)
how-to guide for end-to-end examples.

::: mlforecast.lag_transforms.RollingQuantile
    options:
      show_if_no_docstring: false

::: mlforecast.lag_transforms.RollingMax
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.RollingMin
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.RollingStd
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.RollingMean
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.SeasonalRollingQuantile
    options:
      show_if_no_docstring: false

::: mlforecast.lag_transforms.SeasonalRollingMax
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.SeasonalRollingMin
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.SeasonalRollingStd
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.SeasonalRollingMean
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.ExpandingQuantile
    options:
      show_if_no_docstring: false

::: mlforecast.lag_transforms.ExpandingMax
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.ExpandingMin
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.ExpandingStd
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.ExpandingMean
    options:
      show_if_no_docstring: false
      merge_init_into_class: true
      inherited_members: true

::: mlforecast.lag_transforms.ExponentiallyWeightedMean
    options:
      show_if_no_docstring: false

::: mlforecast.lag_transforms.Offset
    options:
      show_if_no_docstring: false

::: mlforecast.lag_transforms.Combine
    options:
      show_if_no_docstring: false
