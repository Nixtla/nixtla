---
title: Lag transformations
description: Compute lag transforms
---

## Overview

Lag transforms allow you to compute lagged features and rolling statistics over grouped time series data. All transforms work with the `GroupedArray` structure and provide both `transform()` and `update()` methods for batch processing and incremental updates.

## Basic Example

```python
import numpy as np
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import Lag, RollingMean

# Create sample data: two time series
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0])
indptr = np.array([0, 5, 8])  # First series: 5 elements, second: 3 elements
ga = GroupedArray(data, indptr)

# Simple lag
lag2 = Lag(lag=2)
lagged = lag2.transform(ga)

# Rolling mean with lag
rolling_mean = RollingMean(lag=1, window_size=3)
rolling = rolling_mean.transform(ga)
```

## Rolling Window Examples

Rolling window operations compute statistics over a sliding window of observations.

```python
import numpy as np
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import (
    RollingMean, RollingStd, RollingMin, RollingMax, RollingQuantile
)

# Sample time series data
data = np.array([10.0, 12.0, 15.0, 14.0, 18.0, 20.0, 22.0, 19.0])
indptr = np.array([0, 8])
ga = GroupedArray(data, indptr)

# Rolling mean with window size 3, lag 1
rolling_mean = RollingMean(lag=1, window_size=3)
mean_result = rolling_mean.transform(ga)
# Computes mean of last 3 values with lag 1

# Rolling standard deviation
rolling_std = RollingStd(lag=1, window_size=3, min_samples=2)
std_result = rolling_std.transform(ga)

# Rolling minimum and maximum
rolling_min = RollingMin(lag=1, window_size=3)
rolling_max = RollingMax(lag=1, window_size=3)
min_result = rolling_min.transform(ga)
max_result = rolling_max.transform(ga)

# Rolling median (50th percentile)
rolling_median = RollingQuantile(lag=1, p=0.5, window_size=3)
median_result = rolling_median.transform(ga)
```

## Seasonal Rolling Examples

Seasonal rolling operations compute statistics over windows that respect seasonality patterns.

```python
import numpy as np
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import (
    SeasonalRollingMean, SeasonalRollingStd, SeasonalRollingMin,
    SeasonalRollingMax, SeasonalRollingQuantile
)

# Daily data with weekly seasonality (14 days)
data = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 25.0,
                 11.0, 16.0, 13.0, 19.0, 21.0, 23.0, 26.0])
indptr = np.array([0, 14])
ga = GroupedArray(data, indptr)

# Seasonal rolling mean with weekly pattern
seasonal_mean = SeasonalRollingMean(
    lag=1,
    season_length=7,  # Weekly seasonality
    window_size=2     # Use last 2 seasonal observations
)
seasonal_result = seasonal_mean.transform(ga)
# Computes mean using observations from the same day of week

# Seasonal rolling std
seasonal_std = SeasonalRollingStd(lag=1, season_length=7, window_size=2)
seasonal_std_result = seasonal_std.transform(ga)

# Seasonal rolling min/max
seasonal_min = SeasonalRollingMin(lag=1, season_length=7, window_size=2)
seasonal_max = SeasonalRollingMax(lag=1, season_length=7, window_size=2)

# Seasonal rolling quantile
seasonal_q90 = SeasonalRollingQuantile(
    lag=1, p=0.9, season_length=7, window_size=2
)
```

## Expanding Window Examples

Expanding windows compute cumulative statistics from the start of each series.

```python
import numpy as np
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import (
    ExpandingMean, ExpandingStd, ExpandingMin, ExpandingMax, ExpandingQuantile
)

# Sample data: two time series
data = np.array([5.0, 10.0, 8.0, 12.0, 15.0, 20.0, 25.0, 30.0])
indptr = np.array([0, 5, 8])
ga = GroupedArray(data, indptr)

# Expanding mean (cumulative average)
exp_mean = ExpandingMean(lag=1)
cumulative_avg = exp_mean.transform(ga)
# Each value is the mean of all previous observations

# Expanding standard deviation
exp_std = ExpandingStd(lag=1)
cumulative_std = exp_std.transform(ga)

# Expanding min and max
exp_min = ExpandingMin(lag=1)
exp_max = ExpandingMax(lag=1)
running_min = exp_min.transform(ga)
running_max = exp_max.transform(ga)

# Expanding quantile
exp_median = ExpandingQuantile(lag=1, p=0.5)
running_median = exp_median.transform(ga)
```

## Exponentially Weighted Mean Example

The exponentially weighted mean gives more weight to recent observations.

```python
import numpy as np
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import ExponentiallyWeightedMean

# Sample data
data = np.array([10.0, 12.0, 15.0, 14.0, 18.0, 20.0])
indptr = np.array([0, 6])
ga = GroupedArray(data, indptr)

# Exponentially weighted mean with alpha=0.3
# Higher alpha = more weight to recent values
ewm = ExponentiallyWeightedMean(lag=1, alpha=0.3)
smoothed = ewm.transform(ga)
```

## Update Method for Incremental Processing

All transforms provide an `update()` method for efficient incremental computation when new data arrives.

```python
import numpy as np
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import ExpandingMean

# Initial data
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
indptr = np.array([0, 5])
ga = GroupedArray(data, indptr)

# Transform to initialize statistics
exp_mean = ExpandingMean(lag=1)
result = exp_mean.transform(ga)

# New observation arrives
new_data = np.array([6.0])
new_indptr = np.array([0, 1])
new_ga = GroupedArray(new_data, new_indptr)

# Update statistics incrementally (much faster than re-transforming)
updated_value = exp_mean.update(new_ga)
# Returns the updated expanding mean for the new observation
```

## Available lag transformations

::: coreforecast.lag_transforms.Lag
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.RollingMean
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.RollingStd
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.RollingMin
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.RollingMax
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.RollingQuantile
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.SeasonalRollingMean
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.SeasonalRollingStd
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.SeasonalRollingMin
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.SeasonalRollingMax
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.SeasonalRollingQuantile
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true
  
::: coreforecast.lag_transforms.ExpandingMean
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.ExpandingStd
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.ExpandingMin
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.ExpandingMax
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.ExpandingQuantile
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true

::: coreforecast.lag_transforms.ExponentiallyWeightedMean
    handler: python
    options:
      docstring_style: google
      members: [__init__]
      heading_level: 3
      show_root_heading: true
      show_source: true
