---
description: Compute transformations on exogenous regressors
output-file: feature_engineering.html
title: Feature engineering | MLForecast
---


```python
import numpy as np
import pandas as pd
from nbdev import show_doc

from mlforecast.lag_transforms import ExpandingMean
from mlforecast.utils import generate_daily_series
```

## Setup

```python
rng = np.random.RandomState(0)
series = generate_daily_series(100, equal_ends=True)
starts_ends = series.groupby(
    'unique_id', observed=True, as_index=False
)['ds'].agg(['min', 'max'])
prices = []
for r in starts_ends.itertuples():
    dates = pd.date_range(r.min, r.max + 14 * pd.offsets.Day())
    df = pd.DataFrame({'ds': dates, 'price': rng.rand(dates.size)})
    df['unique_id'] = r.Index
    prices.append(df)
prices = pd.concat(prices)
prices['price2'] = prices['price'] * rng.rand(prices.shape[0])
prices.head()
```

|     | ds         | price    | unique_id | price2   |
|-----|------------|----------|-----------|----------|
| 0   | 2000-10-05 | 0.548814 | 0         | 0.345011 |
| 1   | 2000-10-06 | 0.715189 | 0         | 0.445598 |
| 2   | 2000-10-07 | 0.602763 | 0         | 0.165147 |
| 3   | 2000-10-08 | 0.544883 | 0         | 0.041373 |
| 4   | 2000-10-09 | 0.423655 | 0         | 0.391577 |

------------------------------------------------------------------------

::: mlforecast.feature_engineering.transform_exog

```python
transformed = transform_exog(
    prices,
    lags=[1, 2],
    lag_transforms={1: [ExpandingMean()]}
)
transformed.head()
```

|  | ds | price | unique_id | price2 | price_lag1 | price_lag2 | price_expanding_mean_lag1 | price2_lag1 | price2_lag2 | price2_expanding_mean_lag1 |
|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 2000-10-05 | 0.548814 | 0 | 0.345011 | NaN | NaN | NaN | NaN | NaN | NaN |
| 1 | 2000-10-06 | 0.715189 | 0 | 0.445598 | 0.548814 | NaN | 0.548814 | 0.345011 | NaN | 0.345011 |
| 2 | 2000-10-07 | 0.602763 | 0 | 0.165147 | 0.715189 | 0.548814 | 0.632001 | 0.445598 | 0.345011 | 0.395304 |
| 3 | 2000-10-08 | 0.544883 | 0 | 0.041373 | 0.602763 | 0.715189 | 0.622255 | 0.165147 | 0.445598 | 0.318585 |
| 4 | 2000-10-09 | 0.423655 | 0 | 0.391577 | 0.544883 | 0.602763 | 0.602912 | 0.041373 | 0.165147 | 0.249282 |

```python
import polars as pl
```


```python
prices_pl = pl.from_pandas(prices)
transformed_pl = transform_exog(
    prices_pl,
    lags=[1, 2],
    lag_transforms={1: [ExpandingMean()]},
    num_threads=2,
)
transformed_pl.head()
```

| ds | price | unique_id | price2 | price_lag1 | price_lag2 | price_expanding_mean_lag1 | price2_lag1 | price2_lag2 | price2_expanding_mean_lag1 |
|----|----|----|----|----|----|----|----|----|----|
| datetime\[ns\] | f64 | i64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 |
| 2000-10-05 00:00:00 | 0.548814 | 0 | 0.345011 | NaN | NaN | NaN | NaN | NaN | NaN |
| 2000-10-06 00:00:00 | 0.715189 | 0 | 0.445598 | 0.548814 | NaN | 0.548814 | 0.345011 | NaN | 0.345011 |
| 2000-10-07 00:00:00 | 0.602763 | 0 | 0.165147 | 0.715189 | 0.548814 | 0.632001 | 0.445598 | 0.345011 | 0.395304 |
| 2000-10-08 00:00:00 | 0.544883 | 0 | 0.041373 | 0.602763 | 0.715189 | 0.622255 | 0.165147 | 0.445598 | 0.318585 |
| 2000-10-09 00:00:00 | 0.423655 | 0 | 0.391577 | 0.544883 | 0.602763 | 0.602912 | 0.041373 | 0.165147 | 0.249282 |
