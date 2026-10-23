---
output-file: utils.html
title: Utils | MLForecast
---


```python
from fastcore.test import test_eq, test_fail
from nbdev import show_doc
```

::: mlforecast.utils.generate_daily_series

Generate 20 series with lengths between 100 and 1,000.

```python
n_series = 20
min_length = 100
max_length = 1000

series = generate_daily_series(n_series, min_length, max_length)
series
```

|       | unique_id | ds         | y        |
|-------|-----------|------------|----------|
| 0     | id_00     | 2000-01-01 | 0.395863 |
| 1     | id_00     | 2000-01-02 | 1.264447 |
| 2     | id_00     | 2000-01-03 | 2.284022 |
| 3     | id_00     | 2000-01-04 | 3.462798 |
| 4     | id_00     | 2000-01-05 | 4.035518 |
| ...   | ...       | ...        | ...      |
| 12446 | id_19     | 2002-03-11 | 0.309275 |
| 12447 | id_19     | 2002-03-12 | 1.189464 |
| 12448 | id_19     | 2002-03-13 | 2.325032 |
| 12449 | id_19     | 2002-03-14 | 3.333198 |
| 12450 | id_19     | 2002-03-15 | 4.306117 |

We can also add static features to each serie (these can be things like
product_id or store_id). Only the first static feature (`static_0`) is
relevant to the target.

```python
n_static_features = 2

series_with_statics = generate_daily_series(n_series, min_length, max_length, n_static_features)
series_with_statics
```

|       | unique_id | ds         | y          | static_0 | static_1 |
|-------|-----------|------------|------------|----------|----------|
| 0     | id_00     | 2000-01-01 | 7.521388   | 18       | 10       |
| 1     | id_00     | 2000-01-02 | 24.024502  | 18       | 10       |
| 2     | id_00     | 2000-01-03 | 43.396423  | 18       | 10       |
| 3     | id_00     | 2000-01-04 | 65.793168  | 18       | 10       |
| 4     | id_00     | 2000-01-05 | 76.674843  | 18       | 10       |
| ...   | ...       | ...        | ...        | ...      | ...      |
| 12446 | id_19     | 2002-03-11 | 27.834771  | 89       | 42       |
| 12447 | id_19     | 2002-03-12 | 107.051746 | 89       | 42       |
| 12448 | id_19     | 2002-03-13 | 209.252845 | 89       | 42       |
| 12449 | id_19     | 2002-03-14 | 299.987801 | 89       | 42       |
| 12450 | id_19     | 2002-03-15 | 387.550536 | 89       | 42       |

```python
for i in range(n_static_features):
    assert all(series_with_statics.groupby('unique_id')[f'static_{i}'].nunique() == 1)
```

If `equal_ends=False` (the default) then every serie has a different end
date.

```python
assert series_with_statics.groupby('unique_id')['ds'].max().nunique() > 1
```

We can have all of them end at the same date by specifying
`equal_ends=True`.

```python
series_equal_ends = generate_daily_series(n_series, min_length, max_length, equal_ends=True)

assert series_equal_ends.groupby('unique_id')['ds'].max().nunique() == 1
```

------------------------------------------------------------------------

::: mlforecast.utils.generate_prices_for_series

```python
series_for_prices = generate_daily_series(20, n_static_features=2, equal_ends=True)
series_for_prices.rename(columns={'static_1': 'product_id'}, inplace=True)
prices_catalog = generate_prices_for_series(series_for_prices, horizon=7)
prices_catalog
```

|      | ds         | unique_id | price    |
|------|------------|-----------|----------|
| 0    | 2000-10-05 | id_00     | 0.548814 |
| 1    | 2000-10-06 | id_00     | 0.715189 |
| 2    | 2000-10-07 | id_00     | 0.602763 |
| 3    | 2000-10-08 | id_00     | 0.544883 |
| 4    | 2000-10-09 | id_00     | 0.423655 |
| ...  | ...        | ...       | ...      |
| 5009 | 2001-05-17 | id_19     | 0.288027 |
| 5010 | 2001-05-18 | id_19     | 0.846305 |
| 5011 | 2001-05-19 | id_19     | 0.791284 |
| 5012 | 2001-05-20 | id_19     | 0.578636 |
| 5013 | 2001-05-21 | id_19     | 0.288589 |

```python
test_eq(set(prices_catalog['unique_id']), set(series_for_prices['unique_id']))
test_fail(lambda: generate_prices_for_series(series), contains='equal ends')
```

------------------------------------------------------------------------

::: mlforecast.utils.PredictionIntervals
    options:
      show_if_no_docstring: false
