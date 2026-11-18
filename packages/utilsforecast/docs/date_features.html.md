---
title: Date Features | UtilsForecast
description: Backend-agnostic calendar features
---

The `utilsforecast.date_features` module provides calendar features that
compute the same values with the same dtypes for pandas (including pyarrow
backed, which require pandas>=2) and polars dates. They can be passed to
`utilsforecast.feature_engineering.time_features`.

- Time zone aware dates use their local time, as given by each backend's
  time zone database. pandas and polars can disagree, e.g. for daylight
  saving time after 2037.
- Dates without a time (e.g. polars `Date`) have their hour, minute and
  second set to 0.
- Null dates raise an error, also in `time_features` with any kind of feature.

```python
from utilsforecast import date_features as dtf
from utilsforecast.feature_engineering import time_features

train, future = time_features(
    df,
    freq='D',
    features=[dtf.day_of_week, dtf.month, dtf.is_month_end],
    h=7,
)
```

`dtf.available()` lists every calendar feature:

| Feature | Values | dtype |
|---------|--------|-------|
| `year` | Year. | `int16` |
| `quarter` | Quarter of the year, from 1 to 4. | `int8` |
| `month` | Month of the year, from 1 to 12. | `int8` |
| `week` | ISO week of the year, from 1 to 53. | `int8` |
| `day` | Day of the month, from 1 to 31. | `int8` |
| `day_of_week` | Day of the week, from 0 (Monday) to 6 (Sunday). | `int8` |
| `day_of_year` | Day of the year, from 1 to 366. | `int16` |
| `days_in_month` | Number of days in the month, from 28 to 31. | `int8` |
| `hour` | Hour of the day, from 0 to 23. | `int8` |
| `minute` | Minute of the hour, from 0 to 59. | `int8` |
| `second` | Second of the minute, from 0 to 59. | `int8` |
| `is_month_start` | 1 on the first day of the month, 0 otherwise. | `int8` |
| `is_month_end` | 1 on the last day of the month, 0 otherwise. | `int8` |
| `is_quarter_start` | 1 on the first day of the quarter, 0 otherwise. | `int8` |
| `is_quarter_end` | 1 on the last day of the quarter, 0 otherwise. | `int8` |
| `is_year_start` | 1 on January 1st, 0 otherwise. | `int8` |
| `is_year_end` | 1 on December 31st, 0 otherwise. | `int8` |

::: utilsforecast.date_features.available
    handler: python
    options:
      docstring_style: google
      heading_level: 3
      show_root_heading: true
      show_source: true
