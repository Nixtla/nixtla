---
output-file: core.html
title: Core | MLForecast
---


```python
import datetime

import tempfile
from nbdev import show_doc
from fastcore.test import test_eq, test_fail, test_warns

from mlforecast.callbacks import SaveFeatures
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.utils import generate_daily_series, generate_prices_for_series
```

## Data format

The required input format is a dataframe with at least the following
columns:

* `unique_id` with a unique identifier for each time serie
* `ds` with the datestamp and a column
* `y` with the values of theserie

Every other column is considered a static feature unless stated
otherwise in `TimeSeries.fit`

```python
series = generate_daily_series(20, n_static_features=2)
series
```

|      | unique_id | ds         | y          | static_0 | static_1 |
|------|-----------|------------|------------|----------|----------|
| 0    | id_00     | 2000-01-01 | 7.404529   | 27       | 53       |
| 1    | id_00     | 2000-01-02 | 35.952624  | 27       | 53       |
| 2    | id_00     | 2000-01-03 | 68.958353  | 27       | 53       |
| 3    | id_00     | 2000-01-04 | 84.994505  | 27       | 53       |
| 4    | id_00     | 2000-01-05 | 113.219810 | 27       | 53       |
| ...  | ...       | ...        | ...        | ...      | ...      |
| 4869 | id_19     | 2000-03-25 | 400.606807 | 97       | 45       |
| 4870 | id_19     | 2000-03-26 | 538.794824 | 97       | 45       |
| 4871 | id_19     | 2000-03-27 | 620.202104 | 97       | 45       |
| 4872 | id_19     | 2000-03-28 | 20.625426  | 97       | 45       |
| 4873 | id_19     | 2000-03-29 | 141.513169 | 97       | 45       |

For simplicity we’ll just take one time serie here.

```python
uids = series['unique_id'].unique()
serie = series[series['unique_id'].eq(uids[0])]
serie
```

|     | unique_id | ds         | y          | static_0 | static_1 |
|-----|-----------|------------|------------|----------|----------|
| 0   | id_00     | 2000-01-01 | 7.404529   | 27       | 53       |
| 1   | id_00     | 2000-01-02 | 35.952624  | 27       | 53       |
| 2   | id_00     | 2000-01-03 | 68.958353  | 27       | 53       |
| 3   | id_00     | 2000-01-04 | 84.994505  | 27       | 53       |
| 4   | id_00     | 2000-01-05 | 113.219810 | 27       | 53       |
| ... | ...       | ...        | ...        | ...      | ...      |
| 217 | id_00     | 2000-08-05 | 13.263188  | 27       | 53       |
| 218 | id_00     | 2000-08-06 | 38.231981  | 27       | 53       |
| 219 | id_00     | 2000-08-07 | 59.555183  | 27       | 53       |
| 220 | id_00     | 2000-08-08 | 86.986368  | 27       | 53       |
| 221 | id_00     | 2000-08-09 | 119.254810 | 27       | 53       |

------------------------------------------------------------------------

::: mlforecast.core.TimeSeries
    handler: python
    options:
      docstring_style: google
      members:
        - fit_transform
        - predict
        - update
      heading_level: 3
      show_root_heading: true
      show_source: true

```python
def month_start_or_end(dates):
    return dates.is_month_start | dates.is_month_end

flow_config = dict(
    freq='W-THU',
    lags=[7],
    lag_transforms={
        1: [ExpandingMean(), RollingMean(7)]
    },
    date_features=['dayofweek', 'week', month_start_or_end]
)

ts = TimeSeries(**flow_config)
ts
```

``` text
TimeSeries(freq=W-THU, transforms=['lag7', 'expanding_mean_lag1', 'rolling_mean_lag1_window_size7'], date_features=['dayofweek', 'week', 'month_start_or_end'], num_threads=1)
```

The frequency is converted to an offset.

```python
test_eq(ts.freq, pd.tseries.frequencies.to_offset(flow_config['freq']))
```

The date features are stored as they were passed to the constructor.

```python
test_eq(ts.date_features, flow_config['date_features'])
```

The transformations are stored as a dictionary where the key is the name
of the transformation (name of the column in the dataframe with the
computed features), which is built using `build_transform_name` and the
value is a tuple where the first element is the lag it is applied to,
then the function and then the function arguments.

```python
test_eq(
    ts.transforms.keys(),
    ['lag7', 'expanding_mean_lag1', 'rolling_mean_lag1_window_size7'],
)
```

Note that for `lags` we define the transformation as the identity
function applied to its corresponding lag. This is because
[`_transform_series`](https://Nixtla.github.io/mlforecast/grouped_array.html#_transform_series)
takes the lag as an argument and shifts the array before computing the
transformation.

```python
flow_config = dict(
    freq='D',
    lags=[7, 14],
    lag_transforms={
        2: [
            RollingMean(7),
            RollingMean(14),
        ]
    },
    date_features=['dayofweek', 'month', 'year'],
    num_threads=2
)

ts = TimeSeries(**flow_config)
_ = ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y')
```

The series values are stored as a GroupedArray in an attribute `ga`. If
the data type of the series values is an int then it is converted to
`np.float32`, this is because lags generate `np.nan`s so we need a float
data type for them.

```python
np.testing.assert_equal(
    ts.ga.data,
    series.groupby('unique_id', observed=True).tail(ts.keep_last_n)['y'],
)
```

The series ids are stored in an `uids` attribute.

```python
test_eq(ts.uids, series['unique_id'].unique())
```

For each time serie, the last observed date is stored so that
predictions start from the last date + the frequency.

```python
test_eq(ts.last_dates, series.groupby('unique_id', observed=True)['ds'].max().values)
```

The last row of every serie without the `y` and `ds` columns are taken
as static features.

```python
pd.testing.assert_frame_equal(
    ts.static_features_,
    series.groupby('unique_id', observed=True).tail(1).drop(columns=['ds', 'y']).reset_index(drop=True),
)
```

If you pass `static_features` to
[`TimeSeries.fit_transform`](https://Nixtla.github.io/mlforecast/core.html#timeseries.fit_transform)
then only these are kept.

```python
ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y', static_features=['static_0'])

pd.testing.assert_frame_equal(
    ts.static_features_,
    series.groupby('unique_id', observed=True).tail(1)[['unique_id', 'static_0']].reset_index(drop=True),
)
```

You can also specify keep_last_n in TimeSeries.fit_transform, which
means that after computing the features for training we want to keep
only the last n samples of each time serie for computing the updates.
This saves both memory and time, since the updates are performed by
running the transformation functions on all time series again and
keeping only the last value (the update).

If you have very long time series and your updates only require a small
sample it’s recommended that you set keep_last_n to the minimum number
of samples required to compute the updates, which in this case is 15
since we have a rolling mean of size 14 over the lag 2 and in the first
update the lag 2 becomes the lag 1. This is because in the first update
the lag 1 is the last value of the series (or the lag 0), the lag 2 is
the lag 1 and so on.

```python
keep_last_n = 15

ts = TimeSeries(**flow_config)
df = ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y', keep_last_n=keep_last_n)
ts._predict_setup()

expected_lags = ['lag7', 'lag14']
expected_transforms = ['rolling_mean_lag2_window_size7',
                       'rolling_mean_lag2_window_size14']
expected_date_features = ['dayofweek', 'month', 'year']

test_eq(ts.features, expected_lags + expected_transforms + expected_date_features)
test_eq(ts.static_features_.columns.tolist() + ts.features, df.columns.drop(['ds', 'y']).tolist())
# we dropped 2 rows because of the lag 2 and 13 more to have the window of size 14
test_eq(df.shape[0], series.shape[0] - (2 + 13) * ts.ga.n_groups)
test_eq(ts.ga.data.size, ts.ga.n_groups * keep_last_n)
```

[`TimeSeries.fit_transform`](https://Nixtla.github.io/mlforecast/core.html#timeseries.fit_transform)
requires that the *y* column doesn’t have any null values. This is
because the transformations could propagate them forward, so if you have
null values in the *y* column you’ll get an error.

```python
series_with_nulls = series.copy()
series_with_nulls.loc[1, 'y'] = np.nan
test_fail(
    lambda: ts.fit_transform(series_with_nulls, id_col='unique_id', time_col='ds', target_col='y'),
    contains='y column contains null values'
)
```

Once we have a trained model we can use
[`TimeSeries.predict`](https://Nixtla.github.io/mlforecast/core.html#timeseries.predict)
passing the model and the horizon to get the predictions back.

```python
class DummyModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X['lag7'].values

horizon = 7
model = DummyModel()
ts = TimeSeries(**flow_config)
ts.fit_transform(series, id_col='unique_id', time_col='ds', target_col='y')
predictions = ts.predict({'DummyModel': model}, horizon)

grouped_series = series.groupby('unique_id', observed=True)
expected_preds = grouped_series['y'].tail(7)  # the model predicts the lag-7
last_dates = grouped_series['ds'].max()
expected_dsmin = last_dates + pd.offsets.Day()
expected_dsmax = last_dates + horizon * pd.offsets.Day()
grouped_preds = predictions.groupby('unique_id', observed=True)

np.testing.assert_allclose(predictions['DummyModel'], expected_preds)
pd.testing.assert_series_equal(grouped_preds['ds'].min(), expected_dsmin)
pd.testing.assert_series_equal(grouped_preds['ds'].max(), expected_dsmax)
```

If we have dynamic features we can pass them to `X_df`.

```python
class PredictPrice:
    def predict(self, X):
        return X['price']

series = generate_daily_series(20, n_static_features=2, equal_ends=True)
dynamic_series = series.rename(columns={'static_1': 'product_id'})
prices_catalog = generate_prices_for_series(dynamic_series)
series_with_prices = dynamic_series.merge(prices_catalog, how='left')

model = PredictPrice()
ts = TimeSeries(**flow_config)
ts.fit_transform(
    series_with_prices,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
    static_features=['static_0', 'product_id'],
)
predictions = ts.predict({'PredictPrice': model}, horizon=1, X_df=prices_catalog)
pd.testing.assert_frame_equal(
    predictions.rename(columns={'PredictPrice': 'price'}),
    prices_catalog.merge(predictions[['unique_id', 'ds']])[['unique_id', 'ds', 'price']]
)
```
