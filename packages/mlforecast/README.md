# mlforecast
> Scalable machine learning based time series forecasting.


## Install

`pip install mlforecast`

## How to use

### Programmatic API

Store your time series in a pandas dataframe with an index named **unique_id** that is the identifier of each serie, a column **ds** that contains the datestamps and a column **y** with the values.

```python
from mlforecast.utils import generate_daily_series

series = generate_daily_series(20)
display_df(series.head())
```


| unique_id   | ds                  |        y |
|:------------|:--------------------|---------:|
| id_00       | 2000-01-01 00:00:00 | 0.264447 |
| id_00       | 2000-01-02 00:00:00 | 1.28402  |
| id_00       | 2000-01-03 00:00:00 | 2.4628   |
| id_00       | 2000-01-04 00:00:00 | 3.03552  |
| id_00       | 2000-01-05 00:00:00 | 4.04356  |


Then you define your flow configuration. These include lags, transformations on the lags and date features. The transformations are defined as `numba` jitted functions that transform an array. If they have additional arguments you supply a tuple (`transform_func`, `arg1`, `arg2`, ...)

```python
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean

flow_config = dict(
    lags=[7, 14],
    lag_transforms={
        1: [expanding_mean],
        7: [(rolling_mean, 7), (rolling_mean, 14)]
    },
    date_features=['dayofweek', 'month']
)
```

Next define a model, if you're on a single machine this can be any regressor that follows the scikit-learn API. For distributed training there are `LGBMForecast` and `XGBForecast`.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
```

Now instantiate your forecast object with the model and the flow configuration. There are two types of forecasters, `Forecast` and `DistributedForecast`. Since this is a single machine example we'll use the first.

```python
from mlforecast.forecast import Forecast

fcst = Forecast(model, flow_config)
```

To compute the transformations and train the model on the data you call `.fit` on your `Forecast` object.

```python
fcst.fit(series)
```




    Forecast(model=RandomForestRegressor(), flow_config={'lags': [7, 14], 'lag_transforms': {1: [CPUDispatcher(<function expanding_mean at 0x7f145d8691f0>)], 7: [(CPUDispatcher(<function rolling_mean at 0x7f145d8d3ee0>), 7), (CPUDispatcher(<function rolling_mean at 0x7f145d8d3ee0>), 14)]}, 'date_features': ['dayofweek', 'month']})



To get the forecasts for the next 14 days you just call `.predict(14)` on the forecaster.

```python
predictions = fcst.predict(14)

display_df(predictions.head())
```


| unique_id   | ds                  |   y_pred |
|:------------|:--------------------|---------:|
| id_00       | 2000-08-10 00:00:00 | 5.26783  |
| id_00       | 2000-08-11 00:00:00 | 6.2507   |
| id_00       | 2000-08-12 00:00:00 | 0.214484 |
| id_00       | 2000-08-13 00:00:00 | 1.25304  |
| id_00       | 2000-08-14 00:00:00 | 2.29772  |


### CLI

If you're looking for computing quick baselines, want to avoid some boilerplate or just like using CLIs better then you can use the `mlforecast` binary with a configuration file like the following:

```python
!cat sample_configs/local.yaml
```

    data:
      prefix: data
      input: train
      output: outputs
      format: parquet
    features:
      freq: D
      lags: [7, 14]
      lag_transforms:
        1: 
        - expanding_mean
        7: 
        - rolling_mean:
            window_size: 7
        - rolling_min:
            window_size: 7
      date_features: ["dayofweek", "month", "year"]
      num_threads: 2
    backtest:
      n_windows: 2
      window_size: 7
    forecast:
      horizon: 7
    local:
      model:
        name: sklearn.ensemble.RandomForestRegressor
        params:
          n_estimators: 10
          max_depth: 7


This will use the data in `prefix/input` and write the results to `prefix/output`.

```python
!mlforecast sample_configs/local.yaml
```

    Split 1 MSE: 0.0226
    Split 2 MSE: 0.0179
    [0m

```python
!ls data/outputs/
```

    forecast.parquet  valid_0.parquet  valid_1.parquet

