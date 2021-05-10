# mlforecast
> Scalable machine learning based time series forecasting.


[![CI](https://github.com/Nixtla/mlforecast/actions/workflows/ci.yaml/badge.svg)](https://github.com/Nixtla/mlforecast/actions/workflows/ci.yaml)
[![Lint](https://github.com/Nixtla/mlforecast/actions/workflows/lint.yaml/badge.svg)](https://github.com/Nixtla/mlforecast/actions/workflows/lint.yaml)
[![Python](https://img.shields.io/pypi/pyversions/mlforecast)](https://pypi.org/project/mlforecast/)
[![PyPi](https://img.shields.io/pypi/v/mlforecast?color=blue)](https://pypi.org/project/mlforecast/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/mlforecast?color=blue)](https://anaconda.org/conda-forge/mlforecast)
[![License](https://img.shields.io/github/license/Nixtla/mlforecast)](https://github.com/Nixtla/mlforecast/blob/main/LICENSE)

## Install

### PyPI

`pip install mlforecast`

#### Optional dependencies
If you want more functionality you can instead use `pip install mlforecast[extra1, extra2, ...]`. The current extra dependencies are:

* **aws**: adds the functionality to use S3 as the storage in the CLI.
* **cli**: includes the validations necessary to use the CLI.
* **distributed**: installs [dask](https://dask.org/) to perform distributed training. Note that you'll also need to install either [LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package) or [XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python).

For example, if you want to perform distributed training through the CLI using S3 as your storage you'll need all three extras, which you can get using: `pip install mlforecast[aws, cli, distributed]`.

### conda-forge
`conda install -c conda-forge mlforecast`

Note that this installation comes with the required dependencies for the local interface. If you want to:
* Use s3 as storage: `conda install -c conda-forge boto3 s3path`
* Perform distributed training: `conda install -c conda-forge dask` and either [LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package) or [XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python).

## How to use

### Programmatic API

Store your time series in a pandas dataframe with an index named **unique_id** that identifies each time serie, a column **ds** that contains the datestamps and a column **y** with the values.

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


Then define your flow configuration. This includes lags, transformations on the lags and date features. The lag transformations are defined as [numba](http://numba.pydata.org/) *jitted* functions that transform an array, if they have additional arguments you supply a tuple (`transform_func`, `arg1`, `arg2`, ...).

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

Next define a model. If you want to use the local interface this can be any regressor that follows the scikit-learn API. For distributed training there are `LGBMForecast` and `XGBForecast`.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
```

Now instantiate your forecast object with the model and the flow configuration. There are two types of forecasters, `Forecast` which is local and `DistributedForecast` which performs the whole process in a distributed way.

```python
from mlforecast.forecast import Forecast

fcst = Forecast(model, flow_config)
```

To compute the features and train the model using them call `.fit` on your `Forecast` object.

```python
fcst.fit(series)
```




    Forecast(model=RandomForestRegressor(), flow_config={'lags': [7, 14], 'lag_transforms': {1: [CPUDispatcher(<function expanding_mean at 0x7f1264fe6700>)], 7: [(CPUDispatcher(<function rolling_mean at 0x7f1264fe0430>), 7), (CPUDispatcher(<function rolling_mean at 0x7f1264fe0430>), 14)]}, 'date_features': ['dayofweek', 'month']})



To get the forecasts for the next 14 days call `.predict(14)` on the forecaster. This will update the target with each prediction and recompute the features to get the next one.

```python
predictions = fcst.predict(14)

display_df(predictions.head())
```


| unique_id   | ds                  |   y_pred |
|:------------|:--------------------|---------:|
| id_00       | 2000-08-10 00:00:00 | 5.21542  |
| id_00       | 2000-08-11 00:00:00 | 6.26993  |
| id_00       | 2000-08-12 00:00:00 | 0.232467 |
| id_00       | 2000-08-13 00:00:00 | 1.23008  |
| id_00       | 2000-08-14 00:00:00 | 2.29878  |


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
        - rolling_mean:
            window_size: 14
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


The configuration is validated using `FlowConfig`.

This configuration will use the data in `data.prefix/data.input` to train and write the results to `data.prefix/data.output` both with `data.format`.

```python
!mlforecast sample_configs/local.yaml
```

    Split 1 MSE: 0.0239
    Split 2 MSE: 0.0183

```python
list((data_path/'outputs').iterdir())
```




    [PosixPath('data/outputs/valid_1.parquet'),
     PosixPath('data/outputs/valid_0.parquet'),
     PosixPath('data/outputs/forecast.parquet')]


