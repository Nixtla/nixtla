# autotimeseries
> Nixtla SDK. Time Series Forecasting pipeline at scale.


[![CI python sdk](https://github.com/Nixtla/nixtla/actions/workflows/python-sdk.yml/badge.svg)](https://github.com/Nixtla/nixtla/actions/workflows/python-sdk.yml)
[![Python](https://img.shields.io/pypi/pyversions/autotimeseries)](https://pypi.org/project/autotimeseries/)
[![PyPi](https://img.shields.io/pypi/v/autotimeseries?color=blue)](https://pypi.org/project/autotimeseries/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Nixtla/nixtla/blob/main/sdk/python-autotimeseries/LICENSE)

**autotimeseries** is a python SDK to consume the APIs developed in https://github.com/Nixtla/nixtla.

## Install

### PyPI

`pip install autotimeseries`

## How to use

Check the following examples for a full pipeline:

- [M5 state-of-the-art reproduction](https://github.com/Nixtla/autotimeseries/tree/main/examples/m5).
- [M5 state-of-the-art reproduction in Colab](https://colab.research.google.com/drive/1pmp4rqiwiPL-ambxTrJGBiNMS-7vm3v6?ts=616700c4)

### Basic usage

```python
import os

from autotimeseries.core import AutoTS

autotimeseries = AutoTS(bucket_name=os.environ['BUCKET_NAME'],
                        api_id=os.environ['API_ID'], 
                        api_key=os.environ['API_KEY'],
                        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
                        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
```

#### Upload dataset to S3

```python
train_dir = '../data/m5/parquet/train'
# File with target variables
filename_target = autotimeseries.upload_to_s3(f'{train_dir}/target.parquet')
# File with static variables
filename_static = autotimeseries.upload_to_s3(f'{train_dir}/static.parquet')
# File with temporal variables
filename_temporal = autotimeseries.upload_to_s3(f'{train_dir}/temporal.parquet')
```

Each time series of the uploaded datasets is defined by the column `item_id`. Meanwhile the time column is defined by `timestamp` and the target column by `demand`. We need to pass this arguments to each call.

```python
columns = dict(unique_id_column='item_id',
               ds_column='timestamp',
               y_column='demand')
```

#### Send the job to make forecasts

```python
response_forecast = autotimeseries.tsforecast(filename_target=filename_target,
                                              freq='D',
                                              horizon=28, 
                                              filename_static=filename_static,
                                              filename_temporal=filename_temporal,
                                              objective='tweedie',
                                              metric='rmse',
                                              n_estimators=170,
                                              **columns)
```

#### Download forecasts

```python
autotimeseries.download_from_s3(filename='forecasts_2021-10-12_19-04-32.csv', filename_output='../data/forecasts.csv')
```
