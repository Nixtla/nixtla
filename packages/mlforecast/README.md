# mlforecast

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Statistical%20Forecasting%20Algorithms%20by%20Nixtla%20&url=https://github.com/Nixtla/statsforecast&via=nixtlainc&hashtags=StatisticalModels,TimeSeries,Forecasting)
[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white.png)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)


<div align="center">
<center><img src="https://raw.githubusercontent.com/Nixtla/mlforecast/main/nbs/figs/logo.png" /></center>
<h1 align="center">Machine Learning 🤖 Forecast</h1>
<h3 align="center">Scalable machine learning for time series forecasting</h3>

[![CI](https://github.com/Nixtla/mlforecast/actions/workflows/ci.yaml/badge.svg)](https://github.com/Nixtla/mlforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/mlforecast.png)](https://pypi.org/project/mlforecast/)
[![PyPi](https://img.shields.io/pypi/v/mlforecast?color=blue.png)](https://pypi.org/project/mlforecast/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/mlforecast?color=blue.png)](https://anaconda.org/conda-forge/mlforecast)
[![License](https://img.shields.io/github/license/Nixtla/mlforecast.png)](https://github.com/Nixtla/mlforecast/blob/main/LICENSE)

**mlforecast** is a framework to perform time series forecasting using
machine learning models, with the option to scale to massive amounts of
data using remote clusters.

</div>

## Install

### PyPI

`pip install mlforecast`

### conda-forge

`conda install -c conda-forge mlforecast`

For more detailed instructions you can refer to the [installation
page](https://nixtlaverse.nixtla.io/mlforecast/docs/getting-started/install).

## Quick Start

**Get Started with this [quick
guide](https://nixtlaverse.github.io/mlforecast/docs/getting-started/quick_start_local.html).**

**Follow this [end-to-end
walkthrough](https://nixtlaverse.nixtla.io/mlforecast/docs/getting-started/end_to_end_walkthrough)
for best practices.**

### Videos

- [Overview](https://www.youtube.com/live/EnhyJx8l2LE)

### Sample notebooks

- [m5](https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval)
- [m5-polars](https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval-polars)
- [m4](https://www.kaggle.com/code/lemuz90/m4-competition)
- [m4-cv](https://www.kaggle.com/code/lemuz90/m4-competition-cv)
- [favorita](https://www.kaggle.com/code/lemuz90/mlforecast-favorita)
- [VN1](https://colab.research.google.com/drive/1UdhCAk49k6HgMezG-U_1ETnAB5pYvZk9)

## Why?

Current Python alternatives for machine learning models are slow,
inaccurate and don’t scale well. So we created a library that can be
used to forecast in production environments.
[`MLForecast`](https://nixtlaverse.nixtla.io/mlforecast/forecast#class-mlforecast)
includes efficient feature engineering to train any machine learning
model (with `fit` and `predict` methods such as
[`sklearn`](https://scikit-learn.org/stable/)) to fit millions of time
series.

## Features

- Fastest implementations of feature engineering for time series
  forecasting in Python.
- Out-of-the-box compatibility with pandas, polars, spark, dask, and
  ray.
- Probabilistic Forecasting with Conformal Prediction.
- Support for exogenous variables and static covariates.
- Familiar `sklearn` syntax: `.fit` and `.predict`.

Missing something? Please open an issue or write us in
[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white.png)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## Examples and Guides

📚 [End to End
Walkthrough](https://nixtlaverse.nixtla.io/mlforecast/docs/getting-started/end_to_end_walkthrough):
model training, evaluation and selection for multiple time series.

🔎 [Probabilistic
Forecasting](https://nixtlaverse.nixtla.io/mlforecast/docs/tutorials/prediction_intervals_in_forecasting_models):
use Conformal Prediction to produce prediciton intervals.

👩‍🔬 [Cross
Validation](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/cross_validation):
robust model’s performance evaluation.

🔌 [Predict Demand
Peaks](https://nixtlaverse.nixtla.io/mlforecast/docs/tutorials/electricity_peak_forecasting):
electricity load forecasting for detecting daily peaks and reducing
electric bills.

📈 [Transfer
Learning](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/transfer_learning):
pretrain a model using a set of time series and then predict another one
using that pretrained model.

🌡️ [Distributed
Training](https://nixtlaverse.nixtla.io/mlforecast/docs/getting-started/quick_start_distributed):
use a Dask, Ray or Spark cluster to train models at scale.

## How to use

The following provides a very basic overview, for a more detailed
description see the
[documentation](https://nixtlaverse.nixtla.io/mlforecast/).

### Data setup

Store your time series in a pandas dataframe in long format, that is,
each row represents an observation for a specific serie and timestamp.

``` python
from mlforecast.utils import generate_daily_series

series = generate_daily_series(
    n_series=20,
    max_length=100,
    n_static_features=1,
    static_as_categorical=False,
    with_trend=True
)
series.head()
```

|     | unique_id | ds         | y          | static_0 |
|-----|-----------|------------|------------|----------|
| 0   | id_00     | 2000-01-01 | 17.519167  | 72       |
| 1   | id_00     | 2000-01-02 | 87.799695  | 72       |
| 2   | id_00     | 2000-01-03 | 177.442975 | 72       |
| 3   | id_00     | 2000-01-04 | 232.704110 | 72       |
| 4   | id_00     | 2000-01-05 | 317.510474 | 72       |


> Note: The unique_id serves as an identifier for each distinct time
> series in your dataset. If you are using only single time series from
> your dataset, set this column to a constant value.

### Models

Next define your models, each one will be trained on all series. These
can be any regressor that follows the scikit-learn API.

``` python
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
```

``` python
models = [
    lgb.LGBMRegressor(random_state=0, verbosity=-1),
    LinearRegression(),
]
```

### Forecast object

Now instantiate an
[`MLForecast`](https://nixtlaverse.nixtla.io/mlforecast/forecast#class-mlforecast)
object with the models and the features that you want to use. The
features can be lags, transformations on the lags and date features. You
can also define transformations to apply to the target before fitting,
which will be restored when predicting.

``` python
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences
```

``` python
fcst = MLForecast(
    models=models,
    freq='D',
    lags=[7, 14],
    lag_transforms={
        1: [ExpandingMean()],
        7: [RollingMean(window_size=28)]
    },
    date_features=['dayofweek'],
    target_transforms=[Differences([1])],
)
```

### Training

To compute the features and train the models call `fit` on your
`Forecast` object.

``` python
fcst.fit(series)
```

    MLForecast(models=[LGBMRegressor, LinearRegression], freq=D, lag_features=['lag7', 'lag14', 'expanding_mean_lag1', 'rolling_mean_lag7_window_size28'], date_features=['dayofweek'], num_threads=1)

### Predicting

To get the forecasts for the next `n` days call `predict(n)` on the
forecast object. This will automatically handle the updates required by
the features using a recursive strategy.

``` python
predictions = fcst.predict(14)
predictions
```


|     | unique_id | ds         | LGBMRegressor | LinearRegression |
|-----|-----------|------------|---------------|------------------|
| 0   | id_00     | 2000-04-04 | 299.923771    | 311.432371       |
| 1   | id_00     | 2000-04-05 | 365.424147    | 379.466214       |
| 2   | id_00     | 2000-04-06 | 432.562441    | 460.234028       |
| 3   | id_00     | 2000-04-07 | 495.628000    | 524.278924       |
| 4   | id_00     | 2000-04-08 | 60.786223     | 79.828767        |
| ... | ...       | ...        | ...           | ...              |
| 275 | id_19     | 2000-03-23 | 36.266780     | 28.333215        |
| 276 | id_19     | 2000-03-24 | 44.370984     | 33.368228        |
| 277 | id_19     | 2000-03-25 | 50.746222     | 38.613001        |
| 278 | id_19     | 2000-03-26 | 58.906524     | 43.447398        |
| 279 | id_19     | 2000-03-27 | 63.073949     | 48.666783        |

<p>280 rows × 4 columns</p>


### Visualize results

``` python
from utilsforecast.plotting import plot_series
```

``` python
fig = plot_series(series, predictions, max_ids=4, plot_random=False)
```

![](https://raw.githubusercontent.com/Nixtla/mlforecast/main/nbs/figs/index.png)

## How to contribute

See
[CONTRIBUTING.md](https://github.com/Nixtla/mlforecast/blob/main/CONTRIBUTING.md).
