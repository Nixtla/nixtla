---
output-file: auto.html
title: Auto
---

##

::: mlforecast.auto.AutoRandomForest
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoElasticNet
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoLasso
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoRidge
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoLinearRegression
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoCatboost
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoXGBoost
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoLightGBM
    options:
      show_if_no_docstring: false

::: mlforecast.auto.random_forest_space
    options:
      show_if_no_docstring: false

::: mlforecast.auto.elastic_net_space
    options:
      show_if_no_docstring: false

::: mlforecast.auto.lasso_space
    options:
      show_if_no_docstring: false

::: mlforecast.auto.ridge_space
    options:
      show_if_no_docstring: false

::: mlforecast.auto.linear_regression_space
    options:
      show_if_no_docstring: false

::: mlforecast.auto.catboost_space
    options:
      show_if_no_docstring: false

::: mlforecast.auto.xgboost_space
    options:
      show_if_no_docstring: false

::: mlforecast.auto.lightgbm_space
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoModel
    options:
      show_if_no_docstring: false

::: mlforecast.auto.AutoMLForecast
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - save
        - forecast_fitted_values
        - stack
        - take
        - transform
      heading_level: 3
      show_root_heading: true
      show_source: true

```python
import time

import pandas as pd
from datasetsforecast.m4 import M4, M4Evaluation, M4Info
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
```

```python
def train_valid_split(group):
    df, *_ = M4.load(directory='data', group=group)
    df['ds'] = df['ds'].astype('int')
    horizon = M4Info[group].horizon
    valid = df.groupby('unique_id').tail(horizon).copy()
    train = df.drop(valid.index).reset_index(drop=True)
    return train, valid
```


```python
ridge_pipeline = make_pipeline(
    ColumnTransformer(
        [('encoder', OneHotEncoder(), ['unique_id'])],
        remainder='passthrough',
    ),
    Ridge()
)
auto_ridge = AutoModel(ridge_pipeline, lambda trial: {f'ridge__{k}': v for k, v in ridge_space(trial).items()})
```

```python
optuna.logging.set_verbosity(optuna.logging.ERROR)
group = 'Weekly'
train, valid = train_valid_split(group)
train['unique_id'] = train['unique_id'].astype('category')
valid['unique_id'] = valid['unique_id'].astype(train['unique_id'].dtype)
info = M4Info[group]
h = info.horizon
season_length = info.seasonality
auto_mlf = AutoMLForecast(
    freq=1,
    season_length=season_length,
    models={
        'lgb': AutoLightGBM(),
        'ridge': auto_ridge,
    },
    fit_config=lambda trial: {'static_features': ['unique_id']},
    num_threads=2,
)
auto_mlf.fit(
    df=train,
    n_windows=2,
    h=h,
    num_samples=2,
    optimize_kwargs={'timeout': 60},
    fitted=True,
    prediction_intervals=PredictionIntervals(n_windows=2, h=h),
)
auto_mlf.predict(h, level=[80])
```

|  | unique_id | ds | lgb | lgb-lo-80 | lgb-hi-80 | ridge | ridge-lo-80 | ridge-hi-80 |
|----|----|----|----|----|----|----|----|----|
| 0 | W1 | 2180 | 35529.435224 | 35061.835362 | 35997.035086 | 36110.921202 | 35880.445097 | 36341.397307 |
| 1 | W1 | 2181 | 35521.764894 | 34973.035617 | 36070.494171 | 36195.175757 | 36051.013811 | 36339.337702 |
| 2 | W1 | 2182 | 35537.417268 | 34960.050939 | 36114.783596 | 36107.528852 | 35784.062169 | 36430.995536 |
| 3 | W1 | 2183 | 35538.058206 | 34823.640706 | 36252.475705 | 36027.139248 | 35612.635725 | 36441.642771 |
| 4 | W1 | 2184 | 35614.611211 | 34627.023739 | 36602.198683 | 36092.858489 | 35389.690977 | 36796.026000 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 4662 | W99 | 2292 | 15071.536978 | 14484.617399 | 15658.456557 | 15319.146221 | 14869.410567 | 15768.881875 |
| 4663 | W99 | 2293 | 15058.145278 | 14229.686322 | 15886.604234 | 15299.549555 | 14584.269352 | 16014.829758 |
| 4664 | W99 | 2294 | 15042.493434 | 14096.380636 | 15988.606232 | 15271.744712 | 14365.349338 | 16178.140086 |
| 4665 | W99 | 2295 | 15042.144846 | 14037.053904 | 16047.235787 | 15250.070504 | 14403.428791 | 16096.712216 |
| 4666 | W99 | 2296 | 15038.729044 | 13944.821480 | 16132.636609 | 15232.127800 | 14325.059776 | 16139.195824 |


```python
auto_mlf.forecast_fitted_values(level=[95])
```

|  | unique_id | ds | y | lgb | lgb-lo-95 | lgb-hi-95 | ridge | ridge-lo-95 | ridge-hi-95 |
|----|----|----|----|----|----|----|----|----|----|
| 0 | W1 | 15 | 1071.06 | 1060.584344 | 599.618355 | 1521.550334 | 1076.990151 | 556.535492 | 1597.444810 |
| 1 | W1 | 16 | 1073.73 | 1072.669242 | 611.703252 | 1533.635232 | 1083.633276 | 563.178617 | 1604.087936 |
| 2 | W1 | 17 | 1066.97 | 1072.452128 | 611.486139 | 1533.418118 | 1084.724311 | 564.269652 | 1605.178970 |
| 3 | W1 | 18 | 1066.17 | 1065.837828 | 604.871838 | 1526.803818 | 1080.127197 | 559.672538 | 1600.581856 |
| 4 | W1 | 19 | 1064.43 | 1065.214681 | 604.248691 | 1526.180671 | 1080.636826 | 560.182167 | 1601.091485 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 361881 | W99 | 2279 | 15738.54 | 15887.661228 | 15721.237195 | 16054.085261 | 15927.918181 | 15723.222760 | 16132.613603 |
| 361882 | W99 | 2280 | 15388.13 | 15755.943789 | 15589.519756 | 15922.367823 | 15841.599064 | 15636.903642 | 16046.294485 |
| 361883 | W99 | 2281 | 15187.62 | 15432.224701 | 15265.800668 | 15598.648735 | 15584.462232 | 15379.766811 | 15789.157654 |
| 361884 | W99 | 2282 | 15172.27 | 15177.040831 | 15010.616797 | 15343.464864 | 15396.243223 | 15191.547801 | 15600.938644 |
| 361885 | W99 | 2283 | 15101.03 | 15162.090803 | 14995.666770 | 15328.514836 | 15335.982465 | 15131.287044 | 15540.677887 |

```python
import polars as pl
```

```python
train_pl = pl.from_pandas(train.astype({'unique_id': 'str'}))
auto_mlf = AutoMLForecast(
    freq=1,
    season_length=season_length,
    models={'ridge': AutoRidge()},
    num_threads=2,
)
auto_mlf.fit(
    df=train_pl,
    n_windows=2,
    h=h,
    num_samples=2,
    optimize_kwargs={'timeout': 60},
    fitted=True,
    prediction_intervals=PredictionIntervals(n_windows=2, h=h),
)
auto_mlf.predict(h, level=[80])
```

| unique_id | ds   | ridge        | ridge-lo-80  | ridge-hi-80  |
|-----------|------|--------------|--------------|--------------|
| str       | i64  | f64          | f64          | f64          |
| "W1"      | 2180 | 35046.096663 | 34046.69521  | 36045.498116 |
| "W1"      | 2181 | 34743.269216 | 33325.847975 | 36160.690457 |
| "W1"      | 2182 | 34489.591086 | 32591.254559 | 36387.927614 |
| "W1"      | 2183 | 34270.768179 | 32076.507727 | 36465.02863  |
| "W1"      | 2184 | 34124.021857 | 31352.454121 | 36895.589593 |
| …         | …    | …            | …            | …            |
| "W99"     | 2292 | 14719.457096 | 13983.308582 | 15455.605609 |
| "W99"     | 2293 | 14631.552077 | 13928.874336 | 15334.229818 |
| "W99"     | 2294 | 14532.905239 | 13642.840118 | 15422.97036  |
| "W99"     | 2295 | 14446.065443 | 13665.088667 | 15227.04222  |
| "W99"     | 2296 | 14363.049604 | 13654.220051 | 15071.879157 |

```python
auto_mlf.forecast_fitted_values(level=[95])
```

| unique_id | ds   | y        | ridge        | ridge-lo-95  | ridge-hi-95  |
|-----------|------|----------|--------------|--------------|--------------|
| str       | i64  | f64      | f64          | f64          | f64          |
| "W1"      | 14   | 1061.96  | 1249.326428  | 488.765249   | 2009.887607  |
| "W1"      | 15   | 1071.06  | 1246.067836  | 485.506657   | 2006.629015  |
| "W1"      | 16   | 1073.73  | 1254.027897  | 493.466718   | 2014.589076  |
| "W1"      | 17   | 1066.97  | 1254.475948  | 493.914769   | 2015.037126  |
| "W1"      | 18   | 1066.17  | 1248.306754  | 487.745575   | 2008.867933  |
| …         | …    | …        | …            | …            | …            |
| "W99"     | 2279 | 15738.54 | 15754.558812 | 15411.968645 | 16097.148979 |
| "W99"     | 2280 | 15388.13 | 15655.780865 | 15313.190698 | 15998.371032 |
| "W99"     | 2281 | 15187.62 | 15367.498468 | 15024.908301 | 15710.088635 |
| "W99"     | 2282 | 15172.27 | 15172.591423 | 14830.001256 | 15515.18159  |
| "W99"     | 2283 | 15101.03 | 15141.032886 | 14798.44272  | 15483.623053 |

