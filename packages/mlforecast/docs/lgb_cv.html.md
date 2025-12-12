---
description: Time series cross validation with LightGBM.
output-file: lgb_cv.html
title: LightGBMCV
---

##

::: mlforecast.lgb_cv.LightGBMCV
    handler: python
    options:
      docstring_style: google
      members:
        - fit
        - predict
        - setup
        - partial_fit
      heading_level: 3
      show_root_heading: true
      show_source: true

### Example

This shows an example with just 4 series of the M4 dataset. If you want
to run it yourself on all of them, you can refer to [this
notebook](https://www.kaggle.com/code/lemuz90/m4-competition-cv).

```python
import random

from datasetsforecast.m4 import M4, M4Info
from fastcore.test import test_eq, test_fail
from mlforecast.target_transforms import Differences
from nbdev import show_doc

from mlforecast.lag_transforms import SeasonalRollingMean
```

```python
group = 'Hourly'
await M4.async_download('data', group=group)
df, *_ = M4.load(directory='data', group=group)
df['ds'] = df['ds'].astype('int')
ids = df['unique_id'].unique()
random.seed(0)
sample_ids = random.choices(ids, k=4)
sample_df = df[df['unique_id'].isin(sample_ids)]
sample_df
```

|        | unique_id | ds   | y    |
|--------|-----------|------|------|
| 86796  | H196      | 1    | 11.8 |
| 86797  | H196      | 2    | 11.4 |
| 86798  | H196      | 3    | 11.1 |
| 86799  | H196      | 4    | 10.8 |
| 86800  | H196      | 5    | 10.6 |
| ...    | ...       | ...  | ...  |
| 325235 | H413      | 1004 | 99.0 |
| 325236 | H413      | 1005 | 88.0 |
| 325237 | H413      | 1006 | 47.0 |
| 325238 | H413      | 1007 | 41.0 |
| 325239 | H413      | 1008 | 34.0 |

```python
info = M4Info[group]
horizon = info.horizon
valid = sample_df.groupby('unique_id').tail(horizon)
train = sample_df.drop(valid.index)
train.shape, valid.shape
```

``` text
((3840, 3), (192, 3))
```

What LightGBMCV does is emulate [LightGBM’s cv
function](https://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/lightgbm.cv.html#lightgbm.cv)
where several Boosters are trained simultaneously on different
partitions of the data, that is, one boosting iteration is performed on
all of them at a time. This allows to have an estimate of the error by
iteration, so if we combine this with early stopping we can find the
best iteration to train a final model using all the data or even use
these individual models’ predictions to compute an ensemble.

In order to have a good estimate of the forecasting performance of our
model we compute predictions for the whole test period and compute a
metric on that. Since this step can slow down training, there’s an
`eval_every` parameter that can be used to control this, that is, if
`eval_every=10` (the default) every 10 boosting iterations we’re going
to compute forecasts for the complete window and report the error.

We also have early stopping parameters:

- `early_stopping_evals`: how many evaluations of the full window
    should we go without improving to stop training?
- `early_stopping_pct`: what’s the minimum percentage improvement we
    want in these `early_stopping_evals` in order to keep training?

This makes the LightGBMCV class a good tool to quickly test different
configurations of the model. Consider the following example, where we’re
going to try to find out which features can improve the performance of
our model. We start just using lags.

```python
static_fit_config = dict(
    n_windows=2,
    h=horizon,
    params={'verbose': -1},
    compute_cv_preds=True,
)
cv = LightGBMCV(
    freq=1,
    lags=[24 * (i+1) for i in range(7)],  # one week of lags
)
```

```python
hist = cv.fit(train, **static_fit_config)
```

``` text
[LightGBM] [Info] Start training from score 51.745632
[10] mape: 0.590690
[20] mape: 0.251093
[30] mape: 0.143643
[40] mape: 0.109723
[50] mape: 0.102099
[60] mape: 0.099448
[70] mape: 0.098349
[80] mape: 0.098006
[90] mape: 0.098718
Early stopping at round 90
Using best iteration: 80
```

By setting `compute_cv_preds` we get the predictions from each model on
their corresponding validation fold.

```python
cv.cv_preds_
```

|     | unique_id | ds  | y    | Booster   | window |
|-----|-----------|-----|------|-----------|--------|
| 0   | H196      | 865 | 15.5 | 15.522924 | 0      |
| 1   | H196      | 866 | 15.1 | 14.985832 | 0      |
| 2   | H196      | 867 | 14.8 | 14.667901 | 0      |
| 3   | H196      | 868 | 14.4 | 14.514592 | 0      |
| 4   | H196      | 869 | 14.2 | 14.035793 | 0      |
| ... | ...       | ... | ...  | ...       | ...    |
| 187 | H413      | 956 | 59.0 | 77.227905 | 1      |
| 188 | H413      | 957 | 58.0 | 80.589641 | 1      |
| 189 | H413      | 958 | 53.0 | 53.986834 | 1      |
| 190 | H413      | 959 | 38.0 | 36.749786 | 1      |
| 191 | H413      | 960 | 46.0 | 36.281225 | 1      |

The individual models we trained are saved, so calling `predict` returns
the predictions from every model trained.

------------------------------------------------------------------------

<a
href="<https://github.com/Nixtla/mlforecast/blob/main/mlforecast/lgb_cv.py#L485>"
target="_blank" style={{ float: "right", fontSize: "smaller" }}>source</a>


```python
preds = cv.predict(horizon)
preds
```

|     | unique_id | ds   | Booster0  | Booster1  |
|-----|-----------|------|-----------|-----------|
| 0   | H196      | 961  | 15.670252 | 15.848888 |
| 1   | H196      | 962  | 15.522924 | 15.697399 |
| 2   | H196      | 963  | 14.985832 | 15.166213 |
| 3   | H196      | 964  | 14.985832 | 14.723238 |
| 4   | H196      | 965  | 14.562152 | 14.451092 |
| ... | ...       | ...  | ...       | ...       |
| 187 | H413      | 1004 | 70.695242 | 65.917620 |
| 188 | H413      | 1005 | 66.216580 | 62.615788 |
| 189 | H413      | 1006 | 63.896573 | 67.848598 |
| 190 | H413      | 1007 | 46.922797 | 50.981950 |
| 191 | H413      | 1008 | 45.006541 | 42.752819 |

We can average these predictions and evaluate them.

```python
def evaluate_on_valid(preds):
    preds = preds.copy()
    preds['final_prediction'] = preds.drop(columns=['unique_id', 'ds']).mean(1)
    merged = preds.merge(valid, on=['unique_id', 'ds'])
    merged['abs_err'] = abs(merged['final_prediction'] - merged['y']) / merged['y']
    return merged.groupby('unique_id')['abs_err'].mean().mean()
```

```python
eval1 = evaluate_on_valid(preds)
eval1
```

``` text
0.11036194712311806
```

Now, since these series are hourly, maybe we can try to remove the daily
seasonality by taking the 168th (24 \* 7) difference, that is, substract
the value at the same hour from one week ago, thus our target will be
$z_t = y_{t} - y_{t-168}$. The features will be computed from this
target and when we predict they will be automatically re-applied.

```python
cv2 = LightGBMCV(
    freq=1,
    target_transforms=[Differences([24 * 7])],
    lags=[24 * (i+1) for i in range(7)],
)
hist2 = cv2.fit(train, **static_fit_config)
```

``` text
[LightGBM] [Info] Start training from score 0.519010
[10] mape: 0.089024
[20] mape: 0.090683
[30] mape: 0.092316
Early stopping at round 30
Using best iteration: 10
```

```python
assert hist2[-1][1] < hist[-1][1]
```

Nice! We achieve a better score in less iterations. Let’s see if this
improvement translates to the validation set as well.

```python
preds2 = cv2.predict(horizon)
eval2 = evaluate_on_valid(preds2)
eval2
```

``` text
0.08956665504570135
```

```python
assert eval2 < eval1
```

Great! Maybe we can try some lag transforms now. We’ll try the seasonal
rolling mean that averages the values “every season”, that is, if we set
`season_length=24` and `window_size=7` then we’ll average the value at
the same hour for every day of the week.

```python
cv3 = LightGBMCV(
    freq=1,
    target_transforms=[Differences([24 * 7])],
    lags=[24 * (i+1) for i in range(7)],
    lag_transforms={
        48: [SeasonalRollingMean(season_length=24, window_size=7)],
    },
)
hist3 = cv3.fit(train, **static_fit_config)
```

``` text
[LightGBM] [Info] Start training from score 0.273641
[10] mape: 0.086724
[20] mape: 0.088466
[30] mape: 0.090536
Early stopping at round 30
Using best iteration: 10
```

Seems like this is helping as well!

```python
assert hist3[-1][1] < hist2[-1][1]
```

Does this reflect on the validation set?

```python
preds3 = cv3.predict(horizon)
eval3 = evaluate_on_valid(preds3)
eval3
```

``` text
0.08961279023129345
```

Nice! mlforecast also supports date features, but in this case our time
column is made from integers so there aren’t many possibilites here. As
you can see this allows you to iterate faster and get better estimates
of the forecasting performance you can expect from your model.

If you’re doing hyperparameter tuning it’s useful to be able to run a
couple of iterations, assess the performance, and determine if this
particular configuration isn’t promising and should be discarded. For
example, [optuna](https://optuna.org/) has
[pruners](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
that you can call with your current score and it decides if the trial
should be discarded. We’ll now show how to do that.

Since the CV requires a bit of setup, like the LightGBM datasets and the
internal features, we have this `setup` method.

```python
cv4 = LightGBMCV(
    freq=1,
    lags=[24 * (i+1) for i in range(7)],
)
cv4.setup(
    train,
    n_windows=2,
    h=horizon,
    params={'verbose': -1},
)
```

``` text
LightGBMCV(freq=1, lag_features=['lag24', 'lag48', 'lag72', 'lag96', 'lag120', 'lag144', 'lag168'], date_features=[], num_threads=1, bst_threads=8)
```

Once we have this we can call `partial_fit` to only train for some
iterations and return the score of the forecast window.

```python
score = cv4.partial_fit(10)
score
```

``` text
[LightGBM] [Info] Start training from score 51.745632
```

``` text
0.5906900462828166
```

This is equal to the first evaluation from our first example.

```python
assert hist[0][1] == score
```

We can now use this score to decide if this configuration is promising.
If we want to we can train some more iterations.

```python
score2 = cv4.partial_fit(20)
```

This is now equal to our third metric from the first example, since this
time we trained for 20 iterations.

```python
assert hist[2][1] == score2
```

### Using a custom metric

The built-in metrics are MAPE and RMSE, which are computed by serie and
then averaged across all series. If you want to do something different
or use a different metric entirely, you can define your own metric like
the following:

```python
def weighted_mape(
    y_true: pd.Series,
    y_pred: pd.Series,
    ids: pd.Series,
    dates: pd.Series,
):
    """Weighs the MAPE by the magnitude of the series values"""
    abs_pct_err = abs(y_true - y_pred) / abs(y_true)
    mape_by_serie = abs_pct_err.groupby(ids).mean()
    totals_per_serie = y_pred.groupby(ids).sum()
    series_weights = totals_per_serie / totals_per_serie.sum()
    return (mape_by_serie * series_weights).sum()
```

```python
_ = LightGBMCV(
    freq=1,
    lags=[24 * (i+1) for i in range(7)],
).fit(
    train,
    n_windows=2,
    h=horizon,
    params={'verbose': -1},
    metric=weighted_mape,
)
```

``` text
[LightGBM] [Info] Start training from score 51.745632
[10] weighted_mape: 0.480353
[20] weighted_mape: 0.218670
[30] weighted_mape: 0.161706
[40] weighted_mape: 0.149992
[50] weighted_mape: 0.149024
[60] weighted_mape: 0.148496
Early stopping at round 60
Using best iteration: 60
```
