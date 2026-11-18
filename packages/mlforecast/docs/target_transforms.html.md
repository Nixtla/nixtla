---
output-file: target_transforms.html
title: Target transforms
---

##

```python
import pandas as pd
from fastcore.test import test_fail
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from utilsforecast.processing import counts_by_id

from mlforecast import MLForecast
from mlforecast.utils import generate_daily_series
```

::: mlforecast.target_transforms.BaseTargetTransform
    options:
      show_if_no_docstring: false

::: mlforecast.target_transforms.Differences
    options:
      show_if_no_docstring: false

```python
series = generate_daily_series(10, min_length=50, max_length=100)


diffs = Differences([1, 2, 5])
id_counts = counts_by_id(series, 'unique_id')
indptr = np.append(0, id_counts['counts'].cumsum())
ga = GroupedArray(series['y'].values, indptr)

# differences are applied correctly
transformed = diffs.fit_transform(ga)
assert diffs.fitted_ == []
expected = series.copy()
for d in diffs.differences:
    expected['y'] -= expected.groupby('unique_id', observed=True)['y'].shift(d)
np.testing.assert_allclose(transformed.data, expected['y'].values)

# fitted differences are restored correctly
diffs.store_fitted = True
transformed = diffs.fit_transform(ga)
keep_mask = ~np.isnan(transformed.data)
restored = diffs.inverse_transform_fitted(transformed)
np.testing.assert_allclose(ga.data[keep_mask], restored.data[keep_mask])

# test transform
new_ga = GroupedArray(np.random.rand(10), np.arange(11))
prev_orig = [diffs.scalers_[i].tails_[::d].copy() for i, d in enumerate(diffs.differences)]
expected = new_ga.data - np.add.reduce(prev_orig)
updates = diffs.update(new_ga)
np.testing.assert_allclose(expected, updates.data)
np.testing.assert_allclose(diffs.scalers_[0].tails_, new_ga.data)
np.testing.assert_allclose(diffs.scalers_[1].tails_[1::2], new_ga.data - prev_orig[0])
np.testing.assert_allclose(diffs.scalers_[2].tails_[4::5], new_ga.data - np.add.reduce(prev_orig[:2]))
# variable sizes
diff1 = Differences([1])
ga = GroupedArray(np.arange(10), np.array([0, 3, 10]))
diff1.fit_transform(ga)
new_ga = GroupedArray(np.arange(4), np.array([0, 1, 4]))
updates = diff1.update(new_ga)
np.testing.assert_allclose(updates.data, np.array([0 - 2, 1 - 9, 2 - 1, 3 - 2]))
np.testing.assert_allclose(diff1.scalers_[0].tails_, np.array([0, 3]))

# short series
ga = GroupedArray(np.arange(20), np.array([0, 2, 20]))
test_fail(lambda: diffs.fit_transform(ga), contains="[0]")

# stack
diffs = Differences([1, 2, 5])
ga = GroupedArray(series['y'].values, indptr)
diffs.fit_transform(ga)
stacked = Differences.stack([diffs, diffs])
for i in range(len(diffs.differences)):
    np.testing.assert_allclose(
        stacked.scalers_[i].tails_,
        np.tile(diffs.scalers_[i].tails_, 2)
    )
```

::: mlforecast.target_transforms.AutoDifferences
    options:
      show_if_no_docstring: false

::: mlforecast.target_transforms.AutoSeasonalDifferences
    options:
      show_if_no_docstring: false

::: mlforecast.target_transforms.AutoSeasonalityAndDifferences
    options:
      show_if_no_docstring: false

```python
def test_scaler(sc, series):
    id_counts = counts_by_id(series, 'unique_id')
    indptr = np.append(0, id_counts['counts'].cumsum())
    ga = GroupedArray(series['y'].values, indptr)
    transformed = sc.fit_transform(ga)
    np.testing.assert_allclose(
        sc.inverse_transform(transformed).data,
        ga.data,
    )
    transformed2 = sc.update(ga)
    np.testing.assert_allclose(transformed.data, transformed2.data)

    idxs = [0, 7]
    subset = ga.take(idxs)
    transformed_subset = transformed.take(idxs)
    subsc = sc.take(idxs)
    np.testing.assert_allclose(
        subsc.inverse_transform(transformed_subset).data,
        subset.data,
    )

    stacked = sc.stack([sc, sc])
    stacked_stats = stacked.scaler_.stats_
    np.testing.assert_allclose(
        stacked_stats,
        np.tile(sc.scaler_.stats_, (2, 1)),
    )
```

::: mlforecast.target_transforms.LocalStandardScaler
    options:
      show_if_no_docstring: false

::: mlforecast.target_transforms.LocalMinMaxScaler
    options:
      show_if_no_docstring: false

::: mlforecast.target_transforms.LocalRobustScaler
    options:
      show_if_no_docstring: false

::: mlforecast.target_transforms.LocalBoxCox
    options:
      show_if_no_docstring: false

::: mlforecast.target_transforms.GlobalSklearnTransformer
    options:
      show_if_no_docstring: false

```python
# need this import in order for isinstance to work
from mlforecast.target_transforms import Differences as ExportedDifferences
```

```python
sk_boxcox = PowerTransformer(method='box-cox', standardize=False)
boxcox_global = GlobalSklearnTransformer(sk_boxcox)
single_difference = ExportedDifferences([1])
series = generate_daily_series(10)
fcst = MLForecast(
    models=[LinearRegression(), HistGradientBoostingRegressor()],
    freq='D',
    lags=[1, 2],
    target_transforms=[boxcox_global, single_difference]
)
prep = fcst.preprocess(series, dropna=False)
expected = (
    pd.Series(
        sk_boxcox.fit_transform(series[['y']])[:, 0], index=series['unique_id']
    ).groupby('unique_id', observed=True)
    .diff()
    .dropna()
    .values
)
np.testing.assert_allclose(prep['y'].values, expected)
preds = fcst.fit(series).predict(5)
```
