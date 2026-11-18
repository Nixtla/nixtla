---
output-file: hooks.html
title: Stage hooks
description: Observe forecast stages and profile their time and memory use.
---

`MLForecast` accepts `hooks=[...]` to observe preprocessing, fit, prediction,
and cross validation stages. Hooks receive the live arguments passed to each
stage; changing those objects inside a hook is unsupported.

```python
from mlforecast import MLForecast
from mlforecast.hooks import Profiler

profiler = Profiler(memory=True)
fcst = MLForecast(models=..., freq="D", lags=[1], hooks=[profiler])
cv = fcst.cross_validation(df, n_windows=3, h=7)
model_times = (
    profiler.to_df()
    .query("stage == 'fit_model'")
    .groupby("model")["seconds"]
    .sum()
)
```

`fit_model` and `predict_model` are emitted once per model, and `fit_model` once
per horizon for direct models. `predict_model` covers each model's feature
updates as well as its `predict` calls.

Each span's duration includes its children, so summing durations across every
stage counts nested work more than once. During interval calibration, nested
stages use a scratch forecaster. A hook can distinguish them from the original
forecaster with `fcst is original_fcst`. Call `profiler.reset()` only between runs.

::: mlforecast.hooks.Hook
    handler: python
    options:
      docstring_style: google
      heading_level: 3
      show_root_heading: true

::: mlforecast.hooks.Profiler
    handler: python
    options:
      docstring_style: google
      heading_level: 3
      show_root_heading: true
