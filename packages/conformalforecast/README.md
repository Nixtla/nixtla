# Nixtla

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Adaptive%20conformal%20prediction%20intervals%20for%20time%20series%20forecasting%20by%20Nixtla%20&url=https://github.com/Nixtla/conformalforecast&via=nixtlainc&hashtags=ConformalPrediction,TimeSeries,Forecasting)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/company/nixtlainc)

<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png"/>
<h1 align="center">Conformal 🎯 Forecast</h1>
<h3 align="center">Adaptive conformal prediction intervals for time series forecasting</h3>

[![CI](https://github.com/Nixtla/conformalforecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Nixtla/conformalforecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/conformalforecast)](https://pypi.org/project/conformalforecast/)
[![PyPI](https://img.shields.io/pypi/v/conformalforecast?color=blue)](https://pypi.org/project/conformalforecast/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/conformalforecast/blob/main/LICENSE)
[![Docs](https://github.com/Nixtla/conformalforecast/actions/workflows/build-docs.yaml/badge.svg)](https://github.com/Nixtla/conformalforecast/actions/workflows/build-docs.yaml)

**ConformalForecast** turns any point forecaster into one that emits calibrated
prediction intervals. It ships seven conformal methods — split, CQR, ACI, AgACI,
NexCP, SPCI, and weighted split — with a focus on the online and adaptive
variants designed for non-exchangeable, sequential data. It works with
StatsForecast, MLForecast, NeuralForecast, or a plain Python callable.
</div>

> [!NOTE]
> ConformalForecast is in alpha. APIs are unstable until v0.2.0 and may change
> without a deprecation cycle.

## Installation

Install ConformalForecast from PyPI:

```bash
pip install conformalforecast
```

The core install depends only on `numpy`, `narwhals`, `pydantic`, and `scipy`.
It installs neither pandas nor polars: the Nixtla adapters accept whichever one
your forecasting library uses.
Everything else is an extra:

```bash
pip install "conformalforecast[nixtla]"    # statsforecast, mlforecast, neuralforecast, utilsforecast
pip install "conformalforecast[spci]"      # quantile-forest, for SPCI
pip install "conformalforecast[plotting]"  # matplotlib
pip install "conformalforecast[all]"       # all of the above
```

## Quick start

Calibrate a forecaster on its own cross-validation errors, then predict with
intervals:

```python
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from conformalforecast import SplitConformal
from conformalforecast.adapters import StatsForecastAdapter

# A three-series monthly panel in Nixtla long format.
rng = np.random.default_rng(0)
n_series, T, season, horizon = 3, 132, 12, 6
t = np.arange(T)
df = pd.concat(
    [
        pd.DataFrame(
            {
                "unique_id": f"series_{s}",
                "ds": pd.date_range("2000-01-01", periods=T, freq="MS"),
                "y": 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / season) + rng.normal(0, 1, T),
            }
        )
        for s in range(n_series)
    ],
    ignore_index=True,
)

sf = StatsForecast(models=[SeasonalNaive(season_length=season)], freq="MS", n_jobs=1)
sf.fit(df)

forecaster = StatsForecastAdapter(
    sf=sf, train_df=df, horizon=horizon, freq="MS", model_name="SeasonalNaive"
)

method = SplitConformal(alpha=0.1)  # 90% intervals
fitted = method.calibrate_cv(forecaster, n_windows=30, refit=True)

panel = df.pivot(index="unique_id", columns="ds", values="y").to_numpy()
prediction = fitted.predict(panel)

prediction.point  # (n_series, 1, horizon)
prediction.interval  # (n_series, 1, horizon, 2) — (lower, upper)
```

`alpha=0.1` asks for 90% coverage. `calibrate_cv` runs the forecaster's own
cross-validation, calibrates on the resulting errors and returns a fitted
object; `calibrate` takes explicit history/truth arrays instead when you
already have them. The method itself is never modified, so one configured
method can calibrate as many forecasters as you like.

See the [quick start](https://nixtlaverse.nixtla.io/conformalforecast/docs/getting-started/quickstart.html)
for the full walkthrough, including measured coverage on a held-out split.

## Methods

| Method | Handles | Online | Reference |
|---|---|:--:|---|
| `SplitConformal` | the exchangeable baseline | – | Papadopoulos et al. 2002 |
| `ConformalizedQuantileRegression` | heteroscedastic errors, via quantile forecasts | – | Romano et al. 2019 |
| `AdaptiveConformalInference` | distribution shift, by adapting `alpha` | ✅ | Gibbs & Candès 2021 |
| `AggregatedAdaptiveConformalInference` | shift without picking a learning rate | ✅ | Zaffran et al. 2022 |
| `NonexchangeableConformalPrediction` | shift, by down-weighting stale residuals | ✅ | Barber et al. 2023 |
| `SequentialPredictiveConformalInference` | conditional coverage, via a quantile regressor | ✅ | Xu & Xie 2023 |
| `WeightedSplit` | covariate shift and transfer, by reweighting calibration scores | – | Tibshirani et al. 2019 |

The fitted objects of online methods implement `update(prediction, truth)`,
which advances them in place, so intervals keep recalibrating as new
observations arrive.

Every fitted object offers `recalibrate()` for a target domain you have labels
for, returning a new fitted object. A fitted `WeightedSplit` additionally
offers `adapt()`, which needs only target features or target history — no
target labels at all. See
[Transfer and adaptation](https://nixtlaverse.nixtla.io/conformalforecast/docs/capabilities/transfer.html).

## Works with the Nixtlaverse

An adapter is the uniform interface a conformal method talks to. It converts a
library's long-format frames into arrays at the boundary, so the calibration
math never sees a DataFrame. Fit your forecaster as usual, then wrap it.

### StatsForecast

```python
from conformalforecast.adapters import StatsForecastAdapter

forecaster = StatsForecastAdapter(sf=sf, train_df=df, horizon=14, freq="D", model_name="AutoETS")
```

### MLForecast

```python
from conformalforecast.adapters import MLForecastAdapter

forecaster = MLForecastAdapter(
    mlf=mlf, train_df=df, horizon=14, freq="D", model_name="LGBMRegressor"
)
```

### NeuralForecast

```python
from conformalforecast.adapters import NeuralForecastAdapter

# The horizon is read from the model's own `h`.
forecaster = NeuralForecastAdapter(nf=nf, train_df=df, freq="D", model_name="NHITS")
```

### Anything else

```python
from conformalforecast.adapters import CallableAdapter

forecaster = CallableAdapter(predict_fn=my_model.forecast, horizon=14, n_series=100)
```

Install the forecasting libraries you use separately, or through the `nixtla`
extra. Importing `conformalforecast` never imports statsforecast, mlforecast,
neuralforecast, or torch. See the
[adapters guide](https://nixtlaverse.nixtla.io/conformalforecast/docs/adapters/index.html)
for capability requirements and exogenous handling.

## A method and the fitted object it returns

Every method is a pair: a **method** that is pure configuration, and the
**fitted object** its `fit` / `calibrate` / `calibrate_cv` returns. The method
is never modified by fitting; everything learned from calibration lives on the
returned object.

```python
from conformalforecast import AbsoluteResidual, AdaptiveConformalInference

method = AdaptiveConformalInference(alpha=0.1, gamma=0.01, score=AbsoluteResidual())
fitted = method.calibrate_cv(forecaster, n_windows=50)
result = fitted.predict(history)
fitted.update(result.point, truth)  # online methods advance in place
```

The score is configuration too: pass `score=` to the constructor, or leave it
out for the method's default. An incompatible score is rejected right there.
The forecaster is the one collaborator bound at calibration.

Which entry point you call depends on who owns the cross-validation loop.

**The array surface** — `method.fit(predictions, truths)`, then
`fitted.interval(predictions)` and `fitted.update(prediction, truth)`. Pure
array-in/array-out calibration math, taking no forecaster at all. A signature
here may only accept numpy-coercible arrays and plain-Python config — never
adapters, DataFrames, or library types. This is the surface for libraries that
already own their cross-validation and predict orchestration: statsforecast,
mlforecast, neuralforecast.

**The forecaster surface** — `method.calibrate(forecaster, histories, truths)`
or `method.calibrate_cv(forecaster, n_windows)`, then
`fitted.predict(history)`. CV/predict dispatch through a `ForecasterAdapter`,
written once in the base classes. This is the surface for standalone users.

The [architecture page](https://nixtlaverse.nixtla.io/conformalforecast/docs/concepts/architecture.html)
explains when to reach for which.

## Features

- Seven conformal methods, four of them online and adaptive
- Declarative methods whose `fit` returns a fitted object, with an array
  surface for embedding and a forecaster surface for standalone use
- Panels are first class — every method is multi-series and multi-horizon
- Ragged panels through NaN masking, with no imputation
- NaN-aware quantiles identical across numpy versions — the exact
  `ceil((1 - alpha)(n + 1))` order statistic for split and CQR
- `float32` preserved on the array surface (`fit` / `interval` / `update`)
- Exogenous features threaded through `calibrate` and `predict` (see the roadmap
  for the limits)
- Coverage and interval-quality diagnostics in `conformalforecast.diagnostics`
- Deterministic given a config and inputs; `seed` defaults to `0`, never `None`

## Why ConformalForecast?

Conformal prediction gives distribution-free coverage guarantees without
assuming a noise model. The catch for forecasting is that the classic guarantee
assumes exchangeable data, and time series are not exchangeable: they drift,
they have regimes, and their errors are serially correlated. This library
prioritizes the methods built for that setting.

It is designed for:

- Adding calibrated intervals to a point forecaster that has none
- Replacing intervals from a model whose distributional assumptions do not hold
- Keeping coverage under distribution shift, via online recalibration
- Auditing whether the intervals you already ship actually cover
- Embedding calibration math directly inside a forecasting library

Know the limits. Conformal coverage is *marginal*, not conditional: 90%
intervals cover 90% of the time on average, not for every series or every
horizon step. Adaptive methods trade finite-sample exactness for robustness to
shift, and their guarantees are asymptotic or regret-based. Interval width is a
property of your forecaster — a better model gives narrower intervals, not more
valid ones.

## Documentation

- [Getting started](https://nixtlaverse.nixtla.io/conformalforecast/docs/getting-started/quickstart.html)
- [Tutorials and API reference](https://nixtlaverse.nixtla.io/conformalforecast/)
- [Roadmap](https://github.com/Nixtla/conformalforecast/blob/main/ROADMAP.md) — known gaps and limitations
- [Contributing](https://github.com/Nixtla/conformalforecast/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/Nixtla/conformalforecast/blob/main/CHANGELOG.md)
- [Support](https://github.com/Nixtla/conformalforecast/blob/main/SUPPORT.md)
- [Security policy](https://github.com/Nixtla/conformalforecast/blob/main/SECURITY.md)
- [Third-party notices](https://github.com/Nixtla/conformalforecast/blob/main/THIRD_PARTY_NOTICES.md)
- [Code of conduct](https://github.com/Nixtla/conformalforecast/blob/main/CODE_OF_CONDUCT.md)
- [Citation](https://github.com/Nixtla/conformalforecast/blob/main/CITATION.cff)

## 🤝 How to contribute

Bug reports, method implementations, adapters, and documentation are all
welcome. Read
[CONTRIBUTING.md](https://github.com/Nixtla/conformalforecast/blob/main/CONTRIBUTING.md)
for the development setup, the one-class-two-surfaces architecture new
methods have to fit, and the PR expectations. For a new method or adapter, open an issue first — a
short design discussion up front avoids rework.

```bash
git clone https://github.com/Nixtla/conformalforecast.git
cd conformalforecast
uv sync --all-extras --group dev
uv run pytest tests/unit
```

## 🙏 How to cite

If ConformalForecast helps your research, please cite it. The repository ships a
[CITATION.cff](https://github.com/Nixtla/conformalforecast/blob/main/CITATION.cff),
so GitHub's "Cite this repository" button gives you BibTeX directly:

```bibtex
@software{conformalforecast,
  author = {Peixeiro, Marco and Caballero Ramirez, Saul and {Nixtla}},
  title = {{ConformalForecast}: Adaptive conformal prediction for time series forecasting},
  year = {2026},
  license = {Apache-2.0},
  url = {https://github.com/Nixtla/conformalforecast}
}
```

Please also cite the paper behind whichever method you used — the references are
in each method's API documentation.

## AI disclaimer

Parts of this project were developed with assistance from generative AI tools.
All AI-assisted code and documentation are reviewed, tested, and maintained by
human contributors, who remain responsible for correctness, security,
licensing, and design.

## License

ConformalForecast is licensed under the [Apache License 2.0](https://github.com/Nixtla/conformalforecast/blob/main/LICENSE).
