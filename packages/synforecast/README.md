# Nixtla

[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)

<div align="center">
<h1>SynForecast</h1>
<h3>Fast synthetic time series for testing, augmentation, and pretraining</h3>

[![CI](https://github.com/Nixtla/synforecast/actions/workflows/ci.yaml/badge.svg)](https://github.com/Nixtla/synforecast/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/pypi/pyversions/synforecast)](https://pypi.org/project/synforecast/)
[![PyPI](https://img.shields.io/pypi/v/synforecast?color=blue)](https://pypi.org/project/synforecast/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/synforecast/blob/main/LICENSE)
[![Docs](https://github.com/Nixtla/synforecast/actions/workflows/build-docs.yaml/badge.svg)](https://github.com/Nixtla/synforecast/actions/workflows/build-docs.yaml)

**SynForecast** generates validated synthetic time-series panels with 30
statistical, stochastic, multivariate, domain-specific, and pretraining
generators. It follows the Nixtla long format and supports controlled
changepoints, anomalies, missing data, exogenous variables, and augmentation
of observed series.
</div>

> [!NOTE]
> SynForecast is in alpha. APIs and seed-identical outputs may change before
> the first stable release.

## Installation

Install SynForecast from PyPI:

```bash
pip install synforecast
```

Prebuilt wheels include the Rust extension on supported platforms. Building
from the source distribution requires a Rust toolchain; a NumPy fallback is
available when the extension cannot be loaded.

## Quick start

Generate a long-format panel from a balanced collection of generators:

```python
from synforecast import generate_series

synthetic_df = generate_series(
    n_series=100,
    freq="D",
    min_length=100,
    max_length=100,
    seed=42,
)
```

The result is a pandas DataFrame with the standard Nixtla columns
`[unique_id, ds, y]`. Pass `engine="polars"` to return a Polars DataFrame.

Use a specific generator when you need explicit control over the data-generating
process:

```python
from synforecast.generators import SeasonalGenerator

generator = SeasonalGenerator(
    min_length=365,
    max_length=365,
    freq="D",
    seasonality_period=7,
    seasonality_amplitude=10.0,
    seed=42,
)
synthetic_df = generator.generate(n_series=20)
```

See the [quick start](https://nixtlaverse.nixtla.io/synforecast/docs/getting-started/quickstart.html)
and [generator reference](https://github.com/Nixtla/synforecast/blob/main/GENERATORS.md)
for the complete API.

## Augment observed series

`SynAugment` analyzes each input series, fits a suitable generator, and adds
synthetic series with matching timestamps and statistical characteristics:

```python
from synforecast import SynAugment

augmenter = SynAugment(seed=42)
augmented_train_df = augmenter.augment(train_df, n_augment=2)
```

Fit augmentation parameters on the training split only; fitting on validation
or test observations would leak information into model training. See the
[augmentation guide](https://nixtlaverse.nixtla.io/synforecast/docs/capabilities/augmentation.html)
for generator overrides and diagnostics.

## Works with the Nixtlaverse

SynForecast materializes the same `[unique_id, ds, y]` schema used throughout
the Nixtlaverse. The generated or augmented DataFrame can be passed directly
to forecasting libraries without a SynForecast-specific adapter.

### NeuralForecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

nf = NeuralForecast(
    models=[NHITS(h=14, input_size=28, max_steps=100)],
    freq="D",
)
nf.fit(df=synthetic_df)
neural_forecasts = nf.predict()
```

### MLForecast

```python
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

mlf = MLForecast(
    models=LinearRegression(),
    freq="D",
    lags=[1, 7, 14],
)
mlf.fit(synthetic_df)
ml_forecasts = mlf.predict(h=14)
```

### StatsForecast

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

sf = StatsForecast(
    models=[AutoETS(season_length=7)],
    freq="D",
)
sf.fit(synthetic_df)
statistical_forecasts = sf.predict(h=14)
```

Install the forecasting libraries you want to use separately. SynForecast
does not add them to its runtime dependencies. See the
[Nixtlaverse guide](https://nixtlaverse.nixtla.io/synforecast/docs/integrations/nixtlaverse.html)
for materialized pretraining and augmentation workflows.

## Why SynForecast?

Synthetic time series are useful when real observations are scarce, sensitive,
expensive, or do not cover the conditions a system must handle. SynForecast is
designed for:

- Testing forecasting and anomaly-detection pipelines against known behavior
- Augmenting small training panels without altering validation data
- Pretraining global forecasting models on diverse temporal processes
- Stress-testing changepoints, missingness, anomalies, and regime changes
- Reproducible simulation with explicit, validated generator configurations

Synthetic data reflects the assumptions of its generators. Validate those
assumptions and downstream performance for your use case.

## Features

- 31 generators across statistical, stochastic, multivariate, domain-specific,
  and pretraining categories
- Long-format output following Nixtla conventions
- pandas, Polars, cuDF, Modin, and PyArrow output through Narwhals
- Changepoint, anomaly, missingness, and exogenous-variable injection
- Dataset composition with `SynSet` and augmentation with `SynAugment`
- Seed-deterministic parallel generation
- Rust acceleration through PyO3 with a NumPy fallback

## Documentation

- [Getting started](https://nixtlaverse.nixtla.io/synforecast/docs/getting-started/quickstart.html)
- [Tutorials and capabilities](https://nixtlaverse.nixtla.io/synforecast/)
- [Generator reference](https://github.com/Nixtla/synforecast/blob/main/GENERATORS.md)
- [Contributing](https://github.com/Nixtla/synforecast/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/Nixtla/synforecast/blob/main/CHANGELOG.md)

## AI disclaimer

Parts of this project were developed with assistance from generative AI tools.
All AI-assisted code and documentation are reviewed, tested, and maintained by
human contributors, who remain responsible for correctness, security,
licensing, and design.

## License

SynForecast is licensed under the [Apache License 2.0](https://github.com/Nixtla/synforecast/blob/main/LICENSE).
