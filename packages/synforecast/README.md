# SynForecast

SynForecast generates synthetic time series data. It provides 30 statistical,
stochastic, multivariate, domain-specific, and pretraining generators that
produce panels of series in long format, with optional injection of
changepoints, anomalies, and missing data.

> [!NOTE]
> SynForecast is in alpha. APIs and seed-identical outputs may change before
> the first stable release.

## Features

- 30 generators across five categories; see [GENERATORS.md](https://github.com/Nixtla/synforecast/blob/main/GENERATORS.md) for the full reference:
  - Statistical: Random Walk, Seasonal, SARIMA, ETS, INAR
  - Stochastic: GARCH, Ornstein-Uhlenbeck, Geometric Brownian Motion, Jump Diffusion, Poisson Process, Cyclic, Fractional Brownian Motion, Hawkes Process, Stochastic Volatility, Regime Switching, Chaotic System, Bounded Process, Levy Process
  - Multivariate: Copula, VAR, Gaussian Process
  - Domain-specific: Intermittent Demand, IoT Sensor, Energy Load, State Space, Daily Active Users, Vital Signs, Clickstream
  - Pretraining: TSI, Temporal Causal Model (TCM)
- Long-format output `[unique_id, ds, y]`, following the Nixtla data conventions
- Pattern injection: changepoints, anomalies, and missing data for every generator
- Exogenous variables: datetime features, pattern flags, and correlated columns
- Multiple dataframe engines via narwhals: pandas (default), polars, cudf, modin, pyarrow
- Seed-deterministic output, independent of the number of parallel workers
- Rust-accelerated generation via PyO3, with a pure NumPy fallback
- Dataset composition (`SynSet`) and augmentation of real data (`SynAugment`)

## Installation

```sh
pip install synforecast
```

Prebuilt wheels include the Rust extension on supported platforms. If `pip`
must build from the source distribution, a Rust toolchain is required. The
runtime can use its NumPy fallback when the extension is unavailable.

To work on SynForecast itself, follow [CONTRIBUTING.md](https://github.com/Nixtla/synforecast/blob/main/CONTRIBUTING.md).

## Quick start

Generate a panel of series from a balanced pool of generators:

```python
from synforecast import generate_series

df = generate_series(n_series=10, freq="D", min_length=50, max_length=500, seed=0)
```

`df` is a pandas DataFrame in long format:

```
  unique_id         ds         y
0         0 2000-01-01 -1.392372
1         0 2000-01-02 -0.938044
2         0 2000-01-03 -1.433790
3         0 2000-01-04  0.757597
4         0 2000-01-05  0.094236
```

`unique_id` is an integer categorical, `ds` is `datetime64[ns]` (or `int64` when `freq` is an integer), and `y` is `float64`.

Use a specific generator with explicit parameters:

```python
from synforecast.generators import RandomWalkGenerator

gen = RandomWalkGenerator(
    min_length=100,
    max_length=200,
    freq="D",
    drift=0.1,
    volatility=1.5,
    seed=42,
)
df = gen.generate(n_series=3)
```

Combine multiple generators into one dataset with `SynSet`:

```python
from synforecast import SynSet
from synforecast.generators import RandomWalkGenerator, SeasonalGenerator

dataset = SynSet(
    [
        RandomWalkGenerator(min_length=100, max_length=100, freq="D", seed=1),
        SeasonalGenerator(min_length=100, max_length=100, freq="D", seed=2),
    ]
)
df = dataset.generate(n_series_per_generator=5)  # 10 series, ids 0-9
```

## Common parameters

All generators share these parameters:

- `min_length`, `max_length`: bounds on the length of each series (required)
- `freq`: a pandas offset alias (`'D'`, `'h'`, `'5min'`, `'MS'`, `'W-MON'`, ...) or an integer for an integer time index (required)
- `engine`: output dataframe library, one of `'pandas'` (default), `'polars'`, `'cudf'`, `'modin'`, `'pyarrow'`
- `alias`: name of the generator (default: class name)
- `seed`: random seed for reproducibility (default: `None`)
- `start_datetime`: first timestamp of every series, anything `pandas.Timestamp` parses (default: `'2000-01-01'`)
- `id_col`, `time_col`, `target_col`: output column names (defaults: `'unique_id'`, `'ds'`, `'y'`)

`generate(n_series, start_id=0, n_jobs=-1)` returns the panel; output is
seed-deterministic and does not depend on `n_jobs`. Generator-specific
parameters are documented in [GENERATORS.md](https://github.com/Nixtla/synforecast/blob/main/GENERATORS.md).

## Examples

The `examples/` directory contains runnable scripts for the generators and
composition features:

```bash
uv run python examples/garch_example.py
```

## Documentation

- [GENERATORS.md](https://github.com/Nixtla/synforecast/blob/main/GENERATORS.md): reference for all 30 generators and their parameters
- [CONTRIBUTING.md](https://github.com/Nixtla/synforecast/blob/main/CONTRIBUTING.md): development setup and how to add a generator
- [CHANGELOG.md](https://github.com/Nixtla/synforecast/blob/main/CHANGELOG.md): release notes

## AI disclaimer

Parts of this project were developed with assistance from generative AI tools.
All AI-assisted code and documentation are reviewed, tested, and maintained by
human contributors, who remain responsible for correctness, security,
licensing, and design.

## License

Apache License 2.0
