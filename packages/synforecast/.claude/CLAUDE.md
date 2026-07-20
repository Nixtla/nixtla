# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SynForecast is a Python library for generating synthetic time series data with
30 statistical, stochastic, multivariate, domain-specific, and pretraining
generators. It supports pattern injection (changepoints, anomalies, missing
data), multiple dataframe backends via the narwhals abstraction, and Rust
acceleration via PyO3/maturin.

## Development Commands

```bash
# Install dependencies
uv sync --all-extras

# Run all tests with coverage
uv run pytest tests/ -v --cov=synforecast --cov-report=term-missing

# Run a single test file
uv run pytest tests/test_random_walk.py -v

# Run a specific test
uv run pytest tests/test_random_walk.py::test_basic_generation -v

# Lint and format (pre-commit hooks)
uv run pre-commit run --all-files

# Type checking is not yet a repository-wide release gate. When changing a
# typed module, run mypy on the affected files and avoid introducing new errors.
uv run mypy synforecast/path/to/changed_module.py

# Run an example
uv run python examples/random_walk_example.py
```

## Architecture

### Core Classes

**BaseGenerator** (`synforecast/base.py`): Abstract base class for all generators. Handles:
- Timestamp generation based on frequency
- Random state seeding for reproducibility
- Pattern injection pipeline: changepoints → anomalies → missingness
- DataFrame backend abstraction (polars/pandas/cudf/modin/pyarrow)

Every generator must implement `generate_single_series(length: int) -> np.ndarray`.

**SynSet** (`synforecast/dataset.py`): Combines multiple generators into
unified datasets.

### Generators (`synforecast/generators/`)

30 generators organized by type:
- **Statistical**: RandomWalk, Seasonal, SARIMA, ETS, INAR
- **Stochastic**: GARCH, OrnsteinUhlenbeck, GeometricBrownianMotion, JumpDiffusion, PoissonProcess, Cyclic, FractionalBrownianMotion, HawkesProcess, StochasticVolatility, RegimeSwitching, ChaoticSystem, BoundedProcess, LevyProcess
- **Multivariate**: Copula, VAR, GaussianProcess
- **Domain-Specific**: IntermittentDemand, IoTSensor, EnergyLoad, StateSpace, DailyActiveUsers, VitalSigns, Clickstream
- **Pretraining**: TSI, TCM

### Performance Layer (`synforecast/_core.py` + `_lib`)

Pure NumPy pattern injection with optional Rust acceleration via PyO3. The Rust
backend (`_lib`, built with maturin from `rust/`) provides accelerated
implementations of the generators, pattern injection, and distribution
functions.

### Building the Rust Extension

```bash
# One-time: install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and install into venv
maturin develop --release
```

## Creating a New Generator

1. Create `synforecast/generators/my_generator.py`
2. Inherit from `BaseGenerator`
3. Define parameters as Pydantic `Field` attributes with validators
4. Implement `generate_single_series(length: int) -> np.ndarray`
5. Export in `synforecast/generators/__init__.py`
6. Create `tests/test_my_generator.py`
7. Create `examples/my_generator_example.py`

## Code Style

- Ruff handles linting and formatting (88 char line length)
- Pre-commit hooks auto-format on commit
- Pydantic v2 for parameter validation
- Always seed random generation for reproducibility
