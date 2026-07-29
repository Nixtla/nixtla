# Contributing to SynForecast

## Development setup

Requirements: Python 3.10+, [uv](https://github.com/astral-sh/uv), Git, a
[Rust toolchain](https://rustup.rs/), and
[cargo-about](https://github.com/EmbarkStudios/cargo-about).

```bash
git clone https://github.com/Nixtla/synforecast.git
cd synforecast
cargo install cargo-about --version 0.9.1 --locked --features cli
uv sync --all-groups
uv run pre-commit install  # optional: run repository checks on every commit
```

`uv sync` builds and installs the Rust extension. After changing Rust code,
rebuild it in the development environment with:

```bash
uv run maturin develop --release
```

Verify the setup:

```bash
uv run pytest tests/ --no-cov -m "not stats"
```

## Creating a new generator

All generators inherit from `BaseGenerator` ([synforecast/base.py](synforecast/base.py)), which provides the common parameters (`min_length`, `max_length`, `freq`, `engine`, `alias`, `seed`, `start_datetime`, column names), timestamp generation, pattern injection, and the `generate()` method. A generator only declares its parameters as Pydantic fields and implements `generate_single_series()`.

### 1. Implement the generator

Create `synforecast/generators/my_generator.py`:

```python
"""White noise time series generator."""

import numpy as np
from pydantic import Field

from synforecast.base import BaseGenerator


class WhiteNoiseGenerator(BaseGenerator):
    """Generate white noise time series.

    Args:
        mean (float): Mean of the noise (default: 0.0).
        std (float): Standard deviation of the noise (default: 1.0).
    """

    mean: float = Field(default=0.0, description="Mean of the noise")
    std: float = Field(default=1.0, gt=0, description="Standard deviation")

    def generate_single_series(self, length: int) -> np.ndarray:
        return self.rng.normal(self.mean, self.std, length)
```

Conventions:

- Constructors are keyword-only: `WhiteNoiseGenerator(min_length=100, max_length=150, freq="D", std=2.0)`. Do not define `__init__`; declare parameters as Pydantic `Field` attributes with validation constraints (`gt`, `ge`, ...) or `field_validator`s.
- `freq` is a pandas offset alias (`'D'`, `'h'`, `'5min'`, `'MS'`, ...) or an integer; the base class validates it.
- Always draw randomness from `self.rng`, never from the global NumPy state, so output stays seed-deterministic.
- Use `self._sample_innovations(size, scale)` for noise terms so the generator honors `innovation_distribution`.

### 2. Export it

Add the class to `synforecast/generators/__init__.py` (import plus `__all__`).

### 3. Add tests

Create `tests/test_my_generator.py`. Use the shared helpers in [tests/helpers.py](tests/helpers.py) and the `engine` fixture (from [tests/conftest.py](tests/conftest.py)), which parameterizes tests over the pandas and polars engines. Mark statistical property tests with `@pytest.mark.stats`:

```python
import pytest

from synforecast.generators import WhiteNoiseGenerator
from tests.helpers import assert_long_format, assert_mean, assert_std


def test_long_format(engine: str) -> None:
    gen = WhiteNoiseGenerator(
        min_length=50, max_length=100, freq="D", seed=42, engine=engine
    )
    df = gen.generate(n_series=3)
    assert_long_format(df, n_series=3, min_length=50, max_length=100)


@pytest.mark.stats
def test_moments() -> None:
    gen = WhiteNoiseGenerator(
        min_length=5000, max_length=5000, freq="D", mean=5.0, std=2.0, seed=42
    )
    values = gen.generate(n_series=1)["y"].to_numpy()
    assert_mean(values, expected=5.0, std=2.0)
    assert_std(values, expected=2.0)
```

Statistical assertions in `tests/helpers.py` (`assert_mean`, `assert_std`, `assert_acf`, `assert_distribution`) use z-bounds derived from the estimator's sampling error; seed every generator so they are deterministic.

Also cover parameter validation (invalid inputs raise), seed determinism, and integer `freq` where relevant.

### 4. Add an example

Create `nbs/docs/generators/<category>/my_generator.ipynb`: a focused,
executable guide that instantiates the generator with keyword arguments,
explains its behavior, and visualizes representative output. Run
`uv sync --group docs --group integration-docs`, then run
`uv run --no-sync make execute_docs` to refresh committed notebook outputs.

## Testing

```bash
# Quick run of one file (skip coverage)
uv run pytest tests/test_my_generator.py --no-cov

# Skip the slower statistical property tests
uv run pytest tests/ --no-cov -m "not stats"

# Full suite with coverage
uv run pytest tests/ --cov=synforecast --cov-report=term-missing
```

## Code style

Ruff handles linting and formatting (88-character lines). Pre-commit runs both
Ruff hooks and regenerates the bundled Rust license notices when Rust dependency
or license configuration files change. If the generated notice changes, stage
it and run pre-commit again. Static typing is being tightened incrementally;
mypy is not yet a release gate for the whole package.

```bash
uv run pre-commit run --all-files
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets --all-features -- -D warnings
```

## Submitting changes

1. Create a feature branch.
2. Ensure tests pass and pre-commit is clean.
3. Update [GENERATORS.md](GENERATORS.md) and the generator count in [README.md](README.md) when adding a generator, and add an entry to [CHANGELOG.md](CHANGELOG.md).
4. Open a pull request with a clear description and a reference to any related issue.

Report bugs and request features via [GitHub Issues](https://github.com/Nixtla/synforecast/issues).

## Releasing

Publishing a GitHub Release triggers
[`.github/workflows/python-publish.yml`](.github/workflows/python-publish.yml).
The tag must exactly match the package version, such as `v0.1.0` for
`version = "0.1.0"`. The workflow verifies that the tagged commit is on the
protected `main` branch and that its CI, lint, and documentation checks
succeeded. It then builds and smoke-tests CPython 3.10–3.14 wheels for Linux
x86-64 and ARM64, Windows x86-64, and macOS Intel and Apple Silicon, plus the
source distribution. Successful artifacts are published automatically to PyPI
through Trusted Publishing. Publishing the release also triggers the production
documentation deployment.

To make a release:

1. Create a release-preparation branch from the latest `main`. Update the
   version in `pyproject.toml` and `rust/Cargo.toml`, refresh `uv.lock` and
   `rust/Cargo.lock`.
2. Open a pull request titled `Prepare vX.Y.Z release` and merge it after its
   required checks pass.
3. On GitHub, choose **Releases → Draft a new release**. Create the matching
   `vX.Y.Z` tag targeting `main`, generate or edit the release notes, and
   publish the release.
