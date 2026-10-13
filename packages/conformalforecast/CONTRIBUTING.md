# Contributing to conformalforecast

`conformalforecast` is a Python library for adaptive conformal prediction on time
series forecasts. This document is for anyone opening a pull request — bug
fixes, new methods, new adapters, docs, or examples. It covers how to set up
the project, the conventions you need to follow, and what to expect during
review.

## Quick start

```bash
git clone https://github.com/Nixtla/conformalforecast
cd conformalforecast
uv sync --all-extras --all-groups
uv run pre-commit install
uv run pytest -v
```

Branch from `main`, push to your fork, and open a PR against `main`. The CI
matrix runs on Python 3.10–3.14 against `tests/unit`.

## What contributions are welcome

- **Bug fixes** — in any file. Always add a regression test.
- **New conformal methods** — go in `src/conformalforecast/methods/` with a unit
  test in `tests/unit/`.
- **New adapters** — go in `src/conformalforecast/adapters/` with an integration
  test in `tests/integration/` and a new extras entry in `pyproject.toml`.
- **New nonconformity scores** — go in `src/conformalforecast/nonconformity/`.
- **New aggregators or quantile-regression backends** — go in
  `src/conformalforecast/aggregators/` and `src/conformalforecast/quantile_regressors/`
  respectively. Both are pluggable ABCs designed for extension.
- **Documentation and examples** — docstring fixes, notebook examples under
  `examples/`, README clarifications.
- **Performance work** — benchmarked improvements to hot loops. Include
  before/after numbers in the PR description.

Open an issue first for: large architectural changes, new core dependencies,
or anything that would significantly grow install size. These are not
rejections — they need a design conversation before code.

## Project layout

```
src/conformalforecast/
├── base.py              # ForecasterAdapter, ScoreFunction, ConformalMethod ABCs; type aliases
├── capabilities.py      # Capability mixins (SupportsRefit, SupportsQuantiles, ...)
├── methods/             # CP methods (Split, CQR, ACI, AgACI, NexCP, SPCI)
├── adapters/            # Forecaster adapters (Callable, StatsForecast, MLForecast,
│                        #   NeuralForecast, Darts, sktime)
├── nonconformity/       # Score functions (AbsoluteResidual, SignedResidual, QuantileScore)
├── aggregators/         # Online aggregators (EWA) used by AgACI
├── quantile_regressors/ # Quantile regression backends (QRF) used by SPCI
├── diagnostics/         # Coverage, scoring, conditional, reports
├── datasets/            # Small built-in datasets for tests and examples
└── utils/               # Validation, plotting, type helpers

tests/unit/              # Fast tests, no optional dependencies required
tests/integration/       # Slower tests across adapters; gated on extras
examples/                # Runnable notebooks
```

## Conventions that matter

### Array shapes

The leading axis is always `n_series`. The canonical aliases live in
`src/conformalforecast/base.py`:

- `Series` — `(n_series, T)`. A panel of time series.
- `Forecast` — `(n_series, n_samples, horizon)`. Always 3D, even when
  `n_series=1` or `horizon=1`.
- `Interval` — `(n_series, n_samples, horizon, 2)`. The last axis is
  `(lower, upper)`.

```python
# A PredictionResult from a fitted method
fitted = method.calibrate_cv(forecaster, n_windows=50)
result = fitted.predict(history)  # history: (n_series, T)
result.point.shape  # (n_series, 1, horizon)
result.interval.shape  # (n_series, 1, horizon, 2)
```

Document the shape of every array in every public docstring. Reviewers will
ask if you don't.

### Trailing underscore on fitted state

Following sklearn's convention, configuration has no trailing underscore and
fitted state does. The two live on different objects: a method holds only
configuration, and the fitted object its `fit` / `calibrate` / `calibrate_cv`
returns holds the fitted state, plus a copy of the configuration as
`method_`:

```python
method.alpha  # configuration
method.gamma  # configuration
fitted.alpha_t_  # fitted state, set by fit
fitted.scores_  # fitted state, grows in update()
fitted.method_.gamma  # the configuration it was fitted under
```

### Capability mixins

Adapters declare what they can do by inheriting from mixins in
`src/conformalforecast/capabilities.py`: `SupportsRefit`, `SupportsQuantiles`,
`SupportsBootstrap`, `SupportsCrossValidation`,
`SupportsCrossValidationQuantiles`. Methods declare what they need via the
`REQUIRED_CAPABILITIES` class attribute. The check runs where the forecaster
is bound: the first statement of `calibrate` and `calibrate_cv` asks each
requirement through `ForecasterAdapter.supports(capability)` and raises
`UnsupportedCapability` if the adapter is inadequate. A method's constructor
takes no forecaster, so there is nothing to check at construction.

Never check capabilities inside `predict()`. It lives on the fitted object,
which only exists once the check has passed.

### Frames at the boundary only

DataFrames live in `adapters/`, which accept pandas or polars frames and
convert them to numpy panels through narwhals. No pandas, polars or narwhals
inside `methods/`, `nonconformity/`, `diagnostics/`, `panel.py` or
`_quantiles.py`. Those layers work in numpy.

### Online lifecycle

Online methods (`IS_ONLINE = True`) have fitted objects with a non-trivial
`update()`, implemented as the `_update` hook. The contract is: after
`fitted = method.calibrate(...)`, callers alternate `fitted.predict(history)`
and `fitted.update(prediction, truth)`, which mutates `fitted` in place. The caller passes back the `point` from the
`PredictionResult` they received — no hidden state coupling between `predict`
and `update`. See `src/conformalforecast/methods/aci.py` for the canonical
example.

### Dependencies

Core dependencies are `narwhals`, `numpy`, `pydantic`, `scipy`. Anything else
is an extra in `pyproject.toml` (`nixtla`, `spci`, `plotting`).
Adapters soft-import their underlying library and raise an `ImportError` with
the exact install command if it's missing:

```python
try:
    from statsforecast import StatsForecast
except ImportError as _err:
    raise ImportError(
        "StatsForecastAdapter requires the 'statsforecast' package. "
        "Install it with: pip install conformalforecast[nixtla]"
    ) from _err
```

Do not add a new top-level dependency without opening an issue first.

## Adding things

### Adding a new conformal method

1. Read `src/conformalforecast/methods/base.py` for the `ConformalMethod` and
   `ConformalForecaster` contracts.
2. Read `src/conformalforecast/methods/split.py` with `fixed_offset.py`
   (offline reference) and `src/conformalforecast/methods/aci.py` (online
   reference).
3. Create `src/conformalforecast/methods/<name>.py` with a method class
   inheriting `ConformalMethod` and its fitted class inheriting
   `ConformalForecaster`.
4. Declare `REQUIRED_CAPABILITIES`, `CONSUMES_QUANTILES`, `SCORE_TYPE` and
   `IS_ONLINE` as `ClassVar`s on the method.
5. Implement `_default_score` and `fit` on the method, and `interval` (plus
   `_update` if online) on the fitted class. Never override `calibrate`,
   `calibrate_cv`, `predict`, `recalibrate` or `update`.
6. Export both classes from `src/conformalforecast/methods/__init__.py`.
7. Add `tests/unit/methods/test_<name>.py` (array surface: coverage, shape,
   determinism, NaN masking) and `tests/unit/methods/test_<name>_forecaster.py`
   (adapter surface: dispatch, capability errors, binding, coverage smoke test).

### Adding a new adapter

1. Read `src/conformalforecast/adapters/callable.py` (the minimal reference) and
   `src/conformalforecast/adapters/statsforecast.py` (a real-library reference).
2. Create `src/conformalforecast/adapters/<library>.py` inheriting
   `ForecasterAdapter` plus whichever capability mixins it supports.
3. Soft-import the underlying library at the top of the file and raise an
   `ImportError` with the install command on failure.
4. Implement every abstract method from each parent class. Convert
   framework-specific types (DataFrames, TimeSeries, etc.) to numpy at the
   boundary.
5. Add an entry under `[project.optional-dependencies]` in `pyproject.toml`
   and reference it from the install message.
6. Add `tests/integration/test_<library>_adapter.py`. Gate the module on
   `pytest.importorskip("<library>")` so unit-test runs without extras still
   pass.

### Adding a new nonconformity score

1. Read `src/conformalforecast/nonconformity/absolute.py` (the simplest reference).
2. Create `src/conformalforecast/nonconformity/<name>.py` with a class inheriting
   `ScoreFunction`.
3. Implement `score`, `invert`, and `fit` (no-op if there's no internal
   state to learn).
4. Export the class from `src/conformalforecast/nonconformity/__init__.py`.
5. Add a unit test alongside the existing `test_absolute_residual.py` and
   `test_quantile_score.py`.

## Code style and tooling

- Python 3.10+. Use modern syntax: `X | None`, `list[int]`, PEP 695 where
  natural.
- `from __future__ import annotations` at the top of every module.
- Type-annotate all public functions. Use `numpy.typing.NDArray`.
- NumPy-style docstrings (Parameters / Returns / Raises). Document array
  shapes.
- 100-character line limit.
- Tools and commands:
  - `uv run ruff check .` — lint
  - `uv run ruff format .` — format (or `--check` for CI parity)
  - `uv run mypy src/` — strict type-checking on `src/`; tests are looser
  - `uv run pre-commit install` — wire up the pre-commit hooks (ruff,
    ruff-format, trailing whitespace, end-of-file, YAML/TOML syntax)

## Tests

- Every method in `methods/` has a matching `tests/unit/test_<method>.py`.
- Every method test includes:
  - **Coverage** on synthetic data: empirical coverage matches `1 - alpha`
    within a documented tolerance.
  - **Shape**: outputs match the documented shapes for `n_series` in `{1, 3}`
    and `horizon` in `{1, 6}`.
  - **Determinism**: same input plus same seed produces the same output.
  - **Capability errors**: a method that needs `SupportsQuantiles` /
    `SupportsBootstrap` raises `UnsupportedCapability` from `calibrate` /
    `calibrate_cv` when given an inadequate adapter.
- Use `numpy.random.default_rng(seed)` everywhere. Never `np.random.seed`.
- Unit tests use `CallableAdapter` (and its CV variant in
  `tests/_online_helpers.py`) so they run without optional dependencies.
- Integration tests gate on `pytest.importorskip("<library>")` for the
  relevant backend.

Reference: `tests/unit/test_aci.py` for the typical online-method pattern.

Run the suites with:

```bash
uv run pytest tests/unit          # fast, no extras required
make test_integration             # slower, needs extras
```

`make test_integration` runs the integration suite as **two** pytest sessions.
That is not a stylistic choice: torch and scikit-learn each vendor a copy of
`libomp`, and loading both into one process segfaults on macOS arm64. pytest
imports every collected module before running anything, so a marker cannot
prevent it — the torch-backed tests live in
`tests/integration/torch_isolated/` and are collected separately.

## Documentation

The site is built with Quarto and published to Mintlify, the same pipeline the
rest of the Nixtlaverse uses. It has two sources:

- **API reference** — `docs/*.html.md` stubs holding mkdocstrings `:::`
  directives with an explicit `members:` list. `docs/to_mdx.py` renders them, so
  editing a docstring in `src/` updates the page on the next build.
- **Narrative pages** — `nbs/docs/**`, either `.ipynb` (executed, for anything
  showing output) or `.qmd` (prose only). Quarto renders these.

Both land in `docs/mintlify/`, whose only committed file is `docs.json` — the
hand-maintained navigation. Everything else there is generated and gitignored.

### Building

Needs [Quarto](https://quarto.org/docs/get-started/) 1.4.515 on your PATH plus
both dependency groups:

```bash
uv sync --group docs --group integration-docs

make test_docs     # execute every notebook — this is what CI gates on
make all_docs      # full site build into docs/mintlify/
uv run python scripts/check_docs_nav.py   # docs.json vs built pages, both ways
```

`make all_docs` clones the shared Nixtla docs tooling into `docs-scripts/` at a
pinned commit. Bump `DOCS_SCRIPTS_SHA` in the `Makefile` to update it — the pin
exists so a push to that mutable branch cannot change a build unreviewed.

### Adding or changing a page

Add the file under `nbs/docs/`, add one line to `docs/mintlify/docs.json`, and
run `make execute_docs` to write refreshed outputs back into the notebook. Commit
those outputs: CI re-executes every notebook, so a page calling a removed API
fails the build rather than rotting quietly.

#### Keep the models fast

Documentation pages demonstrate **how to conformalize a forecaster**, not how to
forecast well. A `SeasonalNaive` that runs in a second makes the same point as a
tuned model and keeps the docs build under ten minutes. Budgets: statsforecast
pages under 10s, mlforecast under 30s, neuralforecast under 90s on CPU (the
NeuralForecast page trains two models at `max_steps=100` inside that), SPCI
pinned to a single series. If a page genuinely needs something slower, say so in
the page.

Notebooks must not use the network — CI executes them on every pull request, and
a runtime download makes the docs build fail on an unrelated outage. Generate
data with `numpy.random.default_rng(seed)`, or read a committed file from
`nbs/data/`.

## Using AI tools to write code

> Using AI assistants (Claude, Copilot, Cursor, etc.) to write contributions
> is welcome and normal. The expectation is that you understand and verify
> the code before submitting it.
>
> - Run the tests yourself locally. Don't rely on the assistant's claim that
>   they pass.
> - Read what the assistant produced and confirm it does what you intend.
>   AI-generated code can look reasonable while subtly mishandling edge
>   cases — especially shape conventions and the online predict/update
>   lifecycle.
> - If the assistant suggests changes outside the scope of your PR
>   (drive-by refactors, unrelated formatting), revert them. PRs that bundle
>   unrelated changes are hard to review.
> - You are responsible for the correctness, licensing, and design of your
>   contribution regardless of how it was produced.
>
> The maintainers also use AI tools. The standard is the same on both sides:
> the code works and you can defend it in review.
>
> [`AGENTS.md`](AGENTS.md) holds the rules for the assistant itself — most
> tools read it automatically. Point yours at it if it doesn't.

## Reporting bugs and requesting features

Open an issue from the [issue chooser](https://github.com/Nixtla/conformalforecast/issues/new/choose):
bug report, feature request, or documentation issue. Bug reports need a minimal
reproducer, the `conformalforecast` version, your Python version, and the OS.
Usage questions belong in the
[Nixtla community Slack](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)
rather than the issue tracker.

For non-trivial feature ideas (new method, new adapter, API change), open an
issue before a PR. A short design discussion up front avoids rework later.

## Pull request expectations

The PR template in `.github/PULL_REQUEST_TEMPLATE.md` carries the checklist.
Three things to highlight:

- Keep each PR focused on one change. Split unrelated work into separate PRs.
- **Branch from `main` and target `main`. No stacked PRs.** A PR based on
  another unmerged branch is rejected by the `PR target` check. Stacks make
  review order load-bearing, turn one review comment into a rebase of every
  branch above it, and leave the diff of the upper PRs unreadable until the
  lower ones land. If your work genuinely depends on an unmerged PR, either
  wait for it to merge and then branch from `main`, or fold both changes into
  a single PR. Cross-referencing an unmerged PR in the description is fine —
  basing your branch on it is not.
- **Title the PR as a conventional commit** — `<type>[scope][!]: <description>`,
  using the same types as the commit history (`feat`, `fix`, `docs`, `test`,
  `refactor`, `perf`, `ci`, `build`, `chore`, `revert`). The merge commit takes
  its subject from the title, so it becomes permanent history. The `PR title`
  check enforces it.
- **Add exactly one changelog label.** Release notes are generated from labels,
  not from titles, so an unlabelled PR would land in "Other Changes". The
  `Changelog label` check enforces it.

  | Label | Release-note section | Use for |
  | --- | --- | --- |
  | `breaking change` | Breaking Changes | any change to the public API that breaks existing code |
  | `enhancement` | New Features | `feat:` |
  | `bug` | Bug Fixes | `fix:` |
  | `documentation` | Documentation | `docs:`, notebooks, README |
  | `maintenance` | Maintenance | `ci:`, `test:`, `refactor:`, `chore:` |
  | `dependencies` | Dependencies | dependency bumps (dependabot applies this itself) |
  | `skip-changelog` | *omitted* | changes that should not appear in the notes at all |

- Update `CLAUDE.md` if you change project-wide conventions or add a new
  category of files (a new top-level subdirectory under `src/`, a new kind of
  abstract base class, a new testing pattern).
- CI must be green before review. `Lint` runs `pre-commit run --all-files`
  (ruff check, ruff format, the YAML/TOML checks, the notebook normalizer) plus
  `mypy` in strict mode over `src/`. `CI` runs the unit suite on Linux, macOS
  and Windows against Python 3.10–3.14 with core dependencies only, the unit
  suite once more with every extra installed (the only job where the
  `quantile-forest` and `utilsforecast` tests actually execute rather than
  skip), the integration suite on Linux, the unit suite again at the declared
  dependency floor, and an sdist build that is installed and tested from the
  packed archive. `make lint`, `make test`, `make test_extras` and
  `make test_integration` reproduce all of it locally.

## Releasing

Releases are cut by publishing a GitHub release; `.github/workflows/release.yml`
does the rest. There is no manual upload step and no PyPI API token — both
destinations use trusted publishing (OIDC), so the token is minted per job and
never stored.

**A GitHub pre-release publishes to TestPyPI; a full release publishes to PyPI.**
That is the only difference between a rehearsal and the real thing, and both run
through the same workflow so the rehearsal actually proves something.

### One-time setup

Each destination needs its own trusted publisher, matched on owner, repository,
workflow filename (`release.yml`) and environment name. For a project that has
never been published, this must be a *pending* publisher — a regular one cannot
be created for a project that does not exist yet.

| Destination | Environment | Configure at |
| --- | --- | --- |
| TestPyPI | `testpypi` | test.pypi.org → Publishing |
| PyPI | `release` | pypi.org → Publishing |

Both environments must also exist under repository Settings → Environments, with
the names spelled exactly as above.

### Cutting a release

1. Bump `__version__` in `src/conformalforecast/__init__.py`. For a real release,
   bump `version:` in `CITATION.cff` to match and add a `CHANGELOG.md` entry.
2. Merge to `main` and let CI, Lint and Build Docs go green on the merge commit.
   The release gate requires a successful run of all three on exactly that commit.
3. Create the release with the tag `v<version>` — the workflow rejects a tag that
   does not match the package version, and rejects a tag that does not point at a
   commit on `main`.
4. Check the pre-release box for a rehearsal; leave it clear for a real release.

Rehearse before the first release of a new minor version, and any time the
release workflow itself changes:

```bash
# after the v0.1.0.dev1 pre-release finishes
python -m venv /tmp/rehearsal && /tmp/rehearsal/bin/pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  conformalforecast==0.1.0.dev1
/tmp/rehearsal/bin/python -c "import conformalforecast; print(conformalforecast.__version__)"
```

The `--extra-index-url` matters: TestPyPI does not mirror the real dependencies,
so narwhals/numpy/pydantic/scipy have to resolve from PyPI.

Versions are immutable on both indexes and the publish step does not set
`skip-existing`, so a half-failed upload cannot be retried under the same
version. Burn a fresh `.devN` instead of trying to overwrite.

If the gate reports a workflow as not green, read the message before re-tagging:
it distinguishes a real failure from a timeout, and waits up to 45 minutes for
runs that are still in progress.

## License

Contributions are licensed under the Apache License 2.0 (see `LICENSE`).
By submitting a PR, you agree your contribution can be released under that
license.
