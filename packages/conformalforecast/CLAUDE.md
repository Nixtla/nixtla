# conformalforecast — Project Guide for Claude Code

## What this package is

`conformalforecast` is a Python library providing **adaptive conformal prediction methods for time series forecasting**. It wraps statsforecast, mlforecast, neuralforecast, or any plain callable, and produces calibrated prediction intervals using methods designed for non-exchangeable, sequential data.

The package's positioning is **TS-native CP for forecasting**: it prioritizes the online and adaptive methods built for non-exchangeable, sequential data (ACI, AgACI, NexCP, SPCI, CQR) over the split-CP-only surface a general-purpose conformal library exposes.

## Architecture: a declarative method and the fitted forecaster it returns

Every method is a **pair of classes** in `src/conformalforecast/methods/`:

- a **method** inheriting `ConformalMethod` — declarative config only
  (`alpha`, `seed`, `score`, per-method knobs). It holds **no fitted state**
  and is never mutated by fitting;
- a **fitted forecaster** inheriting `ConformalForecaster` — the runtime
  object `fit` / `calibrate` / `calibrate_cv` **return**. It carries every
  piece of calibration state plus a deep copy of the config it was fitted
  under (`method_`).

Both base classes live in `src/conformalforecast/methods/base.py`. The fitted
class lives in the same file as its method, so a method file is still a full
implementation. The one exception is `FixedOffsetForecaster`
(`methods/fixed_offset.py`), **shared** by `SplitConformal` and
`ConformalizedQuantileRegression`: once fitted, both are
`score.invert(inputs, offsets)` and nothing else. Share a fitted class only
under that pressure — identical runtime state *and* behaviour.

| Method | Fitted class |
|---|---|
| `SplitConformal`, `ConformalizedQuantileRegression` | `FixedOffsetForecaster` |
| `WeightedSplit` | `WeightedSplitForecaster` |
| `AdaptiveConformalInference` | `ACIForecaster` |
| `AggregatedAdaptiveConformalInference` | `AgACIForecaster` |
| `NonexchangeableConformalPrediction` | `NexCPForecaster` |
| `SequentialPredictiveConformalInference` | `SPCIForecaster` |

**On the method** (`ConformalMethod`):

- the array-surface entry `fit(predictions, truths, *, X=None) -> F` —
  abstract, pure numpy-in, returns a new fitted forecaster;
- the forecaster-surface entries
  `calibrate(forecaster, histories=None, truths=None, *, X=None) -> F` and
  `calibrate_cv(forecaster, n_windows, *, step_size=1, refit=False,
  aligned=False, X=None) -> F` — concrete, **never overridden**: capability
  checks, CV dispatch and exogenous threading are written once.

`ConformalMethod` is `Generic[F]` over its fitted class, so
`SplitConformal(...).calibrate(...)` is typed `FixedOffsetForecaster`.
Subclasses implement `_default_score` and `fit`.

**On the fitted forecaster** (`ConformalForecaster`):

- the array surface `interval(predictions, *, X=None)` (abstract) and
  `update(prediction, truth, *, X=None)` (concrete; delegates to `_update`);
- the forecaster surface `predict(history, *, X=None)` and
  `recalibrate(target_histories, target_truths, *, X=None)` — concrete,
  **never overridden**;
- `calibration_intervals(predictions, *, X=None)` — concrete default,
  overridden only when the interval is conditional on query-time state (SPCI).

Subclasses implement `interval`, and `_update` when online.

`SupportsTransfer` in `src/conformalforecast/methods/transfer.py` is the one
intermediate ABC, on the **fitted** side: `ConformalForecaster` →
`SupportsTransfer` → `WeightedSplitForecaster`. It is an ABC rather than a
standalone mixin because `adapt` reads `scores_` and `n_calibration_samples_`
— a mixin would have to re-declare as pydantic fields what the base already
owns, and two classes declaring the same field is how a state bug gets in. Add
another intermediate ABC only under the same pressure.

Supporting contracts live in `src/conformalforecast/base.py`: the array type
aliases (`Series`, `Forecast`, `Interval`, `Exog`), the `as_forecast` coercion
helper, `ForecasterAdapter` (uniform interface over forecasting libraries),
`ScoreFunction` (nonconformity scoring), the two result models
(`CalibrationResult`, `PredictionResult`) and the exception hierarchy. Adapters convert library-specific objects (Nixtla
long-format pandas or polars DataFrames) to numpy arrays at the boundary, so methods never see
framework types.

### Typing the pair

- `methods/base.py` defines `ConformalForecaster` **before** `ConformalMethod`,
  binds `F = TypeVar("F", bound=ConformalForecaster)` between them, and calls
  `ConformalForecaster.model_rebuild()` at the end of the module. Keep that
  order: a `TypeVar` whose bound is a forward reference to a later class
  crashes mypy (internal error, "Must not defer during final iteration").
- A method parametrizes the base with its fitted class as a string,
  `class AdaptiveConformalInference(ConformalMethod["ACIForecaster"])`, and the
  fitted class follows the method in the same file with a narrowed
  `method_: AdaptiveConformalInference`. A fitted class whose method is not yet
  defined in its module (e.g. one written in a test) needs an explicit
  `model_rebuild()`.
- The base field is `method_: SkipValidation[ConformalMethod[Any]]`: pydantic
  would otherwise validate against the *parametrized* class, which no concrete
  method subclasses. The narrowed subclass fields are validated normally.
- `Self` comes from `typing_extensions` under `TYPE_CHECKING` only (3.10
  floor; `typing_extensions` is not a declared dependency).

### The array-surface invariant

> `fit`, `interval` and `update` may only accept numpy-coercible arrays and
> plain-Python config — never adapters, DataFrames, or library types.

This is what keeps a method embeddable inside statsforecast, mlforecast and
neuralforecast: they call `method.fit(...)` with their own CV arrays and keep
the **returned** object for `interval` / `update`. It applies to `panel.py`
and `_quantiles.py` as well. The forecaster enters only through `calibrate` /
`calibrate_cv`.

`fit` has **one signature for every method**. CQR consumes quantile forecasts:
its `predictions` argument is the `(n_series, n_windows, horizon, 2)` quantile
cube and its fitted `interval` takes `(n_series, n_samples, horizon, 2)`. The
method declares this with `CONSUMES_QUANTILES = True`; there is no wider
signature, no Protocol and no `cast`.

### The score is config; the forecaster is bound at calibration

`score` is an ordinary constructor field on `ConformalMethod`
(`score: ScoreField = None`, `None` meaning `_default_score()`). It is never a
`fit` / `calibrate*` argument. `fit` resolves it through `_bind_score()`,
which **deep-copies** it — `ScoreFunction.fit` may set internal parameters,
and the copy keeps that off the user's config — and stores the copy as the
fitted object's `score_fn_`; `interval` and the online update invert with it.

Score compatibility is a **construction-time** `ValidationError`: each method
declares `SCORE_TYPE: ClassVar[type[ScoreFunction]]` (`QuantileScore` for CQR,
`SignedResidual` for SPCI, `AbsoluteResidual` for WeightedSplit,
`ScoreFunction` — the default inherited from `ConformalMethod` — otherwise) and a base `model_validator` checks
`isinstance(score, SCORE_TYPE)` **and** that the score's own
`CONSUMES_QUANTILES` `ClassVar` matches the method's — so a point method
rejects `QuantileScore` even though it passes the `ScoreFunction` gate. Both
checks are `ConformalMethod._accepts_score`. Split's cross-field rule (`symmetric=False`
needs a `SignedResidual`) is its own `model_validator`. All gates **narrow**: a
custom score must subclass the required type.

Scores serialize through a `kind` tag. Every concrete `ScoreFunction`
declares its own `kind: Literal[...]`; `__pydantic_init_subclass__` registers
it and raises `TypeError` at class definition if the subclass declares no
`kind`, or if its `kind` is already registered to a different class — typically
a parent's, inherited (otherwise it would silently reload as the parent). `ScoreField`
(`nonconformity/_field.py`, re-exported from the package) is the annotated type every method's `score`
field uses: `SerializeAsAny`, a before-validator resolving a dict by `kind`
through the registry, and a JSON schema listing the built-in kinds.
`ConformalMethod.__get_pydantic_json_schema__` narrows that list per method to
the kinds `_accepts_score` admits, so the schema never advertises a score
construction rejects.

The adapter is the first positional parameter of both calibrate entry points.
They call `fit`, then set `forecaster_` and the retained calibration arrays on
the **returned** object, and return it. The method is never assigned to, so
calibrating twice yields two independent fitted objects and a failed calibrate
cannot disturb an earlier one — by construction, not by ordering. An object
returned by a bare `fit` has no `forecaster_`, so `predict` raises
`CalibrationError` — use `interval` there.

There is **no `is_fitted_` flag** anywhere: a fitted object exists only once
fitted, and the method never is.

### `update` mutates in place

`ConformalForecaster.update` mutates the fitted object and returns `None`. Do
not make it return a new object: a caller who forgets `f = f.update(...)` gets
no error, just intervals that silently stop adapting; and online state holds
objects mutated in place (EWA aggregators, SPCI's per-cell regressor dict), so
cheap structural sharing would alias. Snapshots use
`fitted.model_copy(deep=True)` — never add a `copy()` method, pydantic's
`BaseModel` already defines a deprecated one.

`update` is concrete on the base and **never overridden**: it delegates to
`self._update(...)`, the overridable algorithm hook with a no-op default.
Online fitted classes override `_update`, never `update` — overriding `update`
and calling `super().update()` from it is infinite recursion.

### Fitted objects are values derived from config

- `method_` is a **deep copy** of the method taken at fit. Fitted code reads
  config through it (`self.method_.gamma`, `self.method_.window_size`), and
  later edits to the user's method never leak in.
- Collaborator prototypes on the method (AgACI's `aggregator`, SPCI's
  `regressor`) are deep-copied into fitted state, as before. Nothing a fitted
  object mutates may be shared with the method or another fitted object.
- `SupportsTransfer.adapt(...)` **returns a new fitted object** and leaves the
  receiver untouched. It deep-copies the weighting function before
  `fit_source` / `fit_target`, both of which mutate it.
- `recalibrate(...)` returns `self.method_.calibrate(self.forecaster_, ...)` —
  a new object; the receiver is untouched and the score carries over because
  it is config.

## Repo layout

- `src/conformalforecast/methods/base.py` — the `ConformalMethod` and `ConformalForecaster` contracts (READ FIRST)
- `src/conformalforecast/methods/` — one file per method, each holding the method and its fitted class: `split.py`, `cqr.py`, `aci.py`, `agaci.py`, `nexcp.py`, `spci.py`, `weighted_split.py`
- `src/conformalforecast/methods/fixed_offset.py` — `FixedOffsetForecaster`, the fitted class shared by Split and CQR
- `src/conformalforecast/methods/transfer.py` — `SupportsTransfer` (fitted side; the `adapt` entry point) plus the per-series scale estimators `compute_series_scales` and `scales_from_truths`
- `src/conformalforecast/base.py` — array type aliases, `as_forecast`, `ForecasterAdapter`, `ScoreFunction`, result types, exceptions
- `src/conformalforecast/_model.py` — `ArrayModel`, the pydantic base every model in the package subclasses
- `src/conformalforecast/_quantiles.py` — NaN-aware exact order statistics and the batched quantile helpers (package root, not under `methods/`)
- `src/conformalforecast/panel.py` — public, DataFrame-free long ↔ panel conversion, CV assembly, level↔quantile helpers
- `src/conformalforecast/adapters/` — `callable`, `statsforecast`, `mlforecast`, `neuralforecast`, and the shared private `_nixtla_common`
- `src/conformalforecast/nonconformity/` — score functions (`absolute`, `signed`, `quantile`) and `ScoreField`, the `kind`-tagged annotated type every method's `score` field uses
- `src/conformalforecast/capabilities.py` — adapter capability mixins
- `src/conformalforecast/aggregators/` — online aggregation (`EWA`), used by AgACI
- `src/conformalforecast/quantile_regressors/` — quantile regressors (`QRFQuantileRegressor`), used by SPCI
- `src/conformalforecast/weighting/` — covariate-shift weight functions (`UniformWeights`, `PrecomputedWeights`, `DensityRatio`) and the structural `DensityRatioEstimator` protocol, used by `WeightedSplit`
- `src/conformalforecast/diagnostics/` — coverage (marginal, by-horizon, by-series,
  rolling), interval-quality scoring, conditional coverage, method-specific
  diagnostics and report builders
- `src/conformalforecast/compat/mlforecast_shim.py` — reference migration shim for mlforecast's interval registry
- No datasets are bundled with the package; docs notebooks generate panels with
  `default_rng` or read `nbs/data/`
- `tests/unit/methods/` — the center of gravity: `test_<name>.py` for a method's
  array surface (adapter-free) and `test_<name>_forecaster.py` for its adapter
  surface, plus `test_method_base.py`, `test_quantiles.py`,
  `test_online_loop_perf.py`, `test_pydantic_conversion.py` and
  `test_transfer.py`
- `tests/unit/` — panel, adapters (including lazy-import hygiene), nonconformity,
  aggregators, quantile regressors, diagnostics, `ArrayModel`, the mlforecast
  shim, docs rendering, import hygiene
- `tests/integration/` — slow tests against real Nixtla libraries (needs extras)
- `tests/integration/torch_isolated/` — the torch-backed integration tests, in
  their own directory because they must be collected in a separate pytest
  session (torch and scikit-learn each vendor a `libomp`; one process
  loading both segfaults). Run both via `make test_integration`
- `tests/helpers.py` — shared test adapters (e.g. the quantile-capable callable adapter)
- `nbs/docs/` — the published narrative docs (`.ipynb` executed, `.qmd` prose),
  in five sections: `getting-started/`, `concepts/`, `capabilities/`,
  `adapters/` (published as "Nixtlaverse"), and `methods/` (one page per method)
- `nbs/data/` — small committed datasets so docs never hit the network:
  `air_passengers.csv` for single-series pages, `m4_hourly_panel.csv` (8 ragged
  series) for panel pages. Provenance is documented in `nbs/data/README.md`
- `docs/*.html.md` — mkdocstrings stubs; `docs/to_mdx.py` renders them to MDX.
  All method classes are documented in the single `docs/methods.html.md`
- `docs/mintlify/docs.json` — hand-maintained site navigation (the only committed
  file in `docs/mintlify/`; everything else there is generated and gitignored)
- `AGENTS.md` — the short rule list for AI coding agents of any vendor; the
  cross-tool counterpart to this file
- `ROADMAP.md` — known defects, missing capabilities and deliberate scope
  boundaries, each reproduced against this code. **Read it before concluding
  that something is broken or before proposing a feature** — the answer is often
  already there, including which limits are deliberate

## Type and shape conventions

- **All array shapes documented in docstrings.** Always.
- `Series`: 2D `np.ndarray[float]`, shape `(n_series, T)`. A panel of time series.
  Single-series users pass `n_series=1`.
- `Forecast`: 3D `np.ndarray[float]`, shape `(n_series, n_windows | n_samples, horizon)`.
  Always 3D, even when `n_series=1` or `horizon=1`.
- `Interval`: 4D `np.ndarray[float]`, shape `(n_series, n_samples, horizon, 2)`.
  Last axis is `(lower, upper)`.
- `Exog`: 4D `np.ndarray[float]`, shape
  `(n_series, n_windows | n_samples, horizon, n_features)` — features aligned to
  each predicted step.
- The leading axis is **always** `n_series`. Iterate, slice, and reduce over it
  the way pandas users iterate over groupby keys.
- **Ragged panels use left-NaN padding.** In a `Series`, NaN is allowed only as
  a contiguous *leading* prefix per series; interior or trailing NaN is an error.
  In a calibration `Forecast`, a NaN cell means "this window does not exist for
  this series". Methods propagate NaN — never impute.
- The **array surface** preserves `float32` — `as_forecast` does not upcast, and
  every method has a `test_float32_preserved` case. The **forecaster surface does
  not**: `calibrate`/`calibrate_cv` cast truths to `float64` and
  `ForecasterAdapter._validate_history` casts history, so `predict` returns
  `float64` even from an all-`float32` neuralforecast model. See ROADMAP.md.
- No pandas, polars or narwhals in `methods/`, `nonconformity/`, `diagnostics/`,
  `panel.py` or `_quantiles.py` (pinned by `tests/unit/test_imports.py`). Frames
  live in `adapters/`: each adapter converts native frames with narwhals at its
  edges, and `_nixtla_common.py` works on narwhals frames and numpy only.

## Capabilities are mixins

- Optional adapter features are mixins in `src/conformalforecast/capabilities.py`:
  `SupportsRefit`, `SupportsQuantiles`, `SupportsBootstrap`,
  `SupportsCrossValidation`, `SupportsCrossValidationQuantiles`.
- Adapters inherit only the mixins they implement.
- Methods declare requirements via the class attribute
  `REQUIRED_CAPABILITIES: ClassVar[tuple[type, ...]]`. Only
  `ConformalizedQuantileRegression` declares one — `(SupportsQuantiles,)`; the
  other six are `()`.
- `SupportsBootstrap` is currently implemented by no adapter.
- `ConformalMethod._check_capabilities(forecaster)` — the **first statement** of
  both `calibrate` and `calibrate_cv`, i.e. where the forecaster is bound —
  checks each requirement through `ForecasterAdapter.supports(capability) -> bool`,
  whose default is an `isinstance` check. There is nothing to check at
  construction: the constructor takes no forecaster. Overriding `supports()` is
  **sanctioned** when a capability depends on runtime configuration rather than
  the class, and all three Nixtla adapters do it: `NeuralForecastAdapter`
  inspects both its loss and `exog_features`, `StatsForecastAdapter` and
  `MLForecastAdapter` inspect `exog_features` (all three also gate
  `SupportsRefit` that way). So a point-loss or exogenous forecaster fails CQR
  calibration rather than `predict`. **Never check capabilities inside
  `predict()`** — it lives on the fitted object, which is only ever built
  after the gate passed.
- `calibrate_cv` additionally gates the CV capability the chosen path needs
  (`SupportsCrossValidation`, or `SupportsCrossValidationQuantiles` for
  quantile-consuming methods) — inside the branch it selects, so slightly later
  than `_check_capabilities`, but still before any `fit`.

## Invariants every method must satisfy

1. `alpha` lives in `(0, 1)`, enforced by `Field(gt=0.0, lt=1.0)` on
   `ConformalMethod`. Methods never write an `__init__` or a hand-rolled range
   check.
2. `interval()` / `update()` / `predict()` exist only on the fitted object, so
   "used before fit" is unrepresentable — there is no guard and no
   `is_fitted_`. `predict()` and `recalibrate()` still raise `CalibrationError`
   when the object came from a bare `fit` and has no `forecaster_`.
3. The three declarations live on the method class: whether `fit`'s
   `predictions` are a quantile cube via `CONSUMES_QUANTILES` (`True` for CQR
   only); adapter needs via `REQUIRED_CAPABILITIES`, checked where the
   forecaster is bound (both `calibrate` entry points); and score compatibility
   via `SCORE_TYPE`, checked at **construction** by a base `model_validator`
   (SPCI requires `SignedResidual`, CQR requires `QuantileScore`,
   `WeightedSplit` requires `AbsoluteResidual` because its weighted
   per-test-point path builds the interval as `prediction ± q` — each a
   `pydantic.ValidationError` from the constructor). All score gates
   **narrow**: a custom score must subclass `QuantileScore` /
   `SignedResidual` / `AbsoluteResidual`.
   **Never check any of them inside `predict()`.**
4. `predict()` is deterministic given fitted state and input. `seed` defaults to
   `0`, never `None`.
5. Online fitted classes (method `IS_ONLINE = True`) override `_update` with
   non-trivial logic that **mutates the fitted object in place**. They never
   override `update` — the base owns it. Offline fitted classes inherit the
   no-op `_update` default.
6. Every method validates per-series calibration size in its own `fit` and
   raises `InsufficientCalibrationSamples` naming the offending series indices.
   The threshold is not uniform, because the quantile each method takes is not:
   Split and CQR get the check free from `conformal_quantile`; NexCP from
   `weighted_conformal_quantile`, which tests the weighted *effective* sample
   size `(Σw)² / Σw²` rather than a raw count; ACI and AgACI inline the same
   `ceil(1/alpha)` count check because `masked_quantile` does not validate; SPCI
   requires `max(window_size + 1, ceil(1/alpha))` residuals; `WeightedSplit`
   gets the weighted effective sample size from
   `weighted_quantile_from_weights`, which also rejects a cell whose weights sum
   to zero; and asymmetric `SplitConformal` takes one one-sided quantile per
   tail, so its floor is the stricter
   `max(ceil(1/alpha_lo), ceil(1/(1 - alpha_hi)))` rather than `ceil(1/alpha)`.
   Never add a second check on a path that already has one.
7. Every quantile comes from `_quantiles.py`, never from `np.quantile`. Split
   and CQR take the exact `k = min(ceil((1 - alpha)(n + 1)), n)` order
   statistic via `conformal_quantile`. The adaptive methods need a quantile at a
   *time-varying, per-cell* level, where that order statistic is not defined, so
   ACI and AgACI use `masked_quantile` (the version-stable linear quantile of
   item 8) and NexCP `weighted_conformal_quantile` (recency-weighted selection).
   `WeightedSplit` takes the exact weighted order statistic from
   `weighted_quantile_from_weights`, or from `weighted_quantile_per_test_point`
   when the normalization has to include each test point's own weight.
8. Per-cell Python loops over the `(series × horizon)` grid are a performance
   bug in `methods/`. Use the batched helpers in `_quantiles.py`:
   `_linear_quantile_from_sorted` (one gather, version-stable linear quantile,
   accepts leading batch axes such as AgACI's expert axis) and
   `ExpandingSortedScores` (incrementally-sorted history for an online window
   loop). `SequentialPredictiveConformalInference` is the sanctioned exception —
   it fits one regressor per cell. The two helpers have contract tests in
   `tests/unit/methods/test_quantiles.py` (`_linear_quantile_from_sorted`
   against `np.quantile` in both float dtypes; `ExpandingSortedScores` against
   `masked_quantile` at every prefix) — keep those green when touching the
   quantile math. The contract is that this package's quantile is **identical
   across numpy versions**, which is stronger than tracking `np.quantile`: numpy
   computed gamma in the array dtype through 2.4 and in `float64` from 2.5, so
   following it would shift `float32` interval bounds on a numpy upgrade.
   `_linear_quantile_from_sorted` therefore always uses the float64-gamma form,
   and matches `np.quantile` exactly for `float64` everywhere and for `float32`
   from numpy 2.5 on. The test asserts equality exactly in those cases and
   within a small absolute tolerance for `float32` below 2.5. Do **not** pin
   whole-method output in a committed fixture: `EWA.weights` uses `np.exp` and
   libm is not bit-reproducible across architectures, so such a snapshot passes
   locally and fails in CI.
9. `calibration_intervals()` is what `diagnostics.evaluate_calibration` replays.
   It is concrete on `ConformalForecaster` and the default broadcasts `interval()`
   over every calibration window, which is exact only when the offsets are a
   constant of the whole calibration set — true for Split, CQR, WeightedSplit,
   ACI, NexCP and AgACI. A method whose `interval()` is conditional on
   **query-time state** must override it, or Mode-1 diagnostics report an
   artifact: SPCI conditions on the preceding residual window, and broadcasting
   its final answer scored 23% against a 90% target. Windows the method cannot
   evaluate (an online warm-up) get NaN bounds, which the diagnostics layer
   excludes from its averages rather than scoring as misses.

## Calibration entry points

`calibrate` and `calibrate_cv` are concrete on `ConformalMethod`;
`recalibrate` is concrete on `ConformalForecaster`. None is ever overridden,
and all three **return a new fitted object**:

- `calibrate(forecaster, histories=None, truths=None, *, X=None)` —
  explicit arrays, any adapter, loops via `predict_batch` (or
  `predict_quantiles` when the method `CONSUMES_QUANTILES`). Both arrays
  default to `None`; omitting either raises `ValueError`.
- `calibrate_cv(forecaster, n_windows, *, step_size=1, refit=False,
  aligned=False, X=None)` — delegates to `forecaster.cross_validate`,
  automatically switching to `cross_validate_quantiles` when the method
  `CONSUMES_QUANTILES`. `refit` is `bool | int`.
- `fitted.recalibrate(target_histories, target_truths, *, X=None)` —
  `self.method_.calibrate(self.forecaster_, ...)` pointed at a target-domain
  panel. It takes **no forecaster and no score**: the forecaster is the one
  already bound (never refitted) and the score is config on `method_`. It
  raises `CalibrationError` when there is no `forecaster_`, and leaves the
  receiver untouched.

Fitted classes implementing `SupportsTransfer` additionally offer
`adapt(*, weighting=None, target_X=None, target_series=None)`, which needs no
target labels at all — it recomputes the quantile from the stored source scores
under new weights or a new scale ratio and **returns a new fitted object**, so
it is idempotent by construction. `recalibrate` and `adapt` do not compose: the
re-fit starts from uniform weights, because target labels supersede the
unlabelled proxy.

`X` is not yet threaded through cross-validation: `calibrate_cv` accepts it only
to raise `NotImplementedError`, and `calibrate` raises `NotImplementedError`
when `X` is passed to a quantile-consuming method.

The two calibrate entry points call `_check_capabilities(forecaster)` as their
first statement, hand arrays to `fit`, then set `forecaster_`, the retained
calibration arrays and `calibration_.diagnostics["path"]` on the returned
object — see *The score is config; the forecaster is bound at calibration*.

`aligned=False` (default) gives dense per-series windows — the window axis means
"k steps before this series' own end". `aligned=True` gives calendar-aligned
cutoffs with NaN cells for late-starting series. Recency-weighted methods
(NexCP, SPCI) interpret the axis accordingly. The quantile-CV path (CQR) ignores
`aligned` — it always densifies.

Capability failures raise `UnsupportedCapability` uniformly from the base class.

## Style and tooling

- Python 3.10+. Use modern syntax: `X | None`, `list[int]`, PEP 695 where natural.
- `from __future__ import annotations` at the top of every module.
- Type-annotate all public functions. Use `numpy.typing.NDArray`.
- Docstrings: NumPy style (Parameters / Returns / Raises sections).
- Format and lint with `ruff` — rules in `pyproject.toml` (`E,W,F,I,B,C4,UP,ARG,SIM`),
  version pinned by the pre-commit `rev`; keep the `dev` group floor in step with it.
- Type-check with `uv run mypy`. `[tool.mypy]` sets `strict = true`,
  `files = ["src"]` and the `pydantic.mypy` plugin — no CLI flags. Tests can be looser.
- 100-char line limit.

## Dependency policy

- Core deps only: `narwhals>=2.9`, `numpy>=1.24`, `pydantic>=2.7`, `scipy>=1.11`.
  pandas and polars are both optional: the Nixtla adapters accept either through
  narwhals, and `[nixtla]` adds `utilsforecast`, which does the `freq` arithmetic
  in each backend's own dialect.
  Everything else is an extra — `nixtla`, `spci`, `plotting`, plus `all` which
  pulls in the three. The dependency groups are `dev`, `docs` and
  `integration-docs`; the last is what the docs notebooks execute against, and
  its floors must match the extras'.
- The pydantic floor is `pydantic>=2.7` (pinned in `pyproject.toml`, not
  whatever happens to be installed). Don't document or depend on a pydantic
  feature newer than that floor.
- The three Nixtla adapters use **soft imports**: try-import the library, raise
  an `ImportError` with the exact `pip install conformalforecast[<extra>]`
  command if missing. `compat/` shims and `quantile_regressors/qrf.py` (for
  `[spci]`) do the same. `CallableAdapter` needs none.
- The Nixtla adapters are re-exported lazily from `adapters/__init__.py`, so
  `import conformalforecast` must never import statsforecast, mlforecast,
  neuralforecast or torch. Pinned by
  `tests/unit/test_adapters_lazy.py::test_importing_package_does_not_import_optional_libraries`.
- Never add a new core dependency without explicit user approval.

## Testing conventions

- Every method has two unit-test files in `tests/unit/methods/`.
- `test_<name>.py` — the **array surface**, adapter-free, covering:
  - **Coverage**: synthetic data with known noise; empirical coverage matches
    `1 - alpha` within a tolerance documented in the test.
  - **Shape**: `n_series ∈ {1, 5}` × `horizon ∈ {1, 12}`, plus a
    `float32`-preservation case. `X` threading is covered per-method where it
    matters (`test_spci.py::TestExogenous`), not in the shape matrix.
  - **Determinism**: same config + same arrays → identical output.
  - **NaN masking**: ragged calibration arrays give the same result as fitting
    each series separately on its dense sub-array;
    `InsufficientCalibrationSamples` raised with the right series indices.
- `test_<name>_forecaster.py` — the **adapter surface**, covering plumbing:
  dispatch (`calibrate` vs `calibrate_cv`), uniform `UnsupportedCapability`,
  forecaster binding (`forecaster_` set on the returned object, `predict`
  raising `CalibrationError` on the object a bare `fit` returned), exogenous
  threading, the `supports()` hook, and one coverage smoke test per method.
- Every test fits through the returned object: `f = m.fit(...)` then
  `f.interval(...)`. Online methods additionally pin that
  `f.model_copy(deep=True)` is independent of a later `f.update(...)`.
- Use `numpy.random.default_rng(seed)` everywhere. Never `np.random.seed`.
- The `CallableAdapter` is the test workhorse — it's the simplest adapter and
  lets every method be tested without external dependencies. Quantile-capable
  test adapters live in `tests/helpers.py`.

## Adding a new method (procedure)

Two classes, one file. Start with the array surface — it is what the math
lives in; the forecaster surface is inherited and needs no code.

1. Read `src/conformalforecast/methods/base.py` (both contracts) and
   `src/conformalforecast/methods/split.py` plus `fixed_offset.py` (the
   simplest reference implementation). For an online method also read
   `src/conformalforecast/methods/aci.py`.
2. Create `src/conformalforecast/methods/<name>.py` with a method class
   inheriting `ConformalMethod[<Name>Forecaster]` and its fitted class
   inheriting `ConformalForecaster`. Reuse an existing fitted class only if
   the runtime state and behaviour are identical (as Split and CQR share
   `FixedOffsetForecaster`).

   On the **method**, declare config as `Field(...)` with its constraints and
   description — do **not** write an `__init__`; no class has one. A forecaster
   is never a constructor parameter; the score is the inherited `score` field.

   `Field` only expresses single-field constraints. Anything else is a
   validator:
   - a cross-field rule → `@model_validator(mode="after")`, as in
     `agaci.py`'s `_validate_clips` (both-or-neither clip bounds, lower <
     upper) or `split.py`'s symmetric-score rule;
   - config *derived* from other config → a read-only `@property` when it is
     cheap (CQR's quantile levels), or a `model_validator` writing it back as
     in `spci.py`'s `_default_regressor_from_seed` (a plain `default_factory`
     cannot read the sibling `seed` field, and the data-aware form arrived well
     above our `>=2.7` floor);
   - a warning rather than an error → `@field_validator`, as in `aci.py`'s
     `_warn_on_large_gamma`.

   A derived value written back by a validator must still survive
   `model_validate(model_dump())`.

   On the **fitted class**, declare every piece of state as a **required**
   field with a trailing underscore and a narrowed `method_` type. Read config
   through `self.method_`, never by copying it into a second field.
3. Set the method's class attributes: `REQUIRED_CAPABILITIES` (adapter mixins
   the forecaster surface needs, `()` if none), `CONSUMES_QUANTILES` (usually
   `False`), `SCORE_TYPE` (usually `ScoreFunction`) and `IS_ONLINE`.
4. Implement `_default_score` and `fit` on the method, and `interval` — plus
   `_update` if `IS_ONLINE` — on the fitted class. `fit` calls
   `_bind_score()`, builds the fitted object and returns it; it never assigns
   to `self`. **Override `_update`, never `update`**: the base's `update`
   calls your `_update`, so overriding `update` recurses. `_update` mutates the
   fitted object in place. Coerce inputs with `as_forecast` from
   `conformalforecast.base` and take every quantile through
   `conformalforecast._quantiles` (never `np.quantile`). Deep-copy any
   collaborator prototype the fitted object will mutate.
   Override the concrete `calibration_intervals` too if the interval depends on
   query-time state rather than on a calibration-set constant — see invariant 9.
5. Do **not** override `calibrate`, `calibrate_cv`, `predict`, `recalibrate`
   or `update` — all are concrete on the bases and the forecaster surface comes
   for free.
6. Export both classes from `src/conformalforecast/methods/__init__.py`
   (import plus `__all__`).
7. Two test files in `tests/unit/methods/`:
   `test_<name>.py` for the array surface (coverage, shape, determinism,
   NaN-masking-equals-dense, input errors) and `test_<name>_forecaster.py` for
   the adapter surface (dispatch, capability errors, binding, exogenous
   threading, coverage smoke test) — see *Testing conventions* above.
8. Add both classes to `docs/methods.html.md` (one `:::` block each; there are
   no per-method stub files).
9. Narrative page `nbs/docs/methods/<name>.ipynb`, following the template the
   existing per-method pages share: what it does → the one equation → a worked
   example → an interval plot → measured coverage → when *not* to use it → a
   `## References` section citing the original paper. Reserve `callout-note` /
   `callout-warning` for caveats a reader must not miss, not for the citation.
   Add it to the "Offline" or "Online & adaptive" group in `docs/mintlify/docs.json`, list it
   in `nbs/docs/methods/index.qmd`, then `make execute_docs` to commit its
   outputs. Keep the model fast — the page is about conformalizing, not about
   accuracy. Every claim in the prose must match the numbers the notebook
   actually printed; when a method underperforms, say so and explain why.

## State conventions

Following sklearn's convention, now expressed as pydantic fields:

- **Configuration attributes** — declared as `Field(...)` with a description
  and validators, no trailing underscore. Constraints (`gt=`, `ge=`, `le=`,
  `min_length=`, ...) live on the `Field`, not in hand-written `if` statements,
  e.g. `alpha: float = Field(gt=0.0, lt=1.0, description=...)`. Examples:
  `alpha`, `seed`, `gamma`.
- **Fitted state attributes** — live **only on the fitted class**, with a
  trailing underscore, declared as **required** fields (no `| None = None`,
  no `SkipJsonSchema`): the object exists only once fitted, so there is no
  "not yet set" state to model. The exceptions are `forecaster_` and the
  retained `*_calibration_` arrays, which default to `None` because a bare
  `fit` never sets them. Examples: `score_quantile_`, `n_calibration_samples_`,
  `alpha_t_`, `forecaster_`, `score_fn_`, `method_`, `calibration_`. A method
  class has **no** fitted state at all — no trailing-underscore fields.
- **Class metadata** (`REQUIRED_CAPABILITIES`, `CONSUMES_QUANTILES`,
  `SCORE_TYPE`, `IS_ONLINE`) is `ClassVar` on the method, which keeps it out of
  the schema and out of the constructor while staying readable on the class
  with no instance in hand (`compat/mlforecast_shim.py` reads
  `CONSUMES_QUANTILES` this way). Fitted code reads it through
  `type(self.method_)`.
- Every pydantic-carrying class — methods, adapters, capability mixins, score
  functions, aggregators, quantile regressors, the diagnostics `Report` and the
  two result types — subclasses `ArrayModel` (`src/conformalforecast/_model.py`).
  `PanelIndex` (a frozen dataclass), `ExpandingSortedScores` and the exceptions
  are deliberately plain classes. Pydantic's generated
  equality is a field-dict comparison that calls `bool()` on an elementwise
  array comparison and raises `ValueError` instead of returning `False`, so
  `ArrayModel` overrides `__eq__` to compare array and DataFrame fields with
  `np.array_equal(..., equal_nan=True)` / `DataFrame.equals`, recursing into
  dicts, lists and tuples; instances are explicitly unhashable.
- `model_config = ConfigDict(extra="forbid")` on `ConformalMethod` is
  explicit (and likewise on `ConformalForecaster`): pydantic v2 defaults
  `extra` to `"ignore"`, so without it
  `SplitConformal(alpha=0.1, IS_ONLINE=True)` would construct silently instead
  of raising. `ArrayModel` itself sets `arbitrary_types_allowed=True` (needed
  because `forecaster_`/`score_fn_` hold `ForecasterAdapter`/`ScoreFunction`
  instances, and adapters in turn hold raw library objects — `StatsForecast`,
  `MLForecast`, `NeuralForecast` — and numpy arrays that pydantic can only
  `isinstance`-check, not structurally validate) and
  `validate_assignment=False`, which is what keeps an online `update()` loop
  from re-validating an `(n_series, horizon)` array on every step, and what
  makes `refit` reassigning `train_df` free.
- Every converted class must survive `model_validate(model_dump())`; the
  parametrized test in `tests/unit/methods/test_pydantic_conversion.py`, whose
  method list is the `ConformalMethod` subclasses in
  `conformalforecast.methods.__all__`, pins this — including a non-default
  `score`, which round-trips through its `kind` tag. Because a method holds no
  fitted state, it round-trips whether or not it has been used to fit.
  The fitted classes, the four adapters (`callable`, `statsforecast`, `mlforecast`,
  `neuralforecast`) and the two result types (`CalibrationResult`,
  `PredictionResult`) are the deliberate exceptions — see
  `src/conformalforecast/_model.py`'s module docstring for why. The result types
  additionally cannot generate a JSON schema at all (an all-array payload), so
  their contract is pinned in `tests/unit/test_base.py` instead.
- **Two adapter fields deliberately break the no-trailing-underscore
  convention.** `train_df` is public, non-underscored config that
  **mutates** — `refit` reassigns it. It holds the user's native pandas or polars
  frame, typed `_nixtla_common.TrainFrame`. A private `_last_train_df` twin was
  rejected: it gives one value two names, and `ArrayModel.__eq__` would still
  compare the stale public copy, so two adapters refit to different data
  would compare equal. Pydantic has no init-only field, and `train_df_` would
  give a fitted-state name to something users must pass at construction — it
  isn't fitted state, it's mutable config. `exog_features` is the same
  exception: it isn't assigned by `fit()`/`calibrate()`/`update()` either, so
  it carries no trailing underscore even though `NeuralForecastAdapter`
  derives it via a `model_validator` from the model's `futr_exog_list`
  (`StatsForecastAdapter`/`MLForecastAdapter` instead take it as ordinary
  user config). `SkipJsonSchema` only hides a field from the *schema*, not
  from `ArrayModel.__eq__` — that gap is exactly what produced the
  equal-but-different `train_df` bug above. `sf`/`mlf`/`nf` need no analogous
  exception: the adapters mutate them in place (`self.sf.fit(...)`), never
  reassign them.
- `CalibrationResult` is carried on the fitted object as `calibration_` (the
  calibrate entry points return the fitted object, not the result). It has
  exactly two fields: `n_calibration_samples` — shape `(n_series,)`
  per-series effective counts — and `diagnostics`, a dict of method-specific
  values: `"score_quantile"` (Split, CQR, NexCP), `"gamma"`/`"alpha_t"` (ACI),
  `"gammas"`/`"alpha_t_per_expert"` (AgACI), `"rho"` (NexCP),
  `"window_size"`/`"beta_grid_size"`/`"refit_every"` (SPCI),
  `"score_quantile"`/`"pooling"` (WeightedSplit), `"scale_ratio"`/`"weight_summary"`
  (after `adapt`), and the `"path"` marker: `"loop"`, `"cross_validation"`, `"recalibrate"` or `"adapt"`. It is
  a snapshot taken at fit: `predict()`/`interval()`/`update()` read the fitted
  object's own fields, never the result, and arrays in it are defensively
  copied.

## Adding a new adapter (procedure)

1. Read `adapters/callable.py` (simplest reference).
2. Create `adapters/<library>.py` inheriting `ForecasterAdapter` plus the
   capability mixins it supports.
3. Implement every abstract method from each parent class, including the
   keyword-only `X=None` on `predict`/`predict_batch`/`cross_validate` and the
   `aligned` CV mode.
4. Soft-import the underlying library at the top of the file. If missing,
   raise `ImportError` with the exact `pip install conformalforecast[<extra>]` command.
5. Register the class in `adapters/__init__.py`'s `_LAZY_ADAPTERS` map and
   `__all__` so it stays lazily imported.
6. Tests in `tests/integration/test_<library>_adapter.py` — under
   `torch_isolated/` if the library pulls in torch.
7. Add an extras entry in `pyproject.toml` and document it in the README.

## Exceptions

`ConformalForecastError` is the base. `CalibrationError`,
`UnsupportedCapability` and `InsufficientCalibrationSamples` (a
`CalibrationError` carrying `series_indices`) derive from it.

## What NOT to do

- Don't stack PRs. Branch from `main`, target `main`; the `PR targets main` check
  rejects anything else. See `AGENTS.md` and `CONTRIBUTING.md`.
- Don't leave development-process references in the code: no PR or phase
  numbers, no plan or spec item ids, no branch names, no "in a follow-up PR".
- Don't import pandas, polars or narwhals inside `methods/`, `nonconformity/`,
  `diagnostics/`, `panel.py`, or `_quantiles.py`.
- Don't give `fit`, `interval` or `update` an adapter, DataFrame, or library type.
- Don't override `update` on a fitted class — override `_update`.
- Don't override `calibrate`, `calibrate_cv`, `predict` or `recalibrate`.
- Don't assign to `self` inside a method's `fit` or the calibrate entry
  points — `fit` returns a new fitted object; the method stays config.
- Don't make `update` return a new object, and don't add a `copy()` method —
  snapshots are `model_copy(deep=True)`.
- Don't share a mutable collaborator (score, aggregator, regressor, weighting)
  between a method and its fitted object, or between two fitted objects.
- Don't put `score` back on `fit` / `calibrate*`, and don't check score
  compatibility anywhere but the method's validators.
- Don't make `predict()` raise capability errors. Capability checks happen where
  the forecaster is bound (both `calibrate` entry points).
- Don't add framework-specific types to the public API.
- Don't write a method that only works for `horizon=1`. Multi-horizon is mandatory.
- Don't add a top-level dependency without approval.
- Don't import a Nixtla library eagerly in `adapters/__init__.py`.
- Don't replace numpy operations with pandas inside hot loops.
- Don't declare `REQUIRED_CAPABILITIES`, `CONSUMES_QUANTILES`, `SCORE_TYPE` or
  `IS_ONLINE` without `ClassVar` — undeclared, they become pydantic fields and leak into
  `model_json_schema()` and the constructor.
- Don't declare fitted state on a method class — it belongs on the fitted
  class, which is not LLM-facing and needs no `SkipJsonSchema`.

## Running things

```bash
make devenv                         # all extras + the dev and docs groups
make test                           # fast tests, with coverage
make test_extras                    # unit suite with every extra installed
make test_integration               # slow tests, needs extras; two sessions
make lint                           # pre-commit (ruff, format, yaml/toml) + mypy
uv build                            # build wheel + sdist
```

Documentation needs Quarto plus `--group docs --group integration-docs`. Note
`make devenv` does **not** install `integration-docs` — add it before running
any of these:

```bash
make test_docs        # execute the non-network notebooks under nbs/docs — the CI gate
make execute_docs     # --include-network --write: also runs the network pages,
                      # and writes refreshed outputs back into the notebooks
make all_docs         # full site build into docs/mintlify/, composed of
                      # load_docs_scripts, api_docs, examples_docs, format_docs, brand_docs
uv run python scripts/check_docs_nav.py   # docs.json vs built pages, both ways
```

## When in doubt

Read `methods/base.py` (both `ConformalMethod` and `ConformalForecaster`), then `base.py`. Read an existing implementation in the
same folder and match its style. If the contract doesn't fit what you're
building, that's a design question — flag it, don't work around it.
