# Changelog

## 0.1.0 (unreleased)

### Added

- Thirty-one configurable synthetic time-series generators spanning statistical,
  stochastic, multivariate, domain-specific, and pretraining use cases.
- Strict Pydantic configuration models with validation for shared and
  generator-specific parameters.
- Native Rust generation with deterministic single-series and batch execution.
- Long-format pandas and Polars output with configurable column names,
  datetime or integer time indexes, and deterministic per-series seeding.
- Changepoint, anomaly, and missingness injection, plus exogenous-variable
  generation.
- `SynSet` for composing generator pools and `SynAugment` for fitting
  generators to observed series or creating TSMixup combinations.
- `Multivariatizer` for adding contemporaneous and lead-lag dependence to
  univariate generators.
- TSI, TCM, and KernelSynth meta-generators for diverse foundation-model
  pretraining corpora.
- `generate_series`, `balanced_pool`, and `pretraining_pool` convenience APIs.
- Materialized NeuralForecast, MLForecast, and StatsForecast workflows covering
  observed-only training, augmentation, synthetic-only pretraining, and
  pretraining followed by fine-tuning.
- Executable tutorials, API documentation, mathematical correctness and native
  binding tests, and open-source community and security policies.
- Complete generated API references for every public generator, composition
  helper, and top-level convenience API.
- Citation metadata, support guidance, code ownership, and third-party data
  provenance.

### Changed

- `balanced_pool` now returns its 42 generators interleaved round-robin
  across the 15 behavioral niches instead of grouped by niche, so any prefix
  of the pool spans as many distinct behaviors as possible. `generate_series`
  panels smaller than the pool therefore span one behavioral niche per series
  (previously `n_series=6` produced five SARIMA variants and one ETS).
  Seed-to-variant assignments are unchanged; only the list order — and hence
  which generators a small `generate_series` panel draws — differs.
- `generate_series` accepts `with_generator_col=True` to add a `generator`
  column recording the alias of the generator that produced each series.
- GENERATORS.md documents a verification status for every generator
  (theory-tested / simulator / procedural, with criteria) and states explicitly
  that `SynAugment` matches mean, standard deviation, and lag-1
  autocorrelation by construction. Added a copula tail-dependence property
  test verifying the t copula's coefficient against theory and the Gaussian
  copula's decay toward zero.
- Quality gates: an unexpected `SynAugment` AR(1) substitution during any test
  now fails the suite (tests that exercise the fallback on purpose opt out via
  the `allow_ar1_fallback` marker). mypy runs in the lint workflow on an
  incrementally growing module list (`_fitting`, `utils`, `presets` to start).
  CI gained a benchmark smoke job (model-free scripts under `--quick`) and an
  optional-engine job running the PyArrow and Modin smoke tests; cuDF remains
  a documented local-only check.
- Benchmarks: removed `benchmark_rust_vs_python.py`, which toggled the
  `_HAS_RUST` flag that no longer exists since the pure-Python fallback was
  removed (it silently measured Rust against Rust). The remaining scripts run
  from a clean checkout again: stale pre-release parameter names were updated,
  the batch-vs-threaded toggle now empties `_GEN_TYPE_MAP` instead of nulling
  the batch module, the pretraining benchmark falls back to CPU when no GPU is
  available, and every script supports `--quick` for smoke runs. Saved results
  now embed environment metadata (package versions, platform, exact command),
  and `benchmark_pretraining.py` computes CIs via a hierarchical
  (seed-then-series) bootstrap with win rates and Wilcoxon tests on per-seed
  aggregates rather than treating per-series rows across seeds as independent.
- `SynAugment` now raises when a fitted generator fails for a series instead
  of silently substituting an AR(1) series. Pass `on_error="ar1"` to restore
  the substitution behavior; substitutions are then reported in a single
  summary warning. Unknown generator names in `generator_override` raise a
  `ValueError` instead of silently using a random walk.
- Publishing a GitHub Release now validates, builds, smoke-tests, and publishes
  the package to PyPI automatically. The release also deploys the production
  documentation.
- Package and README descriptions use the same testing, augmentation, and
  pretraining positioning and link reproducible benchmark evidence.

### Fixed

- `SARIMAGenerator.get_model_info` labeled non-seasonal models with a
  misleading `(0,0,0)[seasonal_period]` suffix; they are now reported as
  plain `ARIMA(p,d,q)`.
- `SynAugment` fitted GARCH parameters under the key `mean`, which
  `GARCHGenerator` rejects (its field is `mu`), so every GARCH-classified
  series silently fell back to AR(1) generation. The fitter→generator
  parameter contract is now covered by tests for all supported generators.

### Security

- Updated Pillow to 12.3.0 to incorporate its July 2026 security fixes.
- Documented the runtime versus development dependency boundary and the policy
  against loading untrusted model checkpoints.
