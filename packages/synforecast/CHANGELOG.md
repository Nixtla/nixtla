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

- Publishing a GitHub Release now validates, builds, smoke-tests, and publishes
  the package to PyPI automatically. The release also deploys the production
  documentation.
- Package and README descriptions use the same testing, augmentation, and
  pretraining positioning and link reproducible benchmark evidence.

### Security

- Updated Pillow to 12.3.0 to incorporate its July 2026 security fixes.
- Documented the runtime versus development dependency boundary and the policy
  against loading untrusted model checkpoints.
