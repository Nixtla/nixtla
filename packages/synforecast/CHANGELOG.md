# Changelog

## 0.1.0 (unreleased)

### Added

- Thirty configurable synthetic time-series generators spanning statistical,
  stochastic, multivariate, domain-specific, and pretraining use cases.
- Strict Pydantic configuration models with validation for shared and
  generator-specific parameters.
- Rust-accelerated batch generation with deterministic NumPy fallbacks.
- Long-format pandas and Polars output with configurable column names,
  datetime or integer time indexes, and deterministic per-series seeding.
- Changepoint, anomaly, and missingness injection, plus exogenous-variable
  generation.
- `SynSet` for composing generator pools and `SynAugment` for fitting
  generators to observed series.
- `Multivariatizer` for adding contemporaneous and lead-lag dependence to
  univariate generators.
- `generate_series` and `balanced_pool` convenience APIs.
- Executable tutorials, API documentation, mathematical and Rust/Python
  parity tests, and open-source community and security policies.
