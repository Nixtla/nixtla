# Changelog

## Unreleased

### Added

- `FitAndSimulate`, `Mixup`, `MovingBlockBootstrap` and `DBA`: augmentation
  strategies as frozen specs with `apply()`, plus a module-level `analyze` in
  `synforecast.augment`. `source_col` records provenance; `augment` and
  `mixup` support extra columns.

### Fixed

- `augment` is reproducible on polars input, `_aug_` ids no longer collide
  with existing ids, and unknown `generator_override` keys warn.

### Changed

- A spec's `apply` is reproducible per call: the same spec, data and seed
  return identical output, and specs are frozen. The `SynAugment` facade
  still advances its seed on every call, so a fresh instance with the same
  seed reproduces the same sequence of outputs.
- Invalid strategy arguments raise pydantic `ValidationError` (a `ValueError`)
  when the spec is built (for `SynAugment`, still at the method call);
  numeric fields reject bools and strings such as `"3"`. An
  unknown generator name and bad `scaling`/`scale` values get pydantic's
  message, augmentation warnings log from `synforecast.augment`, and the AR(1)
  summary no longer starts with `SynAugment`.

### Removed

- `SynAugment.rng`.

## 0.2.0 - 2026-10-12

### Added

- `synforecast.evaluation`: `compute_features` (twelve scale-free features per
  series) and `feature_coverage` / `compare_feature_coverage`, which score how
  well a synthetic corpus covers real data.
- Opt-in `seasonal_pool(freq=...)` preset and
  `pretraining_pool(include_seasonal=...)` for strongly seasonal data.
- `MARGenerator` for GRATIS-style mixture-autoregressive simulation, with
  `tune_to_features` for feature-targeted generation. It is also part of
  `pretraining_pool`.
- `SynAugment.mbb` (moving-block bootstrap) and `SynAugment.dba` (DTW
  barycenter averaging).
- Native Rust paths for MAR generation, features, decomposition, MBB, DTW, and
  DBA.

### Changed

- Renamed `balanced_pool` to `interpretable_pool` and `include_balanced` to
  `include_interpretable`. The old names still work but warn.
- `interpretable_pool` derives its seasonal period from `freq` instead of
  hardcoding 12, and accepts a `seasonal_period` override. Its cyclic
  variants take their mean period from it instead of a fixed 50 steps.
- `SynAugment.mbb` and `SynAugment.dba` skip unusable series with a warning.
- Removed the unused dense pairwise-DTW API.

### Fixed

- `MARGenerator` validates parameters and stationarity, including mixtures
  above AR order 32, and raises instead of silently substituting noise.
- MAR fallback noise honors `standardize=True`.
- Native MAR coefficient-offset overflow.
- Overlays (changepoints, anomalies, missingness) no longer affect each
  other or the base series. Previously, enabling one could move another, and
  for IoTSensor and DailyActiveUsers change later series.

### Reproducibility

- Same-seed output differs from 0.1.0 for the cyclic pool variants,
  IoTSensor, DailyActiveUsers, and multi-series ETS/StateSpace
  `generate_with_states`; with overlays enabled also for KernelSynth, Copula,
  VAR and TCM.

## 0.1.0 - Initial release
