# Changelog

## Unreleased

### Added

- `FitAndSimulate`, `Mixup`, `MovingBlockBootstrap` and `DBA`: frozen
  augmentation specs run with `apply()`, plus `analyze` in
  `synforecast.augment`. `source_col` records provenance (`Mixup`: anchor,
  `DBA`: reference; other blended series still contribute values); `augment`
  and `mixup` keep extra columns.
- `BaseGenerator.generate_values(n_series, n_jobs)`: the series `generate`
  would draw, as value arrays without building a DataFrame (native Rust batch
  path where available). Both methods go through a new `_generate_metadata`
  hook, so a custom generator with its own generation scheme overrides that
  hook rather than `generate`.
- `members=` on `pretraining_pool`, `interpretable_pool` and `seasonal_pool`:
  build only the selected pool entries, each identical to the full pool's
  entry. Drawing one member per series no longer constructs the whole pool
  (0.11 ms instead of 3.6 ms per draw). Entry seeds are `seed` plus a fixed
  offset, so a stream of fresh draws should reuse built members or give each
  rebuild a random 64-bit seed. Consecutive seeds repeat series of other
  entries exactly (see the `members` docs).
- Python 3.15 support: CI tests it, and wheels are built for CPython 3.15.
  On 3.15 synforecast requires polars 2.0 or newer, the first release that
  works on Python 3.15.

### Changed

- `KernelSynthGenerator` writes each composition as a polynomial in `x x'`
  with stationary coefficients, `K = sum_d (x x')^d T_d(x - x')`, and samples
  it exactly as `sum_d x^d * f_d` from independent O(n^2) Durbin-Levinson
  draws. No composition forms a dense covariance or an O(n^3) Cholesky any
  more, including those with a linear kernel. Both bulk `generate()` (Rust
  batch path) and `generate_single_series` run the same Rust sampler, so
  `Multivariatizer` and `SynAugment` get it too. `jitter` is now added to each
  polynomial term and escalated tenfold on a numerical breakdown, to at least
  1e-6 of the term's variance even with `jitter=0`.
- `KernelSynthGenerator.max_kernels` is now at most 64.
- `GaussianProcessGenerator(kernel="periodic")` samples exactly without an
  O(n^3) Cholesky, for any length scale. An integer period up to
  `4 * max(length, 1024)` samples one period by a circulant FFT and tiles it;
  any other period uses the kernel's
  cosine series in O(n J), or Durbin-Levinson in O(n^2) for length scales so
  short that J would exceed n. About 1 ms instead of 560 ms at length 2,273.
- A spec's `apply` returns identical output for the same spec, data and seed.
- Invalid arguments raise pydantic `ValidationError` (a `ValueError`, with
  pydantic's messages, including bounds and unknown generator names) when a
  spec is built; numeric fields reject bools and strings. Warnings log from
  `synforecast.augment`, and the AR(1) summary no longer starts with
  `SynAugment`.
- `generator_override` keys are stored and matched as `str(key)`, so a spec
  round-trips through `model_dump`/JSON to an equal spec. Integral floats are
  stored as integers, so `1`, `1.0` and `"1"` are one key and still match a
  `unique_id` upcast to float.
- A spec's `model_copy(update=...)` validates the update like constructor
  arguments, so an updated `generator_override` stays read-only and hashable.
- `SynAugment.augment_single_series` validates `n_augment` (1 to 1000) like
  `FitAndSimulate`; `n_augment=0` used to return an empty list.
- pandas output of a spec's `apply` and of every `SynAugment` method has a
  fresh `RangeIndex`; synthetic rows used to repeat their source's row labels.

### Fixed

- `augment` is reproducible on polars input, `_aug_` ids no longer collide
  with existing ids, and unknown `generator_override` keys warn.
- `mixup(alpha=nan or inf)` is rejected.
- An explicit `engine` converts the input natively (pyarrow when available)
  and skips the conversion when it already matches the input library. pandas 3
  input with missing values in a string column no longer fails with
  `engine="polars"`; missing values arrive as nulls.
- `SynAugment(seed=...)` rejects bools and strings, like spec seeds. It now
  takes only a non-negative int or `None`; `np.random.SeedSequence` and array
  seeds are no longer accepted.
- `analyze` no longer averages an empty list (numpy `RuntimeWarning`) when
  every squared-return autocorrelation is undefined.
- `synforecast.__version__` falls back to `"0+unknown"` when package metadata
  is missing.

### Removed

- Python 3.10 support (end of life 2026-10-31). synforecast now requires
  Python 3.11 or newer.

### Reproducibility

- Same-seed output of periodic `GaussianProcessGenerator` differs from 0.2.0;
  the sampled distribution is unchanged.
- Same-seed output of `KernelSynthGenerator` differs from 0.2.0; the sampled
  distribution is unchanged up to the jitter.
- `pretraining_pool` and `interpretable_pool` build the same generators with
  the same seeds as 0.2.0; only their KernelSynth and periodic
  `GaussianProcessGenerator` entries produce different output. Every other
  generator is bit-identical to 0.2.0.

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
