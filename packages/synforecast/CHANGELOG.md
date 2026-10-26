# Changelog

## Unreleased

### Added

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
