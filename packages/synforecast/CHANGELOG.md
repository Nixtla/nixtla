# Changelog

## Unreleased

- `balanced_pool` derives the seasonal period of its seasonal variants from
  `freq` (hourly 24, daily 7, weekly 52, monthly 12, quarterly 4) instead of
  hardcoding 12, and accepts an explicit `seasonal_period` override that
  `pretraining_pool` forwards.
- Added `MARGenerator` for GRATIS-style MAR simulation, including a validated
  fixed-parameter mode.
- Added `MARGenerator.tune_to_features` and a minimal feature module for
  feature-targeted generation, with `tuning_diagnostics` reporting convergence
  and search budget usage.
- Added non-parametric `SynAugment.mbb` remainder block bootstrapping.
- Added panel-aware `SynAugment.dba` with banded DTW.
- Added `MARGenerator` to `pretraining_pool` after the existing meta-generators.
- Added native Rust paths for MAR batch generation, feature computation,
  decomposition, MBB sampling, DTW, and DBA updates. Banded DTW stores only
  the band, spectral entropy uses cached RealFFT plans for all lengths,
  and `SynAugment.dba` computes pairwise panel
  distances in bounded parallel chunks, retaining only the requested nearest
  neighbors. Moving-average decomposition is linear-time and reused across
  MBB copies.
- Feature search scores populations in parallel native Rust, isolates candidate
  failures, and caches stationarity checks. Covariance certificates with a NumPy
  dense eigenvalue fallback keep validation free of a SciPy runtime dependency.
  It accepts output, standardization and innovation options and `n_jobs`.
- Panel methods sort and partition once. DBA copies share initial alignments
  and refine in parallel, with allocation/iteration limits and interrupt checks.
- Removed the unused dense pairwise-DTW API and moved the convolution test
  oracle out of the package. Fixed native MAR coefficient-offset overflow.
- `MARGenerator` rejects non-finite or oversized parameters, checks
  second-order stationarity of fixed mixtures (Wong and Li 2000), and raises
  instead of substituting noise when a fixed model fails the output guards.
- MAR random-mode fallback noise now honors `standardize=True` in both paths.
- Fixed MAR mixtures no longer bypass stationarity validation above AR order
  32. Trailing zeros are ignored, and larger mixtures require a sufficient
  stability condition; uncertified mixtures raise an explicit error.
- `SynAugment.mbb` and `SynAugment.dba` skip unusable series with a logged
  warning instead of raising or silently dropping them.
- MBB reports explicit seasonal periods that cannot fit two cycles, DBA
  reports near-zero reference scales, and infinity errors identify the source.

## 0.1.0 - Initial release
