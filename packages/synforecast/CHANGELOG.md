# Changelog

## Unreleased

- Added `synforecast.evaluation`: `compute_features` extracts a frozen
  twelve-feature `native_v1` schema per series in one parallel native call,
  and `feature_coverage` / `compare_feature_coverage` score how well synthetic
  corpora occupy a real-anchored PCA grid, with retained/dropped diagnostics,
  out-of-range counts, and a `CoverageResult.summary()` table row. All twelve
  features are computed on the normalized series, so none depends on input
  scale; a declared period that cannot fit two cycles leaves the three
  decomposition-based features undefined. The period is declared through
  `seasonal_period` or derived from `freq` with the preset convention, and a
  warning is logged when undeclared seasonality drops out of the space.
  `n_jobs` defaults to -1, as in the generators.
- Coverage findings, from benchmarks whose scripts land in a follow-up release:
  on fresh M4 queries the pools come within a few points of the real-to-real
  baseline on Yearly and Quarterly, about 5 to 30 points below it on Monthly,
  Weekly, and Daily, and far below it on Hourly (1–2% against 91%). Several
  independent strongly seasonal panels (tourism quarterly and monthly,
  hospital monthly, weather daily) are also under-covered. The gaps are panel-specific rather than
  frequency-specific, with strong regular seasonality on a moving level as
  the recurring theme. Default presets are unchanged.
- Added the opt-in `seasonal_pool(freq=...)` preset: eight configured TSI,
  ETS, and Seasonal generators with a pronounced seasonal cycle on a moving
  level, meant to be added on top of `interpretable_pool` or
  `pretraining_pool`. `freq` is required so the combined corpus never mixes
  frequencies by default. In a pre-declared test it raised coverage on M1
  monthly, was neutral on M4, and did not help intermittent data.
- Added `pretraining_pool(include_seasonal=...)`, which appends the
  `seasonal_pool` instances at the derived period. A pre-declared fifteen-panel
  test found gains on every quarterly and monthly panel but small losses on
  weekly, daily, and intermittent ones, so the default stays False.
- Renamed `balanced_pool` to `interpretable_pool` and `pretraining_pool`'s
  `include_balanced` to `include_interpretable`. Every instance of the pool is
  an interpretable single-mechanism process, while both pools are balanced
  across niches. The old names remain as deprecated aliases that warn.

- `interpretable_pool` derives the seasonal period of its seasonal variants from
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
