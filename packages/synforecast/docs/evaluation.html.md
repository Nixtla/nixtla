---
title: Corpus evaluation
description: Native feature extraction and reference-anchored coverage diagnostics
---

`compute_features` extracts one row of native summaries per series.
`compare_feature_coverage` fits a scaler, PCA projection, and grid on the
retained real reference panel, then projects each synthetic corpus into that
same space. Adding another corpus leaves existing scores unchanged under
these defaults. Corpus size still affects occupancy; compare matched sizes.

The twelve-feature schema is **`native_v1`**. Its feature names, order,
definitions, normalization, and undefined-value rules are fixed; changes to
those contracts require a new schema identifier. Period and window settings
remain explicit extraction parameters, recorded with the schema. Raw-input
coverage metadata records both the requested and effective window size.
These are SynForecast definitions, not a reproduction of an external feature
package.

The first two PCA components keep only part of the real-feature variance:
54–84% on the M4 panels we evaluated, with visible losses for spike and
variance-shift differences. Treat the map as a two-dimensional diagnostic,
not as comprehensive feature-space coverage. Full-feature nearest-neighbour
distances agree with the map on the aggregate ordering of corpora but often
disagree on individual series, and grid boundaries alone can move a score
noticeably. Report full-feature distances and real-calibrated coverage
alongside the map; neither a single radius nor a grid score establishes
general fidelity.

## Choosing a period and window

Extraction never reads timestamps: declare the period yourself, either in
observations (`seasonal_period`) or through `freq`, which uses the same
convention as the preset pools (hourly 24, daily 7, weekly 52, monthly 12,
quarterly 4, yearly none). Without either, no seasonality is declared, the
two seasonal features are zero for every series, and
`compare_feature_coverage` drops them as constant columns and logs a warning.
A period of 1 means no seasonality.

An availability check of all 100,000 M4 training series supports the
following settings, using at most the last 512 observations:

| Panel | `seasonal_period` | `window_size` |
|---|---:|---:|
| Yearly | `None` | 5 |
| Quarterly | 4 | 4 |
| Monthly | 12 | 12 |
| Weekly | `None` | 10 |
| Daily | 7 | 7 |
| Hourly | 24 | 24 |

All M4 series are finite in all twelve features under these settings. Weekly
uses M4's nonseasonal convention: its two seasonal features are constant and
annual seasonality is unmeasured. A 52-week period excludes 65 of the 359
Weekly series, which lack two full cycles; report that as a separate eligible
population. Daily's seven-day cycle is an explicit assumption (M4 metadata
declares it nonseasonal), and Hourly's 24-hour period covers only that cycle.

```python
import pandas as pd

from synforecast import compare_feature_coverage, compute_features

# real, interpretable, and mar are long-format panels: unique_id, ds, y.
# Supply at least two usable real series.
results = compare_feature_coverage(
    real,
    {"interpretable": interpretable, "mar": mar},
    freq="MS",
)
print(pd.DataFrame({name: r.summary() for name, r in results.items()}).T)
results["interpretable"].plot()

# Cache features once for repeated evaluations or different grid resolutions.
real_features = compute_features(real, freq="MS")
interpretable_features = compute_features(interpretable, freq="MS")
cached = compare_feature_coverage(
    real_features, {"interpretable": interpretable_features}, precomputed=True
)
```

Feature frames preserve the input dataframe engine and ID dtype, with a
fresh index for pandas. Observations are sorted by ID and time. Raw targets
must be finite, IDs/times non-null, and timestamps unique within each series.
`n_jobs` sets the native worker threads; the default -1 uses all CPUs and
honours `RAYON_NUM_THREADS`, as in the generators.

## Native feature contracts

All features require at least three observations. Every feature is computed
on the series normalized to zero mean and unit population variance, so none
depends on the input scale or offset. Constant input maps to zero;
normalization scales by the maximum absolute value first to avoid overflow.

| Feature | Definition and additional minimum length |
|---|---|
| spectral_entropy | Normalized non-DC periodogram entropy |
| trend_strength | Classical moving-average decomposition strength; two declared cycles |
| seasonal_strength | Classical decomposition strength; two declared cycles, or zero without a period |
| acf1 | Mean-centered biased lag-one ACF |
| x_acf10 | Squared ACF sum at lags 1 through 10; at least 11 observations |
| diff1_acf1 | Lag-one ACF of first differences |
| seas_acf1 | ACF at the declared period; more observations than the period; zero without a period |
| spike | Population variance of leave-one-out population variances of the decomposition remainder; two declared cycles |
| lumpiness | Population variance of full-tile population variances; incomplete final tile omitted |
| max_level_shift | Largest absolute mean difference between adjacent nonoverlapping windows, checking every valid split |
| max_var_shift | Largest absolute population-variance difference between those windows |
| crossing_points | Raw count of transitions across the median; ties belong to the lower half |

When a declared period cannot fit two cycles, the three decomposition-based
features (`trend_strength`, `seasonal_strength`, `spike`) are NaN, so a panel
of mixed lengths never mixes two decomposition definitions. Without a
declared period, the decomposition uses a trend window near one tenth of the
series length.

For lumpiness and shift features, `window_size` can explicitly set the width
(at least 2). The default is the declared period, or 10 without one; at least
two full windows are required. `window_size` does not change the decomposition
or seasonal ACF. For short nonseasonal panels, a fixed `window_size=5` keeps
series of 10 to 19 observations that a ten-observation window would exclude.
Record the width with cached features and use the same setting for real and
synthetic corpora. ACF uses the full mean-centered sum of squares in its
denominator at every lag and returns zero if that sum is at most machine
epsilon. Crossing counts and other estimates remain length-dependent, so
match length distributions as well as sample sizes.

## Reading the scores

- `miscoverage`: uncovered real cells divided by **all** grid cells.
- `reverse_miscoverage`: synthetic-only cells divided by all grid cells,
  restricted to the grid's bounding box.
- `uncovered_real_cell_fraction`: uncovered real cells divided by occupied
  real cells.
- `uncovered_real_series_fraction`: retained real series in uncovered cells
  divided by all retained real series.
- `synthetic_out_of_range_fraction`: synthetic series outside the box divided
  by all retained synthetic series, including those outside it.

`CoverageResult.summary()` returns these scores with the retained counts as a
flat dict. Results compare by identity, since several fields are arrays.

The default real-range grid adds one percent padding per axis. It excludes
out-of-range synthetic points rather than clipping them onto edge cells.
Out-of-range is evidence of extrapolation in the displayed coordinates, not
proof of unrealistic data. Differences in discarded dimensions may be hidden;
inspect explained variance and the features themselves. Zero miscoverage is
not evidence of predictive fidelity or privacy.

`missing="raise"` rejects undefined feature values. Explicit `missing="drop"`
evaluates complete cases, logs exclusions, and returns retained/dropped IDs
and reasons. Counts and fractions describe the retained population. A column
is constant when its range in the fitting population is at most 1e-12, scaled
by its magnitude when that exceeds one. Real-constant columns cannot enter
reference PCA; synthetic deviations from those reference values are counted
separately.

At least two retained real series and one variable feature are needed for the
default fit. Rank-one PCA uses a zero second coordinate. Result metadata stores
fitted parameters, a space identifier, and a reference fingerprint. A shared
space identifier does not make different reference samples or sample sizes
interchangeable. Persist extraction settings with cached native feature frames;
precomputed frames do not carry an implicit native schema, and
`precomputed=True` rejects the extraction keywords rather than ignoring them.

`fit="pooled"` and `grid_range="pooled"` provide a joint-fitting sensitivity
case. Synthetic outliers can shift that space and its bin widths. Optional
t-SNE requires `embedding="tsne", fit="pooled", seed=...`, scikit-learn,
and more than 30 retained series for its fixed perplexity of 30. It fits all
declared points jointly and does not project new points. Pooled PCA uses the
same occupancy measure but does not reproduce Kang's t-SNE experiment.

## API

::: synforecast.evaluation.compute_features

::: synforecast.evaluation.feature_coverage

::: synforecast.evaluation.compare_feature_coverage

::: synforecast.evaluation.CoverageDiagnostics

::: synforecast.evaluation.CoverageResult
