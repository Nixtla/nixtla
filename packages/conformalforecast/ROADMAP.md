# ConformalForecast Roadmap

Known gaps, defects and rough edges in the current release, and what closing them
looks like. Everything here was reproduced against the code in this repository;
where a limit is deliberate rather than accidental, it says so.

## Defects

- **The forecaster surface does not preserve `float32`.** `calibrate` and
  `calibrate_cv` in `src/conformalforecast/methods/base.py` coerce truths with
  `np.asarray(..., dtype=np.float64)` before handing them to `fit`, and cast
  predictions (and CQR's quantile cube) to `float64` for the retained
  calibration copy. So a `float32` forecaster -- neuralforecast's
  output, in particular -- yields a `float64` `score_quantile_`, `float64`
  retained calibration arrays, and `float64` intervals out of `predict()`,
  even though `float32` preservation matters most for exactly that adapter.
  The array surface (`fit`/`interval`/`update` called directly with `float32`
  arrays) is unaffected and preserves `float32` for every method --
  `tests/unit/methods/test_<name>.py::test_float32_preserved` covers it.
  Fixing it changes the output dtype of `calibrate`/`calibrate_cv`, so it is a
  deliberate numeric-behavior decision rather than a drop-in fix.

## Missing capabilities

- **Asymmetric CQR cannot be reached by any shipped adapter.**
  `ConformalizedQuantileRegression(symmetric=False, alpha_lo=..., alpha_hi=...)`
  is implemented, validated and tested, but the StatsForecast and
  NeuralForecast adapters (the two with a quantile path) request quantiles through the integer `level` API, where `level=90` means the 5th
  *and* 95th percentiles inseparably. `_nixtla_common.quantiles_to_levels`
  therefore rejects `[0.02, 0.92]`. Training a neuralforecast model with
  `MQLoss(quantiles=[0.02, 0.92])` does **not** help — the adapter still routes
  through `level=` and raises before the model is consulted. The feature works
  today only behind a custom `SupportsQuantiles` adapter. The natural fix is a
  non-level quantile path in `NeuralForecastAdapter`, which is the one backend
  that genuinely holds arbitrary quantiles: read `model.loss.quantiles` and index
  the output columns directly.
- **Exogenous features are supported far more narrowly than the `X` array shape
  suggests.** All three Nixtla adapters carry `X` through `predict`, given a
  `train_df` holding the regressors' history — StatsForecast and MLForecast
  opt in with `exog_features=[...]`, NeuralForecast reads the list off the
  model's `futr_exog_list`. What remains: `calibrate_cv(X=...)` always raises
  `NotImplementedError`; CQR rejects `X` in both `calibrate` and `predict`, and
  quantile methods are gated off on any exogenous adapter because
  `predict_quantiles` has no `X` in its contract; `refit` is unavailable on any
  exogenous adapter, and historical and static covariates are still refused,
  both because a history panel carries target values only. Cross-validated
  exogenous calibration is the substantive piece of work.
- **The exogenous history can never leave the training frame.** `Series` is
  `(n_series, T)` — target values only — so every adapter recovers the
  regressors by date-joining the stored `train_df`. A history extending past it
  raises, and `refit` cannot work. Carrying exogenous values on the panel type
  itself is the change that would lift both limits at once.
- **`SequentialPredictiveConformalInference.interval` ignores the `X` it is
  given.** Future features reach the regressors at `fit` time, but the interval
  query always uses the most recent calibration row, so `predict(X=...)` does
  not condition the interval on the step being predicted — the one thing the
  4-D `X` shape most suggests it does. Threading future exogenous rows into the
  query is the fix.
- **Custom column names must match the fitted model, for `mlforecast` and
  `neuralforecast`.** All three Nixtla adapters forward
  `id_col`/`time_col`/`target_col` into every library call that accepts them, so
  an `(item, time, value)` frame works end to end. But `MLForecast.predict` and
  `NeuralForecast.predict` take no column-name arguments — they reuse what their
  own `fit` recorded — so those two adapters cannot rename on the way in. A
  disagreement is refused at construction rather than left to raise a `KeyError`
  from the reshaping code. `sf.forecast` re-reads the names from its kwargs on
  every call, so `StatsForecastAdapter` has no such constraint.
- **`diagnostics.evaluate` has no `X` parameter**, so an exogenous-aware method
  cannot be evaluated with the helper and the holdout loop must be written by
  hand. Small, and it blocks a documented workflow.
- **`SupportsTransfer` is implemented only on `WeightedSplitForecaster`.** Every
  other fitted object simply has no `adapt()` method at all — `SupportsTransfer`
  is an intermediate ABC that only `WeightedSplitForecaster` inherits, so
  calling `adapt()` on another fitted object raises `AttributeError`, not
  `UnsupportedCapability`.
  Extending transfer to `SplitConformal`, `AdaptiveConformalInference` and
  `NonexchangeableConformalPrediction` is a follow-on: the ABC's contract
  (`_recompute_quantile`, retained `scores_`, `source_scales_`) is general
  enough for any of them to implement it later, but none does yet.
- **`recalibrate()` reuses the bound adapter, so it cannot serve a target panel
  with a different series count.** `ForecasterAdapter` fixes `n_series` at
  construction and `recalibrate` does not take a new forecaster, so a target
  panel of a different width raises, naming the fix (build a fresh adapter and
  call the method's `calibrate` on it). `WeightedSplit(pooling="global")` plus `adapt()` is
  the route that does serve a differently-shaped target panel, and the
  [transfer page](nbs/docs/capabilities/transfer.ipynb) demonstrates both.
  Relaxing `n_series` to a per-call value is a follow-on.
- **`recalibrate()` on an online method discards accumulated `update()`
  state.** `recalibrate` re-fits through the method's `calibrate`, so for the
  four online methods (`AdaptiveConformalInference`,
  `AggregatedAdaptiveConformalInference`, `NonexchangeableConformalPrediction`,
  `SequentialPredictiveConformalInference`) the new fitted object it returns
  computes its online state — `alpha_t_`, aggregator weights, recency-weighted
  buffers — from the target panel's own trajectory, so adaptation accumulated
  through `update()` on the receiver does not carry over. The receiver itself
  is left untouched, so nothing is lost, but continuing mid-stream from the
  adapted state on target data is not supported. This is documented on
  `recalibrate`.
- **`DensityRatio`'s default `clip_quantile=0.99` may be too tight under
  strong covariate shift.** On the synthetic covariate-shift benchmark the
  same weights give 0.8213 coverage at the default and 0.9167 with clipping
  off, against a 0.90 target — the median *test* weight saturates at the clip
  value, so the guard binds on over half the points and discards most of the
  weighting signal it was meant to protect. Measured on the sweep in
  [the weighted split page](nbs/docs/methods/weighted_split.ipynb). Revisiting
  the default — or setting it from the target weight distribution rather than
  the source one — is a follow-on.
- **`WeightedSplit` identifies a "symmetric score" as
  `isinstance(score, AbsoluteResidual)`.** Its weighted per-test-point path
  builds the interval as `prediction +/- q`, which is correct only for a
  symmetric score, so anything else is rejected at construction through
  `SCORE_TYPE = AbsoluteResidual`. The general fix is still an `IS_SYMMETRIC`
  class attribute on the `ScoreFunction` contract, which every score would
  declare and the method would read. That changes a shared public contract and
  was deliberately out of scope; the isinstance check is a stand-in, and it
  wrongly rejects a legitimate symmetric score that does not subclass
  `AbsoluteResidual`.
- **`pooling="global"` and per-test-point weights do not combine.** The
  Tibshirani normalization that puts `w(x_test)` in the denominator runs only
  under `pooling="per_series"`; under `pooling="global"` the weights shape the
  pooled calibration quantile but the test point's own weight is ignored. So
  weighting-with-test-point-normalization and serving a target panel of a
  different series count are individually available and not jointly.
- **`SupportsBootstrap` is exported but dead.** No adapter implements
  `bootstrap_predict`, no method requires it, and the only test references assert
  that adapters *do not* implement it. Its docstring points at EnbPI, which is
  not shipped. Either implement EnbPI or drop the mixin from the public API —
  exporting a capability nobody can satisfy is a promise the package does not
  keep.

## Statistical caveats worth fixing or documenting louder

- **`evaluate_calibration` is in-sample, and for SPCI doubly so.** Mode 1
  replays each calibration window against the state that window implied rather
  than broadcasting the final one, so SPCI's number is a real measurement
  (broadcasting would query its quantile regressors against residual histories
  they were never conditioned on, scoring ~23% against a 90% target). It is
  still optimistic, and for SPCI more than for the
  others: the quantile forests were fit on the very residual rows the replay
  queries them with, so it lands near 100% on synthetic data. It is a plumbing
  check. Report `evaluate`.
- **A sequential Mode-1 replay is not available for ACI/AgACI.** Their
  `interval()` is a calibration-set constant, so broadcasting it is legitimate
  and the number is sane. Replaying their `fit` loop step by step would be
  strictly better, but it runs straight into the unfloored `alpha_t` defect
  below — the widths reach ~1e300 within the calibration set. Fix that first.
- **SPCI under-covers badly at multi-step horizons on real data.** At horizon 6
  on both bundled datasets it lands near 65% against a 90% target, while split
  conformal holds 90% on the same forecaster; at horizon 1 it reaches nominal and
  beats split conformal on width. This is not a horizon bug in the method —
  on i.i.d. synthetic data it sits at ~85% for horizons 1, 3 and 6 alike — and it
  does not respond to `window_size`, `n_windows` or `beta_grid_size`. The likely
  cause is that a quantile forest cannot extrapolate to residual magnitudes
  absent from its training window, which multi-step errors on a trending or
  level-shifting series routinely are. Worth a deliberate investigation; the
  [SPCI page](nbs/docs/methods/spci.ipynb) documents the measurement in the
  meantime.
- **`aligned=True` is expensive on panels with many distinct end dates.** The
  calendar axis is the union of every series' cutoffs, and the Nixtla libraries
  count cutoffs back from each series' *own* end date (statsforecast and
  neuralforecast both do), so series that end on different dates contribute
  disjoint cutoff sets. The union then grows with `n_series` rather than with
  `n_windows`, and so does the returned cube — 400 staggered series at 12 windows
  give a 4800-column axis. Axis 1 also stops meaning "chronological age", which
  is what NexCP and SPCI read it as, so the mostly-NaN cube is rarely what you
  want. The dense default is unaffected: it never builds the union. Documented on
  the [ragged panels page](nbs/docs/capabilities/ragged-panels.ipynb); a warning
  above some ratio of union size to `n_windows` would make it harder to walk into.
- **The quantile cross-validation path ignores `aligned`.** `calibrate_cv` on a
  CQR method always densifies, whatever the flag says, so the calendar-aligned
  cube is unreachable there. Ragged panels calibrate fine; it is the aligned
  representation that is missing. Stated in the `calibrate_cv` docstring and
  nowhere else.
- **ACI and AgACI can degenerate to infinite width.** Nothing bounds `alpha_t`
  below zero, and at `alpha_t <= 0` the requested quantile falls beyond the
  largest calibration score. A `gamma` of 0.2 reaches that state within a few
  steps on a 48-month stream, producing widths around `1e300` — technically 100%
  covering, and useless. The constructor warns above `gamma = 0.1` but nothing
  stops it, and the threshold is off by one step: `gamma = 0.1` itself
  degenerates on that same stream (see the expert table on the AgACI page) while
  raising no warning at all, because the check is strictly greater-than. AgACI
  is unharmed by a degenerate expert because the aggregator drops its weight,
  which is a good argument for preferring it; plain ACI has no such protection.
  Clipping `alpha_t` to a small positive floor would make the failure survivable.
- **The weighted conformal quantile silently clamps instead of returning an
  infinite interval.** In Tibshirani et al. (2019) the construction returns an
  unbounded interval when `w(x_test)` is large enough that the normalized
  calibration weights cannot reach `1 - alpha` on their own. This
  implementation clips the target to 1.0 and returns the largest calibration
  score instead, consistent with `weighted_conformal_quantile`. That is a
  defensible choice — an infinite band is useless in production — but it is
  undocumented, and it bites precisely in the high-shift regime where the
  method is supposed to earn its keep: coverage degrades quietly rather than
  announcing that the source data cannot answer the question. It should at
  minimum be surfaced in `adapt()`'s diagnostics.
- **`weighted_quantile_per_test_point` materializes an
  `(n_series, n_windows, n_samples, horizon)` boolean array.** At 100 series,
  500 calibration windows, 500 test samples and horizon 24 that is roughly
  600 MB in one allocation. A scaling limit rather than a correctness issue,
  and the fix is chunking over the sample axis.
- **The method-level coverage tolerance is wider than it looks.** SPCI's own
  contract test allows 0.08 around the target, and on i.i.d. data SPCI
  genuinely sits 3–5 points under nominal. That is inherent to an asymptotic
  method with a beta search, not a bug, but it should be stated in the API
  docstring rather than only in the test comment.

## Packaging and CI

- **Nothing enforces that the extras and `integration-docs` agree.** The
  documentation notebooks execute against the `integration-docs` group, while
  users install the `nixtla`, `spci` and `plotting` extras. A floor that is lower
  in an extra than in the group advertises a version combination no CI job has
  ever run. The four floors are aligned today (`statsforecast>=2.1.1`,
  `mlforecast>=1.1`, `neuralforecast>=3.2.1`, `quantile-forest>=1.4.2`) and a
  comment in `pyproject.toml` says to keep them so, but that is a convention,
  not a check. A few lines of test comparing the two tables would make it one.
- **`quantile-forest` is declared twice and can drift.** It is both the `spci`
  extra and a member of `integration-docs`, because the docs CI job installs
  dependency groups and never extras. The duplication is deliberate — the SPCI
  page cannot execute without it — but it is the concrete instance of the problem
  above, and applies to any future extra a documentation page depends on.
- **The extras have no dependency-floor test.** `Test lowest direct dependencies`
  installs the core package only, so it verifies `narwhals>=2.9`, `numpy>=1.24`,
  `scipy>=1.11` and `pydantic>=2.7` but never resolves an extra. The
  forecasting-library floors are therefore asserted and unverified in both
  directions.

## mlforecast integration

`compat/mlforecast_shim.py` is a reference implementation for an upstream PR, not
a supported API. Two small upstream changes would let it cover more methods (both
recorded in its module docstring):

- `PredictionIntervals.__init__` validates `method` against a hard-coded list
  rather than `_INTERVAL_METHODS.keys()`, so a registered name is reachable
  through `get_conformal_method` but not through `PredictionIntervals(method=...)`.
- `compute_conformity_scores` already accepts `signed=`, but `MLForecast.fit`
  never passes `signed=True` on the ordinary path, so the signed residual is
  unrecoverable and every signed-score method is rejected by the shim.

The shim also does not support `weights` / `target_weights` (density-ratio
reweighting) or `is_transfer=True`. NexCP's `rho` decay is a different mechanism
and must not be substituted for density-ratio weights.

The shim *could* cover `WeightedSplit` — that is what
`weights` / `target_weights` would map onto — given the same two upstream
changes above: `PredictionIntervals.__init__` validating against
`_INTERVAL_METHODS.keys()` so a registered name is constructible, and a signed
path through `MLForecast.fit`. Registering the weighted method is the natural
next increment for the shim.

## Scope boundaries (deliberate, not gaps)

- **Nixtlaverse plus callables only.** There are no darts or sktime
  adapters and no extras for them. `CallableAdapter` is the supported route to
  anything else.
- **Core dependencies are narwhals, numpy, pydantic and scipy.** Everything else is an
  extra, and `import conformalforecast` never imports a forecasting library.
- **Per-series calibration.** A global model shares parameters across a panel;
  the conformal quantile is still taken per `(series, horizon)` cell. Ordinary
  calibration remains strictly per-series. Pooling across the series axis is
  offered **only** under transfer, through `WeightedSplit(pooling="global")`,
  and **only** after scale normalization — the constructor refuses
  `pooling="global"` without a `scale_estimator`, because pooling residuals
  across series of different scales is unsound. That path exists so a target
  series absent from the source panel can be served at all; it is not a
  general pooled-calibration mode.

## Under consideration

- **EnbPI** — would give `SupportsBootstrap` a reason to exist, and is the
  obvious eighth method.
- **Conditional-coverage diagnostics beyond binning** — `coverage_by_magnitude_bin`
  and `coverage_by_group` are coarse instruments for the question SPCI is trying
  to answer.
