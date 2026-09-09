# SynForecast Roadmap

## Augmentation

- **Optional moment matching** — augmented series are pinned to the source mean,
  std, and lag-1 ACF by construction; add `match_moments=False` and test whether
  unpinned should be the default. Highest priority here.
- **Strategy selection** — fit-and-simulate, `mixup`, `mbb`, and `dba` now ship
  as separate methods; add a unified `SynAugment(strategy=...)` switch spanning
  them and the jitter/scale/warp baselines.
- **Wider fitter coverage** — 9 fitters against 32 generators, so only a third of
  the library is reachable; add ETS, StateSpace, INAR, multi-period seasonal.
- **Panel-aware augmentation** — augment jointly to preserve cross-series
  correlation instead of series-by-series.
- **Exogenous passthrough** — carry `X` columns through augmentation.
- **Fit diagnostics** — return ranked candidates with fit scores so a poor
  best-fit is visible rather than silent.
- **Rust batch path** — route fitted-generator simulation through it so
  augmentation scales like generation.

## Evaluation

- **Feature-space coverage** — a minimal native, targeting-scoped feature set
  now ships through `synforecast/_features.py`; add the full tsfeatures/catch22
  diversity and coverage metrics.
- **Nearest-neighbour distance to real data** — promote the memorization check
  from notebook to API.
- **Fidelity scores** — discriminative and predictive scores for comparing
  generation methods.
- **Wider benchmarks** — extend `when_synthetic_helps` beyond one model and
  dataset.

## Performance

- **Rust SARIMA exogenous support**, then drop the Python SARIMA path.
- **Rust batch dispatch** for `Copula`, `VAR`, `DailyActiveUsers`, `IoTSensor`.
- **Streaming generation** — chunked Parquet/Arrow output for corpora larger than
  memory.

## Integrations and docs

- **Hierarchical panels** — panels that aggregate coherently, for testing
  reconciliation.
- **Generator selection guide** — a decision path to a generator or pool, instead
  of reading all 32 pages.

## Under consideration

- **Deep generative models** (GAN/VAE/diffusion) — heavy deps, GPU time, and
  per-dataset training conflict with cheap deterministic CPU generation;
- **Irregular sampling** — non-uniform time grids and channels observed at
  different times.
