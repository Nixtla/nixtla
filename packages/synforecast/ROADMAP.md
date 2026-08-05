# SynForecast Roadmap

## Generators

- **MAR** ([GRATIS, Kang et al. 2020](https://arxiv.org/abs/1903.02787)) — sample
  from randomly parameterized mixtures of AR components, for pretraining breadth.
- **Feature-targeted generation** (also GRATIS) — tune MAR parameters to hit
  requested feature values; depends on the feature-space module below.
- **MBB** ([Bergmeir et al. 2016](https://doi.org/10.1016/j.ijforecast.2015.07.002),
  [Bandara et al. 2021](https://arxiv.org/abs/2008.02663)) — block-bootstrap the
  STL remainder; non-parametric, so it needs no fitter. Ships as a `SynAugment`
  strategy.
- **DBA** ([Petitjean et al. 2011](https://doi.org/10.1016/j.patcog.2010.09.013),
  augmentation use in [Forestier et al. 2017](https://doi.org/10.1109/ICDM.2017.106)
  and [Bandara et al. 2021](https://arxiv.org/abs/2008.02663)) — average a series
  with its DTW nearest neighbours; cross-series, so it needs the panel API below.

## Augmentation

- **Optional moment matching** — augmented series are pinned to the source mean,
  std, and lag-1 ACF by construction; add `match_moments=False` and test whether
  unpinned should be the default. Highest priority here.
- **Strategy selection** — `SynAugment(strategy=...)` over fit-and-simulate,
  `mixup`, `mbb`, `dba`, and the jitter/scale/warp baselines.
- **Wider fitter coverage** — 9 fitters against 31 generators, so only a third of
  the library is reachable; add ETS, StateSpace, INAR, multi-period seasonal.
- **Panel-aware augmentation** — augment jointly to preserve cross-series
  correlation instead of series-by-series.
- **Exogenous passthrough** — carry `X` columns through augmentation.
- **Fit diagnostics** — return ranked candidates with fit scores so a poor
  best-fit is visible rather than silent.
- **Rust batch path** — route fitted-generator simulation through it so
  augmentation scales like generation.

## Evaluation

- **Feature-space coverage** — tsfeatures/catch22 diversity and coverage metrics;
  prerequisite for feature-targeted generation.
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
  of reading all 31 pages.

## Under consideration

- **Deep generative models** (GAN/VAE/diffusion) — heavy deps, GPU time, and
  per-dataset training conflict with cheap deterministic CPU generation;
- **Irregular sampling** — non-uniform time grids and channels observed at
  different times.