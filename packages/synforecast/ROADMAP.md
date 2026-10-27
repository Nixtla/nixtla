# SynForecast Roadmap

## Augmentation

- **Optional moment matching** — augmented series are pinned to the source mean,
  std, and lag-1 ACF by construction; add `match_moments=False` and test whether
  unpinned should be the default. Highest priority here.
- **Strategy selection** — ~~unified API~~ — done: fit-and-simulate, mixup, MBB
  and DBA ship as frozen specs with `apply()`. Still open: the jitter/scale/warp
  baselines.
- **Wider fitter coverage** — 9 fitters against 32 generators, so only a third of
  the library is reachable; add ETS, StateSpace, INAR, multi-period seasonal.
- **Panel-aware augmentation** — augment jointly to preserve cross-series
  correlation instead of series-by-series.
- ~~**Exogenous passthrough**~~ — done: extra columns are carried through
  augmentation.
- **X-conditioned fit-and-simulate** — simulate the target driven by given
  exogenous inputs.
- **Fit diagnostics** — return ranked candidates with fit scores so a poor
  best-fit is visible rather than silent.
- **Rust batch path** — route fitted-generator simulation through it so
  augmentation scales like generation.

## Evaluation

- **Feature-space coverage** — shipped in `synforecast.evaluation`: a frozen
  twelve-feature `native_v1` schema and real-anchored PCA grid coverage. The
  benchmarks score coverage instead by full-feature nearest-neighbour distance
  at a radius calibrated on held-out real series; promote that score to the
  API so users can reproduce them. Then add tsfeatures/catch22 parity and
  per-feature attribution of uncovered regions. Benchmark scripts follow
  separately.
- **Seasonal coverage** — `seasonal_pool` is opt-in. Next: measure the
  forecasting utility of pool + `seasonal_pool` pretraining on a seasonal
  panel, and run a pre-declared test of a period-aware default (on only for
  periods 4 and 12).
- **Nearest-neighbour distance to real data** — promote the memorization check
  from notebook to API; it shares the machinery of calibrated coverage above.
- **Fidelity scores** — discriminative and predictive scores for comparing
  generation methods.
- **Wider benchmarks** — extend `when_synthetic_helps` beyond one model and
  dataset.

## Performance

- **Rust SARIMA exogenous support**, then drop the Python SARIMA path.
- **Rust batch dispatch** for `Copula`, `VAR`, `DailyActiveUsers`, `IoTSensor`.
- **Streaming generation** — `generate_values` and `members=` give
  DataFrame-free, per-draw generation in memory; next is chunked Parquet/Arrow
  output for corpora larger than memory.
- **TCM cost** — TCM takes over 80% of `pretraining_pool` generation time at
  every length from 128 to 8,192 steps, with 3 of about 60 entries; speed up
  its causal-model simulation.
- **Fresh-seed draws** — rebuilding a pool member with a new seed costs more
  than drawing its series; make per-draw reseeding cheaper than a full rebuild.
- **Long-series KernelSynth** — sampling is O(n^2) per kernel term but only
  about 11% of pool time at 8,192 steps; sample fast-decaying terms by
  circulant embedding if corpora get longer.

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
