# Changelog

All notable changes to ConformalForecast are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - Initial release

First public alpha. APIs are unstable until v0.2.0 and may change without a
deprecation cycle.

- Seven conformal methods: split conformal, CQR, weighted split, ACI, AgACI,
  NexCP and SPCI. Each is pure configuration; `fit`, `calibrate` and
  `calibrate_cv` return a fitted forecaster that serves `predict`, `interval`
  and, for online methods, `update`.
- Adapters for StatsForecast, MLForecast, NeuralForecast and plain callables.
- Covariate-shift support through `recalibrate` and `adapt` on fitted objects,
  with pluggable weighting functions.
- Coverage, interval-quality and conditional-coverage diagnostics.
- A migration shim for MLForecast's interval registry.
