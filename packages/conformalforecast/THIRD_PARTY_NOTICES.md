# Third-party notices and data provenance

ConformalForecast is licensed under Apache-2.0. The project vendors no
third-party source code. Every conformal method is implemented independently
from the primary literature; the notices below record where the ideas, the
interfaces, and the documentation data come from.

## Methods implemented from the literature

The calibration procedures in `src/conformalforecast/methods/` are original
implementations of published methods. The papers describe the algorithms; they
are not a source of code. Deviations from a paper are documented in the
corresponding method docstring.

| Module | Method | Reference |
|---|---|---|
| `split.py` | Split (inductive) conformal prediction | Papadopoulos, Proedrou, Vovk & Gammerman (2002); Lei et al. (2018) |
| `cqr.py` | Conformalized Quantile Regression | Romano, Patterson & Candès (2019) |
| `weighted_split.py` | Weighted conformal prediction under covariate shift | Tibshirani, Barber, Candès & Ramdas (2019) |
| `aci.py` | Adaptive Conformal Inference | Gibbs & Candès (2021) |
| `agaci.py` | Aggregated Adaptive Conformal Inference | Zaffran, Féron, Goude, Josse & Dieuleveut (2022) |
| `nexcp.py` | Conformal prediction beyond exchangeability | Barber, Candès, Ramdas & Tibshirani (2023) |
| `spci.py` | Sequential Predictive Conformal Inference | Xu & Xie (2023) |

The `EWA` aggregator in `src/conformalforecast/aggregators/` implements
exponentially weighted averaging as used by AgACI.

## Runtime dependencies

The published package depends only on `narwhals` (MIT), `numpy` (BSD-3-Clause),
`pydantic` (MIT), and `scipy` (BSD-3-Clause). No third-party
code from these projects is copied into this repository.

## Optional dependencies

Optional extras install, but do not bundle, the following projects. Each is
distributed under its own license by its own maintainers.

- `nixtla`: [statsforecast](https://github.com/Nixtla/statsforecast),
  [mlforecast](https://github.com/Nixtla/mlforecast),
  [neuralforecast](https://github.com/Nixtla/neuralforecast), and
  [utilsforecast](https://github.com/Nixtla/utilsforecast) (Apache-2.0)
- `spci`: [quantile-forest](https://github.com/zillow/quantile-forest) (Apache-2.0)
- `plotting`: [matplotlib](https://github.com/matplotlib/matplotlib) (matplotlib license, BSD-compatible)

## Interface compatibility

`src/conformalforecast/compat/mlforecast_shim.py` targets the interval-method
registry of [mlforecast](https://github.com/Nixtla/mlforecast) (Apache-2.0). It
matches that registry's call signature so a method class can be registered as
an interval method, but its implementation is original. No mlforecast source
is copied.

The adapters in `src/conformalforecast/adapters/` consume the Nixtla long
format (`unique_id`, `ds`, `y`) and the public APIs of the libraries they wrap.
That is interface use, not code reuse.

## AirPassengers documentation data

`nbs/data/air_passengers.csv` contains the classic monthly international
airline passenger totals, January 1949 – December 1960, first published in Box,
Jenkins & Reinsel, *Time Series Analysis: Forecasting and Control* (1976, Series
G), and distributed as `AirPassengers` in R's base `datasets` package (licensed
GPL-2 | GPL-3). The values are factual measurements and are reproduced in
numerous open-source packages. The file is used only in executable
documentation and is **not** included in ConformalForecast Python distributions:
the wheel packages `src/conformalforecast` only, and
`[tool.hatch.build.targets.sdist]` excludes `nbs/` from the source
distribution. Provenance is documented in
[`nbs/data/README.md`](nbs/data/README.md).

## M4 hourly documentation data

`nbs/data/m4_hourly_panel.csv` contains eight hourly series (`H1`, `H10`, `H50`,
`H100`, `H150`, `H200`, `H300`, `H400`) from the M4 forecasting competition,
described in Makridakis, Spiliotis & Assimakopoulos, *The M4 Competition:
100,000 time series and 61 forecasting methods*, International Journal of
Forecasting 36(1), 2020. The M4 dataset is published under CC BY 4.0. The subset
was derived from `https://datasets-nixtla.s3.amazonaws.com/m4-hourly.csv`, the
same file that backs the public statsforecast, mlforecast and neuralforecast
documentation, by selecting those eight series, truncating each to its last
504–672 observations, and re-basing the integer time index onto hourly
timestamps with a shared end date. Like the AirPassengers file it is used only
in executable documentation and is **not** included in ConformalForecast Python
distributions. Provenance and the derivation steps are documented in
[`nbs/data/README.md`](nbs/data/README.md).

Any other data used in the documentation is generated at build time with
`numpy.random.default_rng`.
