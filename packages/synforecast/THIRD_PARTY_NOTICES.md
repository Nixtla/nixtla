# Third-party notices and data provenance

SynForecast is licensed under Apache-2.0. The project implements most models
and numerical methods independently from the primary literature listed in
[`GENERATORS.md`](GENERATORS.md).

## Bundled Rust dependencies

Prebuilt SynForecast wheels contain a compiled Rust extension
(`synforecast._lib`) that statically links third-party Rust crates released
under permissive licenses (Apache-2.0, MIT, BSD, Zlib, Unicode, and similar).
The complete license texts for those crates are generated with
[`cargo-about`](https://github.com/EmbarkStudios/cargo-about) and bundled in
each wheel as `synforecast/THIRD_PARTY_RUST.md`. Regenerate that file with
`make rust_licenses` after changing `rust/Cargo.toml`.

## KernelSynth and TSMixup

The KernelSynth and TSMixup recipes refer to the
[Chronos forecasting reference implementation](https://github.com/amazon-science/chronos-forecasting),
which is distributed under Apache-2.0. SynForecast's implementation and its
deviations are described in `GENERATORS.md` and the corresponding source
docstrings.

## AirPassengers documentation data

`nbs/data/air_passengers.csv` contains the classic monthly international
airline passenger totals first published in Box & Jenkins, *Time Series
Analysis: Forecasting and Control* (1976), and distributed as `AirPassengers`
in R's base `datasets` package (licensed GPL-2 | GPL-3). The values are factual
measurements and are reproduced in numerous open-source packages. The file is
used only in executable documentation and is **not** included in SynForecast
Python distributions (sdist or wheel). Provenance is documented in
[`nbs/data/README.md`](nbs/data/README.md).
