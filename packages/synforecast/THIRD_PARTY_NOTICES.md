# Third-party notices and data provenance

SynForecast is licensed under Apache-2.0. The mathematical references and
implementation deviations are listed in [`GENERATORS.md`](GENERATORS.md).
Algorithm attribution and source-code provenance are separate questions.

## MAR, MBB, DBA, and feature kernels (PR #55)

The initial Python implementations entered in commit `44eb088`; native ports
entered in `5769a5b`. Review compared the MAR sampler/search with the
[GRATIS R and Python sources](https://github.com/ykang/gratis), and DBA with
[Petitjean's implementation](https://github.com/fpetitjean/DBA/blob/master/DBA.py).
The comparison covered GRATIS's `R/mar_model.R`, `R/generate_ts_with_target.R`,
`R/simulate_target.R`, `python/gratis/mar.py`, `python/gratis/target.py`, and
Petitjean's `DBA.py`.
Both reference repositories declare GPL-3.0. The inspected code differs in
parameter sampling/search and in DBA initialization, weighted updates, and
alignment storage. No substantial copied block was identified in this spot
check. These repositories are comparison sources, not bundled dependencies.

This is **not a certification of independent authorship**: the local history
does not record the inputs used to generate the original implementations.
Historical sourcing remains unverified for the new Python and Rust code,
including the FFT additions. Confirmation from the original implementation
author, or identification and review of any adapted source, is still needed
to close that question. The new validation tests derive expected results from
explicit arithmetic, exhaustive paths, or statistical identities; no external
test implementation was copied in this review.

## Bundled Rust dependencies

Prebuilt SynForecast wheels contain a compiled Rust extension
(`synforecast._lib`) that statically links third-party Rust crates released
under permissive licenses (Apache-2.0, MIT, BSD, Zlib, Unicode, and similar).
The complete license texts for those crates are generated with
[`cargo-about`](https://github.com/EmbarkStudios/cargo-about) and bundled in
each wheel as `synforecast/THIRD_PARTY_RUST.md`. Regenerate that file with
`make rust_licenses` after changing `rust/Cargo.toml`.

RealFFT 3.5.0 declares MIT in its published manifest and README, but the crate
contains no license file. The generated notice consequently uses generic MIT
text with placeholder copyright fields. This limitation is stated in the
notice template as well; no copyright holder or year has been invented.
The source snapshot is identified by the crate's `.cargo_vcs_info.json` as
[`d0d4eee`](https://github.com/HEnquist/realfft/tree/d0d4eee0525fd27c96c8a046d6d107acd5ed84a6).
An authoritative copyright notice from upstream remains unresolved. The new
transitive crates (`rustfft`, `primal-check`, `strength_reduce`, `transpose`)
are included in the generated notices.

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
