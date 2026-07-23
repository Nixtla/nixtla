# Third-party notices and data provenance

SynForecast is licensed under Apache-2.0. The project implements most models
and numerical methods independently from the primary literature listed in
[`GENERATORS.md`](GENERATORS.md).

## KernelSynth and TSMixup

The KernelSynth and TSMixup recipes refer to the
[Chronos forecasting reference implementation](https://github.com/amazon-science/chronos-forecasting),
which is distributed under Apache-2.0. SynForecast's implementation and its
deviations are described in `GENERATORS.md` and the corresponding source
docstrings.

## AirPassengers documentation data

`nbs/data/air_passengers.csv` contains the Box–Jenkins monthly international
airline passenger totals distributed as `AirPassengers` in R's `datasets`
package. The source and original publication are documented in
[`nbs/data/README.md`](nbs/data/README.md). R distributes the `datasets`
package as part of the R distribution; consult the
[R licensing terms](https://www.r-project.org/Licenses/) for those source
terms. The file is used only in executable documentation and is not included
in SynForecast Python distributions.
