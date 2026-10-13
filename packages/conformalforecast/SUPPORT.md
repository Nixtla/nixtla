# Support

## Usage questions

Ask usage and integration questions in the
[Nixtla community Slack](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ).
Include the ConformalForecast version, Python version, the forecasting library
and adapter you are using, operating system, and a minimal example when
possible.

For calibration problems, the two most useful details are the `alpha` you asked
for and the `calibration_` attribute of the object `calibrate` or
`calibrate_cv` returns — its `n_calibration_samples` and `diagnostics` fields
usually explain an interval that looks wrong.

## Bugs and feature requests

Use [GitHub Issues](https://github.com/Nixtla/conformalforecast/issues) for
reproducible bugs, documentation problems, and feature requests.

Coverage that misses its nominal rate is not automatically a bug: conformal
coverage is marginal rather than conditional, and adaptive methods trade
finite-sample exactness for robustness to distribution shift. A report is most
actionable when it includes a seeded script and the coverage you measured
against the coverage you expected.

ConformalForecast is an alpha open-source project. Maintainers prioritize
security issues, regressions, and broadly useful fixes, but do not guarantee a
response time or implementation timeline.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, testing
conventions, and the procedure for adding a new method or adapter.

## Security

Do not report vulnerabilities in a public issue. Follow the private reporting
instructions in [SECURITY.md](SECURITY.md).
