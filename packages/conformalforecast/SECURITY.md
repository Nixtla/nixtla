# Security Policy

## Supported versions

Security fixes are made on the latest released version and the `main` branch.
Older releases may not receive patches.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use either
private channel:

- GitHub's [private vulnerability reporting](https://github.com/Nixtla/conformalforecast/security/advisories/new),
  which keeps the report, the discussion, and the eventual advisory together; or
- email to `ops@nixtla.io`.

Include:

- the affected version or commit;
- steps to reproduce the issue;
- the expected impact; and
- any suggested mitigation, if available.

We will acknowledge the report, investigate it, and coordinate disclosure with
the reporter. Do not include secrets or personal data beyond what is needed to
reproduce the issue.

## Dependency scope

The published runtime package depends only on the packages listed under
`project.dependencies` in `pyproject.toml` — `narwhals`, `numpy`, `pydantic`
and `scipy`.
Forecasting libraries, quantile regressors, and plotting are optional extras
that are not installed by default. `uv.lock` also records development,
documentation, and integration dependencies that are not installed with the
published package. Dependency alerts are triaged against that scope, but
maintainers still update non-runtime dependencies or document why an upstream
advisory is not exploitable in the repository's workflows. Dependabot raises
those alerts and the version-update PRs behind them (see
`.github/dependabot.yml`), and CodeQL scans the Python source on every pull
request targeting `main`, every push to it, and weekly (see `.github/workflows/codeql.yml`). Code scanning is free
for public repositories but needs GitHub Advanced Security on private ones, so
that workflow skips itself until this repository is public.

## Untrusted inputs

ConformalForecast operates on numeric arrays and DataFrames. It does not
deserialize models, and documentation examples do not load untrusted model
checkpoints or serialized Python objects. Users should never load an untrusted
checkpoint into a wrapped forecaster: Python checkpoint formats may permit
arbitrary code execution during deserialization. The same applies to any
callable passed to `CallableAdapter`, which is invoked as-is.
