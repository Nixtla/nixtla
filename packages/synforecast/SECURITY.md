# Security Policy

## Supported versions

Security fixes are made on the latest released version and the `main` branch.
Older releases may not receive patches.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Report it
privately to `ops@nixtla.io` with:

- the affected version or commit;
- steps to reproduce the issue;
- the expected impact; and
- any suggested mitigation, if available.

We will acknowledge the report, investigate it, and coordinate disclosure with
the reporter. Do not include secrets or personal data beyond what is needed to
reproduce the issue.

## Dependency scope

The published runtime package depends only on the packages listed under
`project.dependencies` in `pyproject.toml`. `uv.lock` also records development,
documentation, and integration dependencies that are not installed with the
published package. Dependency alerts are triaged against that scope, but
maintainers still update non-runtime dependencies or document why an upstream
advisory is not exploitable in the repository's workflows.

Documentation examples do not load untrusted model checkpoints or serialized
Python objects. Users should never load an untrusted checkpoint: Python
checkpoint formats may permit arbitrary code execution during deserialization.
