# Nixtlaverse

Monorepo for Nixtla's Python forecasting stack. Each package under `packages/`
is an independent project with its own `pyproject.toml`, `uv.lock`, tests and
docs, and is released to PyPI on its own tag.

| Package | Path | Description |
|---|---|---|
| [`nixtla`](packages/nixtla) | `packages/nixtla` | TimeGPT SDK |
| [`statsforecast`](packages/statsforecast) | `packages/statsforecast` | Statistical & econometric models (C++) |
| [`neuralforecast`](packages/neuralforecast) | `packages/neuralforecast` | Deep-learning models (PyTorch) |
| [`mlforecast`](packages/mlforecast) | `packages/mlforecast` | Machine-learning forecasting |
| [`hierarchicalforecast`](packages/hierarchicalforecast) | `packages/hierarchicalforecast` | Hierarchical reconciliation (C++) |
| [`utilsforecast`](packages/utilsforecast) | `packages/utilsforecast` | Shared utilities |
| [`coreforecast`](packages/coreforecast) | `packages/coreforecast` | Fast core routines (C++/CMake) |

## Getting started

```sh
git clone --recurse-submodules https://github.com/Nixtla/nixtla.git
cd nixtla/packages/<package>
uv sync
uv run pytest
```

`--recurse-submodules` matters: `coreforecast`, `statsforecast` and
`hierarchicalforecast` build C++ extensions from vendored sources declared in
the root `.gitmodules`. If you already cloned, run
`git submodule update --init --recursive`.

## How the packages relate

```
coreforecast ─┬─► statsforecast
              ├─► neuralforecast
              └─► mlforecast
utilsforecast ─┬─► statsforecast, neuralforecast, mlforecast
               ├─► hierarchicalforecast
               └─► nixtla
```

Published packages depend on each other through ordinary PyPI version ranges.
For local development each dependent also declares a `[tool.uv.sources]` path
entry, so `uv sync` builds its in-repo siblings from source and a change to
`utilsforecast` is exercised by every package that uses it, before release.

## Releases

Tags are namespaced per package — `statsforecast/v2.1.1`, `mlforecast/v1.1.0`,
and so on — and each one triggers only that package's publish workflow.
`nixtla` keeps its historical unprefixed `vX.Y.Z` tags.

## Contributing

See the `CONTRIBUTING.md` inside the package you're changing. CI is scoped by
path, so a PR touching one package only runs that package's workflows.
