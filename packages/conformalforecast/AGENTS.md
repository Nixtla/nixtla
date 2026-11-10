# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, Gemini, Copilot,
Jules, …) working in this repository. Humans should read
[CONTRIBUTING.md](CONTRIBUTING.md) instead — it says all of this at length.

This file is the short list of rules that are easy to break and expensive to
undo. It is not a substitute for the architecture guide: read
[CLAUDE.md](CLAUDE.md) before writing code, whichever tool you are.

## Branching and PRs

- **Branch from `main`. Target `main`. Never stack PRs.** A PR based on another
  unmerged branch is rejected by the `PR targets main` CI check. If the user asks
  you to split work, create *independent* branches off `main`, not a chain.
- If a change genuinely depends on unmerged work, say so and stop. Do not build
  the stack anyway. Either the prerequisite merges first, or both changes go in
  one PR.
- Never rewrite published history: no force-push to `main`, no amending a commit
  someone else may have pulled.
- One PR, one change. Do not bundle a drive-by refactor with a fix.

## Opening a PR

Every PR runs these checks; get them right when you open it, not after CI
fails:

- **`PR title`** — a conventional commit, `<type>[scope][!]: <description>`
  with type one of `feat fix docs test refactor perf ci build chore revert`.
  The merge commit takes its subject from the title.
- **`Changelog label`** — exactly one of `breaking change`, `enhancement`
  (`feat`), `bug` (`fix`), `documentation` (`docs`), `maintenance`
  (`ci`/`test`/`refactor`/`chore`), `dependencies`, or `skip-changelog`.
  Apply it at creation: `gh pr create --label <label> ...`.
- **`PR targets main`** — see above.
- **`CI`** — unit tests across ubuntu/macos/windows × every supported Python,
  unit tests with every extra, the lowest-direct-dependency resolution,
  integration tests and the sdist build. Locally: `make test`,
  `make test_extras`, `make test_integration`.
- **`Lint`** — pre-commit on all files plus mypy. Locally: `make lint`.
- **`Build Docs`** — executes the docs notebooks, builds the site, checks
  `docs.json` navigation and runs `mint validate`. Run `make test_docs` and
  `uv run python scripts/check_docs_nav.py` when you touch `nbs/`, `docs/` or
  a public docstring.
- **`CodeQL`** — static analysis; nothing to run locally.

Fill in [the PR template](.github/PULL_REQUEST_TEMPLATE.md).

## Before you commit

- Run the tests. Do not report a result you have not seen — `make test` for the
  fast suite, `make lint` for ruff plus mypy.
- If tests fail, say so with the output. A failing suite reported as passing is
  worse than no report.
- Do not commit or push unless the user asked you to.

## What not to leave behind

- No references to the development process in code, comments, docstrings or
  docs: no PR or phase numbers, no plan or spec item ids, no branch names, no
  "in a follow-up PR". A reader who never saw the plan must not be left
  wondering what it meant.
- No agent tooling in commits. Your tool's config and scratch directories are
  gitignored; leave them untracked rather than staging them with `git add -A`.
- No `TODO(agent)` markers or commented-out code as a substitute for finishing
  the work.

## Scope

- Do what was asked. Flag problems you find outside that scope rather than
  fixing them uninvited.
- Read [ROADMAP.md](ROADMAP.md) before concluding something is broken or
  proposing a feature. Known defects and deliberate scope boundaries are
  recorded there, each reproduced against this code — including several that
  look like bugs and are not.
- Ask before adding a dependency. Core deps are `narwhals`, `numpy`, `pydantic`
  and `scipy`; everything else is an extra.

## The invariants most often broken

These are the ones an agent gets wrong without noticing. Each is explained in
[CLAUDE.md](CLAUDE.md):

- The array surface (`fit` / `interval` / `update`) takes numpy arrays and
  plain-Python config only — never an adapter, DataFrame or library type.
- A method is config only. `fit` / `calibrate` / `calibrate_cv` **return** a
  new fitted object and never assign to `self`; `interval`, `update`,
  `predict`, `recalibrate` and `adapt` live on that fitted object.
- The score is a constructor field (`score=`), never a `fit` / `calibrate*`
  argument.
- On a fitted class, override `_update`, never `update`. Overriding `update`
  and calling `super().update()` is infinite recursion. `update` mutates in
  place; snapshots are `model_copy(deep=True)`.
- Never override `calibrate`, `calibrate_cv`, `predict` or `recalibrate`.
- Every conformal quantile comes from `conformalforecast._quantiles`, never
  `np.quantile`.
- No pandas, polars or narwhals inside `methods/`, `nonconformity/`, `diagnostics/`,
  `panel.py` or `_quantiles.py`.
- Multi-horizon is mandatory. A method that only works for `horizon=1` is wrong.
