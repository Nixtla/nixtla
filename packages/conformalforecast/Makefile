.PHONY: devenv test test_integration test_extras lint

# `--group docs` is not optional here even though these targets do not build the
# site: the `normalize-doc-notebooks` pre-commit hook runs
# `scripts/execute_notebooks.py` as a system command, and nbformat/nbclient are
# declared only in that group. Without it `make lint` dies with
# ModuleNotFoundError while CI's Lint job stays green.
devenv:
	uv sync --quiet --all-extras --group dev --group docs
	uv run pre-commit install

test:
	uv run pytest tests/unit --cov=conformalforecast --cov-report=term-missing

# The unit suite is adapter-free by design, so CI runs it core-only across the
# matrix to prove that. The tests behind an extra (quantile-forest, utilsforecast)
# only execute with everything installed -- this is that run.
test_extras:
	uv run pytest tests/unit

# Two sessions on purpose: torch and scikit-learn each vendor a copy of libomp,
# and loading both into one process segfaults on macOS arm64. pytest imports
# every collected module before running anything, so the split has to be by
# collection scope -- see tests/integration/torch_isolated/__init__.py.
test_integration:
	uv run pytest tests/integration --ignore=tests/integration/torch_isolated
	uv run pytest tests/integration/torch_isolated

lint:
	uv run pre-commit run --all-files
	uv run mypy

# Documentation build targets
.PHONY: load_docs_scripts api_docs examples_docs format_docs brand_docs execute_docs test_docs all_docs

# Pin the docs tooling to an immutable commit so builds are reproducible and not
# exposed to unreviewed changes on the mutable `scripts` branch. Bump when updating.
DOCS_SCRIPTS_SHA := 487857a7c2de443365fe38841a4b29d7995bc168

load_docs_scripts:
	rm -rf docs-scripts
	git clone -b scripts https://github.com/Nixtla/docs.git docs-scripts --single-branch
	git -C docs-scripts checkout --quiet $(DOCS_SCRIPTS_SHA)

api_docs:
	python docs/to_mdx.py

examples_docs:
	mkdir -p nbs/_extensions
	rm -rf nbs/_extensions/mintlify
	cp -r docs-scripts/mintlify/ nbs/_extensions/mintlify
	python docs-scripts/update-quarto.py
	quarto render nbs --output-dir ../docs/mintlify/
	find docs/mintlify/docs -name "*.mdx" ! -name "*.html.mdx" -type f -exec sh -c 'mv "$$1" "$${1%.mdx}.html.mdx"' _ {} \;

# `docs-final-formatting.bash` hardcodes the `_docs` output directory and uses
# GNU `sed -i`, so it is rewritten to point at ours with perl, which behaves the
# same on macOS and Linux (BSD `sed -i` requires an argument GNU `sed` rejects).
format_docs:
	perl -pi -e 's|_docs|docs/mintlify|g' docs-scripts/docs-final-formatting.bash
	bash docs-scripts/docs-final-formatting.bash
	python docs-scripts/docs_replace_imgs.py --path docs/mintlify

# The site's logo and favicon are the shared Nixtlaverse marks, taken from the
# pinned docs-scripts commit rather than committed here as a second copy: they
# are identical across statsforecast, neuralforecast, mlforecast and
# hierarchicalforecast, and a duplicate would silently drift from the house
# assets. `docs.json` references them as /light.png, /dark.png and /favicon.svg.
# Runs last, after quarto has written the output directory.
brand_docs:
	cp docs-scripts/imgs/light.png docs-scripts/imgs/dark.png docs-scripts/imgs/favicon.svg docs/mintlify/

execute_docs:
	python scripts/execute_notebooks.py --include-network --write

test_docs:
	python scripts/execute_notebooks.py

all_docs: load_docs_scripts api_docs examples_docs format_docs brand_docs
