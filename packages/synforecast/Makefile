devenv:
	uv sync --quiet --dev --frozen
	uv run pre-commit install

init_codespace:
	npm install -g @anthropic-ai/claude-code
	git pull || true
	uv sync --quiet --dev --frozen

# Regenerate the bundled third-party Rust license notices. Requires cargo-about
# (`cargo install cargo-about --features cli`). Run after changing rust/Cargo.toml.
rust_licenses:
	cd rust && cargo about generate about.hbs -o ../synforecast/THIRD_PARTY_RUST.md

# Documentation build targets
.PHONY: load_docs_scripts api_docs examples_docs format_docs execute_docs test_docs all_docs

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

format_docs:
	sed -i -e 's|_docs|docs/mintlify|g' docs-scripts/docs-final-formatting.bash
	bash docs-scripts/docs-final-formatting.bash
	python docs-scripts/docs_replace_imgs.py --path docs/mintlify

execute_docs:
	python scripts/execute_notebooks.py --include-network --write

test_docs:
	python scripts/execute_notebooks.py

all_docs: load_docs_scripts api_docs examples_docs format_docs
