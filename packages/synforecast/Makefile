devenv:
	uv sync --quiet --dev --frozen
	uv run pre-commit install

init_codespace:
	npm install -g @anthropic-ai/claude-code
	git pull || true
	uv sync --quiet --dev --frozen

# Documentation build targets
.PHONY: load_docs_scripts api_docs examples_docs format_docs execute_docs test_docs all_docs

load_docs_scripts:
	rm -rf docs-scripts
	git clone -b scripts https://github.com/Nixtla/docs.git docs-scripts --single-branch

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
