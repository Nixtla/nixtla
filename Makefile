devenv:
	uv sync --quiet --all-groups --all-extras --frozen
	uv run --no-sync pre-commit install

init_codespace:
	npm install -g @anthropic-ai/claude-code@1.0.127
	npm i -g mint
	git pull || true
	uv sync --quiet --all-groups --all-extras --frozen

jupyter:
	mkdir -p tmp
	jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'


load_docs_scripts:
# 	load processing scripts
	if [ ! -d "docs-scripts" ] ; then \
		git clone -b scripts https://github.com/Nixtla/docs.git docs-scripts --single-branch; \
	fi

api_docs:
	lazydocs .nixtla --no-watermark
	python docs/to_mdx.py

examples_docs:
	mkdir -p nbs/_extensions
	cp -r docs-scripts/mintlify/ nbs/_extensions/mintlify
	quarto render nbs --output-dir ../docs/mintlify/

format_docs:
	# replace _docs with docs
	sed -i -e 's/_docs/docs/g' ./docs-scripts/docs-final-formatting.bash
	bash ./docs-scripts/docs-final-formatting.bash
	find docs/mintlify -name "*.mdx" -exec sed -i.bak '/^:::/d' {} + && find docs/mintlify -name "*.bak" -delete
	find docs/mintlify -name "*.mdx" -exec sed -i.bak 's/<support@nixtla\.io>/\\<support@nixtla.io\\>/g' {} + && find docs/mintlify -name "*.bak" -delete

preview_docs:
	cd docs/mintlify && mintlify dev

clean:
	rm -f docs/*.md
	find docs/mintlify -name "*.mdx" -exec rm -f {} +


all_docs: load_docs_scripts api_docs examples_docs format_docs

licenses:
	pip-licenses --format=csv --with-authors --with-urls > third_party_licenses.csv
	python scripts/filter_licenses.py
	rm -f third_party_licenses.csv
	@echo "✓ THIRD_PARTY_LICENSES.md updated"

lint:
	@echo "Running pre-commit hooks..."
	uv run pre-commit run --show-diff-on-failure --files nixtla/*

format:
	@echo "Running formatter on staged files..."
	@git diff --cached --name-only --diff-filter=ACMR | grep '\.py$$' | xargs -r uv run ruff format

deploy-snowflake:
	@echo "Deploying Nixtla components to Snowflake..."
	uv run python -m nixtla.scripts.snowflake_install_nixtla