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