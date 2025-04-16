devenv:
	uv venv
	. .venv/bin/activate; uv pip install -Ue .[dev,distributed]
	. .venv/bin/activate; pre-commit install
	. .venv/bin/activate; nbdev_install_hooks


jupyter:
	mkdir -p tmp
	jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'
