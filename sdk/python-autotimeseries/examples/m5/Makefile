init:
	virtualenv venv && \
		. venv/bin/activate && \
		pip install -r requirements.txt

prepare_data: 
	. venv/bin/activate && \
		mkdir -p data && \
		for FORMAT in parquet csv; do \
			python -m src.download_data --directory data --output-format $$FORMAT; \
		done

jupyter:
	. venv/bin/activate && jupyter lab
