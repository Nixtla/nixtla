TS_FILES := Hourly_H.parquet Daily_D.parquet Weekly_W-MON.parquet Monthly_MS.parquet 
FILTERED_TS_FILES := $(patsubst %,./data/filtered_datasets/%,$(TS_FILES))

filter_data:
	@for file in $(TS_FILES); do \
		python -m src.utils.filter_data --dataset_path ./data/$$file; \
	done

run_timegpt: .require-dataset_path
	@echo Running TimeGPT with dataset_path=$(dataset_path)
	@python -m src.nixtla_timegpt --dataset_path $(dataset_path)

run_sn: .require-dataset_path
	@echo Running SN with dataset_path=$(dataset_path)
	@python -m src.statsforecast_sn --dataset_path $(dataset_path)

run_automl: .require-dataset_path
	@echo Running AutoML with dataset_path=$(dataset_path)
	@python -m src.azure_automl.forecasting --dataset_path $(dataset_path)

run_methods:
	@for file in $(TS_FILES); do \
		echo "Running methods for $$file"; \
		$(MAKE) run_timegpt dataset_path=./data/filtered_datasets/$$file; \
		$(MAKE) run_sn dataset_path=./data/filtered_datasets/$$file; \
		$(MAKE) run_automl dataset_path=./data/filtered_datasets/$$file; \
	done

download_automl_forecasts:
	@python -m src.azure_automl.download_forecasts

evaluate_experiments:
	@python -m src.evaluation --datasets_paths "$(shell echo $(FILTERED_TS_FILES) | tr ' ' ',')"

.require-dataset_path:
ifndef dataset_path
	$(error dataset_path is required)
endif

