SRC_DIR := data
EXCLUDE_STRINGS := catalogue
TS_FILES := $(filter-out $(wildcard $(SRC_DIR)/*$(foreach str,$(EXCLUDE_STRINGS),*$(str)*)), $(wildcard $(SRC_DIR)/*.parquet))

evaluate: .require-method
	@echo "Evaluation for $${method}..."
	@for file in $(TS_FILES); do \
		echo $$file; \
		python -m src.$${method}_exp --file $$file; \
	done
	@echo "Evaluation for $${method} complete."

summarize_results:
	@echo "Summarize results..."
	@python -m src.results_summary --dir ./data/results/
	@echo "Summarize results complete."

.require-method:
ifndef method
	$(error method is required)
endif
