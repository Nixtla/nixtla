IMAGE := tsbenchmarks
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PARENT_ROOT := $(shell dirname ${ROOT})

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

init:
	docker build -t ${IMAGE} .

run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \
		-v ${PARENT_ROOT}/data/:/opt/ml/processing/input/ \
		-v ${PARENT_ROOT}/data/output/:/opt/ml/processing/output/ \
		${IMAGE} ${module}

test_compute_benchmarks: 
	$(MAKE) run_module \
		module="benchmarks/compute_benchmarks.py \
			--dataset M4-Daily \
			--filename train/M4-forecasts-dummy.csv \
			--unique-id-column unique_id \
			--ds-column ds \
			--y-column y"

test_tsbenchmarks: test_compute_benchmarks


.require-module:
ifndef module
	$(error module is required)
endif
