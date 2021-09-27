IMAGE := tsforecast
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

test_tsforecast: 
	$(MAKE) run_module \
		module="python forecast/make_forecast.py \
			--filename train/target-dummy.csv \
			--filename-static train/static-dummy.csv \
			--filename-temporal train/temporal-dummy.csv \
			--filename-temporal-future test/temporal-future-dummy.csv \
			--backtest-windows 2 \
			--unique-id-column item_id \
			--ds-column timestamp \
			--y-column demand"

.require-module:
ifndef module
	$(error module is required)
endif
