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
		-v ${PARENT_ROOT}/data/:/opt/ml/input/data/train/ \
		-v ${PARENT_ROOT}/data/output/:/opt/ml/output/data/ \
		${IMAGE} ${module}

test_tsforecast: 
	$(MAKE) run_module \
		module="python /opt/ml/code/train.py --filename train/target-dummy.parquet \
			--unique-id-column item_id \
			--ds-column timestamp \
			--y-column demand"

.require-module:
ifndef module
	$(error module is required)
endif
