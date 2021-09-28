IMAGE := tsfeatures
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

test_calendar: 
	$(MAKE) run_module \
		module="calendar/make_holidays.py \
			--filename train/target-dummy.csv \
			--country USA \
			--events calendar-holidays.txt \
			--unique-id-column item_id \
			--ds-column timestamp \
			--y-column demand"

test_features: 
	$(MAKE) run_module \
		module="features/make_features.py \
			--filename train/target-dummy.csv \
			--freq 7 \
			--kind static \
			--unique-id-column item_id \
			--ds-column timestamp \
			--y-column demand"

test_tsfeatures: test_calendar test_features


.require-module:
ifndef module
	$(error module is required)
endif
