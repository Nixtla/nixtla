IMAGE := nixtla-influxdb
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PARENT_ROOT := $(shell dirname ${ROOT})
PORT := 8888

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

init:
	docker build -t ${IMAGE} .

get_dataset:
	mkdir -p dataset/ && \
		wget https://raw.githubusercontent.com/influxdata/tg-brew-anomaly/master/data/temps.txt.gz -O dataset/temps.txt.gz && \
		gunzip dataset/temps.txt.gz 

run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \
		${IMAGE} ${module}

jupyter:
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p ${PORT}:8888 ${IMAGE} \
		bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

.require-module:
ifndef module
	$(error module is required)
endif
