IMAGE := apinixtla
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))


DOCKER_PARAMETERS := \
        --user $(shell id -u) \
        -v ${ROOT}:/app \
        -w /app \
        -e HOME=/tmp

init_api_docker:
	docker build -t ${IMAGE} .

create_api_zip: .require-route
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} bash -c \
		"rm -f /app/api.zip && \
		cd /usr/local/lib/python3.7/site-packages && \
		zip -r9 /app/api.zip . && \
		cd /app/api && zip -g /app/api.zip -r . && \
		cd /app/${route} && zip -g /app/api.zip -r ."

init:
	virtualenv venv && \
		. venv/bin/activate && \
		pip install -r requirements.txt

app:
	. venv/bin/activate && \
		cd api && uvicorn main:app --reload

workflow:
	./workflows.sh docker-image && ./workflows.sh lambda


init_dockers:
	for ROUTE in tsfeatures tsforecast tsbenchmarks tspreprocess; do \
		make -C $$ROUTE init; \
	done

.require-route:
ifndef route
	$(error route is required)
endif
