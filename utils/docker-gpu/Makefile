IMAGE := gpucontainer
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PARENT_ROOT := $(shell dirname ${ROOT})
PORT := 8888

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	--gpus all \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

init:
	docker build -t ${IMAGE} .

install_nvidia_docker2:
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   		&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   		&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
		&& sudo apt-get update \
		&& sudo apt-get install -y nvidia-docker2 \
		&& sudo systemctl restart docker

jupyter:
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p ${PORT}:8888 ${IMAGE} \
		bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \
		${IMAGE} ${module}

bash_docker:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE}

.require-module:
ifndef module
	$(error module is required)
endif
