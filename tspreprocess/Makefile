IMAGE := tspreprocess
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

test_onehot: 
	$(MAKE) run_module \
		module="preprocess/make_preprocess.py \
			--filename train/static-dummy.csv \
			--kind onehot \
			--unique-id-column item_id \
			--ds-column timestamp \
			--y-column demand"

test_balance: 
	$(MAKE) run_module \
		module="preprocess/make_preprocess.py \
			--filename train/target-unbalanced-dummy.csv \
			--kind balance \
			--unique-id-column item_id \
			--ds-column timestamp \
			--y-column demand"
	$(MAKE) run_module \
		module="preprocess/test_balanced_data.py \
			--filename1 /opt/ml/processing/output/balance-preprocessed.csv \
		        --filename2 /opt/ml/processing/input/train/target-dummy.csv"	
	
test_tspreprocess: test_onehot test_balance


.require-module:
ifndef module
	$(error module is required)
endif
