ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))

init:
	virtualenv venv && \
		. venv/bin/activate && \
		pip install -r requirements.txt

app:
	. venv/bin/activate && \
		cd api && uvicorn main:app --reload

workflow:
	. ./workflows.sh 
