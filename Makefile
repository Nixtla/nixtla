ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

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


create_api_zip:
	if [ -d "$(ROOT)/package" ]; then rm -Rf $(ROOT)/package; fi && mkdir $(ROOT)/package
	if [ -f "$(ROOT)/api.zip" ]; then rm $(ROOT)/api.zip; fi
	docker run --rm \
		--user $(CURRENT_UID):$(CURRENT_GID) \
		--volume="/etc/group:/etc/group:ro" \
		--volume="/etc/passwd:/etc/passwd:ro" \
		--volume="/etc/shadow:/etc/shadow:ro" \
		--volume $(ROOT)/requirements.txt:/tmp/requirements.txt \
		--volume $(ROOT)/package:/tmp/package \
		python:3.7 pip install -r /tmp/requirements.txt --target /tmp/package
	cd ./package && zip -r9 ../api.zip . && cd ..
	cd ./api && zip ../api.zip . -r && cd ..
	cp api.zip tsbenchmarks_api.zip
	cd ./tsbenchmarks && zip ../tsbenchmarks_api.zip . -r && cd ..
	cp api.zip tsfeatures_api.zip
	cd ./tsfeatures && zip ../tsfeatures_api.zip . -r && cd ..
	cp api.zip tsforecast_api.zip
	cd ./tsfeatures && zip ../tsfeatures_api.zip . -r && cd ..
	cp api.zip tspreprocess_api.zip
	cd ./tspreprocess && zip ../tspreprocess_api.zip . -r && cd ..

clean_api_zip:
	if [ -d "$(ROOT)/package" ]; then rm -Rf $(ROOT)/package; fi
	if [ -f "$(ROOT)/api.zip" ]; then rm $(ROOT)/api.zip; fi
	if [ -f "$(ROOT)/tsbenchmarks_api.zip" ]; then rm $(ROOT)/tsbenchmarks_api.zip; fi
	if [ -f "$(ROOT)/tsfeatures_api.zip" ]; then rm $(ROOT)/tsfeatures_api.zip; fi
	if [ -f "$(ROOT)/tsforecast_api.zip" ]; then rm $(ROOT)/tsforecast_api.zip; fi
	if [ -f "$(ROOT)/tspreprocess_api.zip" ]; then rm $(ROOT)/tspreprocess_api.zip; fi

upload_api_zip:
	aws s3 cp tsbenchmarks_api.zip s3://${s3_bucket}/functions/tsbenchmarks/api.zip
	aws s3 cp tsfeatures_api.zip s3://${s3_bucket}/functions/tsfeatures/api.zip
	aws s3 cp tsforecast_api.zip s3://${s3_bucket}/functions/tsforecast/api.zip
	aws s3 cp tspreprocess_api.zip s3://${s3_bucket}/functions/tspreprocess/api.zip

create_docker_image:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${src_docker_image}
	docker build -t ${src_docker_image}/${route}:latest ./${route}
	docker push ${src_docker_image}/${route}:latest
	