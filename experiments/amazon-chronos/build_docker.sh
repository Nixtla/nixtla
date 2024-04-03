#!/bin/bash
set -euxo pipefail

TAG=${1:-"nixtla-eval"}

export METAFLOW_BATCH_CONTAINER_REGISTRY=$(cat ~/.metaflowconfig/config.json | jq -r .METAFLOW_BATCH_CONTAINER_REGISTRY)
export METAFLOW_BATCH_CONTAINER_IMAGE=$(cat ~/.metaflowconfig/config.json | jq -r .METAFLOW_BATCH_CONTAINER_IMAGE)

docker build -t $METAFLOW_BATCH_CONTAINER_REGISTRY/$METAFLOW_BATCH_CONTAINER_IMAGE:$TAG .
aws ecr get-login-password --region $(aws configure get region) | docker login --username AWS --password-stdin $METAFLOW_BATCH_CONTAINER_REGISTRY
docker push $METAFLOW_BATCH_CONTAINER_REGISTRY/$METAFLOW_BATCH_CONTAINER_IMAGE:$TAG
