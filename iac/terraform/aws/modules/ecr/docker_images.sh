#! /bin/bash
set -e

cd ../../../../../

# create docker images
make create_docker_image src_docker_image=$SRC_DOCKER_IMAGE.dkr.ecr.$AWS_REGION.amazonaws.com route=$ROUTE