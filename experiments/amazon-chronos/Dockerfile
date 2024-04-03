FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

ADD . /app/
WORKDIR /app

RUN python -m pip install "."
