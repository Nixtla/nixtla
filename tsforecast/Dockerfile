FROM continuumio/miniconda3:4.7.12

ARG ENV_YML=./environment.yml
ADD $ENV_YML ./environment.yml

RUN conda install -n base -c conda-forge mamba && \
    mamba env update -n base -f ./environment.yml && \
    conda clean -afy

COPY ./forecast/make_forecast.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py 
