FROM continuumio/miniconda3:4.7.12

ARG ENV_YML=./environment.yml
ADD $ENV_YML ./environment.yml

RUN conda install -n base -c conda-forge mamba && \
    mamba env update -n base -f ./environment.yml && \
    conda clean -afy

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python"]
