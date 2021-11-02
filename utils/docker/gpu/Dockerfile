FROM nvidia/cuda:10.2-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    unzip \
    libnvinfer6=6.0.1-1+cuda10.2 \
    libnvinfer-dev=6.0.1-1+cuda10.2 \
    libnvinfer-plugin6=6.0.1-1+cuda10.2 \
    git && \
    apt-get clean -y

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b  && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -n base -c conda-forge mamba

ADD ./environment.yml ./environment.yml
RUN mamba env update -n base -f ./environment.yml && \
    conda clean -afy

