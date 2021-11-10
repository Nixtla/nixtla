FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unrar \
    unzip \
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

