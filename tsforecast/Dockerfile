FROM python:3.8-buster

ADD ./requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

COPY ./forecast/make_forecast.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py
