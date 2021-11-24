FROM python:3.7.12-slim-buster

ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update \
	&& apt-get install zip -y
