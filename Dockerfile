FROM python:3.10.6-buster
#using the FROM command to select the base layer of our docker image
#this layer often contains naked operating system amongst other things

WORKDIR /app
#setting the working directory within the container

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
#copying in the the required packages listed out in requirements.txt
#into the docker image so that the docker container will have access to these packages
#when instantiated

COPY ./lateguru_ml /app/lateguru_ml
COPY ./model /app/model
COPY Makefile Makefile
COPY app.py app.py
COPY setup.py setup.py
COPY main.py main.py
COPY test_pipeline.py test_pipeline.py

CMD uvicorn lateguru_ml.api.fast:app
