FROM python:3.10

ENV PIP_DEFAULT_TIMEOUT=1000    

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt