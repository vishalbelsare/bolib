FROM ubuntu:xenial

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-tk \
    gfortran \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash pyuser

USER pyuser

WORKDIR /app

ADD . /app

RUN python3 -m pip install --upgrade pip

################################
#TODO deal with DIRECT package #
RUN python3 -m pip install numpy
################################
RUN python3 -m pip install bolib

ENTRYPOINT ["python3"]
