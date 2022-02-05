FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ENV DEV="git curl wget build-essential gcc g++ make libffi-dev libssl-dev zlib1g-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev libgl1-mesa-dev libopencv-dev python3 python3-pip tmux"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install $DEV

RUN apt search python-is-python3 \
    && apt-get -y install python-is-python3 \
    && apt-get -y install python3-distutils

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-distutils \
    python3.8 \
    python3.8-dev \
    python3-pip && \
    wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py

RUN ln -sf /usr/bin/python3.8 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.8 /usr/local/bin/python && \
    python -m pip --no-cache-dir install --upgrade setuptools

RUN /usr/bin/python3.8 -m pip install --upgrade pip

RUN pip install poetry

WORKDIR /app

COPY poetry.lock pyproject.toml Makefile ./

RUN make init