FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt update && apt install -y python3 python3-pip git

COPY requirements.txt .
RUN pip install -r requirements.txt