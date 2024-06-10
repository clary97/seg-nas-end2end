FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV PIP_DEFAULT_TIMEOUT=1000

RUN apt-get update && apt-get install -y libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY . /seg-nas-end2end/segmentation_nas
WORKDIR /seg-nas-end2end/segmentation_nas

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

CMD ["python", "train.py"]