# Base Images
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN apt-get update -y
RUN apt-get install -y vim tmux openssh-server git
RUN pip install redis ujson msgpack h5py transformers accelerate
RUN pip install setproctitle pandas scikit-learn
RUN pip install ruamel.yaml

RUN pip install record
RUN pip install wandb nltk datasets