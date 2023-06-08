FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y python3-pip && apt-get install -y libsndfile1 libsndfile1-dev
RUN apt-get update && apt-get install -y git
ADD requirements.txt /QDL/requirements.txt
WORKDIR /QDL/
RUN python3 -m pip install --no-cache-dir -r requirements.txt
WORKDIR /
RUN git clone https://github.com/mit-han-lab/torchquantum
WORKDIR /torchquantum
RUN python3 -m pip install -r requirements.txt && python3 -m pip install -e .
WORKDIR /QDL
ADD src/qdl/data /QDL/src/qdl/data
ADD src/qdl/datasets /QDL/src/qdl/datasets
ADD src/qdl/models /QDL/src/qdl/models
ADD src/qdl/training /QDL/src/qdl/training
ADD src/qdl/utils /QDL/src/qdl/utils
ADD src/qdl/layers /QDL/src/qdl/layers
ADD src/qdl/__init__.py /QDL/src/qdl/__init__.py
ADD pyproject.toml /QDL/pyproject.toml
ADD LICENSE /QDL/LICENSE
ADD README.md /QDL/README.md
ADD tests/ /QDL/tests
WORKDIR /QDL/
RUN python3 -m pip install .
WORKDIR /QDL/src/qdl/training
ENTRYPOINT ["python3"]
