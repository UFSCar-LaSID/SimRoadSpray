FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install common dependencies
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create a new environment with Python 3.8
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -y -n py38 python=3.8 && conda clean --all

# Activate env by default
SHELL ["conda", "run", "-n", "py38", "/bin/bash", "-c"]

# Install PyTorch and torchvision
RUN conda install -y pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install MMEngine, MMCV and MMDetection
RUN pip install openmim && \
    mim install "mmengine" "mmcv>=2.0.0rc4, <2.2.0" "mmdet>=3.0.0"

RUN pip install --upgrade pip

# Install MMDetection3D
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /mmdetection3d \
    && cd /mmdetection3d \
    && pip install --no-cache-dir -e .

WORKDIR /mmdetection3d

# Adicionar modificações da lib
COPY ./mmdet3d_modifications /mmdetection3d

# Instalar outras libs necessárias para rodar o SECOND
RUN pip install cumm-cu113 && \
    pip install spconv-cu113 && \
    pip install numpy --upgrade && \
    python projects/BEVFusion/setup.py develop

RUN apt-get update && \
    apt-get install wget && \
    mkdir models

RUN echo "source activate py38" >> ~/.bashrc

# Ensure the Python 3.8 conda environment is used by default in the container
CMD [ "bash", "-c", "source activate py38 && exec bash" ]
