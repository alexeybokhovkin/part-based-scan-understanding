FROM nvidia/cuda:11.1-devel-ubuntu20.04

MAINTAINER Alexey Bokhovkin <a.bokhovkin@skoltech.ru>

## Base packages for ubuntu

# clean the libs list
RUN apt-get clean
RUN apt-get update -qq
RUN apt update
RUN apt upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y \
    git \
    wget \
    bzip2 \
    htop \
    vim \
    nano \
    cmake \
    g++ \
    make \
    build-essential \
    software-properties-common \
    apt-transport-https \
    sudo \
    gosu \
    libgl1-mesa-glx \
    graphviz \
    tmux \
    openconnect \
    openssh-server

## Download and install miniconda

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH
RUN echo $PATH
## Install base conda packages

RUN conda config --add channels conda-forge 
RUN conda config --add channels open3d-admin
RUN conda config --add channels anaconda

RUN conda install -y numpy>=1.19 \
                     scipy>=1.5 \
                     pandas>=1.1 \
                     matplotlib>=3.3.3 \
                     seaborn>=0.9.0 \
                     pillow>=8.0 \
                     scikit-learn>=0.23 \
                     jupyter>=1.0.0 \
                     open3d>=0.11 \
                     opencv>=4.5 \
                     nltk>=3.3 \
                     h5py \
                     tqdm

RUN pip install trimesh \
	            jupyterthemes \
                scikit-image \
                plyfile \
                umap-learn \
                sklearn

RUN conda install -c pytorch pytorch==1.7.0 torchvision==0.8
RUN pip install pytorch_lightning==0.8.5

RUN conda install -c conda-forge jupyterlab
RUN pip install k3d==2.8.0

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN jupyter nbextension install --py --sys-prefix k3d
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix

RUN pip install numpy-quaternion
RUN pip install numba

# Final stage, staring jupyter and change user and group

COPY jupyter_notebook_config.py /root/.jupyter/

ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID user && useradd -m -s /bin/bash -u $UID -g user -G root user
RUN usermod -aG sudo user
RUN echo "user:user" | chpasswd
WORKDIR /home/user

COPY --chown=user:user jupyter_notebook_config.py /home/user/.jupyter/
COPY runuser.sh /opt/run/
RUN echo "export PATH='/opt/conda/bin:${PATH}'" >> /home/user/.bashrc
RUN chmod +x /opt/run/runuser.sh


# start custom entrypoint
ENTRYPOINT ["/opt/run/runuser.sh"]
