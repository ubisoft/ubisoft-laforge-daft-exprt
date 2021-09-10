FROM nvidia/cuda:11.2.0-base-ubuntu20.04

# set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda3/bin:$PATH

# install linux packages
RUN apt-get update && apt-get install -y curl libsndfile1 libopenblas-dev
RUN rm -rf /var/lib/apt/lists/*

# install miniconda and python 3.8
RUN curl -sLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ./miniconda.sh \
    && ./miniconda.sh -b -p /root/miniconda3 \
    && rm miniconda.sh \
    && conda install -y python==3.8 \
    && conda clean -ya

# install conda and pip packages
COPY ./environment/conda_requirements.txt /opt/daft_exprt/environment/
COPY ./environment/pip_requirements.txt /opt/daft_exprt/environment/
WORKDIR /opt/daft_exprt/environment
RUN conda install -c conda-forge -y --file conda_requirements.txt
RUN pip install -r pip_requirements.txt

# install PyTorch
RUN pip install torch==1.9.0+cu111 torchaudio==0.9.0 tensorboard -f https://download.pytorch.org/whl/torch_stable.html

# install MFA thirdparty packages
RUN mfa thirdparty download
RUN mfa thirdparty validate

# download pre-trained MFA models for english
RUN mfa download acoustic english
RUN mfa download g2p english_g2p
RUN mfa download dictionary english

# set working directory
WORKDIR /workdir
