# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Legate with GASNet and CUDA for Infiniband

FROM ubuntu:16.04

MAINTAINER Michael Bauer <mbauer@nvidia.com>

# Install dependencies.
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq && \
    apt-get install -qq software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update -qq && \
    apt-get install -qq \
      build-essential git python-pip time wget \
      g++-4.8 g++-4.9 g++-5 g++-6 \
      gcc-4.9-multilib g++-4.9-multilib \
      libomp-dev libncurses5-dev zlib1g-dev \
      mpich libmpich-dev \
      libblas-dev liblapack-dev libhdf5-dev \
      module-init-tools \
      gdb vim \
      openmpi-bin openssh-client openssh-server libopenmpi-dev && \
    apt-get clean
# Make sure there are no AVX512 headers floating around as they mess with nvcc
RUN sed -i '/avx512/d' /usr/lib/gcc/x86_64-linux-gnu/5/include/immintrin.h

# InfiniBand
RUN apt-get update && apt-get install -y --no-install-recommends \
    dapl2-utils \
    ibutils \
    ibverbs-utils \
    infiniband-diags \
    libdapl-dev \
    libibcm-dev \
    libibverbs1-dbg \
    libibverbs-dev \
    libmlx4-1-dbg \
    libmlx4-dev \
    libmlx5-1-dbg \
    libmlx5-dev \
    librdmacm-dev \
    opensm

# Build GASNet
RUN git clone -b master https://github.com/StanfordLegion/gasnet.git /usr/local/gasnet_build
RUN make -C /usr/local/gasnet_build/ -e CONDUIT=ibv -e RELEASE_DIR=/usr/local/gasnet/
ENV GASNET /usr/local/gasnet

# Install CUDA
RUN wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb && apt-get update -qq && \
    apt-get -y --allow-unauthenticated install cuda-command-line-tools-9-2 cuda-core-9-2 cuda-cublas-dev-9-2 && \
    ln -s /usr/local/cuda-9.2 /usr/local/cuda && \
    wget http://download.nvidia.com/XFree86/Linux-x86_64/396.24/NVIDIA-Linux-x86_64-396.24.run && \
    sh ./NVIDIA-Linux-x86_64-396.24.run -s -N --no-kernel-module && \
    rm cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb && rm NVIDIA-Linux-x86_64-396.24.run
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA /usr/local/cuda

# Instal OpenBLAS
RUN git clone https://github.com/xianyi/OpenBLAS.git /usr/local/openblas_build
RUN USE_OPENMP=1 make -C /usr/local/openblas_build
RUN PREFIX=/usr/local/openblas make -C /usr/local/openblas_build install
ENV OPEN_BLAS_DIR /usr/local/openblas

# Install Legate
RUN git clone -b control_replication https://gitlab.com/StanfordLegion/legion.git /usr/local/legion
ENV LG_RT_DIR /usr/local/legion/runtime
RUN git clone -b stable https://gitlab-master.nvidia.com/mbauer/Legate.git /usr/local/Legate
RUN CONDUIT=ibv /usr/local/Legate/install.py --gasnet --cuda --arch volta --openmp && \
    ln -s /usr/local/Legate/driver.py /usr/local/bin/legate

# Configure container startup.
CMD ["/usr/local/bin/legate"]
