ARG CUDA_VERSION=12.8.1
FROM docker.1ms.run/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG GRACE_BLACKWELL=0
ARG HOPPER_SBO=0
ARG CPU_VARIANT=x86-intel-multi
ARG BUILD_ALL_CPU_VARIANTS=1

# Proxy settings for build-time network access
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy}

ARG GRACE_BLACKWELL_DEEPEP_BRANCH=gb200_blog_part_2
ARG HOPPER_SBO_DEEPEP_COMMIT=9f2fc4b3182a51044ae7ecb6610f7c9c3258c4d6
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.19
ARG SGL_VERSION=0.5.6.post1
ARG USE_LATEST_SGLANG=0
ARG GDRCOPY_VERSION=2.5.1
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com
ARG FLASHINFER_VERSION=0.5.3

# ktransformers wheel version (cu128torch28 for CUDA 12.8 + PyTorch 2.8)
ARG KTRANSFORMERS_VERSION=0.4.2
ARG KTRANSFORMERS_WHEEL=ktransformers-0.4.2+cu128torch28fancy-cp312-cp312-linux_x86_64.whl

# flash_attn wheel for fine-tune env
ARG FLASH_ATTN_WHEEL=flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}

# Add GKE default lib and bin locations
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Replace Ubuntu sources with Tsinghua mirror for Ubuntu 24.04 (noble)
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources; \
fi

# Install system dependencies (organized by category for better caching)
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # Core system utilities
    tzdata \
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    # Build essentials
    build-essential \
    cmake \
    perl \
    patchelf \
    ccache \
    git \
    git-lfs \
    # MPI and NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM
    ffmpeg \
    # InfiniBand/RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    # Package building tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    # Development tools
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    zsh \
    tree \
    less \
    rdma-core \
    # NCCL
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl --retry 3 --retry-delay 2 -fsSL -o v${GDRCOPY_VERSION}.tar.gz \
        https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
    && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
    && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
########## Install Miniconda ###########################
########################################################

RUN mkdir -p /opt/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/miniconda3/bin:${PATH}"

# Accept conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Configure conda to use Tsinghua mirror
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free \
    && conda config --set show_channel_urls yes

########################################################
########## Dual Conda Environment Setup ################
########################################################

FROM base AS framework

ARG CUDA_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL
ARG SGL_KERNEL_VERSION
ARG SGL_VERSION
ARG USE_LATEST_SGLANG
ARG FLASHINFER_VERSION
ARG GRACE_BLACKWELL
ARG GRACE_BLACKWELL_DEEPEP_BRANCH
ARG HOPPER_SBO
ARG HOPPER_SBO_DEEPEP_COMMIT
ARG DEEPEP_COMMIT
ARG GITHUB_ARTIFACTORY
ARG KTRANSFORMERS_VERSION
ARG KTRANSFORMERS_WHEEL
ARG FLASH_ATTN_WHEEL

WORKDIR /workspace

# Create two conda environments with Python 3.12
RUN conda create -n serve python=3.12 -y \
    && conda create -n fine-tune python=3.12 -y

# Set pip mirror for both conda envs
RUN /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && /opt/miniconda3/envs/fine-tune/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Clone repositories
# Use kvcache-ai/sglang fork with kimi_k2 branch
RUN git clone https://${GITHUB_ARTIFACTORY}/kvcache-ai/sglang.git /workspace/sglang \
    && cd /workspace/sglang && git checkout kimi_k2

RUN git clone --depth 1 https://${GITHUB_ARTIFACTORY}/hiyouga/LLaMA-Factory.git /workspace/LLaMA-Factory \
    && git clone -b add-cd-djw --depth 1 https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers.git /workspace/ktransformers \
    && cd /workspace/ktransformers && git submodule update --init --recursive

# Download ktransformers wheel and flash_attn wheel for fine-tune env
RUN curl --retry 3 --retry-delay 2 -fsSL -o /workspace/${KTRANSFORMERS_WHEEL} \
    https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers/releases/download/v${KTRANSFORMERS_VERSION}/${KTRANSFORMERS_WHEEL} \
    && curl --retry 3 --retry-delay 2 -fsSL -o /workspace/${FLASH_ATTN_WHEEL} \
    https://${GITHUB_ARTIFACTORY}/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_ATTN_WHEEL}

########################################################
# Environment 1: serve (sglang + kt-kernel)
########################################################

# Upgrade pip and install basic tools in serve env
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/serve/bin/pip install --upgrade pip setuptools wheel html5lib six

# Install sgl-kernel
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.6.1) CUINDEX=126 ;; \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && if [ "$CUDA_VERSION" = "12.6.1" ]; then \
        /opt/miniconda3/envs/serve/bin/pip install https://${GITHUB_ARTIFACTORY}/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu124-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps \
    ; \
    elif [ "$CUDA_VERSION" = "12.8.1" ] || [ "$CUDA_VERSION" = "12.9.1" ]; then \
        /opt/miniconda3/envs/serve/bin/pip install sgl-kernel==${SGL_KERNEL_VERSION} \
    ; \
    elif [ "$CUDA_VERSION" = "13.0.1" ]; then \
        /opt/miniconda3/envs/serve/bin/pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps \
    ; \
    fi

# Install SGLang in serve env
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.6.1) CUINDEX=126 ;; \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
    esac \
    && cd /workspace/sglang \
    && /opt/miniconda3/envs/serve/bin/pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}

# Download FlashInfer cubin for serve env
RUN --mount=type=cache,target=/root/.cache/pip \
    FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} FLASHINFER_LOGGING_LEVEL=warning \
    /opt/miniconda3/envs/serve/bin/python -m flashinfer --download-cubin

# Install DeepEP in serve env
RUN set -eux; \
    if [ "$GRACE_BLACKWELL" = "1" ]; then \
      git clone https://github.com/fzyzcjy/DeepEP.git /workspace/DeepEP && \
      cd /workspace/DeepEP && \
      git checkout ${GRACE_BLACKWELL_DEEPEP_BRANCH} && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh; \
    elif [ "$HOPPER_SBO" = "1" ]; then \
      git clone https://github.com/deepseek-ai/DeepEP.git -b antgroup-opt /workspace/DeepEP && \
      cd /workspace/DeepEP && \
      git checkout ${HOPPER_SBO_DEEPEP_COMMIT} && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh; \
    else \
      curl --retry 3 --retry-delay 2 -fsSL -o /tmp/${DEEPEP_COMMIT}.zip \
          https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip && \
      unzip -q /tmp/${DEEPEP_COMMIT}.zip -d /tmp && rm /tmp/${DEEPEP_COMMIT}.zip && \
      mv /tmp/DeepEP-${DEEPEP_COMMIT} /workspace/DeepEP && \
      cd /workspace/DeepEP && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/DeepEP && \
    case "$CUDA_VERSION" in \
        12.6.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0' ;; \
        12.8.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0' ;; \
        12.9.1|13.0.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac && \
    . /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve && \
    TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} \
    pip install --no-build-isolation .

# Install NCCL for serve env
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${CUDA_VERSION%%.*}" = "12" ]; then \
        /opt/miniconda3/envs/serve/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps ; \
    elif [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        /opt/miniconda3/envs/serve/bin/pip install nvidia-nccl-cu13==2.28.3 --force-reinstall --no-deps ; \
    fi

# Install kt-kernel in serve env with all CPU variants
RUN . /opt/miniconda3/etc/profile.d/conda.sh && conda activate serve \
    && cd /workspace/ktransformers/kt-kernel \
    && CPUINFER_BUILD_ALL_VARIANTS=1 ./install.sh build

########################################################
# Environment 2: fine-tune (LLaMA-Factory + ktransformers)
########################################################

# Install dependency libraries for ktransformers (CUDA 11.8 runtime required)
RUN conda install -n fine-tune -y -c conda-forge libstdcxx-ng gcc_impl_linux-64 \
    && conda install -n fine-tune -y -c nvidia/label/cuda-11.8.0 cuda-runtime

# Install PyTorch 2.8 in fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.6.1) CUINDEX=126 ;; \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
    esac \
    && /opt/miniconda3/envs/fine-tune/bin/pip install --upgrade pip setuptools wheel \
    && /opt/miniconda3/envs/fine-tune/bin/pip install \
        torch==2.8.0 \
        torchvision \
        torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}

# Install LLaMA-Factory in fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/LLaMA-Factory \
    && /opt/miniconda3/envs/fine-tune/bin/pip install -e ".[torch,metrics]" --no-build-isolation

# Install ktransformers wheel in fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/fine-tune/bin/pip install /workspace/${KTRANSFORMERS_WHEEL}

# Install flash_attn wheel in fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/miniconda3/envs/fine-tune/bin/pip install /workspace/${FLASH_ATTN_WHEEL}

# Install NCCL for fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${CUDA_VERSION%%.*}" = "12" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps ; \
    elif [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip install nvidia-nccl-cu13==2.28.3 --force-reinstall --no-deps ; \
    fi

########################################################
# Cleanup and final setup
########################################################

# Clean up downloaded wheels
RUN rm -f /workspace/${KTRANSFORMERS_WHEEL} /workspace/${FLASH_ATTN_WHEEL}

# Initialize conda for bash
RUN /opt/miniconda3/bin/conda init bash

# Create shell aliases for convenience
RUN echo '\n# Conda environment aliases\nalias serve="conda activate serve"\nalias finetune="conda activate fine-tune"' >> /root/.bashrc

########################################################
# Extract version information for image naming
########################################################

# Extract versions from each component and save to versions.env
RUN set -x && \
    # SGLang version (from version.py file)
    cd /workspace/sglang/python/sglang && \
    SGLANG_VERSION=$(python3 -c "exec(open('version.py').read()); print(__version__)" 2>/dev/null || echo "unknown") && \
    echo "SGLANG_VERSION=$SGLANG_VERSION" > /workspace/versions.env && \
    echo "Extracted SGLang version: $SGLANG_VERSION" && \
    \
    # KTransformers version (from version.py in repo)
    cd /workspace/ktransformers && \
    KTRANSFORMERS_VERSION=$(python3 -c "exec(open('version.py').read()); print(__version__)" 2>/dev/null || echo "unknown") && \
    echo "KTRANSFORMERS_VERSION=$KTRANSFORMERS_VERSION" >> /workspace/versions.env && \
    echo "Extracted KTransformers version: $KTRANSFORMERS_VERSION" && \
    \
    # LLaMA-Factory version (from fine-tune environment)
    . /opt/miniconda3/etc/profile.d/conda.sh && conda activate fine-tune && \
    cd /workspace/LLaMA-Factory && \
    LLAMAFACTORY_VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); from llamafactory import __version__; print(__version__)" 2>/dev/null || echo "unknown") && \
    echo "LLAMAFACTORY_VERSION=$LLAMAFACTORY_VERSION" >> /workspace/versions.env && \
    echo "Extracted LLaMA-Factory version: $LLAMAFACTORY_VERSION" && \
    \
    # Display all versions
    echo "=== Version Summary ===" && \
    cat /workspace/versions.env

WORKDIR /workspace

CMD ["/bin/bash"]
