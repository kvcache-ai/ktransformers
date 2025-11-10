FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel as compile_server


ARG CPU_INSTRUCT=NATIVE

# 设置工作目录和 CUDA 路径
WORKDIR /workspace
ENV CUDA_HOME=/usr/local/cuda



# 安装依赖
RUN apt update -y
RUN apt install -y --no-install-recommends \
    libtbb-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libaio1 \
    libaio-dev \
    libfmt-dev \
    libgflags-dev \
    zlib1g-dev \
    patchelf \
    git \
    wget \
    vim \
    gcc \
    g++ \
    cmake
# 拷贝代码
RUN git clone https://github.com/kvcache-ai/ktransformers.git 
# 清理 apt 缓存
RUN rm -rf /var/lib/apt/lists/*

# 进入项目目录
WORKDIR /workspace/ktransformers
# 初始化子模块
RUN git submodule update --init --recursive

# 升级 pip
RUN pip install --upgrade pip

# 安装构建依赖
RUN pip install ninja pyproject numpy cpufeature aiohttp zmq openai

# 安装 flash-attn（提前装可以避免后续某些编译依赖出错）
RUN pip install flash-attn

# 安装 ktransformers 本体（含编译）
RUN CPU_INSTRUCT=${CPU_INSTRUCT} \
    USE_BALANCE_SERVE=1 \
    KTRANSFORMERS_FORCE_BUILD=TRUE \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0+PTX" \
    pip install . --no-build-isolation --verbose

RUN pip install third_party/custom_flashinfer/
# 清理 pip 缓存
RUN pip cache purge

# 拷贝 C++ 运行时库
RUN cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/

# 保持容器运行（调试用）
ENTRYPOINT ["tail", "-f", "/dev/null"]