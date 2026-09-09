# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG DSV4_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
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
ARG KTRANSFORMERS_VERSION=0.5.3
ARG KTRANSFORMERS_WHEEL=ktransformers-0.5.3+cu128torch28fancy-cp312-cp312-linux_x86_64.whl

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
########## DeepSeek-V4-Flash one-click image ############
########################################################

# This target is self-contained so the DSV4 image and its merged SGLang API
# support can be reviewed as a Dockerfile-only change.
FROM ${DSV4_BASE_IMAGE} AS dsv4

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG IMAGE_VERSION=dev
ARG BUILD_DATE=unknown
ARG KTRANSFORMERS_REPOSITORY=https://github.com/kvcache-ai/ktransformers.git
ARG KTRANSFORMERS_REF=6d460cc10780f2e0ad541b5d58ad28086dd32ef0
ARG SGLANG_REPOSITORY=https://github.com/kvcache-ai/sglang.git
ARG SGLANG_REF=0980b6da258192896a9d884fbe564fd1f9ec07e3
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_VERSION=2.9.1
ARG SGL_KERNEL_VERSION=0.3.21
ARG FLASHINFER_VERSION=0.6.15.post1
ARG TRANSFORMERS_VERSION=4.57.1
ARG TILELANG_VERSION=0.1.10
ARG TVM_FFI_VERSION=0.1.11
ARG CPUINFER_CUDA_ARCHS=80;86;89;90;100;120
ARG BUILD_JOBS=16

ENV CUDA_HOME=/usr/local/cuda \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false \
    MODEL_PATH=/model \
    PORT=30000 \
    DSV4_REASONING_EFFORT_PROFILE=official \
    REASONING_PARSER=deepseek-v4 \
    TOOL_CALL_PARSER=deepseekv4 \
    PATH=/opt/venv/bin:/usr/local/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libhwloc-dev \
        libnuma-dev \
        libopenmpi-dev \
        numactl \
        pkg-config \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv

RUN python -m pip config set global.index-url "${PIP_INDEX_URL}" \
    && python -m pip install --upgrade setuptools wheel \
    && python -m pip install "cmake>=3.31,<4" ninja

RUN python -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==0.24.1" \
        "torchaudio==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url "${PIP_INDEX_URL}" \
        --retries 10 \
        --timeout 120

ARG FLASHINFER_CUBIN_INDEX_URL=https://flashinfer.ai/whl
ARG FLASHINFER_CUBIN_TRUSTED_HOST=flashinfer.ai

RUN python -m pip install \
        "sgl-kernel==${SGL_KERNEL_VERSION}" \
        "flashinfer-python==${FLASHINFER_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "tilelang==${TILELANG_VERSION}" \
        "apache-tvm-ffi==${TVM_FFI_VERSION}" \
        "cuda-python==13.2.0" \
        "nvidia-cutlass-dsl==4.6.1" \
        IPython \
        aiohttp \
        'anthropic>=0.20.0' \
        blobfile==3.0.0 \
        build \
        compressed-tensors \
        datasets \
        decord2 \
        einops \
        fastapi \
        gguf \
        'grpcio>=1.78.0' \
        'grpcio-health-checking>=1.78.0' \
        'grpcio-reflection>=1.78.0' \
        hf_transfer \
        'huggingface_hub<1.0' \
        interegular \
        'llguidance>=0.7.11,<0.8.0' \
        modelscope \
        msgspec \
        numpy \
        nvidia-ml-py \
        openai==2.6.1 \
        openai-harmony==0.0.4 \
        orjson \
        outlines==0.1.11 \
        packaging \
        partial_json_parser \
        pillow \
        'prometheus-client>=0.20.0' \
        psutil \
        py-spy \
        pybase64 \
        pydantic \
        python-multipart \
        'pyzmq>=25.1.2' \
        requests \
        scipy \
        sentencepiece \
        setproctitle \
        'smg-grpc-proto>=0.3.3' \
        soundfile==0.13.1 \
        tiktoken \
        timm==1.0.16 \
        torch_memory_saver==0.0.9 \
        torchao==0.9.0 \
        torchcodec==0.8.0 \
        tqdm \
        uvicorn \
        uvloop \
        xgrammar==0.1.27 \
    && python -m pip install \
        --index-url "${FLASHINFER_CUBIN_INDEX_URL}" \
        --trusted-host "${FLASHINFER_CUBIN_TRUSTED_HOST}" \
        --no-deps \
        "flashinfer-cubin==${FLASHINFER_VERSION}" \
    && python -m pip install --no-deps quack-kernels==0.2.4

RUN git clone --filter=blob:none "${KTRANSFORMERS_REPOSITORY}" /workspace/ktransformers \
    && git -C /workspace/ktransformers checkout --detach "${KTRANSFORMERS_REF}" \
    && git -C /workspace/ktransformers submodule update --init --recursive \
    && git -C /workspace/ktransformers/third_party/sglang fetch \
        --depth 1 "${SGLANG_REPOSITORY}" "${SGLANG_REF}" \
    && git -C /workspace/ktransformers/third_party/sglang checkout --detach FETCH_HEAD

RUN export SGLANG_KT_VERSION="$(python -c "exec(open('/workspace/ktransformers/version.py').read()); print(__version__)")" \
    && python -m pip install --no-deps \
        /workspace/ktransformers/third_party/sglang/python \
    && cd /workspace/ktransformers/kt-kernel \
    && CPUINFER_BUILD_ALL_VARIANTS=1 \
       CPUINFER_USE_CUDA=1 \
       CPUINFER_CUDA_ARCHS="${CPUINFER_CUDA_ARCHS}" \
       CPUINFER_PARALLEL="${BUILD_JOBS}" \
       ./install.sh build \
    && python -m pip install --no-deps /workspace/ktransformers \
    && rm -rf /workspace/ktransformers/kt-kernel/build

RUN python - <<'PY'
import importlib.metadata

from sglang.srt.entrypoints.openai import encoding_dsv4
from sglang.srt.function_call.function_call_parser import FunctionCallParser

expected = {
    "ktransformers": "0.7.0.post1",
    "kt-kernel": "0.7.0.post1",
    "sglang-kt": "0.7.0.post1",
    "torch": "2.9.1+cu128",
    "flashinfer-python": "0.6.15.post1",
    "transformers": "4.57.1",
}
for distribution, version in expected.items():
    actual = importlib.metadata.version(distribution)
    if actual != version:
        raise RuntimeError(f"{distribution}: expected {version}, got {actual}")

if set(encoding_dsv4.REASONING_EFFORT_PROFILES["official"]) != {
    "low",
    "high",
    "max",
}:
    raise RuntimeError("missing official DeepSeek-V4 reasoning effort profile")
if "deepseekv4" not in FunctionCallParser.ToolCallParserEnum:
    raise RuntimeError("missing DeepSeek-V4 tool-call parser")
PY

COPY --chmod=0755 <<'ENTRYPOINT' /usr/local/bin/entrypoint-dsv4
#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[dsv4-entrypoint] %s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  exec "$@"
fi

MODEL_PATH="${MODEL_PATH:-/model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
KT_GPU_EXPERTS="${KT_GPU_EXPERTS:-0}"
KT_GPU_PREFILL_TOKEN_THRESHOLD="${KT_GPU_PREFILL_TOKEN_THRESHOLD:-2048}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$CHUNKED_PREFILL_SIZE}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.4}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1200}"
DSV4_REASONING_EFFORT_PROFILE="${DSV4_REASONING_EFFORT_PROFILE:-official}"
REASONING_PARSER="${REASONING_PARSER:-deepseek-v4}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseekv4}"

[[ "$TP" =~ ^[1-9][0-9]*$ ]] || die "TP must be a positive integer"
[[ -r "$MODEL_PATH/config.json" ]] || die "missing $MODEL_PATH/config.json"
find "$MODEL_PATH" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name '*.gguf' \) -print -quit | grep -q . ||
  die "no model weight files found in $MODEL_PATH"

GPU_INFO="$(TP="$TP" python - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
tp = int(os.environ["TP"])
if torch.cuda.device_count() < tp:
    raise SystemExit(f"TP={tp} exceeds the visible GPU count")

supported = {
    (8, 0): ("8.0", "8.0"),
    (8, 6): ("8.6", "8.6"),
    (8, 9): ("8.9", "8.9"),
    (9, 0): ("9.0+PTX", "9.0a"),
    (10, 0): ("10.0+PTX", "10.0a"),
    (12, 0): ("12.0+PTX", "12.0a"),
}
torch_arches = []
flashinfer_arches = []
for index in range(tp):
    capability = torch.cuda.get_device_capability(index)
    if capability not in supported:
        raise SystemExit(f"unsupported GPU capability: SM{capability[0]}{capability[1]}")
    torch_arch, flashinfer_arch = supported[capability]
    if torch_arch not in torch_arches:
        torch_arches.append(torch_arch)
    if flashinfer_arch not in flashinfer_arches:
        flashinfer_arches.append(flashinfer_arch)
print("|".join((";".join(torch_arches), ";".join(flashinfer_arches))))
PY
)" || die "GPU validation failed"
IFS='|' read -r DEFAULT_TORCH_ARCH DEFAULT_FLASHINFER_ARCH <<< "$GPU_INFO"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$DEFAULT_TORCH_ARCH}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-$DEFAULT_FLASHINFER_ARCH}"
export SGLANG_DSV4_MODE="${SGLANG_DSV4_MODE:-2604}"
export SGLANG_DSV4_2604_SUBMODE="${SGLANG_DSV4_2604_SUBMODE:-2604B}"

if [[ -z "${KT_CPUINFER_THREADS:-}" ]]; then
  PHYSICAL_CORES="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l)"
  (( PHYSICAL_CORES > 4 )) && KT_CPUINFER_THREADS=$((PHYSICAL_CORES - 4)) || KT_CPUINFER_THREADS="$PHYSICAL_CORES"
fi
if [[ -z "${KT_THREADPOOL_COUNT:-}" ]]; then
  KT_THREADPOOL_COUNT="$(lscpu -p=NODE | awk -F, '!/^#/ && $1 >= 0 {print $1}' | sort -u | wc -l)"
  (( KT_THREADPOOL_COUNT > 0 )) || KT_THREADPOOL_COUNT=1
fi

cmd=(
  python -m sglang.launch_server
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL_PATH"
  --kt-weight-path "$MODEL_PATH"
  --kt-method MXFP4
  --kt-num-gpu-experts "$KT_GPU_EXPERTS"
  --kt-cpuinfer "$KT_CPUINFER_THREADS"
  --kt-threadpool-count "$KT_THREADPOOL_COUNT"
  --kt-gpu-prefill-token-threshold "$KT_GPU_PREFILL_TOKEN_THRESHOLD"
  --kt-enable-dynamic-expert-update
  --tensor-parallel-size "$TP"
  --context-length "$CONTEXT_LENGTH"
  --attention-backend flashinfer
  --mem-fraction-static "$MEM_FRACTION"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"
  --watchdog-timeout "$WATCHDOG_TIMEOUT"
  --disable-shared-experts-fusion
  --trust-remote-code
  --cuda-graph-bs 1
  --cuda-graph-max-bs 1
  --disable-radix-cache
  --skip-server-warmup
  --reasoning-parser "$REASONING_PARSER"
  --tool-call-parser "$TOOL_CALL_PARSER"
  --json-model-override-args "{\"dsv4_reasoning_effort_profile\":\"$DSV4_REASONING_EFFORT_PROFILE\"}"
)
cmd+=("$@")

printf '[dsv4-entrypoint] launching:' >&2
printf ' %q' "${cmd[@]}" >&2
printf '\n' >&2
exec numactl --interleave=all "${cmd[@]}"
ENTRYPOINT

LABEL org.opencontainers.image.title="KTransformers DeepSeek-V4-Flash" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.ktransformers.source-revision="${KTRANSFORMERS_REF}" \
      io.ktransformers.sglang-revision="${SGLANG_REF}" \
      io.ktransformers.reasoning-effort-profile="official"

WORKDIR /workspace
EXPOSE 30000
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=20 \
  CMD curl --fail --silent --show-error "http://127.0.0.1:$PORT/health" >/dev/null || exit 1
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/entrypoint-dsv4"]
CMD []

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
ARG FUNCTIONALITY=sft

WORKDIR /workspace

# Create conda environments (fine-tune only needed for sft mode)
RUN conda create -n serve python=3.12 -y \
    && if [ "$FUNCTIONALITY" = "sft" ]; then conda create -n fine-tune python=3.12 -y; fi

# Set pip mirror for conda envs
RUN /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && if [ "$FUNCTIONALITY" = "sft" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple; \
    fi

# Clone repositories (sglang is included as a submodule in ktransformers)
RUN git clone --depth 1 https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers.git /workspace/ktransformers \
    && cd /workspace/ktransformers && git submodule update --init --recursive \
    && ln -s /workspace/ktransformers/third_party/sglang /workspace/sglang \
    && if [ "$FUNCTIONALITY" = "sft" ]; then \
        git clone --depth 1 https://${GITHUB_ARTIFACTORY}/hiyouga/LLaMA-Factory.git /workspace/LLaMA-Factory; \
    fi

# Download ktransformers wheel and flash_attn wheel for fine-tune env (sft mode only)
RUN if [ "$FUNCTIONALITY" = "sft" ]; then \
        curl --retry 3 --retry-delay 2 -fsSL -o /workspace/${KTRANSFORMERS_WHEEL} \
            https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers/releases/download/v${KTRANSFORMERS_VERSION}/${KTRANSFORMERS_WHEEL} \
        && curl --retry 3 --retry-delay 2 -fsSL -o /workspace/${FLASH_ATTN_WHEEL} \
            https://${GITHUB_ARTIFACTORY}/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_ATTN_WHEEL}; \
    fi

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

# Install SGLang in serve env (version aligned with ktransformers)
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.6.1) CUINDEX=126 ;; \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
    esac \
    && export SGLANG_KT_VERSION=$(python3 -c "exec(open('/workspace/ktransformers/version.py').read()); print(__version__)") \
    && echo "Installing sglang-kt v${SGLANG_KT_VERSION}" \
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
# Environment 2: fine-tune (LLaMA-Factory + ktransformers) - sft mode only
########################################################

# Install dependency libraries for ktransformers (CUDA 11.8 runtime required)
RUN if [ "$FUNCTIONALITY" = "sft" ]; then \
        conda install -n fine-tune -y -c conda-forge libstdcxx-ng gcc_impl_linux-64 \
        && conda install -n fine-tune -y -c nvidia/label/cuda-11.8.0 cuda-runtime; \
    fi

# Install PyTorch 2.8 in fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        case "$CUDA_VERSION" in \
            12.6.1) CUINDEX=126 ;; \
            12.8.1) CUINDEX=128 ;; \
            12.9.1) CUINDEX=129 ;; \
            13.0.1) CUINDEX=130 ;; \
        esac \
        && /opt/miniconda3/envs/fine-tune/bin/pip install --upgrade pip setuptools wheel hatchling \
        && /opt/miniconda3/envs/fine-tune/bin/pip install \
            torch==2.8.0 \
            torchvision \
            torchaudio \
            --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}; \
    fi

# Install LLaMA-Factory in fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        cd /workspace/LLaMA-Factory \
        && /opt/miniconda3/envs/fine-tune/bin/pip install -e ".[torch,metrics]" --no-build-isolation; \
    fi

# Install ktransformers wheel in fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip install /workspace/${KTRANSFORMERS_WHEEL}; \
    fi

# Install flash_attn wheel in fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        /opt/miniconda3/envs/fine-tune/bin/pip install /workspace/${FLASH_ATTN_WHEEL}; \
    fi

# Install NCCL for fine-tune env
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        if [ "${CUDA_VERSION%%.*}" = "12" ]; then \
            /opt/miniconda3/envs/fine-tune/bin/pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps ; \
        elif [ "${CUDA_VERSION%%.*}" = "13" ]; then \
            /opt/miniconda3/envs/fine-tune/bin/pip install nvidia-nccl-cu13==2.28.3 --force-reinstall --no-deps ; \
        fi; \
    fi

########################################################
# Cleanup and final setup
########################################################

# Clean up downloaded wheels
RUN if [ "$FUNCTIONALITY" = "sft" ]; then \
        rm -f /workspace/${KTRANSFORMERS_WHEEL} /workspace/${FLASH_ATTN_WHEEL}; \
    fi

# Initialize conda for bash
RUN /opt/miniconda3/bin/conda init bash

# Create shell aliases for convenience
RUN echo '\n# Conda environment aliases\nalias serve="conda activate serve"' >> /root/.bashrc \
    && if [ "$FUNCTIONALITY" = "sft" ]; then \
        echo 'alias finetune="conda activate fine-tune"' >> /root/.bashrc; \
    fi

########################################################
# Extract version information for image naming
########################################################

# Extract versions from each component and save to versions.env
RUN set -x && \
    # KTransformers version (single source of truth for both kt-kernel and sglang-kt)
    cd /workspace/ktransformers && \
    KTRANSFORMERS_VERSION=$(python3 -c "exec(open('version.py').read()); print(__version__)" 2>/dev/null || echo "unknown") && \
    echo "KTRANSFORMERS_VERSION=$KTRANSFORMERS_VERSION" > /workspace/versions.env && \
    echo "Extracted KTransformers version: $KTRANSFORMERS_VERSION" && \
    \
    # sglang-kt version = ktransformers version (aligned)
    echo "SGLANG_KT_VERSION=$KTRANSFORMERS_VERSION" >> /workspace/versions.env && \
    echo "sglang-kt version (aligned): $KTRANSFORMERS_VERSION" && \
    \
    # LLaMA-Factory version (from fine-tune environment, sft mode only)
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        . /opt/miniconda3/etc/profile.d/conda.sh && conda activate fine-tune && \
        cd /workspace/LLaMA-Factory && \
        LLAMAFACTORY_VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); from llamafactory import __version__; print(__version__)" 2>/dev/null || echo "unknown") && \
        echo "LLAMAFACTORY_VERSION=$LLAMAFACTORY_VERSION" >> /workspace/versions.env && \
        echo "Extracted LLaMA-Factory version: $LLAMAFACTORY_VERSION"; \
    else \
        echo "LLAMAFACTORY_VERSION=none" >> /workspace/versions.env && \
        echo "LLaMA-Factory not installed (infer mode)"; \
    fi && \
    \
    # Display all versions
    echo "=== Version Summary ===" && \
    cat /workspace/versions.env

WORKDIR /workspace

CMD ["/bin/bash"]
