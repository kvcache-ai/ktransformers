# Running KTransformers on AVX2 CPUs

This tutorial explains how to run KTransformers on machines that only support AVX2 (without AVX512 or AMX).

## Table of Contents

- [Supported Precision Formats](#supported-precision-formats)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Verification](#verification)
- [Starting the Inference Server](#starting-the-inference-server)
  - [Example: Qwen3-30B-A3B (BF16)](#example-qwen3-30b-a3b-bf16)
  - [Example: Qwen3.5-35B-A3B-FP8 (FP8)](#example-qwen35-35b-a3b-fp8-fp8)
  - [Example: Qwen3-30B-A3B-GPTQ-Int4 (GPTQ_INT4)](#example-qwen3-30b-a3b-gptq-int4-gptq_int4)
  - [Example: Kimi-K2.5 (RAWINT4)](#example-kimi-k25-rawint4)
  - [Sending Requests](#sending-requests)
- [Performance Tuning](#performance-tuning)
- [FAQ](#faq)

## Supported Precision Formats

| `--kt-method` | Precision | Description |
|---------------|-----------|-------------|
| `BF16` | BF16 native precision | Zero precision loss, uses BF16 weights directly |
| `FP8` | FP8 block quantization |  |
| `GPTQ_INT4` | INT4 GPTQ |  |
| `RAWINT4` | Raw INT4 with BF16 scales | Used by Kimi-K2.5; weights stored in compressed SafeTensor format |


## Hardware Requirements

- **CPU**: x86-64 + AVX2 + FMA (Intel Haswell 2013+ / AMD Zen+)
- **GPU**: NVIDIA 24GB+ VRAM (RTX 3090/4090/5090, etc.)
- **Memory**: At least the size of the model weights (e.g., Qwen3-30B-A3B BF16 requires 64GB+)
- **OS**: Linux

## Installation

Build and install from source (one-click install for kt-kernel + SGLang):

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive

# One-click install
./install.sh
```

On AVX512 or AMX machines, you can also manually force AVX2 compilation:

```bash
export KT_RAWINT4_BACKEND=avx2
export CPUINFER_CPU_INSTRUCT=AVX2
export CPUINFER_ENABLE_AMX=OFF
./install.sh kt-kernel --manual
```



## Verification

```bash
# Check if the CPU supports AVX2
lscpu | grep -i avx2

# Check the loaded kt-kernel variant
python -c "import kt_kernel; print(kt_kernel.__cpu_variant__)"
# Expected output: avx2

# System diagnostics
kt doctor
```

## Starting the Inference Server

Use `--kt-method BF16`, `FP8`, `GPTQ_INT4`, or `RAWINT4`. KT-Kernel will **automatically detect** the CPU and fall back to the AVX2 backend when AVX512/AMX is unavailable.

### Example: Qwen3-30B-A3B (BF16)

```bash
# Download the model
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /path/to/Qwen3-30B-A3B

# Check physical core count and NUMA node count
lscpu | grep -E "^CPU\(s\)|Thread\(s\) per core|NUMA node\(s\)"

# Start the server (adjust kt-cpuinfer and kt-threadpool-count based on your hardware)
python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/Qwen3-30B-A3B \
  --kt-weight-path /path/to/Qwen3-30B-A3B \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 32 \
  --kt-method BF16 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 \
  --max-running-requests 2 \
  --served-model-name Qwen3 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --enable-p2p-check \
  --disable-shared-experts-fusion
```

### Example: Qwen3.5-35B-A3B-FP8 (FP8)

```bash
# Download the model
huggingface-cli download Qwen/Qwen3.5-35B-A3B-FP8 --local-dir /path/to/Qwen3.5-35B-A3B-FP8

# Start the server
python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --kt-weight-path /path/to/Qwen3.5-35B-A3B-FP8 \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 2 \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 400 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --max-running-requests 1 \
  --max-total-tokens 32000 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --disable-shared-experts-fusion
```

### Example: Qwen3-30B-A3B-GPTQ-Int4 (GPTQ_INT4)

```bash
# Download the model
huggingface-cli download Qwen/Qwen3-30B-A3B-GPTQ-Int4 --local-dir /path/to/Qwen3-30B-A3B-GPTQ-Int4

# Start the server
python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --kt-weight-path /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 2 \
  --kt-method GPTQ_INT4 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --max-running-requests 1 \
  --max-total-tokens 32000 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --disable-shared-experts-fusion
```

### Example: Kimi-K2.5 (RAWINT4)

> **Note**: The following command is optimized for 4x RTX PRO 6000 Blackwell (96GB each) + AMD Threadripper PRO 5995WX (64 cores, 1 NUMA node) + 256GB RAM.

```bash
# Download the model
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /path/to/Kimi-K2.5

# Start the server
python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/Kimi-K2.5 \
  --kt-weight-path /path/to/Kimi-K2.5 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 228 \
  --kt-enable-dynamic-expert-update \
  --kt-method RAWINT4 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --chunked-prefill-size 8192 \
  --max-running-requests 4 \
  --context-length 262144 \
  --enable-mixed-chunk \
  --tensor-parallel-size 4 \
  --enable-p2p-check \
  --disable-shared-experts-fusion
```

### Sending Requests

```bash
# Interactive chat
kt chat

# OpenAI-compatible API
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```



## Performance Tuning

- `--kt-cpuinfer`: set to the number of **physical cores**
- `--kt-threadpool-count`: set to the number of **NUMA nodes**
- `--kt-num-gpu-experts`: higher values reduce CPU load but increase GPU VRAM usage
- Memory bandwidth is often the bottleneck; high-frequency DDR5 memory helps significantly

## FAQ



**GPU OOM**
- Reduce `--kt-num-gpu-experts`, `--chunked-prefill-size`, `--max-total-tokens`
- Lower `--mem-fraction-static`

For more questions, see [FAQ](../FAQ.md).
