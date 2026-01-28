# Running Native Precision Models with SGLang and KT-Kernel

This tutorial demonstrates how to run native precision MoE model inference using SGLang integrated with KT-Kernel. KTransformers v0.5.1+ supports multiple native precision formats, enabling efficient inference across various model architectures.

## Table of Contents

- [Supported Precision Formats](#supported-precision-formats)
- [Supported Models](#supported-models)
- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Launch Server](#launch-server)
  - [Example Configurations](#example-configurations)
  - [Key Parameters Reference](#key-parameters-reference)
- [Send Inference Requests](#send-inference-requests)
- [Technical Highlights](#technical-highlights)
  - [Experts Scheduling](#experts-scheduling)
  - [Dual Prefill Mechanism](#dual-prefill-mechanism)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Supported Precision Formats

KTransformers supports multiple native precision formats via the `--kt-method` parameter:

| kt-method | Precision Format | Description | Instruction Set |
|-----------|-----------------|-------------|-----------------|
| `BF16` | BF16 Native | Zero precision loss, original weights | AMX + AVX512 |
| `FP8` | FP8 Blockwise | Block-wise scale quantization | AVX512 |
| `FP8_PERCHANNEL` | FP8 Per-Channel | Per-channel scale quantization | AVX512 |
| `RAWINT4` | INT4 Native | Same INT4 weights for CPU and GPU | AVX512 |

## Supported Models

| Model(sorted by lexicographical order) | kt-method | Precision | 
|-------|-----------|------------|
| **DeepSeek-V3/R1/V3.2** | `FP8` | FP8 |
| **GLM-4.7** | `FP8_PERCHANNEL`, `BF16` | FP8, BF16 |
| **Kimi-K2-Thinking** | `RAWINT4` | INT4 Native |
| **MiniMax-M2/M2.1** | `FP8` | FP8 |
| **Qwen3-235B-A22B** | `FP8`, `BF16` | FP8, BF16 |
| **Qwen3-30-A3B** | `FP8`, `BF16` | FP8, BF16 |
| **Qwen3-Next-80B-A3B** | `FP8`, `BF16` | FP8, BF16 |

## Hardware Requirements

**Minimum Configuration:**
- **GPU**: 1-2 x NVIDIA GPU with at least 24GB VRAM (RTX 4090/5090 or equivalent, depending on model)
- **CPU**: x86 CPU with AVX512 support (Intel Sapphire Rapids+, AMD EPYC)
  - BF16 additionally benefits from AMX support
- **RAM**: At least as much RAM as model size (e.g., 256GB+ for MiniMax-M2.1)
- **Storage**: Sufficient space for model weights (varies by model)

**Recommended Configuration:**
- **GPU**: 1-8 x NVIDIA RTX 5090 (32 GB) or equivalent
- **CPU**: 2 x AMD EPYC 9355 32-Core / Intel Xeon Platinum 8488C
- **RAM**: 1TB DDR5 5600MT/s ECC
- **PCIe**: PCIe 5.0 for optimal CPU-GPU data transfer
- **OS**: Linux (Ubuntu 20.04+ recommended)

## Prerequisites

Before starting, ensure you have:

1. **SGLang installed**

    Clone and install the custom SGLang repository:

    ```bash
    git clone https://github.com/kvcache-ai/sglang.git
    cd sglang
    pip install -e "python[all]"
    ```

2. **KT-Kernel installed**

    Follow the [kt-kernel installation guide](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md):

    ```bash
    git clone https://github.com/kvcache-ai/ktransformers.git
    cd ktransformers/kt-kernel
    ./install.sh
    ```

    Verify the installation:

    ```bash
    kt version
    ```

3. **CUDA toolkit** - CUDA 12.0+ recommended
4. **Hugging Face CLI** - For downloading models:
   ```bash
   pip install -U huggingface-hub
   ```
   
## Launch Server

### Example Configurations
For now, only `MiniMax-M2/M2.1`, `DeepSeek-V3/R1-0528/V3.2`, `Kimi-K2-Thinking` can run with kt-cli.

**DeepSeek-V3.2**

```bash
kt run V3.2 --kt-enable-dynamic-expert-update
```

**GLM-4.7**

```bash
python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 30000 \
    --model /path/to/GLM-4.7/ \
    --kt-weight-path /path/to/GLM-4.7/ \
    --kt-cpuinfer 100 \
    --kt-threadpool-count 2 \
    --kt-num-gpu-experts 15 \
    --kt-method BF16 \
    --kt-enable-dynamic-expert-update \
    --attention-backend flashinfer \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 16384 \
    --max-running-requests 2 \
    --max-total-tokens 32768 \
    --trust-remote-code \
    --served-model-name GLM-4.7 \
    --enable-mixed-chunk \
    --tensor-parallel-size 8 \
    --enable-p2p-check \
    --disable-shared-experts-fusion \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --watchdog-timeout 3000 \
    --kt-gpu-prefill-token-threshold 1024
```

**GLM-4.7-FP8**

```bash
python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 30000 \
    --model /path/to/GLM-4.7-FP8/ \
    --kt-weight-path /path/to/GLM-4.7-FP8/ \
    --kt-cpuinfer 100 \
    --kt-threadpool-count 2 \
    --kt-num-gpu-experts 80 \
    --kt-method FP8_PERCHANNEL \
    --kt-enable-dynamic-expert-update \
    --attention-backend flashinfer \
    --mem-fraction-static 0.75 \
    --chunked-prefill-size 16384 \
    --max-running-requests 4 \
    --max-total-tokens 100000 \
    --trust-remote-code \
    --served-model-name GLM-4.7 \
    --enable-mixed-chunk \
    --tensor-parallel-size 8 \
    --enable-p2p-check \
    --disable-shared-experts-fusion \
    --watchdog-timeout 3000 \
    --fp8-gemm-backend triton \
    --kt-gpu-prefill-token-threshold 2048
```

**Qwen3-235B-A22B**

```bash
python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 30000 \
    --model /path/to/Qwen3-235B-A22B \
    --kt-weight-path /path/to/Qwen3-235B-A22B \
    --kt-cpuinfer 100 \
    --kt-threadpool-count 2 \
    --kt-num-gpu-experts 20 \
    --kt-method FP8 \
    --kt-enable-dynamic-expert-update \
    --kt-expert-placement-strategy uniform \
    --attention-backend flashinfer \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 16384 \
    --max-running-requests 4 \
    --max-total-tokens 100000 \
    --trust-remote-code \
    --served-model-name Qwen3-235B-A22B \
    --enable-mixed-chunk \
    --tensor-parallel-size 8 \
    --enable-p2p-check \
    --kt-gpu-prefill-token-threshold 2048
```

### Key Parameters Reference

| Parameter | Description |
|-----------|-------------|
| `--kt-method` | Precision format: `BF16`, `FP8_PERCHANNEL`, `FP8`, `RAWINT4`, `AMXINT4` |
| `--kt-cpuinfer` | Number of CPU inference threads (set to ~90% of physical cores) |
| `--kt-threadpool-count` | Number of thread pools (set to NUMA node count) |
| `--kt-num-gpu-experts` | Number of experts kept on GPU per layer |
| `--kt-enable-dynamic-expert-update` | Enable dynamic expert placement updates during Layerwise Prefill |
| `--kt-expert-placement-strategy` | Expert placement strategy |
| `--kt-gpu-prefill-token-threshold` | Token threshold for triggering Layerwise Prefill |
| `--chunked-prefill-size` | Maximum tokens per prefill batch |
| `--max-total-tokens` | Maximum total tokens in KV cache |

## Send Inference Requests

Once the server is running (default: `http://localhost:30000`), you can interact with the model:

### Option A: Interactive Chat with KT CLI

```bash
kt chat
```

This opens an interactive terminal chat session. Type your messages and press Enter to send. Use `Ctrl+C` to exit.

### Option B: OpenAI-Compatible API

The server exposes an OpenAI-compatible API at `http://localhost:30000/v1`.

**curl example (streaming):**

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_NAME",
    "messages": [{"role": "user", "content": "Hello! What can you help me with?"}],
    "stream": true
  }'
```

**Python example:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="MODEL_NAME",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Technical Highlights

### Experts Scheduling

See [CPU-GPU Expert Scheduling Tutorial](./experts-sched-Tutorial.md) for details.

### Dual Prefill Mechanism

KTransformers implements an adaptive dual prefill mechanism based on input token count:

| Mode | Trigger Condition | Computation |
|------|-------------------|-------------|
| **CPU-GPU Hybrid** | num_tokens < threshold | GPU + CPU |
| **Layerwise Prefill** | num_tokens >= threshold | GPU (CPU weights transferred to GPU) |

Set the `kt-gpu-prefill-token-threshold` parameter for best performance based on your workload.

## Troubleshooting

### OOM (Out of Memory) Issues

Layerwise prefill requires extra VRAM. If you encounter OOM, adjust these parameters:

| Parameter | VRAM Impact |
|-----------|-------------|
| `--kt-num-gpu-experts` | Reduces expert weight VRAM usage |
| `--chunked-prefill-size` | Reduces prefill extra VRAM allocation |
| `--max-total-tokens` | Reduces KV cache VRAM usage |
| `--mem-fraction-static` | Adjusts static memory fraction |

**Tips:**
- Test with an input of length `chunked-prefill-size` to verify configuration
- Reduce `--kt-num-gpu-experts` if GPU memory is limited
- For multi-GPU setups, ensure `--enable-p2p-check` is enabled
- For FP8 models, `--fp8-gemm-backend triton` may be required

## Additional Resources

- [KT-Kernel Documentation](../../../kt-kernel/README.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [MiniMax-M2.1 Tutorial](./MiniMax-M2.1-Tutorial.md) - Detailed guide for MiniMax-M2.1 and other FP8 models
- [Kimi-K2-Thinking Tutorial](./Kimi-K2-Thinking-Native.md) - Detailed guide for Kimi-K2-Thinking
