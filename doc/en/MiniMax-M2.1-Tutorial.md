# Running MiniMax-M2.1 with Native Precision using SGLang and KT-Kernel

This tutorial demonstrates how to run MiniMax-M2.1 model inference using SGLang integrated with KT-Kernel. MiniMax-M2.1 provides native FP8 weights, enabling efficient GPU inference with reduced memory footprint while maintaining high accuracy.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Step 1: Download Model Weights](#step-1-download-model-weights)
- [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
- [Step 3: Send Inference Requests](#step-3-send-inference-requests)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Overview

MiniMax-M2.1 is a large MoE (Mixture of Experts) model that provides native FP8 weights. This tutorial uses KT-Kernel's FP8 support to enable CPU-GPU heterogeneous inference:

- **FP8 GPU Inference**: Native FP8 precision for GPU-side computation, providing both memory efficiency and computational accuracy
- **CPU-GPU Heterogeneous Architecture**:
  - Hot experts and attention modules run on GPU with FP8 precision
  - Cold experts offloaded to CPU for memory efficiency

## Hardware Requirements

**Minimum Configuration:**
- **GPU**: NVIDIA RTX 4090 24 GB (or equivalent with at least 24GB VRAM available)
- **CPU**: x86 CPU with AVX512 support (e.g., Intel Sapphire Rapids, AMD EPYC)
- **RAM**: At least <!-- TODO: RAM requirement -->GB system memory
- **Storage**: 220 GB for model weights (same weight dir for GPU and CPU)

**Tested Configuration:**

- **GPU**: 1/2 x NVIDIA GeForce RTX 5090 (32 GB)
- **CPU**: 2 x AMD EPYC 9355 32-Core Processor (128 threads)
- **RAM**: 1TB DDR5 5600MT/s ECC
- **OS**: Linux (Ubuntu 20.04+ recommended)

## Prerequisites

Before starting, ensure you have:

1. **SGLang installed** - Follow [SGLang integration steps](./kt-kernel_intro.md#integration-with-sglang)
2. **KT-Kernel installed** - Follow the [installation guide](./kt-kernel_intro.md#installation)

Note: Currently, please clone our custom SGLang repository:

```bash
git clone https://github.com/kvcache-ai/sglang.git
cd sglang
pip install -e "python[all]"
```

3. **CUDA toolkit** - CUDA 12.0+ recommended for FP8 support
4. **Hugging Face CLI** - For downloading models:
   ```bash
   pip install -U huggingface-hub
   ```

## Step 1: Download Model Weights

<!-- TODO: using kt-cli -->
## Step 2: Launch SGLang Server


<!-- TODO: using kt-cli -->

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for detailed parameter tuning guidelines.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--kt-method FP8` | Enable FP8 inference mode for MiniMax-M2.1 native FP8 weights. |
| `--kt-cpuinfer` | Number of CPU inference threads. Set to physical CPU cores (not hyperthreads). |
| `--kt-threadpool-count` | Number of thread pools. Set to NUMA node count. |
| `--kt-num-gpu-experts` | Number of experts kept on GPU for decoding. |
| `--chunked-prefill-size` | Maximum tokens per prefill batch. |
| `--max-total-tokens` | Maximum total tokens in KV cache. |
| `--kt-gpu-prefill-token-threshold` | Token threshold for layerwise prefill strategy. |

## Step 3: Send Inference Requests


## Performance

### Throughput (tokens/s)

The following benchmarks were measured with single concurrency (Prefill tps / Decode tps):

| GPU  | CPU  | PCIe |  2048 tokens | 8192 tokens | 32768 tokens |
|------------|-------------|-------------|-------------|-------------|--------------|
| 1 x RTX 4090 (48 GB) | 2 x Intel Xeon Platinum 8488C| PCIe 4.0 | 129 / 21.8 | 669 / 20.9 | 1385 / 18.5 |
| 2 x RTX 4090 (48 GB) | 2 x Intel Xeon Platinum 8488C| PCIe 4.0 | 139 / 23.6 | 1013 / 23.3 | 2269 / 21.6 |
| 1 x RTX 5090 (32 GB) | 2 x AMD EPYC 9355 | PCIe 5.0 | 408 / 32.1 | 1196 / 31.4 | 2540 / 27.6 |
| 2 x RTX 5090 (32 GB) | 2 x AMD EPYC 9355 | PCIe 5.0 | 414 / 34.3 | 1847 / 33.1 | 4007 / 31.8 |

### Comparison with llama.cpp

We benchmarked KT-Kernel + SGLang against llama.cpp to demonstrate the performance advantages of our CPU-GPU heterogeneous inference approach.


<!-- TODO: Add prefill performance comparison chart -->
<!-- ![Prefill Performance Comparison](./images/minimax-m2.1-prefill-comparison.png) -->

| Input Length | llama.cpp (tokens/s) | KT-Kernel (tokens/s) | Speedup |
|--------------|----------------------|----------------------|---------|
| <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

### Key Observations

<!-- TODO: Add key observations and analysis, e.g.:
- KT-Kernel achieves Xx speedup in prefill compared to llama.cpp
- Decode performance shows Xx improvement due to GPU expert caching
- Memory efficiency comparison
- Scalability with different batch sizes
-->

## Troubleshooting
<!-- TODO: -->

## Advance Use Casee: Running Claude Code with MiniMax-M2.1 Local Backend
<!-- TODO: -->

## Additional Resources

- [KT-Kernel Documentation](../../kt-kernel/README.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [KT-Kernel Parameters Reference](../../kt-kernel/README.md#kt-kernel-parameters)