# Running DeepSeek-V4-Flash with SGLang and KT-Kernel

This tutorial demonstrates how to run **DeepSeek-V4-Flash** model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. The hybrid path splits MXFP4 routed experts between CPU (KT-Kernel `cpuinfer`) and GPU (sglang `kt-num-gpu-experts`), enabling deployment on consumer-grade hardware.

## Table of Contents

- [Running DeepSeek-V4-Flash with SGLang and KT-Kernel](#running-deepseek-v4-flash-with-sglang-and-kt-kernel)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download Model Weights](#step-1-download-model-weights)
  - [Step 2: Quantize CPU Weights (Optional, for AMXINT4 mode)](#step-2-quantize-cpu-weights-optional-for-amxint4-mode)
  - [Step 3: Launch SGLang Server](#step-3-launch-sglang-server)
    - [Launch Command (8× RTX 5090 Example)](#launch-command-8-rtx-5090-example)
    - [Optional: Enable MTP (Multi-Token Prediction) Speculative Decoding](#optional-enable-mtp-multi-token-prediction-speculative-decoding)
  - [Step 4: Send Inference Requests](#step-4-send-inference-requests)
    - [Decode](#decode)
    - [Interactive Chat (kt chat)](#interactive-chat-kt-chat)

## Hardware Requirements

**Validated Configuration (this tutorial):**
- **GPU**: 8× NVIDIA RTX 5090 (32GB VRAM each, SM_120)
- **CPU**: x86 CPU with AVX512 support
- **RAM**: ≥256GB system memory
- **Storage**: ~340GB for model weights

**Supported GPU architectures** (auto-detected at startup; non-validated configurations should work but have not been benchmarked end-to-end):

| Arch | Compute Cap | MXFP4 MoE | NSA sparse MLA | Validated |
|------|------------|-----------|----------------|-----------|
| Hopper (H100 / H200) | SM_90 | triton_kernels | flash_mla wheel | — |
| Datacenter Blackwell (B100 / B200) | SM_100 | trtllm-fp4 | Triton fallback | — |
| Consumer Blackwell (RTX 5090) | SM_120 | triton_kernels | Triton fallback | ✓ |
| Ada Lovelace (RTX 4090 / L20 / L40) | SM_89 | triton_kernels | Triton fallback | — |
| Ampere (A100 / A6000) | SM_80 / SM_86 | triton_kernels | Triton fallback | ✗ (not supported) |


## Prerequisites

1. **KT-Kernel installed**:
   ```bash
   git clone https://github.com/kvcache-ai/ktransformers.git
   cd ktransformers
   git submodule update --init --recursive
   cd kt-kernel && ./install.sh
   ```

2. **SGLang installed** (kvcache-ai fork):
   ```bash
   ./install.sh   # from ktransformers root
   ```

3. **CUDA 12.8+** and **flashinfer ≥ 0.6.9** (`flashinfer-python` and `flashinfer-cubin` must be the same version):
   ```bash
   pip install --upgrade flashinfer-python flashinfer-cubin
   ```
   This upgrade is required (even though `sglang-kt` pins `flashinfer_python==0.6.3`) because V4-Flash's MXFP4 MoE module imports `mxfp8_quantize`, `trtllm_fp4_block_scale_routed_moe`, etc., which only exist in flashinfer ≥ 0.6.9.

4. **transformers==4.57.1** (V4-Flash is incompatible with the 5.x series):
   ```bash
   pip install "transformers==4.57.1"
   ```
   `transformers` 5.x adds default-valued fields to `PretrainedConfig` that make `DeepSeekV4Config`'s dataclass declaration raise `TypeError: non-default argument 'quantization_config' follows default argument` at import time. `sglang-kt`'s pyproject does not pin `transformers`, so a fresh `pip install` will pull the latest 5.x and break server startup; pinning explicitly to `4.57.1` is required until the upstream fix lands.

5. **tilelang** (manual install — required for the NSA sparse-MLA tilelang indexer path used on non-Hopper GPUs):
   ```bash
   pip install tilelang
   ```
   `sglang-kt`'s pyproject does not declare `tilelang` as a dependency, so `pip install ./python[all]` will not pull it in. Validated with `tilelang==0.1.8`.


## Step 1: Download Model Weights

```bash
mkdir -p /path/to/models
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /path/to/models/DeepSeek-V4-Flash
```

## Step 2: Quantize CPU Weights (Optional, for AMXINT4 mode)

This step is only needed if you want to run the CPU experts in **AMXINT4** mode instead (e.g., on Intel Xeon with AMX where INT4 is preferred over MXFP4).

### Conversion Command

For a 4-NUMA system with 64 physical cores assigned to CPU inference:

```bash
cd /path/to/ktransformers/kt-kernel

python scripts/convert_cpu_weights_ds4.py \
  --input-path /path/to/models/DeepSeek-V4-Flash \
  --input-type fp4 \
  --output /path/to/models/DeepSeek-V4-Flash-AMXINT4 \
  --quant-method int4 \
  --cpuinfer-threads 64 \
  --threadpool-count 4 \
  --no-merge-safetensor
```

The script auto-detects `model_type=deepseek_v4` and `expert_dtype=fp4` from `config.json`, dequantizes the MXFP4 routed experts (group size 32) on GPU, and re-quantizes them to AMX-INT4 layout on CPU. Both HF (`model.layers.{L}.mlp.experts.{E}.{proj}.weight`) and V4 inference (`layers.{L}.ffn.experts.{E}.{w1,w2,w3}.weight`) key formats are supported.

To use the converted weights, replace the relevant flags in Step 3's launch command:

```bash
  --kt-weight-path /path/to/models/DeepSeek-V4-Flash-AMXINT4 \
  --kt-method AMXINT4 \
```

## Step 3: Launch SGLang Server

### Launch Command (8× RTX 5090 Example)

```bash
numactl --interleave=all python -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30000 \
  --model /path/to/models/DeepSeek-V4-Flash \
  --kt-weight-path /path/to/models/DeepSeek-V4-Flash \
  --kt-method MXFP4 \
  --kt-num-gpu-experts 144 \
  --kt-cpuinfer 8 \
  --kt-threadpool-count 2 \
  --kt-gpu-prefill-token-threshold 4096 \
  --kt-enable-dynamic-expert-update \
  --tensor-parallel-size 8 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 2048 \
  --max-running-requests 4 \
  --max-total-tokens 32768 \
  --watchdog-timeout 3000 \
  --disable-shared-experts-fusion \
  --cuda-graph-bs 1 2 4 \
  --cuda-graph-max-bs 4 \
  --trust-remote-code
```

It takes about 4-5 minutes to start the server (weight load + CUDA Graph capture).

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for detailed parameter tuning guidelines.

### Optional: Enable MTP (Multi-Token Prediction) Speculative Decoding

V4-Flash ships a NextN draft head that can be run as EAGLE-style speculative decoding for ~1.2× throughput on single-request decode (validated 26.5 → 32.74 tok/s on 8× RTX 5090, 90% accept rate at chain depth 1).

Append the following flags to the launch command above:

```bash
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-moe-runner-backend auto \
```

## Step 4: Send Inference Requests

### Decode

```bash
curl -s -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain quantum computing in detail:",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 256}
  }'
```

### Interactive Chat (kt chat)

The `kt` CLI ships with an OpenAI-compatible chat client that talks to the SGLang server's `/v1/chat/completions` endpoint:

```bash
kt chat --host 127.0.0.1 --port 30000 --temperature 0.7 --max-tokens 2048
```


