# Weight Quantization Tools

KT-Kernel provides weight conversion tools for CPU-GPU hybrid inference (e.g., integrating KTransformers with SGLang). Both tools work together to enable heterogeneous expert placement:

- **CPU Weights (`convert_cpu_weights.py`)**: Quantize weights to INT4/INT8 with AMX optimization for CPU-resident "cold" experts
- **GPU Weights (`convert_gpu_weights.py`)**: Apply GPTQ/RTN quantization (W4A16/W8A16) for GPU-resident "hot" experts
- **KT Fused Expert LoRA (`convert_kt_to_sglang_adapter.py`)**: Convert KT SFT fused expert LoRA checkpoints into adapter-only SafeTensors directories

---

## KT Fused Expert LoRA Adapter Conversion

KT SFT fused expert LoRA saves MoE expert LoRA tensors in `fused_expert_lora.safetensors` using compact 3D tensors:

```
layers.{L}.experts.gate_lora_a
layers.{L}.experts.gate_lora_b
layers.{L}.experts.up_lora_a
layers.{L}.experts.up_lora_b
layers.{L}.experts.down_lora_a
layers.{L}.experts.down_lora_b
```

Use `convert_kt_to_sglang_adapter.py` to convert raw KT SFT output into one merged SGLang adapter directory:

```bash
python scripts/convert_kt_to_sglang_adapter.py /path/to/kt_adapter /path/to/sglang_adapter \
  --base-model-name-or-path /path/to/base_model \
  --lora-alpha 16 \
  --overwrite
```

Output:

```
sglang_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

The converter merges the existing non-expert `adapter_model.safetensors` with expanded expert tensors from `fused_expert_lora.safetensors`. Pass this merged directory to SGLang with:

```bash
--enable-lora \
--lora-paths my_lora=/path/to/sglang_adapter
```

The KTransformers SGLang fork will auto-split the merged adapter internally at server startup. Users do not need to pass separate expert and non-expert adapter paths in the normal workflow.

Optional split outputs for debugging:

```bash
python scripts/convert_kt_to_sglang_adapter.py /path/to/kt_adapter /path/to/sglang_adapter \
  --base-model-name-or-path /path/to/base_model \
  --expert-output-dir /path/to/expert_adapter \
  --nonexpert-output-dir /path/to/nonexpert_adapter \
  --overwrite
```

Existing PEFT prefixes such as `base_model.model.` are stripped to match SGLang's loader. Scaling is not folded into the LoRA B tensors. Runtime scaling remains `lora_alpha / r`; if the input directory has no `adapter_config.json`, pass `--lora-alpha` explicitly.

This script only converts adapter files. Serving compatibility depends on the KTransformers SGLang runtime branch being used.

### Optional Integration Validation

The unit tests use synthetic tensors and run without model files. To validate a real KT adapter directory, set these environment variables:

```bash
export KT_LORA_ADAPTER_DIR=/path/to/kt_adapter
export KT_LORA_BASE_MODEL=/path/to/base_model
export KT_LORA_ALPHA=16  # required only if the input has no adapter_config.json
```

Then run:

```bash
python -m pytest kt-kernel/test/per_commit/test_convert_kt_to_sglang_adapter_integration.py -q
```

To run a large adapter conversion smoke test, also set:

```bash
export KT_LORA_LARGE_ADAPTER_DIR=/path/to/large_kt_adapter
```

These integration tests check real fused tensor splitting, optional `adapter_model.safetensors` merging, `adapter_config.json` compatibility with `sglang.srt.lora.lora_config.LoRAConfig`, and large-file readability. They intentionally do not start an SGLang server or validate runtime `FusedMoE` LoRA application.

---

## CPU Weight Quantization

Convert weights to INT4/INT8 format optimized for AMX inference on CPU. These quantized weights are used for "cold" experts (less frequently accessed) that run on CPU in hybrid inference scenarios.

### Quantization Methods

- **INT4**: 4-bit quantization for maximum memory efficiency
- **INT8**: 8-bit quantization for better accuracy

### Supported Input Formats

- **FP8**: 8-bit floating point with automatic dequantization
- **FP16**: 16-bit floating point
- **BF16**: BFloat16 format

> **⚠️ Precision Warning:** Quantizing directly from FP8 to INT4/INT8 may cause significant accuracy degradation. For best results, use the original **BF16** model as the source for INT4/INT8 quantization.

## Basic Usage

### Quantize BF16 model to INT4

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/bf16/model \
  --input-type bf16 \
  --output /path/to/output \
  --quant-method int4
```

### Quantize FP16 model to INT8

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/fp16/model \
  --input-type fp16 \
  --output /path/to/output \
  --quant-method int8
```

### Quantize FP8 model to INT4

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/fp8/model \
  --input-type fp8 \
  --output /path/to/output \
  --quant-method int4
```

## Output Format

By default, the converted weights are saved in SafeTensors format with NUMA-aware layout:

```
output_dir/
├── model-00001-of-00050.safetensors
├── model-00002-of-00050.safetensors
├── ...
├── config.json
└── tokenizer files...
```

Each expert's weights are split across NUMA nodes for optimal memory access:
- `blk.{layer}.ffn_{proj}_exps.{expert}.numa.{numa_idx}.weight`: Quantized weights
- `blk.{layer}.ffn_{proj}_exps.{expert}.numa.{numa_idx}.scale`: Quantization scales

## Advanced Options

### Low Memory Mode

For systems with insufficient memory to complete full model quantization, use the `--no-merge-safetensor` flag to keep weights in layer folder structure without merging into safetensor files:

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/model \
  --input-type bf16 \
  --output /path/to/output \
  --quant-method int4 \
  --no-merge-safetensor
```

This will save quantized weights in the following folder structure:

```
output_dir/
├── _layer_0/
│   ├── _numa_0/
│   │   ├── INT4_down_0_*.kt
│   │   ├── INT4_gate_0_*.kt
│   │   └── INT4_up_0_*.kt
│   └── _numa_1/
│       └── ...
├── _layer_1/
│   └── ...
└── ...
```

**When to use `--no-merge-safetensor`:**
- Machine runs out of memory during the merge step
- Need to process very large models on memory-constrained systems
- Want to preserve intermediate layer-wise quantized weights

### Resume Layer

For memory-constrained systems that are unable to complete quantization despite enabling low memory mode with `--no-merge-safetensor`, restart the script with the `--resume-layer` arg to specify the layer from which to continue the conversion process. In the example below, we skip layers 0-11 and resume conversion starting with layer 12.

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/model \
  --input-type bf16 \
  --output /path/to/output \
  --quant-method int4 \
  --no-merge-safetensor
  --resume-layer 12
```

## Examples

### Example 1: Quantize DeepSeek-V3.1 (FP8 → INT4)

```bash
python scripts/convert_cpu_weights.py \
  --input-path /mnt/data/models/DeepSeek-V3.1 \
  --input-type fp8 \
  --output /mnt/data/models/DeepSeek-V3.1-INT4 \
  --quant-method int4 \
  --cpuinfer-threads 60 \
  --threadpool-count 2
```

### Example 2: Quantize Qwen3-Next-80B (BF16 → INT4, Low Memory)

```bash
python scripts/convert_cpu_weights.py \
  --input-path /mnt/data/models/Qwen3-Next-80B-A3B-Instruct \
  --input-type bf16 \
  --output /mnt/data/models/Qwen3-Next-80B-A3B-Instruct-INT4 \
  --quant-method int4 \
  --cpuinfer-threads 60 \
  --threadpool-count 2 \
  --no-merge-safetensor
```

---

## GPU Weight Quantization

### Prerequisites

GPU weight quantization requires additional dependencies. Install them before proceeding:

```bash
pip install accelerate transformers llmcompressor datasets
```

**Required packages:**
- `accelerate`: For distributed model loading and device mapping
- `transformers`: For model and tokenizer loading
- `llmcompressor`: For quantization (supports GPTQ and RTN methods)
- `datasets`: For calibration data loading (GPTQ only)

**Documentation:** This tool is based on llmcompressor. For more details, see [llmcompressor quantization guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/getting-started/compress/#select-a-quantization-method-and-scheme).

### Overview

Apply weight quantization to model weights for GPU-resident "hot" experts (frequently accessed) in CPU-GPU hybrid inference. This tool works together with `convert_cpu_weights.py` to enable heterogeneous expert placement:

- **GPU-resident experts** ("hot" experts) use GPTQ/RTN quantization (this tool) for efficient GPU memory usage
- **CPU-resident experts** ("cold" experts) use AMX-optimized INT4/INT8 quantization (convert_cpu_weights.py)
- **Attention layers, gates, and shared experts** remain in higher precision

This approach maximizes throughput and resource utilization by intelligently distributing experts across CPUs and GPUs.

### Quantization Methods

#### 1. GPTQ (Calibration-based, Default)
**Pros:**
- Higher accuracy through calibration-based quantization
- Recommended for production deployments

**Cons:**
- Requires calibration dataset
- Slower quantization process
- Higher memory requirements (needs Hessian matrix)

#### 2. RTN (Round-To-Nearest)
**Pros:**
- Fast quantization (no calibration needed)
- Lower memory requirements
- Good for quick testing and prototyping

**Cons:**
- Slightly lower accuracy compared to GPTQ
- No calibration optimization

### Quantization Types

- **W4A16**: 4-bit weights, 16-bit activations (INT4)
- **W8A16**: 8-bit weights, 16-bit activations (INT8)

### Basic Usage

#### GPTQ Quantization (Recommended for Production)
```bash
python scripts/convert_gpu_weights.py \
  --model_id /path/to/model \
  --output_dir /path/to/output \
  --quant_method GPTQ \
  --quant_type W4A16
```

#### RTN Quantization (Fast, for Testing)
```bash
python scripts/convert_gpu_weights.py \
  --model_id /path/to/model \
  --output_dir /path/to/output \
  --quant_method RTN \
  --quant_type W4A16
```

### Memory Requirements

Understanding memory requirements is crucial for successful quantization. The requirements differ significantly between RTN and GPTQ methods.

#### RTN Memory Requirements

RTN only requires memory for quantization parameters (scales/zero-points):

| Component | Requirement |
|-----------|-------------|
| **DRAM (CPU Memory)** | ≥ Total model parameters |
| **VRAM (GPU Memory)** | ≥ Single layer parameters |

**Example: DeepSeek-R1-0528-BF16 (684B parameters)**
- DRAM: ~1368 GB (684B params × 2 bytes)
- VRAM: ~22.4 GB (1 layer)

#### GPTQ Memory Requirements

GPTQ requires additional memory for Hessian matrices during calibration:

| Component | Requirement |
|-----------|-------------|
| **DRAM (CPU Memory)** | ≥ Total model parameters |
| **VRAM (GPU Memory)** | ≥ Single layer parameters × 2 |

The Hessian matrix is approximately the same size as the layer weights and is used to increase accuracy recovery.

**Example: DeepSeek-R1-0528-BF16 (684B parameters)**
- DRAM: ~1368 GB (684B params × 2 bytes)
- VRAM: ~44.8 GB (1 layer × 2 for Hessian matrix)

#### Method Comparison

| Method | Speed | VRAM | Accuracy | Use Case |
|--------|-------|------|----------|----------|
| **RTN** | Fast | Low (~22GB) | Good | Testing, prototyping |
| **GPTQ** | Slow | High (~45GB) | Better | Production deployment |

### Advanced Options

#### Calibration Configuration (GPTQ Only)

For GPTQ quantization, control the calibration process for better quantization quality:

```bash
python scripts/convert_gpu_weights.py \
  --model_id /path/to/model \
  --output_dir /path/to/output \
  --quant_method GPTQ \
  --quant_type W4A16 \
  --num_calibration_samples 512 \
  --max_sequence_length 2048 \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset_split train_sft
```

**Options (GPTQ only):**
- `--num_calibration_samples`: Number of samples for calibration (default: 512)
- `--max_sequence_length`: Maximum sequence length (default: 2048)
- `--dataset`: HuggingFace dataset for calibration
- `--dataset_split`: Dataset split to use
- `--dampening_frac`: Dampening fraction to reduce quantization noise (default: 0.1)

#### Memory Management

Use `--max_gpu_memory` to limit GPU memory usage and offload remaining layers to CPU:

```bash
python scripts/convert_gpu_weights.py \
  --model_id /path/to/model \
  --output_dir /path/to/output \
  --quant_method GPTQ \
  --quant_type W4A16 \
  --max_gpu_memory "40GiB"
```

**Recommended settings for GPTQ:**

| GPU VRAM | Suggested `--max_gpu_memory` | Notes |
|----------|------------------------------|-------|
| 24 GiB   | 10-12 GiB | Reserve ~50% for Hessian |
| 48 GiB   | 24-30 GiB | Reserve ~40% for Hessian |
| 80 GiB   | 40-50 GiB | Reserve ~40% for Hessian |

**Recommended settings for RTN:**

| GPU VRAM | Suggested `--max_gpu_memory` | Notes |
|----------|------------------------------|-------|
| 24 GiB   | 18-20 GiB | No Hessian needed |
| 48 GiB   | 40-45 GiB | No Hessian needed |
| 80 GiB   | 70-75 GiB | No Hessian needed |

**Options:**
- `--max_gpu_memory`: Maximum GPU memory for model weights per device (e.g., '40GiB')
- `--max_cpu_memory`: Maximum CPU memory (default: 1000GiB when `--max_gpu_memory` is set)

**Important:** llmcompressor does not support disk offloading. Ensure your machine has enough GPU + CPU memory to load the entire model. If you still encounter OOM:
1. Use RTN instead of GPTQ (requires less memory)
2. Reduce `--num_calibration_samples` (GPTQ only, e.g., 256)
3. Reduce `--max_sequence_length` (GPTQ only, e.g., 1024)
4. Use `--force_cpu` to run entirely on CPU (slower but avoids GPU OOM)

### Examples

#### Example 1: GPTQ Quantization for Production (Qwen3-Next-80B, W4A16)

```bash
python scripts/convert_gpu_weights.py \
  --model_id /mnt/data/models/Qwen3-Next-80B-A3B-Instruct \
  --output_dir /mnt/data/models/Qwen3-Next-80B-A3B-Instruct-GPTQ-W4A16 \
  --quant_method GPTQ \
  --quant_type W4A16 \
  --num_calibration_samples 512 \
  --max_sequence_length 2048 \
  --max_gpu_memory "40GiB" \
  --trust_remote_code
```

#### Example 2: RTN Quantization for Fast Testing (DeepSeek-R1, W4A16)

```bash
python scripts/convert_gpu_weights.py \
  --model_id /mnt/data/models/DeepSeek-R1-0528-BF16 \
  --output_dir /mnt/data/models/DeepSeek-R1-0528-RTN-W4A16 \
  --quant_method RTN \
  --quant_type W4A16 \
  --max_gpu_memory "70GiB" \
  --trust_remote_code
```

#### Example 3: GPTQ with Custom Calibration Dataset (GLM-4.5-Air, W8A16)

```bash
python scripts/convert_gpu_weights.py \
  --model_id /mnt/data/models/GLM-4.5-Air \
  --output_dir /mnt/data/models/GLM-4.5-Air-GPTQ-W8A16 \
  --quant_method GPTQ \
  --quant_type W8A16 \
  --dataset "tatsu-lab/alpaca" \
  --dataset_split "train" \
  --num_calibration_samples 256 \
  --max_gpu_memory "40GiB" \
  --trust_remote_code
```
