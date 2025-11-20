# Weight Quantization Tools

KT-Kernel provides weight conversion tools for CPU-GPU hybrid inference (e.g., integrating KTransformers with SGLang). Both tools work together to enable heterogeneous expert placement:

- **CPU Weights (`convert_cpu_weights.py`)**: Quantize weights to INT4/INT8 with AMX optimization for CPU-resident "cold" experts
- **GPU Weights (`convert_gpu_weights.py`)**: Apply GPTQ quantization (W4A16/W8A16) for GPU-resident "hot" experts

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
- `llmcompressor`: For GPTQ quantization
- `datasets`: For calibration data loading

### Overview

Apply GPTQ quantization to model weights for GPU-resident "hot" experts (frequently accessed) in CPU-GPU hybrid inference. This tool works together with `convert_cpu_weights.py` to enable heterogeneous expert placement:

- **GPU-resident experts** ("hot" experts) use GPTQ quantization (this tool) for efficient GPU memory usage
- **CPU-resident experts** ("cold" experts) use AMX-optimized INT4/INT8 quantization (convert_cpu_weights.py)
- **Attention layers, gates, and shared experts** remain in higher precision

This approach maximizes throughput and resource utilization by intelligently distributing experts across CPUs and GPUs.

### Quantization Types

- **W4A16**: 4-bit weights, 16-bit activations (GPTQ4)
- **W8A16**: 8-bit weights, 16-bit activations (GPTQ8)

### Basic Usage

```bash
python scripts/convert_gpu_weights.py \
  --model_id /path/to/model \
  --output_dir /path/to/output \
  --quant_type W4A16
```

### Advanced Options

#### Calibration Configuration

Control the calibration process for better quantization quality:

```bash
python scripts/convert_gpu_weights.py \
  --model_id /path/to/model \
  --output_dir /path/to/output \
  --quant_type W4A16 \
  --num_calibration_samples 512 \
  --max_sequence_length 2048 \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset_split train_sft
```

**Options:**
- `--num_calibration_samples`: Number of samples for calibration (default: 512)
- `--max_sequence_length`: Maximum sequence length (default: 2048)
- `--dataset`: HuggingFace dataset for calibration
- `--dataset_split`: Dataset split to use

### Examples

#### Example 1: Quantize Qwen3-Next-80B for Hybrid Inference (W4A16)

```bash
python scripts/convert_gpu_weights.py \
  --model_id /mnt/data/models/Qwen3-Next-80B-A3B-Thinking \
  --output_dir /mnt/data/models/Qwen3-Next-80B-A3B-Thinking-GPTQ4 \
  --quant_type W4A16 \
  --num_calibration_samples 512 \
  --max_sequence_length 2048 \
  --trust_remote_code
```
