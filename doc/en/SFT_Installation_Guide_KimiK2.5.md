# Kimi-K2.5 LoRA SFT Tutorial

This tutorial demonstrates how to perform **LoRA Supervised Fine-Tuning (SFT)** on **Kimi-K2.5** using **LlamaFactory** with **KTransformers** as the backend, and then serve the fine-tuned model using **SGLang**.

The workflow is:

```txt
KTransformers + LlamaFactory LoRA SFT → (Optional) LlamaFactory Verification → SGLang Serving
```

## Table of Contents

- [Hardware Requirements](https://chatgpt.com/c/6975bb7f-52e0-839c-a727-ec4b5d6723b5#hardware-requirements)
- [Prerequisites](https://chatgpt.com/c/6975bb7f-52e0-839c-a727-ec4b5d6723b5#prerequisites)
- [Step 0: Environment Setup (Method 1: Source Install)](https://chatgpt.com/c/6975bb7f-52e0-839c-a727-ec4b5d6723b5#step-0-environment-setup-method-1-source-install)
- [Step 1: Prepare Model Weights (BF16 for SFT)](https://chatgpt.com/c/6975bb7f-52e0-839c-a727-ec4b5d6723b5#step-1-prepare-model-weights-bf16-for-sft)
- [Step 2: Prepare YAML for LoRA SFT (KTransformers Backend)](https://chatgpt.com/c/6975bb7f-52e0-839c-a727-ec4b5d6723b5#step-2-prepare-yaml-for-lora-sft-ktransformers-backend)
- [Step 3: Run LoRA SFT](https://chatgpt.com/c/6975bb7f-52e0-839c-a727-ec4b5d6723b5#step-3-run-lora-sft)
- [Step 4: Post-SFT Quick Verification with LlamaFactory (Optional)](https://chatgpt.com/c/6975bb7f-52e0-839c-a727-ec4b5d6723b5#step-4-post-sft-quick-verification-with-LlamaFactory-optional)
- [Step 5: SGLang Serving with LoRA (Recommended Delivery Path)](https://chatgpt.com/c/6975bb7f-52e0-839c-a727-ec4b5d6723b5#step-5-sglang-serving-with-lora-recommended-delivery-path)

## Hardware Requirements

### Training (LoRA SFT)

- **LlamaFactory + KTransformers**
- **GPU**: 4 * NVIDIA RTX 4090 24GB (or equivalent with at least total 48GB VRAM available)
- **CPU**: x86 CPU with AMX support
- **RAM**: At least 2TGB system memory
- Swap can be used if CPU memory is insufficient

### Inference (LoRA Adapter + Original Model)

- **SGLang + KTransformers**
- **GPU**: 2 * NVIDIA RTX 4090 24GB (or equivalent with at least total 48GB VRAM available)
- **CPU**: x86 CPU with AVX512F support (e.g., Intel Sapphire Rapids)
- **RAM**: At least 600GB system memory
- **Storage**: ~600GB for model weights (native INT4 weight, same weight dir for CPU and GPU)



## Step 0: Environment Setup

We recommend to separate **two conda environments**:

| Environment | Purpose                                             |
| ----------- | --------------------------------------------------- |
| `kt-kernel` | Inference & serving (KTransformers + SGLang)        |
| `kt-sft`    | Training (LlamaFactory + KTransformers SFT backend) |

### 0.1 Inference Environment: `kt-kernel`

```bash
conda create -n kt-kernel python=3.11
conda activate kt-kernel

git clone https://github.com/kvcache-ai/ktransformers.git
git checkout kimi_k2.5
git submodule update --init --recursive
cd kt-kernel && ./install.sh
```

### 0.2 Install SGLang (Inference / Serving)

**Recommended for Kimi-K2.5:**

```bash
git clone https://github.com/kvcache-ai/sglang.git
cd sglang
git checkout kimi_k2.5
pip install -e "python[all]"
```

### 0.3 Training Environment: `kt-sft`

```bash
conda create -n kt-sft python=3.11
conda activate kt-sft

git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
```

### 0.4 Install KTransformers SFT Dependencies

```bash
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
conda install -y -c nvidia/label/cuda-11.8.0 cuda-runtime

# Install matching wheels (recommended), from https://github.com/kvcache-ai/ktransformers/releases
pip install ktransformers-<matching-version>.whl
pip install flash_attn-<matching-version>.whl
```

## Step 1: Prepare Model Weights (BF16 for SFT)

### 1.1 Download INT4 Weights

KTransformers **requires BF16 weights for SFT**.

```bash
# Download Kimi-K2.5 (RAW-INT4 for both CPU and GPU)
huggingface-cli download moonshotai/Kimi-K2.5 \
  --local-dir /path/to/kimi-k2.5
```

### 1.2 Convert INT4 → BF16

Kimi-K2.5 base model is in **INT4** format, convert it to **BF16** before SFT.

## Step 2: Prepare YAML for LoRA SFT (KTransformers Backend)

### 2.1 Training YAML (LoRA SFT)

Example file:
`examples/train_lora/kimik2_lora_sft_kt.yaml`

Required fields:

```yaml
stage: sft
finetuning_type: lora
bf16: true

use_kt: true
kt_optimize_rule: <rule.yaml>
cpu_infer: 32
chunk_size: 8192
```

Other fields (dataset, output_dir, learning rate, epochs) can be adjusted as usual.

### 2.2 Inference YAML (LlamaFactory Verification)

Key requirements:

- `adapter_name_or_path`: LoRA output directory
- `infer_backend: ktransformers`
- **Same `use_kt` and `kt_optimize_rule` as training**

This YAML is used only for **quick verification**, not production serving.

## Step 3: Run LoRA SFT

```bash
conda activate kt-sft
cd LlamaFactory

USE_KT=1 llamafactory-cli train examples/train_lora/kimik2_lora_sft_kt.yaml
```

After training, the LoRA adapter is saved to `output_dir`.

## Step 4: Post-SFT Quick Verification with LlamaFactory (Optional)

Before production deployment, the new PDF recommends a **lightweight sanity check**.

```bash
conda activate kt-sft
cd LlamaFactory

llamafactory-cli chat examples/inference/kimik2_lora_sft_kt.yaml
```

Purpose:

- Validate LoRA correctness
- Ensure reproducibility
- Not for throughput benchmarking

## Step 5: SGLang Serving with LoRA (Recommended Delivery Path)

This is the **major runtime update** introduced by the new PDF.

### 5.1 Convert LoRA for SGLang

```bash
python ktransformers/kt-kernel/scripts/convert_lora.py \
  --base_path /path/to/kimi-base-model \
  --lora_path /path/to/llamafactory/output_dir \
  --output_path /path/to/lora_converted
```

### 5.2 (Optional) Convert CPU Weights to INT8

To reduce CPU memory usage:

```bash
python ktransformers/kt-kernel/scripts/convert_cpu_weights.py \
  --base_path /path/to/kimi-base-model \
  --output_dir /path/to/kimi-base-model-int8
```

This produces:

```text
/path/to/kimi-base-model-int8/int8
```

### 5.3 Launch SGLang Server with LoRA

```bash
conda activate kt-kernel

python -m sglang.launch_server \
  --enable-lora \
  --lora-paths lora1=/path/to/lora_converted \
  --lora-backend triton \
  --model-path /path/to/kimi-base-model \
  --tp 1 \
  --trust-remote-code \
  --context-length 4096 \
  --kt-weight-path /path/to/kimi-base-model-int8/int8 \
  --mem-fraction-static 0.9
```

Notes:

- `--kt-weight-path` points to CPU INT8 weights
- Adjust `tp`, `context-length`, and memory parameters per machine
- RAWINT4 inference paths can follow **Kimi-K2.5-Native** directly