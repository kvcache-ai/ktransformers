# KTransformers v0.6.1 KT SFT Quick Start

Last updated: 2026-04-30

This Quick Start is for LLaMA-Factory users who want to run KTransformers-backed MoE LoRA SFT. KTransformers has supported fine-tuning since v0.4.1 and later releases; v0.6.1 refactors and upgrades the existing KT SFT path with a pip-installable package entry, a cleaner dependency boundary, and LLaMA-Factory-compatible examples.

## 1. Scope

- Task: MoE LoRA SFT
- Training entry: LLaMA-Factory `src/train.py` and YAML configs
- KT package entry: `ktransformers[sft]`
- Example models: Qwen3 MoE / Qwen3.5 MoE
- Recommended baseline: Python 3.11 and `torch==2.9.1`

KT inference uses a separate SGLang-KT package path. See [KT Inference Packages](#7-kt-inference-packages).

## 2. Environment Setup

Start from a clean environment:

```bash
conda create -n kt-sft python=3.11 -y
conda activate kt-sft
```

Install the PyTorch baseline:

```bash
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1
```

If you already have a working `torch==2.9.1` environment, you can reuse it. Plain PyPI or mirror installs usually resolve `torch==2.9.1` to a CUDA 12.8 wheel; the PyTorch `cu130` index resolves to the CUDA 13.0 wheel. In both cases, the Python package version remains `torch==2.9.1`.

## 3. Install LLaMA-Factory and KT SFT

Use a LLaMA-Factory checkout that contains the KT examples and `requirements/ktransformers.txt`. Related integration PR:

https://github.com/hiyouga/LLaMA-Factory/pull/10430

Confirm the checkout contains:

- `requirements/ktransformers.txt`
- `examples/ktransformers/`

Install in this order:

```bash
cd /path/to/LLaMA-Factory
pip install -e .
pip install -r requirements/ktransformers.txt
```

`requirements/ktransformers.txt` should contain one line:

```text
ktransformers[sft]
```

This entry installs the KT SFT stack:

- `ktransformers`
- `kt-kernel`
- `transformers-kt`
- `accelerate-kt`

If your LLaMA-Factory checkout does not yet provide `requirements/ktransformers.txt`, you can install the KT SFT package entry directly:

```bash
pip install "ktransformers[sft]"
```

For LLaMA-Factory users, the requirements file is preferred because it keeps the examples and optional dependency flow together.

## 4. Post-Install Check

```bash
python - <<'PY'
import importlib.metadata as md
import torch
import transformers
import accelerate
import kt_kernel
import ktransformers
from accelerate.utils.dataclasses import KTransformersPlugin

print("torch         =", torch.__version__)
print("transformers  =", transformers.__version__)
print("accelerate    =", accelerate.__version__)
print("kt_kernel     =", kt_kernel.__version__)
print("ktransformers =", ktransformers.__version__)
print("transformers-kt dist =", md.version("transformers-kt"))
print("accelerate-kt dist   =", md.version("accelerate-kt"))
print("KTransformersPlugin  =", KTransformersPlugin.__name__)
PY
```

Expected values should be close to:

- `torch = 2.9.1+cu128` or `2.9.1+cu130`
- `transformers = 5.6.0`
- `accelerate = 1.14.0`
- `kt_kernel = 0.6.1`
- `ktransformers = 0.6.1`
- `KTransformersPlugin = KTransformersPlugin`

## 5. Run Qwen3.5 MoE LoRA SFT

Reference configs:

- accelerate config: `examples/ktransformers/accelerate/fsdp2_kt_int8.yaml`
- train yaml: `examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml`
- starting point for resource planning: 4 x 24GB GPU + 512GB CPU memory; real requirements depend on model, context length, batch size, and LoRA config.

Before running, open the train YAML and adjust these fields for your local setup:

- `model_name_or_path`
- `dataset`
- `output_dir`
- `cutoff_len`
- `max_steps`

Run:

```bash
cd /path/to/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
  src/train.py \
  examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml
```

## 6. Other MoE Examples

| Model path | accelerate config | train yaml |
| --- | --- | --- |
| Qwen3 MoE BF16 | `examples/ktransformers/accelerate/fsdp2_kt_bf16.yaml` | `examples/ktransformers/train_lora/qwen3moe_lora_sft_kt.yaml` |
| Qwen3.5 MoE INT8 | `examples/ktransformers/accelerate/fsdp2_kt_int8.yaml` | `examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml` |
| DeepSeek-V2 MoE | `examples/ktransformers/accelerate/` KT configs | `examples/ktransformers/train_lora/deepseek_v2_lora_sft_kt.yaml` |
| DeepSeek-V3 MoE | `examples/ktransformers/accelerate/` KT configs | `examples/ktransformers/train_lora/deepseek_v3_lora_sft_kt.yaml` |

Start from the example `kt_config`, `use_kt: true`, and LoRA settings, then adjust model paths and resource-related fields for your hardware.

## 7. KT Inference Packages

KT inference uses the SGLang-KT path:

```bash
pip install kt-kernel sglang-kt
```

LLaMA-Factory KT SFT continues to use:

```bash
pip install -r requirements/ktransformers.txt
```

with:

```text
ktransformers[sft]
```

Keep `sglang-kt` out of LLaMA-Factory SFT requirements.

## 8. Troubleshooting

### `ValueError: unknown keys (['kt_config'])`

This usually means the environment is still using upstream `accelerate`. Reinstall in this order from the LLaMA-Factory checkout:

```bash
pip install -e .
pip install -r requirements/ktransformers.txt
```

Then rerun the post-install check and confirm `accelerate = 1.14.0` and `KTransformersPlugin` imports successfully.

### `kt_kernel` or `ktransformers` is missing

Confirm that the KT SFT dependency step was run from the LLaMA-Factory checkout:

```bash
pip install -r requirements/ktransformers.txt
```

### CPU or GPU memory is insufficient

Try the following first:

- reduce `cutoff_len`
- reduce batch size
- start from INT8 / INT4 KT configs
- increase CPU memory or reduce other concurrent workloads

## 9. Related Links

- KTransformers: https://github.com/kvcache-ai/ktransformers
- KTransformers v0.6.1 Release: https://github.com/kvcache-ai/ktransformers/releases/tag/v0.6.1
- LLaMA-Factory KT PR: https://github.com/hiyouga/LLaMA-Factory/pull/10430
- KTransformers docs: https://ktransformers.net/docs
