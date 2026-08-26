# Heterogeneous VLM LoRA Fine-Tuning with KTransformers and LLaMA-Factory

Last updated: 2026-08-14

This guide covers MoE VLM LoRA SFT with LLaMA-Factory for training and
KTransformers for CPU/GPU heterogeneous expert execution. The workflow has been
validated with Qwen3-VL-30B-A3B-Instruct and Qwen3.5-35B-A3B.

KTransformers is optional. Use LLaMA-Factory alone for dense VLMs, pure-GPU
training, or full-parameter fine-tuning.

## Quick Start

### 1. Prerequisites

- Linux, NVIDIA GPUs, and a working CUDA driver
- Python 3.11
- An AVX-512 or AMX CPU recommended for KT expert execution
- Sufficient GPU memory, CPU memory, and storage for the selected model
- LLaMA-Factory with `requirements/ktransformers.txt` and the Qwen3-VL KT example
- A KTransformers release with Qwen3-VL MoE and instance-scoped VLM Conv3D support

### 2. Create the environment

```bash
conda create -n kt-vlm-lora python=3.11 -y
conda activate kt-vlm-lora

pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1
```

KT SFT pins torch 2.9.1. Do not mix the torch 2.10 pure-GPU stack into this
environment.

### 3. Install LLaMA-Factory and KT

```bash
git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/ktransformers.txt
pip check
```

`requirements/ktransformers.txt` should contain:

```text
ktransformers[sft]
```

Install in this order so the KT-enabled Transformers and Accelerate modules are
applied after the base LLaMA-Factory dependencies.

When building KTransformers from source, clone its submodules as well:

```bash
git clone --recursive https://github.com/kvcache-ai/ktransformers.git
# For an existing checkout:
git submodule update --init --recursive
```

Without the pinned `llama.cpp` and `pybind11` submodules, `kt-kernel` cannot be
built.

### 4. Verify the installation

```bash
python - <<'PY'
import importlib.metadata as md
import accelerate, kt_kernel, torch, transformers
from accelerate.utils.dataclasses import KTransformersPlugin
from kt_kernel.sft.conv3d_compat import is_vlm_conv3d_compatible, patch_vlm_conv3d
from transformers.integrations.kt import is_kt_expert_loading_enabled

print("torch                    =", torch.__version__)
print("transformers dist/module =", md.version("transformers"), transformers.__version__)
print("accelerate dist/module   =", md.version("accelerate"), accelerate.__version__)
print("transformers-kt          =", md.version("transformers-kt"))
print("accelerate-kt            =", md.version("accelerate-kt"))
print("kt-kernel                =", kt_kernel.__version__)
print("KT plugin                =", KTransformersPlugin.__name__)
print("Conv3D API               =", patch_vlm_conv3d.__name__, is_vlm_conv3d_compatible.__name__)
print("Transformers KT hook     =", is_kt_expert_loading_enabled.__name__)
PY

pip check
```

The KT-enabled modules may report different module and distribution versions
because the current KT fork wheels provide the `transformers` and `accelerate`
import packages under separate distribution names. The capability imports above
are the authoritative installation check.

### 5. Configure and launch training

Start from:

```text
examples/ktransformers/train_lora/qwen3vlmoe_lora_sft_kt.yaml
```

Review at least these fields:

```yaml
model_name_or_path: Qwen/Qwen3-VL-30B-A3B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: all

dataset: mllm_demo
template: qwen3_vl
cutoff_len: 512

use_kt: true
```

For Qwen3.5 VLM, change the model path, use `template: qwen3_5`, and select a
multimodal dataset. The existing Qwen3.5 KT YAML uses text data by default.

Set `num_processes` in
`examples/ktransformers/accelerate/fsdp2_kt_bf16.yaml` to the visible GPU count,
then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_bf16.yaml \
  src/train.py \
  examples/ktransformers/train_lora/qwen3vlmoe_lora_sft_kt.yaml
```

Use the BF16 configuration for original BF16 checkpoints. INT8/INT4 KT configs
require matching converted expert weights and `kt_weight_path`.

### 6. Check the result

A successful smoke test should show:

- finite loss and a finite, nonzero gradient norm;
- `Wrapped ... MoE layers with KTMoEWrapper` in the log;
- Conv3D compatibility activation under torch 2.9;
- regular PEFT adapter files and `fused_expert_lora.safetensors`;
- nonzero LoRA-B values after an optimizer step.

## Full Documentation

### Integration boundary

LLaMA-Factory owns multimodal preprocessing, templates, FSDP, LoRA configuration,
training, and checkpoints. With `use_kt: true`, KTransformers wraps supported MoE
experts, runs them on CPU backends, and stores their fused LoRA tensors.

The VLM-specific KT code only:

1. Maps Qwen3-VL MoE experts and their checkpoint prefix.
2. On torch 2.9.x, patches only the supported Conv3D instances on the loaded KT
   VLM and marks them after validating the operator contract.

It is not used when `use_kt: false`.

### Models and LoRA scope

| Model | Template | Notes |
| --- | --- | --- |
| Qwen3-VL-30B-A3B-Instruct | `qwen3_vl` | Reference KT VLM example |
| Qwen3.5-35B-A3B | `qwen3_5` | Requires multimodal data settings |

Use LLaMA-Factory's native VLM freeze flags:

| LoRA scope | `freeze_vision_tower` | `freeze_multi_modal_projector` | `freeze_language_model` |
| --- | ---: | ---: | ---: |
| Text only | `true` | `true` | `false` |
| Vision only | `false` | `true` | `true` |
| Text and vision | `false` | `true` | `false` |

With `lora_target: all`, LLaMA-Factory excludes the multimodal projector from
automatic target discovery. Projector training requires explicit compatible
targets and `freeze_multi_modal_projector: false`.

The KT path supports `finetuning_type: lora`. Use pure-GPU LLaMA-Factory with
`use_kt: false` for full-parameter VLM fine-tuning.

### Data and resource settings

`mllm_demo` is suitable for a smoke test. Register production multimodal data in
`data/dataset_info.json` using a format supported by the selected LLaMA-Factory
revision, and verify that image/video paths resolve correctly.

In the FSDP config, keep `kt_config.lora_rank` equal to the training YAML's
`lora_rank`. Tune `num_processes`, KT CPU thread counts, pixel limits,
`cutoff_len`, batch size, and gradient accumulation for the host.

The validation setup was eight RTX 4090 48 GB GPUs, an Intel Xeon Platinum 8488C,
and 2 TiB RAM. It is not a minimum hardware requirement.

### Checkpoints

KT saves `fused_expert_lora.safetensors` alongside the normal PEFT adapter files.
Keep both together when copying or resuming a checkpoint.

### End-to-end validation

One-step BF16 FSDP2 tests with LoRA rank 2 and real images completed on
2026-08-13:

| Model | Loss | Gradient norm | Runtime | KT evidence |
| --- | ---: | ---: | ---: | --- |
| Qwen3-VL-30B-A3B-Instruct | 13.6875 | 7.493 | 9.42 s | 48 MoE layers; 288 fused LoRA tensors |
| Qwen3.5-35B-A3B | 1.6299 | 0.6675 | 14.10 s | 40 MoE layers; 240 fused LoRA tensors |

Both runs completed multimodal preprocessing, distributed loading,
forward/backward, optimizer update, and adapter saving. These are functional
checks, not convergence or performance benchmarks.

### Troubleshooting

#### `ValueError: unknown keys (['kt_config'])`

The KT-enabled Accelerate build is not active. Reinstall in order:

```bash
pip install -e .
pip install -r requirements/ktransformers.txt
```

#### `kt_kernel.sft.conv3d_compat` is missing

Reinstall `kt-kernel` from a KTransformers revision containing VLM support.

#### The KT Conv3D compatibility check fails

The instance-scoped KT fallback rejects Conv3D modules whose stride, padding,
dilation, or groups are unsupported. No ms-swift installation is required.

#### Qwen3.5 cannot import `causal_conv1d_cuda`

Either uninstall the optional package or rebuild it after installing the final
torch version:

```bash
MAX_JOBS=16 pip install \
  causal-conv1d==1.6.2.post1 \
  --force-reinstall \
  --no-deps \
  --no-build-isolation
```

#### CPU or GPU memory is exhausted

Reduce pixel limits, `cutoff_len`, and batch size. Quantized KT configurations
also reduce memory use but require correctly converted expert weights.

### Pure-GPU alternative

In a separate clean environment, install LLaMA-Factory using its normal
procedure, do not install `requirements/ktransformers.txt`, and set:

```yaml
use_kt: false
```

This uses LLaMA-Factory's existing VLM LoRA and full-parameter training paths
without importing KTransformers.

## Related documentation

- [KT SFT Quick Start](./KTransformers-Fine-Tuning_Quick-Start.md)
- [KT SFT User Guide](./KTransformers-Fine-Tuning_User-Guide.md)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
