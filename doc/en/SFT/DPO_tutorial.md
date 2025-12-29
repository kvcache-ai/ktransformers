# DPO Training with LLaMA-Factory

This tutorial demonstrates how to use Direct Preference Optimization (DPO) to fine-tune a language model using the LLaMA-Factory framework. DPO is a method for training models based on human preferences, allowing for more aligned and user-centric outputs.

## Installation

### Step 1: Create a conda environment and suit it for KTransformers

```Bash
conda create -n Kllama python=3.12 # choose from : [3.11, 3.12, 3.13]
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
conda install -y -c nvidia/label/cuda-12.8.0 cuda-runtime
```

### Step 2: Install the LLaMA-Factory environment

```Bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```


### Step 3: Install KTransformers
#### Option 1: Install the KTransformers wheel that matches your Torch and Python versions, from https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.4

(Note: The CUDA version can differ from that in the wheel filename.)

```Bash
pip install ktransformers-0.4.4+cu128torch28fancy-cp312-cp312-linux_x86_64.whl
```

#### Option 2: Install KTransformers from source

```Bash
git clone --depth 1 https://github.com/kvcache-ai/ktransformers.git
cd ktransformers/kt-sft
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" # set according to your GPU

pip install -r "requirements-sft.txt"
KTRANSFORMERS_FORCE_BUILD=TRUE pip install -v . --no-build-isolation

```

### Step 4: Install the Flash-attention wheel that matches your Torch and Python versions, from: https://github.com/Dao-AILab/flash-attention/releases

```Bash
# abi=True/False can find from below
# import torch
# print(torch._C._GLIBCXX_USE_CXX11_ABI)

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

### Step 5: (Optional) If you want to use flash_infer (otherwise it defaults to triton)

```Bash
git clone https://github.com/kvcache-ai/custom_flashinfer.git
pip install custom_flashinfer/
```

## Prepare Models

We use `deepseek-ai/DeepSeek-V2-Lite` as an example here. You can replace it with other models such as Kimi K2.

## How to start

```Python
# For LoRA SFT
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek2_lora_dpo_kt.yaml
# For Chat with model after LoRA SFT
llamafactory-cli chat examples/inference/deepseek2_lora_dpo_kt.yaml
# For API with model after LoRA SFT
llamafactory-cli api examples/inference/deepseek2_lora_dpo_kt.yaml
```

For example, we provide the YAML file as follows: 

（1）examples/train_lora/deepseek2_lora_dpo_kt.yaml

```YAML
### model
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]

### dataset
dataset: dpo_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/Kllama_deepseekV2_DPO
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### ktransformers
use_kt: true # use KTransformers as LoRA sft backend
kt_optimize_rule: examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat-sft-amx.yaml
cpu_infer: 64
chunk_size: 8192
```

For more details about --kt_optimize_rule, please refer to https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/KTransformers-Fine-Tuning_User-Guide.md 

Then, you can use the lora adapter saved in `saves/Kllama_deepseekV2_DPO` for inference the same as the sft training. For example,

```YAML
model_name_or_path: DeepSeek-V2-Lite-Chat 
adapter_name_or_path: saves/Kllama_deepseekV2_DPO
template: deepseek
infer_backend: ktransformers  # choices: [huggingface, vllm, sglang, ktransformers]
trust_remote_code: true

use_kt: true # use KTransformers as LoRA sft backend to inference
kt_optimize_rule: examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat-sft-amx.yaml
cpu_infer: 32
chunk_size: 8192

```
