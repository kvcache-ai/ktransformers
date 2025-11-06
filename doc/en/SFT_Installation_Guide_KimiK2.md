## Installation

### Step 1: Create a conda environment and suit it for KTransformers

```Bash
conda create -n Kllama python=3.10 # choose from : [3.10, 3.11, 3.12, 3.13]
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
conda install -y -c nvidia/label/cuda-11.8.0 cuda-runtime
```

### Step 2: Install the LLaMA-Factory environment

```Bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Step 3: Install the KTransformers wheel that matches your Torch and Python versions, from https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.1

(Note: The CUDA version can differ from that in the wheel filename.)

```Bash
pip install ktransformers-0.4.1+cu128torch28fancy-cp310-cp310-linux_x86_64.whl
```

### Step 4: Install the Flash-attention wheel that matches your Torch and Python versions, from: https://github.com/Dao-AILab/flash-attention/releases

```Bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# abi=True/False can find from below
# import torch
# print(torch._C._GLIBCXX_USE_CXX11_ABI)
```

### Step 5: (Optional) If you want to use flash_infer (otherwise it defaults to triton)

```Bash
git clone https://github.com/kvcache-ai/custom_flashinfer.git
pip install custom_flashinfer/
```

## Download Model

Download the official KIMI weights. If the weights are in FP8 format, please refer to https://github.com/kvcache-ai/ktransformers/pull/1559 to convert them to BF16 weights.

## How to start

```Python
# For LoRA SFT
USE_KT=1 llamafactory-cli train examples/train_lora/kimik2_lora_sft_kt.yaml
# For Chat with model after LoRA SFT
llamafactory-cli chat examples/inference/kimik2_lora_sft_kt.yaml
# For API with model after LoRA SFT
llamafactory-cli api examples/inference/kimik2_lora_sft_kt.yaml
```

**If your** **CPU** **memory is insufficient to exceed 2T to support the Kimi K2, you can use the swap method additionally:**

```Plain
sudo fallocate -l 200G /data/swapfile
sudo chmod 600 /data/swapfile
sudo mkswap /data/swapfile
sudo swapon /data/swapfile
```

For example, we provide the YAML file as follows: (Since the structures of Kimi and DeepSeek are relatively similar, we use deepseek as template in llamafactory)

（1）examples/train_lora/kimik2_lora_sft_kt.yaml

```YAML
### model
model_name_or_path: KimiK2-model
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity
template: deepseek
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/Kllama_kimik2
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### ktransformers
use_kt: true # use KTransformers as LoRA sft backend
kt_optimize_rule: examples/kt_optimize_rules/DeepSeek-V3-Chat-sft-amx-multi-gpu.yaml
cpu_infer: 32
chunk_size: 8192
```

For more details about --kt_optimize_rule, please refer to https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/KTransformers-Fine-Tuning_User-Guide.md

（2）examples/inference/kimik2_lora_sft_kt.yaml

```YAML
model_name_or_path: opensourcerelease/DeepSeek-V3-bf16
adapter_name_or_path: saves/Kllama_deepseekV3
template: deepseek
infer_backend: ktransformers  # choices: [huggingface, vllm, sglang, ktransformers]
trust_remote_code: true

use_kt: true # use KTransformers as LoRA sft backend to inference
kt_optimize_rule: examples/kt_optimize_rules/DeepSeek-V3-Chat-sft-amx-multi-gpu.yaml
cpu_infer: 32
chunk_size: 8192
```