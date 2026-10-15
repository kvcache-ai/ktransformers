# DeepSeek-V4-Flash LoRA SFT

KTransformers fine-tunes the DeepSeek-V4-Flash checkpoint while keeping routed-expert weights in native MXFP4 on CPU. This example trains on two GPUs and loads the saved LoRA adapter for inference.

## 1. Prepare the environment

Use Python 3.11 or 3.12, with `python` pointing to that interpreter.

```bash
python -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install 'ktransformers[sft]==0.7.2' 'torchaudio==2.9.1'
python -m pip check
```

Keep the complete DeepSeek-V4-Flash checkpoint at `/models/DeepSeek-V4-Flash`. Replace this path with the location on your machine.

## 2. Prepare the data

Save Alpaca-format examples as `data/train.json` (replace the example with your data):

```json
[
  {"instruction": "Translate into English.", "input": "设备正常。", "output": "The device is operating normally."},
  {"instruction": "Translate into English.", "input": "请检查连接。", "output": "Please check the connection."}
]
```

Register the dataset in `data/dataset_info.json`:

```json
{"v4_demo": {"file_name": "train.json"}}
```

## 3. Train

Save as `train.yaml`:

```yaml
model_name_or_path: /models/DeepSeek-V4-Flash
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
lora_target: self_attn.q_a_proj,self_attn.q_b_proj,self_attn.kv_proj,self_attn.o_b_proj,mlp.shared_experts.gate_proj,mlp.shared_experts.up_proj,mlp.shared_experts.down_proj
dataset: v4_demo
dataset_dir: data
template: deepseek4_nothink
enable_thinking: false
cutoff_len: 1024
packing: false
neat_packing: false
output_dir: outputs/dsv4-flash-2gpu
max_steps: 20
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 0.0001
bf16: true
flash_attn: disabled
disable_gradient_checkpointing: false
use_reentrant_gc: false
save_strategy: steps
save_steps: 20
save_only_model: false
ddp_timeout: 3600
use_kt: true
kt_cpu_activation: retain
kt_weight_path: /models/DeepSeek-V4-Flash
kt_non_expert_weight_path: /cache/dsv4-non-experts
kt_config:
  kt_backend: auto
  kt_expert_weight_format: mxfp4
  kt_num_threads: 64
  kt_tp_enabled: true
  kt_threadpool_count: 2
  kt_share_backward_bb: false
  kt_model_max_length: 1024
  kt_max_cache_depth: 2
  kt_force_fused_expert_lora: true
  kt_skip_expert_loading: true
```

Choose `kt_num_threads`, `kt_threadpool_count`, and `kt_tp_enabled` for your CPU and NUMA layout; `kt_backend: auto` selects the available CPU kernel. Tune the LoRA rank, learning rate, batch size, accumulation, and steps for your task. This example has an effective batch size of eight; 20 steps is only a smoke test.

Save the two-GPU Accelerate configuration as `fsdp2.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_machines: 1
num_processes: 2
mixed_precision: bf16
rdzv_backend: static
same_network: true
use_cpu: false
fsdp_config:
  fsdp_version: 2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: DeepseekV4DecoderLayer
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
```

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file fsdp2.yaml \
  -m llamafactory.launcher train.yaml
```

## 4. Load the adapter

Use a separate serving environment:

```bash
python3.11 -m venv .venv-serve
source .venv-serve/bin/activate
python -m pip install 'ktransformers[sglang]==0.7.2'
python -m pip check
```

Replace `outputs/dsv4-flash-2gpu/checkpoint-20` with your actual saved checkpoint directory. It must contain both ordinary and fused expert LoRA weights plus `kt_adapter_manifest.json`. `trained` is the name used by inference requests.

```bash
export SGLANG_DSV4_MODE=2604 SGLANG_DSV4_2604_SUBMODE=2604B
export SGLANG_APPLY_CONFIG_BACKUP=none SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_V4_USE_TRITON_KERNELS=1
export SGLANG_FLASHMLA_BACKEND_OVERRIDE=triton
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash --kt-weight-path /models/DeepSeek-V4-Flash \
  --lora-paths trained=outputs/dsv4-flash-2gpu/checkpoint-20 --max-lora-rank 8 \
  --kt-method MXFP4 --kt-cpuinfer 64 --kt-threadpool-count 2 \
  --kt-num-gpu-experts 0 --kt-gpu-prefill-token-threshold 0 \
  --tp-size 2 --dtype bfloat16 --attention-backend flashinfer \
  --mem-fraction-static 0.65 --max-running-requests 2 \
  --context-length 2056 --chunked-prefill-size 1024 \
  --disable-cuda-graph --disable-custom-all-reduce --disable-shared-experts-fusion \
  --served-model-name dsv4-flash --host 127.0.0.1 --port 31300
```

The V4 mode flags select the DeepSeek-V4-Flash checkpoint loader; the remaining overrides follow the tested RTX 5090 path. Adjust CPU threads and GPU memory settings for your hardware.

Send a smoke-test request to the named adapter. The `input_ids` below are an encoded DeepSeek-V4-Flash prompt asking for the capital of China:

```python
import json
from urllib.request import Request, urlopen

body = {
    "input_ids": [0, 128803, 17607, 2793, 3139, 12108, 1148, 2788, 102475, 9090, 320, 128804, 128822],
    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    "lora_path": "trained",
}
request = Request(
    "http://127.0.0.1:31300/generate",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urlopen(request))["text"])
```
