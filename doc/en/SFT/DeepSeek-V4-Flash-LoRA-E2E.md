# April DeepSeek-V4-Flash LoRA integration

This integration builds on the [native MXFP4 expert kernel](./DeepSeek-V4-MXFP4-Routed-Expert-LoRA-SFT.md).
KT trains CPU routed-expert LoRA while Transformers retains attention, routing,
mHC and shared experts. Routed-expert base weights remain frozen native MXFP4;
non-expert FP8 weights are decoded to BF16 for training.

Use the matching [Transformers-KT](https://github.com/kvcache-ai/transformers/pull/8)
and [SGLang-KT](https://github.com/kvcache-ai/sglang/pull/99) integrations and the
companion LF branch providing `deepseek4_nothink`. Install a validated companion
wheel set with its dependency lock. This integration PR does not select release
versions; the existing default package extras do not supply these pending branches.

## Supported recipe

The validated recipe is April Flash ordinary non-thinking Alpaca/ShareGPT chat,
including multiple turns: LoRA r8/alpha16/dropout0, BF16 autocast, training length
1024 and serving input plus output length 2048. Multi-GPU training uses FSDP2.
Packing, `neat_packing`, `lora_target: all`, dynamic adapters, Flash-0731/Pro and
Transformers cached generation are outside this recipe.

Add the following to a normal LF LoRA SFT YAML with `dataset`, `dataset_dir`
and `output_dir` set. Register raw conversations through `dataset_info.json`.

```yaml
model_name_or_path: /models/DeepSeek-V4-Flash
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
lora_target: self_attn.q_a_proj,self_attn.q_b_proj,self_attn.kv_proj,self_attn.o_b_proj,self_attn.compressor.kv_proj,self_attn.compressor.gate_proj,mlp.shared_experts.gate_proj,mlp.shared_experts.up_proj,mlp.shared_experts.down_proj
bf16: true
flash_attn: disabled
cutoff_len: 1024
template: deepseek4_nothink
enable_thinking: false
packing: false
neat_packing: false
disable_gradient_checkpointing: false
use_reentrant_gc: false
use_kt: true
kt_cpu_activation: retain
kt_weight_path: /models/DeepSeek-V4-Flash
kt_non_expert_weight_path: /cache/v4-non-experts
kt_config:
  kt_expert_weight_format: mxfp4
  kt_backend: auto
  kt_num_threads: 64
  kt_tp_enabled: true
  kt_threadpool_count: 2
  kt_force_fused_expert_lora: true
  kt_share_backward_bb: false
  kt_skip_expert_loading: true
```

First startup prepares the non-expert cache; later startups validate and reuse
it. The April cache uses approximately 13.7 GiB of disk space. Use a dedicated
directory separate from the original model, which must remain accessible.
Preparation serializes concurrent writers, publishes only complete caches and
retries an owned interrupted build. Invalid or damaged caches are rejected.

For FSDP2, wrap `DeepseekV4DecoderLayer` with CPU RAM efficient loading enabled,
parameter offload disabled and reshard after forward enabled. Keep gradient
checkpointing and BF16 autocast enabled; `pure_bf16` and `bf16_full_eval` are
unsupported. GPU activations are recomputed; CPU expert activations are retained.
Indexer, grouped `o_a_proj`, router, mHC, embedding and head stay frozen, with
their existing differentiable paths preserved. CPU expert LoRA and its optimizer
state belong to rank 0; ordinary GPU LoRA uses FSDP2.

## Save, resume and export

Keep `save_only_model: false` and retain the complete checkpoint directory.
The ordinary adapter file alone omits CPU expert LoRA. Resume in a new process
using `resume_from_checkpoint` with the same model, cache, rank count, targets
and scheduler horizon. FSDP2 requires per-rank optimizer files and
`kt_optimizer.index.json`; single-GPU runs use the ordinary optimizer checkpoint.

```bash
python -m kt_kernel.sft.export_dsv4_sglang_adapter \
  --source /models/DeepSeek-V4-Flash \
  --cache /cache/v4-non-experts \
  --adapter /runs/v4/checkpoint-20 \
  --output /deploy/v4-adapter
```

The exporter validates source identity, hashes, LoRA shapes and consumption.
It writes `model/` with ordinary LoRA merged in FP32 then rounded to BF16,
`experts/` with CPU expert LoRA, and `deployment_manifest.json`.
Both components are required for the complete adapter; original weights remain
unchanged. The model snapshot contains no routed-expert base weights.

## Static serving

Use the matched serving lock, including `tilelang==0.1.10` and
`apache-tvm-ffi==0.1.11`, for the compressed indexer on consumer GPUs.

```bash
export SGLANG_DSV4_MODE=2604 SGLANG_DSV4_2604_SUBMODE=2604B
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_DSV4_FP4_EXPERTS=1 SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0 SGLANG_V4_USE_TRITON_KERNELS=1
export SGLANG_FLASHMLA_BACKEND_OVERRIDE=triton

python -m sglang.launch_server \
  --model-path /deploy/v4-adapter/model \
  --kt-weight-path /models/DeepSeek-V4-Flash \
  --kt-expert-lora-path /deploy/v4-adapter/experts \
  --kt-method MXFP4 --kt-cpuinfer 64 --kt-threadpool-count 2 \
  --kt-num-gpu-experts 0 --kt-gpu-prefill-token-threshold 0 \
  --tp 2 --dtype bfloat16 --attention-backend flashinfer \
  --mem-fraction-static 0.65 --max-running-requests 2 \
  --context-length 2056 --chunked-prefill-size 1024 \
  --disable-cuda-graph --disable-custom-all-reduce \
  --disable-shared-experts-fusion --host 127.0.0.1 --port 31300
```

Send official chat-mode encoded prompts to `/generate`. The usable request
budget is 2048 input plus output tokens; 2056 accounts for internal reserved
slots. Keep routed experts on CPU and missing-weight checks enabled.
For forced AVX2, set `KT_KERNEL_CPU_VARIANT=avx2` and `KT_MXFP4_BACKEND=avx2`;
AVX512-BF16 uses `KT_KERNEL_CPU_VARIANT=avx512_bf16`.
