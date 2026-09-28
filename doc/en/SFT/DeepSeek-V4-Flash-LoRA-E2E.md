# DeepSeek-V4-Flash: native MXFP4 LoRA SFT

This recipe targets the **April DeepSeek-V4-Flash checkpoint**, not Flash-0731.
Routed-expert base weights remain frozen native MXFP4 on CPU. GPU non-expert
linear weights are decoded from FP8 into BF16; this is not whole-model FP4
arithmetic. The training model uses differentiable Transformers attention.

End-to-end acceptance is still in progress. See the experiment record before
treating this development branch as a released integration.

## Dependencies and scope

Use the coordinated `yyj/dsv4-flash-delivery` branches of KTransformers,
Transformers-KT, SGLang-KT, and the companion LLaMA-Factory branch. The stack
continues the v0.7.1 companion release with PyTorch 2.9.1/CUDA 12.8 and PEFT
0.18.1. Install a matching, validated wheel set with its dependency lock;
leave normal package-version and provider checks enabled. Development
candidate versions are not a public release or evidence of completed acceptance.

The LF companion adds the `deepseek4_nothink` raw-conversation template and
validates the supported recipe before loading weights. It accepts ordinary
Alpaca and ShareGPT conversations, including multiple turns. Thinking, tool
calls, `packing`, and `neat_packing` are outside the first delivery.

The acceptance matrix covers Python 3.11 and 3.12 on qj5090: AVX512-BF16
with 1/2/4/8 RTX 5090 GPUs, and forced AVX2 with 2 GPUs. Multi-GPU training
uses FSDP2; CPU TP2, S1024, and LoRA rank 8 / alpha 16 / dropout 0 are the
reference recipe. These are acceptance requirements, not completed results.
Forced AVX2 on this host does not certify a CPU that lacks AVX512.
CPU TP1/TP2 are supported by the native kernel.
Full/Hybrid training, GPU routed experts, adapter hot swapping, grouped
`o_a_proj` LoRA, router/indexer LoRA, and MTP training are outside this recipe.

## 1. Configure automatic non-expert cache preparation

Set `kt_non_expert_weight_path` to a new cache directory in the LF YAML below.
The first training startup prepares it automatically; later startups verify
and reuse it. Use a dedicated cache path separate from the source model.
An incompatible or damaged cache is rejected rather than silently overwritten.
A lock serializes preparation, and an owned incomplete conversion can be retried.

The converter streams only non-expert
weights, preserves FP32 mHC/norm state and integer hash tables, and publishes
a ready manifest after validating the complete tensor inventory. The April
checkpoint produces approximately 13.7 GiB of cache files; this is a disk-space
estimate, not additional per-GPU VRAM. No dense expert copy is produced.
Keep the original checkpoint unchanged and accessible at `kt_weight_path`.
Cache and adapter manifests are bound to that source checkpoint.

## 2. Train through the existing LF entrypoint

Add these settings to a normal LF LoRA SFT recipe:

```yaml
model_name_or_path: /models/DeepSeek-V4-Flash
stage: sft
do_train: true
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

Keep LF gradient checkpointing enabled: GPU activations are recomputed, while
CPU expert activations are retained across that recomputation. MXFP4 backward
does not need a transposed base-weight buffer.

Use FSDP2's transformer auto-wrap policy with
`fsdp_transformer_layer_cls_to_wrap: DeepseekV4DecoderLayer`,
`fsdp_cpu_ram_efficient_loading: true`, `fsdp_offload_params: false`, and
`fsdp_reshard_after_forward: true`.

KT wraps **only the routed-expert callable**. V4's hash/learned routing,
shared experts, mHC, and attention remain in Transformers. CPU expert LoRA
parameters and optimizer state belong to rank 0; standard GPU LoRA uses FSDP2.
Only LoRA parameters are optimized; original base files are not rewritten.

Register raw conversation JSON through the ordinary LF `dataset_info.json`
and set `dataset` / `dataset_dir` in the YAML. No V4-specific preprocessing
script or pretokenized internal dataset is required. Both packing options must
remain false. A full S1024 test must be one complete conversation; concatenating
independent samples does not establish support for packing.

Keep BF16 autocast enabled for training and loss evaluation. `pure_bf16`,
`bf16_full_eval`, and Transformers cached generation are not supported by this
recipe. Use explicit LoRA targets from the listed set; `lora_target: all` is
not supported. Indexer, grouped `o_a_proj`, router, mHC, embedding and head
remain frozen. The existing differentiable main-network paths retain input
gradients; discrete index selection and integer embedding inputs do not have
the same gradient semantics.

## 3. Save and resume

Keep `save_only_model: false`. A resumable checkpoint includes both adapter
files, `kt_adapter_manifest.json`, scheduler state and RNG state. FSDP2 also
requires per-rank optimizer files and `kt_optimizer.index.json`; a single-GPU
run uses the ordinary optimizer checkpoint. Do not copy
only `adapter_model.safetensors`: it omits the CPU experts.

Start a **new training process** with the same native model, cache, rank count,
LoRA targets, and scheduler horizon, plus:

```yaml
resume_from_checkpoint: /runs/v4/checkpoint-10
```

Changing `max_steps` changes the cosine schedule; it is not an equivalent
resume comparison. RAM-filesystem checkpoints survive a process restart but
**not a host reboot**. Choose persistent checkpoint storage for durable runs.

## 4. Export one fixed SGLang deployment

```bash
python -m kt_kernel.sft.export_dsv4_sglang_adapter \
  --source /models/DeepSeek-V4-Flash \
  --cache /cache/v4-non-experts \
  --adapter /runs/v4/checkpoint-20 \
  --output /deploy/v4-adapter
```

The exporter verifies hashes, source identity, all LoRA shapes, and complete
consumption. It produces:

- `model/`: independent BF16 non-expert snapshot with standard LoRA merged
  in FP32 then rounded to BF16; no routed-expert base weights.
- `experts/`: explicit per-expert LoRA adapter for the native CPU kernel.
- `deployment_manifest.json`: source binding and consumption audit.

This first deployment is static, not `--lora-paths` hot swapping. Both
components are required. The base model remains read-only. Use
`--component base`, `experts`, or `nonexperts` only for controlled ablations;
all use the same BF16 non-expert convention.
For numerical ablations, add `--match-expert-kernel`: baseline/non-expert-only
exports include an expert adapter with zero B matrices. This keeps the same
native SFT forward path in all conditions, avoiding inference-kernel rounding
differences as a confounder.

The April SGLang path requires the following environment settings:

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
  --context-length 2048 --chunked-prefill-size 1024 \
  --disable-cuda-graph --disable-custom-all-reduce \
  --disable-shared-experts-fusion --host 127.0.0.1 --port 31300
```

Do not enable dynamic expert promotion or suppress missing-weight checks.
Send official chat-mode encoded prompts to `/generate`. The acceptance helper
`kt-kernel/test/generate_dsv4_acceptance.py` records prompts, outputs, finish
reasons, timing, and token log-probabilities for the prepared held-out split.

The total serving context limit is 2048 input plus output tokens. For the
forced AVX2 acceptance path, set both `KT_KERNEL_CPU_VARIANT=avx2` and
`KT_MXFP4_BACKEND=avx2` before starting the process. AVX512-BF16 uses
`KT_KERNEL_CPU_VARIANT=avx512_bf16`. Final tested commands and wheel hashes
must be supplied with the candidate acceptance record.

## Acceptance criteria

- S1024 GAS1/GAS4: at least three optimizer steps for each setting, finite
  loss/gradients, CPU/GPU LoRA updates, frozen-base checks and save completion.
- Train to step 10 with a 20-step scheduler horizon, exit, and resume in a
  new process to step 20. Verify adapter, optimizer, scheduler, progress and
  RNG restoration before resumed updates. Independent training trajectories
  are not promised to be bitwise identical.
- A fixed 200-step reference run records train-probe and held-out loss plus
  paired answers. Short smoke runs do not prove convergence.
- New-process SGLang generation: compare base and trained output on the same
  32 held-out prompts; independently check expert/non-expert adapter effects.

Record cold/hot startup, actual non-padding and supervised token counts, full
length throughput, step times, CPU/GPU memory and disk use with the candidate
wheel hashes. Distinguish sampled host-memory peaks and audit overhead from
instantaneous measurements. No additional minimum-performance SLA is set.
