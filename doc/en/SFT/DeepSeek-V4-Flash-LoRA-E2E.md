# DeepSeek-V4-Flash: native MXFP4 LoRA SFT

This recipe targets the **April DeepSeek-V4-Flash checkpoint**, not Flash-0731.
Routed-expert base weights remain frozen native MXFP4 on CPU. GPU non-expert
linear weights are decoded from FP8 into BF16; this is not whole-model FP4
arithmetic. The training model uses differentiable Transformers attention.

End-to-end acceptance is still in progress. See the experiment record before
treating this development branch as a released integration.

## Dependencies and scope

Use the coordinated `yyj/dsv4-flash-mxfp4-e2e` branches of KTransformers,
Transformers, and KT-SGLang. The V4 model was backported from Transformers
`c583a3a116b9394038c951627e2427b5e2b62b11`; its modular source regenerates
the same model implementation on the pinned KT training base.

The reference environment uses PyTorch 2.9.1/CUDA 12.8, PEFT 0.18.1,
LLaMA-Factory `5226b026`, and Accelerate `aae86aac`. No LLaMA-Factory or
Accelerate source changes are required. The training fork retains version
5.6.0, which LF explicitly excludes: the pinned acceptance environment uses
`DISABLE_VERSION_CHECK=1`. This is a development-environment exception, not a
recommendation to bypass checks with arbitrary package versions.

The initial recipe uses two RTX 5090 GPUs, FSDP2, CPU TP2, S1024, and LoRA
rank 8 / alpha 16 / dropout 0. CPU TP1/TP2 are supported by the native kernel.
Full/Hybrid training, GPU routed experts, adapter hot swapping, grouped
`o_a_proj` LoRA, router/indexer LoRA, and MTP training are outside this recipe.

## 1. Prepare the non-expert cache

Build/install the branch's native extension, then run:

```bash
python -m kt_kernel.sft.deepseek_v4 \
  --source /models/DeepSeek-V4-Flash \
  --output /cache/v4-non-experts
```

The output must be an empty directory. The converter streams only non-expert
weights, preserves FP32 mHC/norm state and integer hash tables, and publishes
a ready manifest after validating the complete tensor inventory. The April
checkpoint produces about 14 GiB of cache; no dense expert copy is produced.
Keep the original checkpoint unchanged and accessible at `kt_weight_path`.
Cache and adapter manifests are bound to that source checkpoint.

## 2. Train through the existing LF entrypoint

Add these settings to a normal LF LoRA SFT recipe:

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

For the internal NekoQA acceptance run, `kt-kernel/test/prepare_dsv4_nekoqa.py`
uses the model's official `encoding/encoding_dsv4.py` in chat mode, fixed
train/held-out splits, and assistant-only labels. `template: empty` is used
**only with this already-tokenized dataset**; it is not a V4 raw-chat template.
The data's redistribution license has not been verified; do not publish it.

## 3. Save and resume

Keep `save_only_model: false`. A resumable checkpoint includes both adapter
files, `kt_adapter_manifest.json`, FSDP state, per-rank optimizer files,
`kt_optimizer.index.json`, scheduler state, and per-rank RNG state. Do not copy
only `adapter_model.safetensors`: it omits the CPU experts.

Start a **new training process** with the same native model, cache, rank count,
LoRA targets, and scheduler horizon, plus:

```yaml
resume_from_checkpoint: /runs/v4/checkpoint-100
```

Changing `max_steps` changes the cosine schedule; it is not an equivalent
resume comparison. RAM-filesystem checkpoints survive a process restart but
**not a host reboot**. Choose persistent checkpoint storage for durable runs.

## 4. Export one fixed SGLang deployment

```bash
python kt-kernel/scripts/export_dsv4_sglang_adapter.py \
  --source /models/DeepSeek-V4-Flash \
  --cache /cache/v4-non-experts \
  --adapter /runs/v4/checkpoint-200 \
  --output /deploy/v4-nekoqa
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

python -m sglang.launch_server \
  --model-path /deploy/v4-nekoqa/model \
  --kt-weight-path /models/DeepSeek-V4-Flash \
  --kt-expert-lora-path /deploy/v4-nekoqa/experts \
  --kt-method MXFP4 --kt-cpuinfer 16 --kt-threadpool-count 2 \
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

## Acceptance criteria

- S1024 GAS1/GAS4: finite loss/gradients, all CPU/GPU LoRA groups update,
  frozen-base identity unchanged, checkpoint save completes.
- Fixed 200-step run: train-probe NLL decreases at least 20%, held-out NLL
  also decreases; batch losses need not be monotonic.
- Continuous 20 steps versus 10 + new-process 10: compare both adapters,
  per-rank AdamW state, scheduler, RNG, global step, and loss history with
  `kt-kernel/test/compare_dsv4_resume.py` (exact comparison by default).
- New-process SGLang generation: compare base and trained output on the same
  32 held-out prompts; independently check expert/non-expert adapter effects.

This is functional training/deployment validation, not a throughput benchmark.
