# DeepSeek V4 native MXFP4 routed-expert LoRA SFT

## Status and scope

This development milestone adds a standalone CPU forward/backward path for the
native MXFP4 routed experts in DeepSeek-V4-Flash-0731. It is intended for
kernel and integration developers, not as an end-user whole-model training
guide.

- Baseline: upstream `origin/main@31985f40bcc40da08107efdb1f81bf88cb38c6b2`
- Development branch: `yyj/dsv4-mxfp4-routed-sft-kernel`
- Python entry point: `KTMoEWrapper(..., mode="sft", method="MXFP4_SFT")`
- CPU path: AVX512-BF16, with `amx` and `avx512_bf16` extension variants accepted
- Training mode: frozen MXFP4 base plus BF16 LoRA on gate, up, and down projections

The supported unit is one routed-expert MoE layer. It computes the routed
output, input gradient, router-weight gradient, and all six LoRA gradients.
TP1 and TP2 use the same API and gradient lifecycle.

The following are explicitly outside this milestone:

- whole-model DeepSeek V4 SFT, including attention backward;
- LLaMA-Factory, Transformers, Accelerate, FSDP, or optimizer integration;
- full or hybrid base-weight training;
- GPU experts or mixed CPU/GPU expert placement;
- other FP4 layouts, group sizes, zero points, or non-x86 backends;
- checkpoint save/resume and end-to-end model-quality validation.

## Native storage and computation

For a logical row-major matrix `W[N, K]`, the checkpoint representation is:

| Projection | Packed weight | Group scale |
| --- | --- | --- |
| `w1` (gate) | E2M1 `uint8[I, H/2]` | UE8M0 `[I, H/32]` |
| `w3` (up) | E2M1 `uint8[I, H/2]` | UE8M0 `[I, H/32]` |
| `w2` (down) | E2M1 `uint8[H, I/2]` | UE8M0 `[H, I/32]` |

Each byte stores two E2M1 values, low nibble first. Each consecutive group of
32 values shares one UE8M0 scale. The loader retains the packed E2M1 weights
and stages each UE8M0 scale losslessly in BF16 storage for the kernel (`1..254`
use normal exponent bits, `0` uses the exact `2^-127` BF16 subnormal, and
reserved `0xff` is rejected). It does not create a persistent dense BF16 or
FP32 base-weight copy.

The native boundary reads those BF16 storage bits directly, so host FTZ/DAZ
mode cannot turn UE8M0 code `0` into a true zero while loading.

For routed expert `e`, the forward equation is:

```text
g = X W1[e]^T + (alpha / rank) (X A1[e]^T) B1[e]^T
u = X W3[e]^T + (alpha / rank) (X A3[e]^T) B3[e]^T
z = SiLU(min(g, L)) * clamp(u, -L, L)
y = z W2[e]^T + (alpha / rank) (z A2[e]^T) B2[e]^T
output[token] = sum(route_weight[token, slot] * y[expert_id[token, slot]])
```

`L` is `swiglu_limit` and must be finite and positive. LoRA is added before
the DeepSeek V4 asymmetric clamp. Backward uses inclusive derivative masks at
the clamp boundaries.

The base contribution to `dX` is computed directly from the original packed
row-major weights. The kernel decodes one group at a time, accumulates in FP32,
and writes BF16 output; it does not require a transposed or fully dequantized
base matrix. Only routed experts receive LoRA gradient work. The six caller-
provided BF16 LoRA gradient buffers follow the existing authoritative-gradient
window: overwrite on the first microbatch, in-place accumulation afterward,
and lazy clearing across optimizer windows.

Attempts to request base gradients, replace base-weight pointers, or prepare a
backward copy fail immediately for `MXFP4_SFT`. This protects the frozen native
base from accidental mutation or silent fallback.

## Direct API example

The following shows the standalone boundary. `lora` contains six contiguous
CPU BF16 `torch.nn.Parameter` objects and `grad` contains their six contiguous
CPU BF16 gradient buffers. Both dictionaries use the names `gate_lora_a`,
`gate_lora_b`, `up_lora_a`, `up_lora_b`, `down_lora_a`, and `down_lora_b`, with
an expert dimension first.

```python
import torch

from kt_kernel import KTMoEWrapper
from kt_kernel.utils.loader import MXFP4SafeTensorLoader

model_path = "/path/to/DeepSeek-V4-Flash-0731"
layer = 3

loader = MXFP4SafeTensorLoader(model_path)
try:
    packed = loader.load_experts(
        f"model.layers.{layer}",
        device="cpu",
        reject_non_finite_scales=True,
    )

    moe = KTMoEWrapper(
        layer_idx=layer,
        num_experts=256,
        num_experts_per_tok=6,
        hidden_size=4096,
        moe_intermediate_size=2048,
        gpu_experts_mask=None,
        num_gpu_experts=0,
        cpuinfer_threads=64,
        threadpool_count=2,
        weight_path=model_path,
        chunked_prefill_size=32,
        method="MXFP4_SFT",
        mode="sft",
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        max_cache_depth=1,
        group_size=32,
        zero_point=False,
        full_weight_grad=False,
        swiglu_limit=10.0,
    )
    moe.init_lora_weights(
        lora["gate_lora_a"], lora["gate_lora_b"],
        lora["up_lora_a"], lora["up_lora_b"],
        lora["down_lora_a"], lora["down_lora_b"],
        grad["gate_lora_a"], grad["gate_lora_b"],
        grad["up_lora_a"], grad["up_lora_b"],
        grad["down_lora_a"], grad["down_lora_b"],
    )
    moe.load_mxfp4_weights(packed, torch.arange(256, dtype=torch.int64))
finally:
    loader.close_all_handles()

optimizer = torch.optim.AdamW(lora.values(), lr=1e-4)

# hidden/output/d_output: contiguous CPU BF16 [tokens, hidden_size]
# expert_ids: CPU int64 [tokens, top_k]; route_weights: CPU FP32 [tokens, top_k]
output = moe.forward(hidden, expert_ids, route_weights, save_for_backward=True)
grad_input, grad_route_weights = moe.backward(d_output)

# C++ has published each authoritative grad buffer as Parameter.grad.
optimizer.step()
moe.release_authoritative_optimizer_grads()
moe.update_lora_weights()
optimizer.zero_grad(set_to_none=False)
```

Passing all six LoRA weights as `nn.Parameter` objects makes
`init_lora_weights()` register their authoritative C++ gradient views. The
first backward in an optimizer window overwrites them; later microbatches
accumulate in place. After `optimizer.step()`, release the views before calling
`zero_grad(set_to_none=False)` so that PyTorch does not scan these buffers.
Plain tensors remain accepted only for integrations that register the six
optimizer parameters and gradient views explicitly through
`register_authoritative_optimizer_grad()`.

Required dimensions are multiples of 32. This milestone supports
`threadpool_count` 1 or 2, and each TP intermediate slice
(`moe_intermediate_size / threadpool_count`) must be a multiple of 32.
The wrapper and native load boundary reject unsupported devices, layouts,
modes, and scale values before computation begins.

## Validation

Source-level tests cover E2M1 packing/dequantization, UE8M0 conversion,
transpose-free `dX`, the asymmetric clamp and its boundary gradients, API
dispatch, validation, and frozen-base fail-fast behavior.

The compiled extension and a real checkpoint layer were validated on `qj5090`
(`qujing`): dual AMD EPYC 9355, 64 physical cores / 128 hardware threads,
AVX512-BF16, no AMX.

| Test | Configuration | Result |
| --- | --- | --- |
| Python reference | source-only reference suite | 11 passed |
| Python API | dispatch and contract suite | 11 passed |
| Python gradient lifecycle | authoritative alias, failure, and legacy-backend contracts | 27 passed |
| C++ numerical lifecycle | 4 experts, H=256, I=256, top-2, rank 8; TP1/TP2; GAS 1/2/8 | passed |
| Gradient lifecycle | different active sets, lazy clear, checkpoint cache replacement | passed |
| Inference regression | existing packed `AMXFP4_KGroup_MOE` path | passed |
| Real V4 layer | layer 3, 256 experts, H=4096, I=2048, top-6, rank 8, qlen=2; TP1 and TP2 | passed |

For the compiled synthetic comparison, all reported forward/backward/LoRA
signals had relative L2 error at most `0.008501` and cosine similarity at least
`0.999965` against the FP32 reference. The existing inference path had relative
L2 `0.002364` and cosine similarity `0.999997`. GAS accumulation, stale-expert
clearing, TP equivalence, checkpoint cache replacement, and unchanged
caller-owned packed checkpoint tensors were checked in the same executable. It
exited 0 in 2.80 seconds with 764,544 KiB maximum RSS.

Both real-layer runs loaded 3,623,878,656 bytes of caller-owned packed layer
storage. Their resident-set peaks include the source tensors, temporary TP
staging, and the native kernel's packed copy. The SHA-256 of the source tensors
before and after forward, backward, and an optimizer-like LoRA update was
identical:

```text
9555d919d7fbc4dfe6cc6864d53ecc0acb9116150fb19e9e5f95f5107712f889
```

Their maximum resident set high-water marks were 11.699 GiB (TP1) and 11.623
GiB (TP2). All output, `dX`, router-gradient, and six LoRA-gradient
signals were finite and nonzero where expected; inactive experts remained
zero. Against a dense oracle for token 0's six routed experts, the worst
relative L2 was `0.007374` and the minimum cosine similarity was `0.999973`.
Publishing a visible AdamW-like LoRA update changed the subsequent output
while the source packed-base hash remained unchanged.

These qlen=2 timings are correctness-smoke observations, not throughput or
latency claims.

Remote evidence is retained at:

```text
/mnt/sft_yyj_yyj/dsv4-mxfp4-routed-sft-kernel-20260902/artifacts/
  final_rebuild_ownership.log
  final_numerical_ownership.log
  final_python_contracts_ownership.log
  dsv4_mxfp4_layer3_tp1_ownership.json
  dsv4_mxfp4_layer3_tp2_ownership.json
  real_layer_tp1_ownership.log
  real_layer_tp2_ownership.log
```

## Next phase

The next milestone is whole-model integration: add or adopt DeepSeek V4
attention backward, connect this layer API to the model and training framework,
define optimizer/checkpoint ownership, and then run end-to-end loss, resume,
quality, memory, and throughput validation. None of those results should be
inferred from the standalone routed-expert evidence above.
