# Running XingChen4 with SGLang in KTransformers

This guide shows how to run XingChen4 (`XingChen4ForCausalLM`) with
KTransformers and the bundled `third_party/sglang` code.

The names and configuration values below follow the current implementation in
`sglang/srt/configs/xingchen4.py` and `sglang/srt/models/xingchen4.py`.

## Model profile

XingChen4 uses a 40-layer decoder architecture with MLA attention, MoE feed-forward
layers, and mHC (Manifold-constrained Hyper-Connection) residual streams.

- Architecture: `XingChen4ForCausalLM`
- Configuration class: `XingChen4Config`
- Model type: `xingchen4`
- Decoder layers: `num_hidden_layers = 40`
- Hidden size: `hidden_size = 3584`
- Dense MLP size: `intermediate_size = 9216`
- Vocabulary size: `vocab_size = 131072`
- Maximum position embeddings: `max_position_embeddings = 262144`
- RMSNorm epsilon: `rms_norm_eps = 1e-6`

### MLA attention

- Attention heads: `num_attention_heads = 32`
- KV heads: `num_key_value_heads = 32`
- Query LoRA rank: `q_lora_rank = 768`
- KV LoRA rank: `kv_lora_rank = 512`
- Non-RoPE query/key head dimension: `qk_nope_head_dim = 128`
- RoPE query/key head dimension: `qk_rope_head_dim = 64`
- Value head dimension: `v_head_dim = 128`
- RoPE base: `rope_theta = 10000`
- Interleaved RoPE layout: `rope_interleave = true`

The runtime attention module is `DeepseekV2AttentionMLA`. When Query LoRA is
enabled, `q_a_proj` and `kv_a_proj_with_mqa` are packed as
`fused_qkv_a_proj_with_mqa` by the SGLang loader.

### MoE layers

- Routed experts: `n_routed_experts = 64`
- Shared experts: `n_shared_experts = 1`
- Experts selected per token: `num_experts_per_tok = 4`
- Expert intermediate size: `moe_intermediate_size = 1024`
- MoE layer frequency: `moe_layer_freq = 1`
- Dense layers before MoE: `first_k_dense_replace = 0`
- Router scoring function: `scoring_func = "sigmoid"`
- Normalized top-k probabilities: `norm_topk_prob = true`
- Routed output scale: `routed_scaling_factor = 2.0`
- Top-k method: `topk_method = "noaux_tc"`

Each decoder layer therefore uses `DeepseekV2MoE`. Its expert `gate_proj` and
`up_proj` weights may be packed as `gate_up_proj` by the runtime loader.

### mHC residual streams

- Number of residual streams: `hc_mult = 4`
- Sinkhorn iterations: `hc_sinkhorn_iters = 20`
- mHC epsilon: `hc_eps = 1e-6`
- Contract residual streams before the final norm and draft model:
  `hc_contract_for_draft = true`

The checkpoint stores the mHC operands directly. No `mapping_proj`, `alpha_*`,
or `bias` materialization is required. The operands are converted to contiguous
FP32 buffers after loading and consumed by the fused mHC kernels.

### Runtime module and checkpoint names

| Component | Runtime module or parameter name |
| --- | --- |
| Token embedding | `model.embed_tokens` |
| Decoder layer | `model.layers.<layer_id>` |
| MLA attention | `model.layers.<layer_id>.self_attn` |
| Attention-side mHC | `model.layers.<layer_id>.attn_hc` |
| Attention mHC operands | `attn_hc.hc_fn`, `attn_hc.hc_scale`, `attn_hc.hc_base` |
| Attention input norm | `model.layers.<layer_id>.input_layernorm` |
| MoE/MLP | `model.layers.<layer_id>.mlp` |
| FFN-side mHC | `model.layers.<layer_id>.ffn_hc` |
| FFN mHC operands | `ffn_hc.hc_fn`, `ffn_hc.hc_scale`, `ffn_hc.hc_base` |
| Post-attention norm | `model.layers.<layer_id>.post_attention_layernorm` |
| Final norm | `model.norm` |
| Output head | `lm_head` |

## Hardware requirements

Minimum example configuration:

- GPU: one NVIDIA GeForce RTX 4090 or another GPU with at least 24 GB VRAM
- CPU: x86-64 with AVX2 or newer instructions; a high core count is recommended
- Storage: approximately 200 GB for weights and temporary caches
- CUDA: CUDA 12.2 or newer is recommended

## Installation

From the KTransformers repository root, initialize the SGLang submodule and
install KTransformers and KT-Kernel:

```bash
git submodule update --init --recursive third_party/sglang
./install.sh
cd kt-kernel
./install.sh
```

Alternatively, use the packaged components when they are available for your
environment:

```bash
pip install kt-kernel sglang-kt
```

## Prepare model weights

Place the XingChen4 checkpoint in a local directory. The examples below use:

```text
/data/models/xingchen4
```

The checkpoint configuration must declare:

```json
{
  "architectures": ["XingChen4ForCausalLM"],
  "model_type": "xingchen4"
}
```

## Start the server

The following example runs XingChen4 on one RTX 4090 and places 24 routed
experts on the GPU:

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path /data/models/xingchen4 \
  --served-model-name xingchen4 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --attention-backend triton \
  --kt-weight-path /data/models/xingchen4 \
  --kt-method BF16 \
  --kt-cpuinfer 50 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 24 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 512 \
  --max-total-tokens 65540 \
  --tool-call-parser xingchen4 \
  --reasoning-parser xingchen4
```

See the [KT-Kernel parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters)
for additional CPU/GPU expert placement options.

## Send a request

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "xingchen4",
    "messages": [
      {"role": "user", "content": "Hi, who are you?"}
    ],
    "temperature": 1,
    "stream": false,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Response:

```json
{
  "id": "6a0d6434acde47d087a240810e1034cb",
  "object": "chat.completion",
  "created": 1787124661,
  "model": "xingchen4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I am the TeleChat.\nI was developed by China Telecom Artificial Intelligence Technology Co., Ltd.\n\nAs an all-around AI assistant, I have a vast knowledge base and can help you with general Q&A, content creation, language translation, deep logical reasoning, programming assistance, and much more.\n\nI am dedicated to providing you with efficient and professional support in your work, studies, and creative endeavors.\n\nHow can I assist you today?",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "matched_stop": 2
    }
  ],
  "usage": {
    "prompt_tokens": 11,
    "total_tokens": 108,
    "completion_tokens": 97,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}
```

With thinking disabled, a successful response returns assistant text in
`choices[0].message.content` and normally leaves `reasoning_content` as `null`.

## Tool-call and reasoning parsers

- `--tool-call-parser xingchen4` parses XingChen4 tool calls represented by
  `<tool_call>...</tool_call>` blocks.
- `--reasoning-parser xingchen4` separates content enclosed by
  `<think>...</think>` into `reasoning_content`.

The parser implementations are registered in:

- `sglang/srt/function_call/xingchen4_detector.py`
- `sglang/srt/function_call/function_call_parser.py`
- `sglang/srt/parser/reasoning_parser.py`

## Implementation paths

- Model: `third_party/sglang/python/sglang/srt/models/xingchen4.py`
- Configuration: `third_party/sglang/python/sglang/srt/configs/xingchen4.py`
- Tool-call parser: `third_party/sglang/python/sglang/srt/function_call/xingchen4_detector.py`
- Model-specific runtime defaults: `third_party/sglang/python/sglang/srt/server_args.py`
