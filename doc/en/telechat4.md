# Running TeleChat4 with SGLang in ktransformers

This guide shows how to run TeleChat4 (TeleChat4ForCausalLM) with the local
`ktransformers` submission branch and its bundled `third_party/sglang` code.
The commands below use the in-repo docs/paths and are aligned with the current
`telechat4.py`, `models/telechat4.py`, and `telechat4` parser registrations.

## Table of Contents

- [Running TeleChat4 with SGLang in ktransformers](#running-telechat4-with-sglang-in-ktransformers)
  - [Table of Contents](#table-of-contents)
  - [Model profile](#model-profile)
  - [Hardware Requirements](#hardware-requirements)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download/Prepare model weights](#step-1-downloadprepare-model-weights)
  - [Step 2: Start SGLang server](#step-2-start-sglang-server)
    - [Launch Command (1x RTX 4090 Example)](#launch-command-1x-rtx-4090-example)
  - [Step 3: Send requests](#step-3-send-requests)
    - [Basic chat completion](#basic-chat-completion)
    - [Tool call + reasoning parser options](#tool-call--reasoning-parser-options)
  - [Notes for ktransformers/TeleChat4](#notes-for-ktransformerstelechat4)

## Model profile

TeleChat4 is implemented in this branch with:

- 40 transformer layers
- `hidden_size = 3584`, `intermediate_size = 9216`
- `num_attention_heads = 32`, `num_key_value_heads = 32`
- MLA attention with `qk_rope_head_dim = 64`, `v_head_dim = 128`
- MoE config:
  - `n_routed_experts = 64`
  - `n_shared_experts = 1`
  - `num_experts_per_tok = 4`
- 40-layer mHC residual stream path (`num_residual_streams = 4`)
- Context and other config values from current `TeleChat4Config`:
  - `kv_lora_rank = 512`, `q_lora_rank = 768`
  - `moe_intermediate_size = 1024`
  - `max_position_embeddings = 262144`
  - `routed_scaling_factor = 2.0`

Note:
- `qk_nope_head_dim` is `128` in this config and `qk_rope_head_dim` is `64`.
- `num_hidden_layers = 40` means a full 40-layer stack.

The TeleChat4 parser integration is present in:

- Tool-call parser: `telechat4` (in `sglang/srt/function_call/telechat4_detector.py`)
- Reasoning parser: `telechat4` (in `sglang/srt/parser/reasoning_parser.py`)

## Hardware Requirements

**Minimum Configuration:**

- **GPU**: NVIDIA GeForce RTX 4090 (or equivalent 24GB) for bf16 TeleChat4 inference
- **CPU**: x86_64 CPU with AVX2+ support (recommended high core count for preprocessing and loader)
- **RAM**: At least 64GB system memory (more is better for large checkpoints and higher parallelism)
- **Storage**: ~200GB (model files + temporary cache)

## Prerequisites

Before starting, ensure you have:


1. **KT-Kernel installed**:

```bash
cd ./ktransformer-telechat

git remote -v
git remote add coworker https://github.com/PaddyXj/sglang.git
git fetch coworker

cd kt-kernel && ./install.sh
```

2. **SGLang installed** - Install the kvcache-ai fork of SGLang (one of):

```bash
# Option A: One-click install (from ktransformers root)
./install.sh

# Option B: pip install
pip install kt-kernel sglang-kt
```

3. **CUDA toolkit** - Compatible with your GPU (CUDA 12.2+ recommended)


4. Prepare model files locally at `/data/models/telechat4`.

## Step 1: Download/Prepare model weights

Download or place your TeleChat4 checkpoint to a local directory.

Example:

```bash
mkdir -p /data/models/telechat4
# put TeleChat4 29b files under /data/models/29b
```

## Step 2: Start SGLang server

### Launch Command (1x RTX 4090 Example)

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path /data/models/telechat4 \
  --served-model-name telechat4 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --attention-backend triton \
  --kt-weight-path /data/models/telechat4 \
  --kt-method BF16 \
  --kt-cpuinfer 50 \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-num-gpu-experts 16 \
  --mem-fraction-static 0.9 \
  --tool-call-parser telechat4 \
  --reasoning-parser telechat4
```

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for detailed parameter tuning guidelines.

## Step 3: Send requests

### Basic chat completion

Request:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "29b",
    "messages": [
      {"role": "user", "content": "hi,who are you?"}
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
  "model": "29b",
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

### Tool call + reasoning parser options

- `--tool-call-parser telechat4` parses TeleChat4-style tool calls in `<tool_call>...</tool_call>` blocks.
- `--reasoning-parser telechat4` extracts reasoning channel output 


## Notes for ktransformers/TeleChat4

- `server_args.py` contains model-specific handling for `TeleChat4ForCausalLM` to
  route mHC to the TileLang path when needed.
- The TeleChat4 implementation is in `third_party/sglang/python/sglang/srt/models/telechat4.py`.
- The model config is in `third_party/sglang/python/sglang/srt/configs/telechat4.py`.
