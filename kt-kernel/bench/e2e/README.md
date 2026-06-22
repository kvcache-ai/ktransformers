# KT E2E Benchmark Harness

This directory contains reusable harnesses for the PR performance report. Keep
model launch commands and private dataset paths outside these scripts; pass them
through shell wrappers or CI/job configs.

## Metrics

- **SFT TPS**: training tokens per second from `KT_E2E_TIMING_JSONL`.
  Prefer `label_tokens_global` for effective supervised tokens. Use
  `attention_tokens_global` when reporting total processed sequence tokens.
- **Prefill TPS**: prompt tokens divided by streaming TTFT. Run with
  `max_tokens=1` so decode work does not pollute prefill timing.
- **Decode TPS**: generated tokens after first streamed token divided by time
  from first streamed token to request end. Use a short prompt and long output.

## Recommended Sapphire4 Protocol

Run prefill and decode as separate server phases. Do not mix decode-heavy
requests into a prefill run; SGLang scheduling can interleave decode with later
prefill and understate pure prefill throughput.

For each target model, collect four inference rows:

1. base model prefill
2. base model decode
3. LoRA model prefill
4. LoRA model decode

Suggested initial request parameters:

| Metric | Concurrency | Prompt | Output |
| --- | ---: | ---: | ---: |
| prefill | 1, then 2/4 if stable | 8k-16k tokens | 1 token |
| decode | 1, then 2/4 if stable | 128-512 tokens | 256-512 tokens |

For Qwen3-235B and Qwen3.5-397B, use the largest prompt length that avoids
chunked-prefill spillover/OOM under the chosen `max_total_tokens`. Record the
server launch parameters with every result: TP size, KT method, CPU threads,
threadpool count, GPU expert count, chunked prefill size, max running requests,
and LoRA adapter path.

## Inference Examples

Prefill, base model:

```bash
python kt-kernel/bench/e2e/bench_openai_tps.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model qwen3_235b_base \
  --mode prefill \
  --prompt-chars 48000 \
  --max-tokens 1 \
  --requests 8 \
  --concurrency 1 \
  --output-prefix /path/to/results/qwen3_235b_base_prefill
```

Decode, LoRA model:

```bash
python kt-kernel/bench/e2e/bench_openai_tps.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model qwen3_235b:lora_name \
  --mode decode \
  --prompt-chars 1600 \
  --max-tokens 512 \
  --requests 8 \
  --concurrency 1 \
  --output-prefix /path/to/results/qwen3_235b_lora_decode
```

## SFT Example

Set `KT_E2E_TIMING_JSONL` before launching LLaMA-Factory training:

```bash
KT_E2E_TIMING_JSONL=/path/to/results/qwen35_sft_timing.jsonl \
accelerate launch --config_file fsdp2.yaml src/train.py train.yaml
```

Summarize:

```bash
python kt-kernel/bench/e2e/summarize_sft_timing.py \
  /path/to/results/qwen35_sft_timing.jsonl \
  --token-field label_tokens_global \
  --drop-first 1 \
  --output /path/to/results/qwen35_sft_summary.json
```

Old timing logs without token fields can still be summarized with
`--tokens-per-micro-step`.
