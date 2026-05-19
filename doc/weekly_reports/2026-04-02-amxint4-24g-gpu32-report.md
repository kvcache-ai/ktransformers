# AMXINT4 Tiered Report: 24G + GPU Experts 32

## Selected Configuration

- Model path: `/mnt/fr0/qwen_copy_test`
- CPU weight path: `/mnt/fr0/qwen_copy_test-AMXINT4-rerun-aftertpfix-20260402-143950`
- Method: `AMXINT4`
- Weight strategy: `tiered`
- CPU threads: `64`
- Threadpool count: `2`
- GPU experts: `32`
- Resident experts: `256`
- CPU memory budget: `24G`
- GPU experts update: enabled

## Why This Config Was Selected

On the 24G CPU-budget line, `gpu_experts=32` was the highest stable GPU-expert configuration before GPU-memory failures started at `64+`.

| gpu_experts | status | avg tok/s | median | min | max |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | success | 33.068 | 41.084 | 6.354 | 44.734 |
| 4 | success | 39.962 | 48.575 | 8.264 | 50.230 |
| 8 | success | 41.531 | 50.381 | 10.663 | 51.681 |
| 16 | success | 43.174 | 50.378 | 14.550 | 52.698 |
| 32 | success | 45.061 | 52.478 | 14.837 | 54.014 |
| 64 | fail_ready | - | - | - | - |
| 128 | fail_ready | - | - | - | - |
| 256 | fail_ready | - | - | - | - |

## Five Responses Captured

The service was queried with these 5 prompts:

1. `Count from 1 to 40 separated by commas only.`
2. `Explain mmap in 5 short sentences.`
3. `Give 8 concise bullet points about NUMA optimization.`
4. `Summarize BF16 inference in about 60 tokens.`
5. `Compare tiered and legacy loading in concise prose.`

Observed response heads:

1. `count`
   Raw head:
   `Thinking Process: ... Task: Count from 1 to 40 ... Sequence: 1, 2, 3, ...`

2. `mmap`
   Raw head:
   `Thinking Process: ... Topic: mmap ... Maps a file or device into memory ...`

3. `numa`
   Raw head:
   `Thinking Process: ... Topic: NUMA optimization ... Memory placement / local vs remote ...`

4. `bf16`
   Raw head:
   `Thinking Process: ... Topic: BF16 inference ... Similar FP32 range ...`

5. `compare`
   Raw head:
   `Thinking Process: ... Compare tiered loading and legacy loading ...`

Note: output quality is now semantically correct, but the model/template still emits `Thinking Process` before the final answer.

## Single-Prompt Split

This table is for the 5 standard prompts above under `24G + gpu_experts=32`.

Definitions:

- `ttft_s`: time to first token
- `prefill_tok_s`: `prompt_tokens / ttft`
- `decode_tok_s`: `completion_tokens / decode_window`
- `e2e_tok_s`: `completion_tokens / total_request_time`

| prompt | prompt_tokens | completion_tokens | total_s | ttft_s | decode_s | prefill tok/s | decode tok/s | e2e tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count | 23 | 64 | 3.465 | 0.157 | 1.074 | 146.676 | 59.586 | 18.469 |
| mmap | 19 | 64 | 1.250 | 0.144 | 1.009 | 132.032 | 63.434 | 51.188 |
| numa | 21 | 64 | 1.198 | 0.146 | 1.011 | 144.223 | 63.312 | 53.442 |
| bf16 | 24 | 64 | 1.196 | 0.146 | 1.006 | 164.711 | 63.637 | 53.519 |
| compare | 20 | 64 | 1.185 | 0.145 | 1.007 | 138.394 | 63.548 | 54.028 |

## Aggregate Split

This is the aggregate view across the 5 prompts above.

| metric | value |
| --- | ---: |
| avg e2e tok/s | 46.129 |
| avg prefill tok/s | 145.207 |
| avg decode tok/s | 62.703 |
| total prompt tokens | 107 |
| total completion tokens | 320 |
| overall prefill tok/s | 144.986 |
| overall decode tok/s | 62.659 |
| overall e2e tok/s | 38.582 |

Interpretation:

- The first request is still much slower than the other 4.
- Once in steady decode, the service stays around `~63 tok/s`.
- End-to-end average is lower because TTFT and the first request dominate.

## Split Tables

The split metrics below use two views:

- **Single 5-prompt average**: average of per-prompt `prefill_tok_s` / `decode_tok_s`
- **Overall aggregate**: total prompt tokens divided by total TTFT, and total completion tokens divided by total decode window

### A. Budget Sweep (`gpu_experts=0`)

| budget | avg prefill tok/s | avg decode tok/s | overall prefill tok/s | overall decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 40G | 144.502 | 60.307 | 144.399 | 60.309 |
| 32G | 146.111 | 60.274 | 146.175 | 60.264 |
| 24G | 144.142 | 60.790 | 144.205 | 60.779 |

### B. GPU Experts Sweep (fixed `24G`)

| gpu_experts | avg prefill tok/s | avg decode tok/s | overall prefill tok/s | overall decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 2 | 145.816 | 61.173 | 145.975 | 61.150 |
| 4 | 146.119 | 61.526 | 145.975 | 61.503 |
| 8 | 143.841 | 60.810 | 143.817 | 60.802 |
| 16 | 144.580 | 62.779 | 144.790 | 62.782 |
| 32 | 145.434 | 62.723 | 145.380 | 62.684 |

### How to read these two tables

- Budget changes (`40G -> 24G`) barely affect steady-state decode. The decode line stays around `~60-61 tok/s`.
- The bigger effect of lower budget is not steady decode itself, but slower first-request / TTFT behavior.
- Adding GPU experts under `24G` does not materially change prefill throughput.
- Decode throughput does improve a bit with more GPU experts, from `~61.2` to `~62.7 tok/s`, but the gain is incremental rather than dramatic.

## Cold-State SSD and GPU Metrics

Cold-state procedure:

1. stop service
2. `drop_caches`
3. start service
4. measure:
   - startup -> ready
   - ready -> 5 hard prompts

Hard-prompt result under `24G + gpu_experts=32`:

| metric | value |
| --- | ---: |
| avg tok/s | 43.157 |
| startup SSD read avg (MiB/s) | 349.069 |
| prompt-window SSD read avg (MiB/s) | 353.288 |
| prompt-window SSD read peak (MiB/s) | 1043.578 |
| prompt-window SSD write avg/peak (MiB/s) | 0 / 0 |
| GPU experts | 0 |
| GPU util avg/max (%) | 25.517 / 52 |
| GPU RX avg/max (MB/s) | 112.931 / 196 |
| GPU TX avg/max (MB/s) | 56.310 / 96 |
| memory page read-in avg/peak (MiB/s) | 757.988 / 2086.702 |
| memory page write-out avg/peak (MiB/s) | 0.010 / 0.082 |

Important:

- The GPU-expert scan above uses `gpu_experts=32`.
- The cold-state high-difficulty SSD probe that was already completed earlier was run with `gpu_experts=0`.
- It still gives a correct answer for the question “does this scheme eat SSD throughput in cold state?” and the answer is yes.

## Bottom Line

- `24G + gpu_experts=32` is the strongest stable GPU-expert point tested so far.
- Standard-prompt steady decode is about `~63 tok/s`.
- End-to-end average is lower because the first request is much slower.
- Cold-state startup and prompt window both consume significant SSD bandwidth.
- Hot-state work windows can show near-zero incremental block IO, but cold-state windows clearly do not.
