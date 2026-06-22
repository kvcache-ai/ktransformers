# KT SFT and Serving Performance Report

## Environment

| Field | Value |
| --- | --- |
| Date | |
| Machine | Sapphire4 |
| GPUs | |
| CPU / NUMA | |
| ktransformers commit | |
| sglang commit | |
| LLaMA-Factory commit | |
| KT method | |
| TP size | |
| CPU infer threads | |
| Threadpool count | |
| GPU experts | |
| Chunked prefill size | |
| Max running requests | |

## Models

| Model | Base path | KT weight path | LoRA adapter |
| --- | --- | --- | --- |
| Qwen3-235B | | | |
| Qwen3.5-397B | | | |

## SFT Throughput

Report effective supervised-token throughput with `label_tokens_global` unless
otherwise stated.

| Model | LoRA rank/alpha | Seq len | Global batch | Grad acc | Samples | Aggregate TPS | P50 micro TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-235B | | | | | | | |
| Qwen3.5-397B | | | | | | | |

## Serving Throughput

Run prefill and decode as separate phases. Prefill uses `max_tokens=1`. Decode
TPS excludes TTFT when streaming usage is available.

| Model | LoRA | Phase | Prompt tokens | Output tokens | Concurrency | Requests | TPS mean | TPS p50 | TPS p90 | TTFT p50 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-235B | no | prefill | | 1 | | | | | | |
| Qwen3-235B | no | decode | | | | | | | | |
| Qwen3-235B | yes | prefill | | 1 | | | | | | |
| Qwen3-235B | yes | decode | | | | | | | | |
| Qwen3.5-397B | no | prefill | | 1 | | | | | | |
| Qwen3.5-397B | no | decode | | | | | | | | |
| Qwen3.5-397B | yes | prefill | | 1 | | | | | | |
| Qwen3.5-397B | yes | decode | | | | | | | | |

## Notes

- Prefill and decode were not mixed within the same run.
- Warmup requests/steps were excluded from reported aggregates.
- Include any OOM/retry/server restart details here.
