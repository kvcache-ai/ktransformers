# KTransformers+SGLang Inference Deployment
Please Note This is Quantization Deployment. For Native Kimi K2 Thinking deployment please refer to [here](./Kimi-K2-Thinking-Native.md).

## Installation

Step 1: Install SGLang

Follow the [official SGLang installation](https://docs.sglang.ai/get_started/install.html) guide to install SGLang:
```
pip install "sglang[all]"
```

Step 2: Install KTransformers CPU Kernels

The KTransformers CPU kernels (kt-kernel) provide AMX-optimized computation for hybrid inference, for detailed installation instructions and troubleshooting, refer to the official [kt-kernel installation guide](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md).

## Download Model

Download the official KIMI weights as GPU weights.

* huggingface: https://huggingface.co/moonshotai/Kimi-K2-Thinking
* modelscope: https://modelscope.cn/models/moonshotai/Kimi-K2-Thinking

Download the AMX INT4 quantized weights from https://huggingface.co/KVCache-ai/Kimi-K2-Thinking-CPU-weight as CPU weights.

## How to start
```
python -m sglang.launch_server   --host 0.0.0.0   --port 60000   --model path/to/Kimi-K2-Thinking/   --kt-weight-path path/to/Kimi-K2-Instruct-CPU-weight/   --kt-cpuinfer 56   --kt-threadpool-count 2   --kt-num-gpu-experts 200   --kt-method AMXINT4   --attention-backend flashinfer   --trust-remote-code   --mem-fraction-static 0.98   --chunked-prefill-size 4096   --max-running-requests 37   --max-total-tokens 37000   --enable-mixed-chunk   --tensor-parallel-size 8   --enable-p2p-check   --disable-shared-experts-fusion
```
tips:

`--kt-cpuinfer`: is recommended to be set to (number of physical CPU cores - 8 (number of GPUs)).

`--kt-num-gpu-experts`: refers to the number of experts retained on GPUs, which should be adjusted according to your available GPU memory and expected KV cache space.

## Test

When testing, you need to add `--disable-radix-cache` and `--disable-chunked-prefix-cache` when starting the server.

### bench prefill
```
python -m sglang.bench_serving   --backend sglang   --host 127.0.0.1   --port 60000   --num-prompts 37 --random-input-len 1024 --random-output-len 1 --random-range-ratio 1.0 --dataset-name random
```

### bench decode
```
python -m sglang.bench_serving   --backend sglang   --host 127.0.0.1   --port 60000   --num-prompts 37 --random-input-len 10 --random-output-len 512 --random-range-ratio 1.0 --dataset-name random
```

## Performance

### System Configuration:

- GPUs: 8× NVIDIA L20
- CPU: Intel(R) Xeon(R) Gold 6454S

### Bench prefill
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 not set
Successful requests:                     37
Benchmark duration (s):                  65.58
Total input tokens:                      37888
Total input text tokens:                 37888
Total input vision tokens:               0
Total generated tokens:                  37
Total generated tokens (retokenized):    37
Request throughput (req/s):              0.56
Input token throughput (tok/s):          577.74
Output token throughput (tok/s):         0.56
Total token throughput (tok/s):          578.30
Concurrency:                             23.31
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   41316.50
Median E2E Latency (ms):                 41500.35
---------------Time to First Token----------------
Mean TTFT (ms):                          41316.48
Median TTFT (ms):                        41500.35
P99 TTFT (ms):                           65336.31
---------------Inter-Token Latency----------------
Mean ITL (ms):                           0.00
Median ITL (ms):                         0.00
P95 ITL (ms):                            0.00
P99 ITL (ms):                            0.00
Max ITL (ms):                            0.00
==================================================
```

### Bench decode

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 not set
Successful requests:                     37
Benchmark duration (s):                  412.66
Total input tokens:                      370
Total input text tokens:                 370
Total input vision tokens:               0
Total generated tokens:                  18944
Total generated tokens (retokenized):    18618
Request throughput (req/s):              0.09
Input token throughput (tok/s):          0.90
Output token throughput (tok/s):         45.91
Total token throughput (tok/s):          46.80
Concurrency:                             37.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   412620.35
Median E2E Latency (ms):                 412640.56
---------------Time to First Token----------------
Mean TTFT (ms):                          3551.87
Median TTFT (ms):                        3633.59
P99 TTFT (ms):                           3637.37
---------------Inter-Token Latency----------------
Mean ITL (ms):                           800.53
Median ITL (ms):                         797.89
P95 ITL (ms):                            840.06
P99 ITL (ms):                            864.96
Max ITL (ms):                            3044.56
==================================================
```
