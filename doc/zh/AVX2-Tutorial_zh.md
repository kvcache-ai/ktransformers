# 在 AVX2 CPU 上使用 KTransformers

本教程介绍如何在仅支持 AVX2 的机器上运行 KTransformers（无需 AVX512 或 AMX）。

## 目录

- [支持的精度格式](#支持的精度格式)
- [硬件要求](#硬件要求)
- [安装](#安装)
- [验证](#验证)
- [启动推理服务](#启动推理服务)
  - [示例：Qwen3-30B-A3B (BF16)](#示例qwen3-30b-a3b-bf16)
  - [示例：Qwen3.5-35B-A3B-FP8 (FP8)](#示例qwen35-35b-a3b-fp8-fp8)
  - [示例：Qwen3-30B-A3B-GPTQ-Int4 (GPTQ_INT4)](#示例qwen3-30b-a3b-gptq-int4-gptq_int4)
  - [示例：Kimi-K2.5 (RAWINT4)](#示例kimi-k25-rawint4)
  - [发送请求](#发送请求)
- [性能调优](#性能调优)
- [常见问题](#常见问题)

## 支持的精度格式

| `--kt-method` | 精度 | 说明 |
|---------------|------|------|
| `BF16` | BF16 原精度 | 零精度损失，直接使用 BF16 权重 |
| `FP8` | FP8 分块量化 |  |
| `GPTQ_INT4` | INT4 GPTQ |  |
| `RAWINT4` | Raw INT4 + BF16 缩放因子 | Kimi-K2.5 专用；权重以压缩 SafeTensor 格式存储 |


## 硬件要求

- **CPU**：x86-64 + AVX2 + FMA（Intel Haswell 2013+ / AMD Zen+）
- **GPU**：NVIDIA 24GB+ 显存（RTX 3090/4090/5090 等）
- **内存**：不少于模型权重大小（如 Qwen3-30B-A3B BF16 需 64GB+）
- **系统**：Linux

## 安装

从源码编译安装（一键安装 kt-kernel + SGLang）：

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive

# 一键安装
./install.sh
```

在AVX512， AMX机器上， 也可以手动强制 AVX2 编译：

```bash
export KT_RAWINT4_BACKEND=avx2
export CPUINFER_CPU_INSTRUCT=AVX2
export CPUINFER_ENABLE_AMX=OFF
./install.sh kt-kernel --manual
```



## 验证

```bash
# 检查 CPU 是否支持 AVX2
lscpu | grep -i avx2

# 检查 kt-kernel 加载的变体
python -c "import kt_kernel; print(kt_kernel.__cpu_variant__)"
# 预期输出：avx2

# 系统诊断
kt doctor
```

## 启动推理服务

使用 `--kt-method BF16`、`FP8`、`GPTQ_INT4` 或 `RAWINT4`，KT-Kernel 会**自动检测** CPU 并在缺少 AVX512/AMX 时回退到 AVX2 后端。

### 示例：Qwen3-30B-A3B (BF16)

```bash
# 下载模型
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /path/to/Qwen3-30B-A3B

# 查看物理核心数和 NUMA 节点数
lscpu | grep -E "^CPU\(s\)|Thread\(s\) per core|NUMA node\(s\)"

# 启动服务（按实际硬件调整 kt-cpuinfer 和 kt-threadpool-count）
python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/Qwen3-30B-A3B \
  --kt-weight-path /path/to/Qwen3-30B-A3B \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 32 \
  --kt-method BF16 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 \
  --max-running-requests 2 \
  --served-model-name Qwen3 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --enable-p2p-check \
  --disable-shared-experts-fusion
```

### 示例：Qwen3.5-35B-A3B-FP8 (FP8)

```bash
# 下载模型
huggingface-cli download Qwen/Qwen3.5-35B-A3B-FP8 --local-dir /path/to/Qwen3.5-35B-A3B-FP8

# 启动服务
python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/Qwen3.5-35B-A3B-FP8 \
  --kt-weight-path /path/to/Qwen3.5-35B-A3B-FP8 \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 2 \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 400 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --max-running-requests 1 \
  --max-total-tokens 32000 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --disable-shared-experts-fusion
```

### 示例：Qwen3-30B-A3B-GPTQ-Int4 (GPTQ_INT4)

```bash
# 下载模型
huggingface-cli download Qwen/Qwen3-30B-A3B-GPTQ-Int4 --local-dir /path/to/Qwen3-30B-A3B-GPTQ-Int4

# 启动服务
python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --kt-weight-path /path/to/Qwen3-30B-A3B-GPTQ-Int4 \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 2 \
  --kt-method GPTQ_INT4 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --max-running-requests 1 \
  --max-total-tokens 32000 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --disable-shared-experts-fusion
```

### 示例：Kimi-K2.5 (RAWINT4)

> **说明**：以下命令针对 4x RTX PRO 6000 Blackwell（各 96GB）+ AMD Threadripper PRO 5995WX（64 核，1 NUMA 节点）+ 256GB RAM 优化。

```bash
# 下载模型
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /path/to/Kimi-K2.5

# 启动服务
python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model /path/to/Kimi-K2.5 \
  --kt-weight-path /path/to/Kimi-K2.5 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 228 \
  --kt-enable-dynamic-expert-update \
  --kt-method RAWINT4 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --chunked-prefill-size 8192 \
  --max-running-requests 4 \
  --context-length 262144 \
  --enable-mixed-chunk \
  --tensor-parallel-size 4 \
  --enable-p2p-check \
  --disable-shared-experts-fusion
```


### 发送请求

```bash
# 交互聊天
kt chat

# OpenAI 兼容 API
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3","messages":[{"role":"user","content":"你好"}],"stream":true}'
```



## 性能调优

- `--kt-cpuinfer` 设为**物理核心数**
- `--kt-threadpool-count` 设为 **NUMA 节点数**
- `--kt-num-gpu-experts` 越大 CPU 负担越小，但 GPU 显存占用越高
- 内存带宽往往是瓶颈，DDR5 高频内存有明显帮助

## 常见问题



**GPU OOM**
- 减小 `--kt-num-gpu-experts`、`--chunked-prefill-size`、`--max-total-tokens`
- 降低 `--mem-fraction-static`

更多问题参见 [FAQ](../en/FAQ.md)。
