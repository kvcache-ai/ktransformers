# 在 Intel iGPU 上使用 SYCL GPTQ INT4 后端

本教程介绍如何使用 KTransformers 的 `SYCL_GPTQ_INT4` 后端，将 MoE 模型中的对称 GPTQ INT4 experts 放到 Intel 集成显卡（iGPU）上计算。

这是一个异构推理方案：Intel iGPU 负责未放在 CUDA GPU 上的 GPTQ INT4 experts；Attention、Embedding、LM Head 以及通过 `--kt-num-gpu-experts` 保留的 experts 仍由 CUDA GPU 执行。因此，当前实现不能替代 NVIDIA GPU，也不是纯 Intel GPU 推理后端。

## 目录

- [支持范围](#支持范围)
- [硬件和软件要求](#硬件和软件要求)
- [安装](#安装)
- [验证安装](#验证安装)
- [下载模型](#下载模型)
- [启动推理服务](#启动推理服务)
- [发送请求](#发送请求)
- [参数说明](#参数说明)
- [常见问题](#常见问题)

## 支持范围

当前 SYCL 后端支持：

- Linux x86-64 平台。
- 通过 Intel Level Zero 驱动访问的 Intel GPU，主要面向共享系统内存的 Intel iGPU。
- MoE expert 的对称 GPTQ INT4 权重。
- `sym=true`、`desc_act=false`，权重包含 `qweight` 和 `scales`，不包含 `qzeros`。
- 已重点验证 `group_size=128` 的模型。
- 单机、`tensor_parallel_size=1`、以 batch size 1 为主的端侧推理。

不支持把 BF16、FP8 或非对称 GPTQ 权重交给此后端。启动时必须使用：

```text
--kt-method SYCL_GPTQ_INT4
```

## 硬件和软件要求

- **Intel GPU**：支持 Level Zero 和 shared USM 的 Intel iGPU。
- **CUDA GPU**：一张受当前 PyTorch 和 SGLang 支持的 NVIDIA GPU，用于模型的其余部分。
- **CPU**：x86-64，支持 AVX2 和 FMA。
- **系统内存**：至少能够容纳模型权重、KV Cache 和 iGPU 运行缓冲区。iGPU 会与 CPU 共享系统内存带宽。
- **操作系统**：Linux。
- **Intel oneAPI**：需要包含 `icpx`、SYCL runtime 和 `sycl-ls` 的 oneAPI Base Toolkit。
- **Intel GPU 驱动**：需要安装 Level Zero 用户态驱动。
- **CUDA Toolkit**：当前 SYCL 构建仍需要 CUDA runtime 提供 SGLang 与 CPUInfer 之间的 host callback 调度。

先确认 Intel GPU 能被 SYCL 正常识别：

```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

输出中应当能看到类似下面的 Level Zero GPU：

```text
[level_zero:gpu:0] ...
```

还需要确认当前用户可以读写 GPU render node：

```bash
ls -l /dev/dri/renderD*
groups
```

如果当前用户不在 `render` 用户组中，可以执行：

```bash
sudo usermod -aG render "$USER"
```

修改用户组后需要注销并重新登录。

## 安装

从源码构建时，`CPUINFER_USE_SYCL` 默认关闭，需要显式设置为 `1`：

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive

source /opt/intel/oneapi/setvars.sh
CPUINFER_USE_SYCL=1 ./install.sh
```

安装脚本会使用 `icpx` 构建 SYCL GPTQ INT4 后端。普通的 CPU/CUDA 构建不需要 oneAPI；只有启用 `CPUINFER_USE_SYCL=1` 时才需要上述环境。

SYCL、CUDA、ROCm、MUSA 和 MACA 是互斥的 kt-kernel 编译后端。如果当前 shell 曾经显式设置过其他后端，请先取消对应变量，例如：

```bash
unset CPUINFER_USE_CUDA CPUINFER_USE_ROCM CPUINFER_USE_MUSA CPUINFER_USE_MACA
```

未显式设置 `CPUINFER_USE_CUDA` 时不需要手动取消，安装脚本会在启用 SYCL 后端时自动关闭 CUDA kernel 后端。CUDA Toolkit 仍会用于链接 host callback 所需的 `cudart`。

## 验证安装

安装完成后，检查 Python 扩展是否包含 SYCL 后端：

```bash
source /opt/intel/oneapi/setvars.sh
python -c "import kt_kernel_ext.moe as moe; print(hasattr(moe, 'SYCLGPTQInt4_MOE'))"
```

预期输出：

```text
True
```

如果机器上存在多个 SYCL GPU，可以在启动服务前指定设备：

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
```

只有一个可用 Intel GPU 时通常不需要设置该变量，后端会优先选择 Level Zero GPU。

## 下载模型

下面使用 `Qwen3.5-35B-A3B-GPTQ-Int4` 作为示例：

```bash
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --local-dir /path/to/Qwen3.5-35B-A3B-GPTQ-Int4
```

使用其他模型时，请先确认它是 MoE 模型，并且 GPTQ 配置为 `sym=true`、`desc_act=false`。目前推荐使用已经验证过的 `group_size=128` 权重。

## 启动推理服务

下面的命令面向单用户、batch size 1 的端侧推理。请将模型路径和显存相关参数改为适合本机的值：

```bash
source /opt/intel/oneapi/setvars.sh

SGLANG_MAMBA_CONV_DTYPE=float16 \
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model /path/to/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --kt-weight-path /path/to/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --served-model-name qwen3.5 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype float16 \
  --attention-backend triton \
  --mem-fraction-static 0.90 \
  --max-running-requests 1 \
  --max-total-tokens 32000 \
  --max-prefill-tokens 16000 \
  --chunked-prefill-size 4096 \
  --watchdog-timeout 1200 \
  --kt-method SYCL_GPTQ_INT4 \
  --kt-cpuinfer 8 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 24 \
  --disable-shared-experts-fusion \
  --disable-cuda-graph
```

正常使用时不需要关闭 radix cache。`--disable-radix-cache` 主要用于确保每次请求都执行完整 prefill 的性能测试，不建议加入日常启动命令。

## 发送请求

服务启动后，可以通过 OpenAI 兼容接口发送请求：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "stream": false,
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ]
  }'
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--kt-method SYCL_GPTQ_INT4` | 让未放在 CUDA GPU 上的 GPTQ INT4 experts 使用 SYCL 后端。 |
| `--kt-num-gpu-experts` | 保留在 CUDA GPU 上的 experts 数量。增大该值会增加 CUDA 显存占用，但减少 iGPU 的计算量。 |
| `--kt-cpuinfer` | CPUInfer 的 host worker 数量。示例中的 `8` 是一个保守起点，可根据 CPU 核心数测试调整。 |
| `--kt-threadpool-count` | CPUInfer thread pool 数量。单 NUMA 节点通常设置为 `1`。 |
| `--chunked-prefill-size` | prefill 分块大小。值过大会增加显存和系统内存压力。 |
| `--max-running-requests 1` | 适合端侧单用户场景，也是当前 SYCL 后端的主要优化目标。 |
| `--disable-cuda-graph` | 当前异构路径包含 CUDA host callback 和外部 SYCL queue，使用 CUDA Graph 可能无法正确覆盖这段调度，因此当前推荐关闭。 |

SYCL kernel 的 tile、subgroup 和 prefill 阈值已经使用当前验证过的默认值，不需要额外设置调试或调优环境变量。

## 常见问题

### 提示 `SYCL_GPTQ_INT4 backend not available`

当前安装没有包含 SYCL 扩展。重新加载 oneAPI 环境并构建：

```bash
source /opt/intel/oneapi/setvars.sh
CPUINFER_USE_SYCL=1 ./install.sh
```

### 找不到 `icpx` 或 `sycl-ls`

确认已经安装 Intel oneAPI Base Toolkit，并在当前 shell 执行：

```bash
source /opt/intel/oneapi/setvars.sh
```

### 无法访问 `/dev/dri/renderD*`

将当前用户加入 `render` 用户组，注销并重新登录。不要通过长期设置 `chmod 666` 绕过设备权限。

### 提示没有可用的 SYCL GPU

先运行 `sycl-ls` 检查 Level Zero GPU。如果存在多个设备，可以显式选择：

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
```

### CMake 找不到 CUDA Toolkit

当前 SYCL 后端仍需要 `cudart` 完成 host callback 调度。确认 CUDA Toolkit 已安装；如果不在默认路径，可以设置：

```bash
export CUDA_HOME=/usr/local/cuda
```

然后重新运行安装命令。

### 提示多个 GPU backend 同时启用

SYCL 与 kt-kernel 的其他 GPU 编译后端互斥。检查并取消之前显式设置的 `CPUINFER_USE_CUDA=1`、`CPUINFER_USE_ROCM=1` 等变量。

### 模型提示 `qzeros`、`sym=false` 或 `desc_act=true` 不受支持

当前后端只支持对称 GPTQ INT4：`sym=true`、`desc_act=false`，且不使用 `qzeros`。需要更换为兼容的模型权重。

### CUDA 显存不足

- 降低 `--kt-num-gpu-experts`。
- 降低 `--mem-fraction-static`、`--max-total-tokens` 或 `--chunked-prefill-size`。
- 降低 `--kt-num-gpu-experts` 会把更多 expert 计算交给 iGPU，可能降低速度并增加系统内存带宽压力。

### iGPU 性能不稳定

- 确认系统处于高性能电源模式，并避免同时运行占用 iGPU 的桌面或视频任务。
- iGPU 与 CPU 共享内存带宽，双通道或更高带宽的内存通常有明显帮助。
- 首次请求包含权重准备和运行时预热，性能比较应在 warmup 后进行。
