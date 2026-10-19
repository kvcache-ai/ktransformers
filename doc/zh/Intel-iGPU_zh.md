# Intel iGPU XMX 后端：使用方法

本后端在 KTransformers 混合推理中将 routed experts 交给 Intel iGPU，Attention、
Embedding、LM Head 和配置在 NVIDIA 上的专家仍由 CUDA 执行。当前实现不是纯 Intel GPU 模型运行器。

已验证硬件为 Core Ultra 7 356H（Panther Lake、Intel `b0a0`）和 RTX 5060 Laptop。
调优配置针对 Qwen3.5-35B-A3B-GPTQ-Int4：256 experts、top-k8、hidden2048、intermediate512、
group128，单个 CPUInfer/NUMA pool、单请求。其它平台和模型形状尚未获得相同的性能验证。

源码构建、环境版本和驱动检查见 [环境安装](Intel-iGPU-Installation_zh.md)。

## 完整启动命令

完成源码安装后，激活构建时所用的 Python 环境，从仓库根目录执行以下命令。
将模型路径和 oneAPI 初始化脚本路径替换为实际位置；`python` 使用当前环境中的解释器。

```bash
source /opt/intel/oneapi/setvars.sh --force
MODEL=/path/to/Qwen3.5-35B-A3B-GPTQ-Int4

KT_SYCL_XMX_PROFILE=panther-lake \
SGLANG_DISABLE_CUDNN_CHECK=1 \
SGLANG_MAMBA_CONV_DTYPE=float16 \
python -m sglang.launch_server \
  --model "$MODEL" \
  --kt-weight-path "$MODEL" \
  --kt-method SYCL_GPTQ_INT4 \
  --host 127.0.0.1 \
  --port 30000 \
  --kt-cpuinfer 10 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 8 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --attention-backend triton \
  --chunked-prefill-size 512 \
  --max-prefill-tokens 4096 \
  --max-running-requests 1 \
  --max-total-tokens 8192 \
  --context-length 8192 \
  --mem-fraction-static 0.9 \
  --disable-radix-cache \
  --tool-call-parser qwen3_coder \
  --disable-shared-experts-fusion \
  --watchdog-timeout 1200 \
  --random-seed 20261016 \
  --trust-remote-code
```

`KT_SYCL_XMX_PROFILE=panther-lake` 是唯一需要设置的后端配置变量。它自动选择 XMX，
并启用已验证的 INT8 激活量化、decode 整专家融合、prefill M8N32/M16N32 gather/vector
路径和 NUMA 分派。原来的 rows、splits、quant、merge 等细项开关用于调试和对照测试，
日常启动无需逐项配置；graph、CPU input quant 和诊断 profiling 默认关闭。

上面另两个 `SGLANG_` 变量属于本机 SGLang 运行环境设置：`SGLANG_DISABLE_CUDNN_CHECK=1`
用于绕过当前 CUDA/cuDNN 组合的版本检查，`SGLANG_MAMBA_CONV_DTYPE=float16` 为本模型
选择 FP16 Mamba 卷积。这两个设置沿用本机已验证环境，不属于 XMX 的分块调优参数。
`MODEL` 只是减少路径重复的 shell 变量，无需 export。

单 Intel GPU 环境会优先选择 Level Zero 设备，无需指定 `ONEAPI_DEVICE_SELECTOR`；
多 SYCL GPU 时可显式指定设备。

推荐配置在算子构造时确定，要求 XMX implementation revision≥7、hidden2048、intermediate512、
group128；不兼容配置会明确报错。它不会修改进程的环境变量，显式设置的细项参数优先于配置
默认值。首次使用请在未保留旧调优变量的新终端启动，避免旧实验设置覆盖推荐配置。

已验证服务配置为 GPU experts8、CPUInfer10、chunk512、最多一个请求；保持这些设置进行首次验收。
`--disable-radix-cache` 用于让每次性能请求都实际执行完整 prefill，日常使用可按需求移除。

服务就绪后，可在另一个终端检查并发送请求：

```bash
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-35B-A3B-GPTQ-Int4","prompt":"请简要说明混合推理的作用。","max_tokens":64,"temperature":0}'
```

## 支持范围和后端选择

普通 `--kt-method SYCL_GPTQ_INT4` 仍默认使用 legacy 实现。
设置 `KT_SYCL_XMX_PROFILE=panther-lake` 会自动选择新实现；
也可单独设置 `KT_SYCL_INT4_BACKEND=xmx` 使用通用执行器并自行调优。
使用普通 SGLang 命令时，不设置推荐配置并设为 `KT_SYCL_INT4_BACKEND=legacy` 可切回原实现。
推荐配置与显式 legacy 选择冲突时会报错，不会静默忽略配置。

权重必须为对称 GPTQ INT4、`desc_act=false`。通用执行器要求 group size 为正的32倍数，
投影维度满足整除和对齐约束；上面的融合调优配置还要求 hidden2048、intermediate512、group128。
不支持的模式和配置会明确报错。专家放置 mask 在加载时确定，不支持运行中动态迁移专家，
也不支持将同一专家拆到多个 NUMA part。
