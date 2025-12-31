# kt-kernel 代码库架构分析文档

## 概述

kt-kernel 是 KTransformers 项目的核心计算内核库，提供高性能的 CPU MoE (Mixture of Experts) 推理能力。它支持多种量化后端（AMX INT4/INT8、Llamafile GGUF、FP8等），并通过 NUMA 感知的线程池实现高效的并行计算。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API 层                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    KTMoEWrapper (工厂类)                      │ │
│  │    根据 method 参数选择: AMX/Native/Llamafile/General        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    BaseMoEWrapper (基类)                      │ │
│  │  - CPUInfer 单例管理                                          │ │
│  │  - KExpertsCPUBuffer 缓冲区管理                               │ │
│  │  - submit_forward / sync_forward 异步执行                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌────────────┬────────────┬────────────┬────────────┐          │
│  │AMXMoEWrapper│NativeMoE  │LlamafileMoE│GeneralMoE │          │
│  │ (INT4/INT8) │(RAWINT4/  │  (GGUF)    │(MOE_INT4/ │          │
│  │             │  FP8)     │            │ MOE_INT8) │          │
│  └────────────┴────────────┴────────────┴────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                    pybind11 (kt_kernel_ext)
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        C++ 后端层                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      CPUInfer                                 │ │
│  │  - submit(): 提交任务到队列                                   │ │
│  │  - sync(): 等待任务完成                                       │ │
│  │  - submit_with_cuda_stream(): GPU-CPU 同步                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌──────────────────────┬──────────────────────┐                │
│  │     WorkerPool       │     TaskQueue        │                │
│  │  - NUMA 感知线程池     │  - 无锁任务队列       │                │
│  │  - Work Stealing     │  - 单生产者多消费者    │                │
│  │  - InNumaPool 子池    │                      │                │
│  └──────────────────────┴──────────────────────┘                │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    TP_MOE<T> 模板类                           │ │
│  │  T = LLAMA_MOE_TP / AMX_MOE_TP / MOE_KERNEL_TP              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、核心 Python 类详解

### 2.1 KTMoEWrapper (python/experts.py)

**作用**: 工厂类，根据 `method` 参数自动选择合适的后端实现。

**关键参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| layer_idx | int | 层索引 |
| num_experts | int | 专家总数 |
| num_experts_per_tok | int | 每个 token 选择的专家数 (top-k) |
| hidden_size | int | 隐藏层维度 |
| moe_intermediate_size | int | MoE 中间层维度 |
| num_gpu_experts | int | GPU 上运行的专家数量 |
| cpuinfer_threads | int | CPU 推理线程数 |
| threadpool_count | int | NUMA 子池数量 |
| weight_path | str | 权重文件路径 |
| method | str | 后端方法 (AMXINT4/AMXINT8/LLAMAFILE/etc.) |

**后端选择逻辑**:
```python
if method in ["AMXINT4", "AMXINT8"]:
    backend_cls = AMXMoEWrapper
elif method in ["RAWINT4", "FP8"]:
    backend_cls = NativeMoEWrapper
elif method == "LLAMAFILE":
    backend_cls = LlamafileMoEWrapper
elif method in ["MOE_INT4", "MOE_INT8"]:
    backend_cls = GeneralMoEWrapper
```

---

### 2.2 BaseMoEWrapper (python/experts_base.py)

**作用**: 所有 MoE Wrapper 的抽象基类，提供通用功能。

**核心组件**:

1. **CPUInfer 单例管理**:
   ```python
   if BaseMoEWrapper._cpu_infer_instance is None:
       worker_config = kt_kernel_ext.WorkerPoolConfig()
       worker_config.subpool_count = threadpool_count
       BaseMoEWrapper._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)
   ```

2. **KExpertsCPUBuffer**: 管理 CPU 端的 pinned memory 缓冲区
   - `input_tensor_cpu`: 输入张量 (bf16)
   - `immediate_experts_ids_cpu`: 即时执行的专家 ID
   - `deferred_experts_ids_cpu`: 延迟执行的专家 ID
   - `weights_cpu`: 专家权重
   - `output_cpu`: 输出张量
   - `output_gpu`: GPU 端输出

3. **核心方法**:
   - `submit_forward()`: 异步提交前向计算任务
   - `sync_forward()`: 同步等待结果并复制到 GPU
   - `forward()`: submit + sync 的组合
   - `select_deferred_experts()`: 选择延迟执行的专家 (用于流水线优化)

---

### 2.3 AMXMoEWrapper (python/utils/amx.py)

**作用**: Intel AMX (Advanced Matrix Extensions) 加速的 INT4/INT8 量化推理。

**特点**:
- 使用 SafeTensorLoader 加载权重
- 支持 NUMA 分片的权重存储格式
- 通过 AMXInt4_MOE / AMXInt8_MOE C++ 类执行计算

**权重加载流程**:
1. 从 SafeTensor 文件加载量化权重
2. 获取 gate/up/down 投影矩阵的指针
3. 配置 MOEConfig 并创建 C++ MoE 实例
4. 调用 `load_weights_task()` 完成权重初始化

---

### 2.4 NativeMoEWrapper (python/utils/amx.py)

**作用**: 支持 RAWINT4 和 FP8 格式的原生量化推理。

**特点**:
- 使用 CompressedSafeTensorLoader 或 FP8SafeTensorLoader
- 权重已预量化，无需在线量化
- 通过 AMXInt4_KGroup_MOE / AMXFP8_MOE 执行计算

---

### 2.5 LlamafileMoEWrapper (python/utils/llamafile.py)

**作用**: 基于 Llamafile 的 GGUF 量化权重推理。

**特点**:
- 使用 GGUFLoader 加载 GGUF 格式权重
- 支持多种 GGML 量化类型 (Q4_K, Q6_K 等)
- 需要 QK_K (256) 对齐的 TP 分片

**关键配置**:
```python
moe_config.m_block = 32          # 并行块大小
moe_config.group_min_len = 10    # qlen < 10 时使用 forward_one
moe_config.group_max_len = chunked_prefill_size
```

---

### 2.6 GeneralMoEWrapper (python/utils/moe_kernel.py)

**作用**: 通用的 INT4/INT8 量化推理，使用 BLIS/AOCL 矩阵库。

**特点**:
- 支持 ARM (KML) 和 x86 (BLIS) 平台
- 权重可在线量化或从文件加载
- 通过 Int4_KERNEL_MOE / Int8_KERNEL_MOE 执行计算

---

## 三、权重加载器详解 (python/utils/loader.py)

### 3.1 SafeTensorLoader

**用途**: 加载标准 SafeTensor 格式的 NUMA 分片权重。

**权重命名格式**:
```
blk.{layer_idx}.ffn_{up,gate,down}_exps.{expert_id}.numa.{numa_id}.weight
blk.{layer_idx}.ffn_{up,gate,down}_exps.{expert_id}.numa.{numa_id}.scale
```

**返回格式**: `{up, gate, down, up_scale, gate_scale, down_scale}`
每个值是 `[numa_id][expert_id] -> numpy array`

---

### 3.2 FP8SafeTensorLoader

**用途**: 加载 FP8 格式权重 (DeepSeek/Mixtral 风格)。

**自动检测命名格式**:
- DeepSeek: `{base}.mlp.experts.{id}.{gate,up,down}_proj.weight`
- Mixtral: `{base}.block_sparse_moe.experts.{id}.{w1,w3,w2}.weight`

---

### 3.3 CompressedSafeTensorLoader

**用途**: 加载 RAWINT4 压缩权重。

**权重命名格式**:
```
{base}.mlp.experts.{expert_id}.{up,gate,down}_proj.weight_packed
{base}.mlp.experts.{expert_id}.{up,gate,down}_proj.weight_scale
```

---

### 3.4 GGUFLoader

**用途**: 加载 GGUF 格式的量化权重 (llama.cpp 兼容)。

**支持的量化类型**:
- Q4_K, Q5_K, Q6_K, Q8_K
- Q4_0, Q5_0, Q8_0
- IQ2_XXS, IQ3_XXS, IQ4_NL 等

---

## 四、C++ 后端核心类

### 4.1 CPUInfer (cpu_backend/cpuinfer.h)

**作用**: CPU 推理协调器，管理任务提交和同步。

**核心成员**:
```cpp
WorkerPool* backend_;    // 线程池
TaskQueue* task_queue_;  // 任务队列
```

**关键方法**:

| 方法 | 说明 |
|------|------|
| `submit(params)` | 提交任务到队列 |
| `sync(allow_n_pending)` | 等待任务完成 |
| `submit_with_cuda_stream(stream, params)` | 从 CUDA stream 提交任务 |
| `sync_with_cuda_stream(stream)` | GPU-CPU 同步 |

---

### 4.2 WorkerPool (cpu_backend/worker_pool.h)

**作用**: NUMA 感知的多级线程池。

**架构**:
```
WorkerPool
├── NumaJobDistributor    // NUMA 节点间任务分发
└── InNumaPool[]          // 每个 NUMA 节点的线程池
    ├── worker_thread[]   // 工作线程
    └── work_stealing     // 任务窃取机制
```

**配置结构**:
```cpp
struct WorkerPoolConfig {
    int subpool_count;                    // 子池数量
    std::vector<int> subpool_numa_map;    // NUMA 映射
    std::vector<int> subpool_thread_count; // 每个子池线程数
};
```

---

### 4.3 TaskQueue (cpu_backend/task_queue.h)

**作用**: 无锁单生产者任务队列。

**核心机制**:
- 使用原子操作实现无锁入队
- 单独的 worker 线程执行任务
- 支持 pending 计数的同步

---

### 4.4 TP_MOE<T> (operators/moe-tp.hpp)

**作用**: 张量并行 MoE 的模板基类。

**泛型参数 T 可以是**:
- `LLAMA_MOE_TP` - Llamafile 后端
- `AMX_MOE_TP<Kernel>` - AMX 后端
- `MOE_KERNEL_TP<Kernel>` - 通用 kernel 后端

**核心流程**:
```
forward()
├── 1. 分发输入到各 NUMA 节点的 TP 实例
│     pool->dispense_backend()->do_numa_job(...)
├── 2. 每个 TP 实例执行 forward
│     tps[numa_id]->forward(qlen, k, expert_ids, ...)
└── 3. 合并结果
      merge_results(qlen, output)
```

---

## 五、MoE 后端实现详解

### 5.1 LLAMA_MOE_TP (operators/llamafile/moe.hpp)

**计算流程**:
```
forward_one() / forward_many()
├── 1. 输入类型转换 (BF16 -> vec_dot_type)
├── 2. Gate GEMM: input × gate_proj → gate_output
├── 3. Up GEMM: input × up_proj → up_output
├── 4. 激活: silu(gate_output) * up_output → intermediate
├── 5. Down GEMM: intermediate × down_proj → output
└── 6. 加权求和: Σ(weight_i * output_i)
```

**使用 llamafile_sgemm 进行矩阵乘法**。

---

### 5.2 AMX_MOE_TP (operators/amx/moe.hpp)

**特点**:
- 使用 Intel AMX 指令集加速矩阵运算
- 支持 INT4/INT8 量化
- 权重布局针对 AMX tile 优化

**关键 Kernel 类型**:
- `GemmKernel224Int4` - INT4 量化
- `GemmKernel224Int8` - INT8 量化
- `GemmKernel224BF` - BF16 精度
- `GemmKernel224FP8` - FP8 量化

---

### 5.3 MOE_KERNEL_TP (operators/moe_kernel/moe.hpp)

**特点**:
- 使用 BLIS/AOCL 矩阵库 (支持 ARM/AMD)
- 权重在线量化或预加载
- 支持 decode/prefill 两种模式

**计算流程**:
```
forward_unified(mode, qlen, k, expert_ids, weights, input, output)
├── 1. 准备: 统计每个专家的 token 数量
├── 2. 复制输入到专家本地缓冲区
├── 3. 量化输入 (BF16 → INT8)
├── 4. Up/Gate GEMM + 反量化
├── 5. 激活: silu(gate) * up
├── 6. 量化中间结果
├── 7. Down GEMM + 反量化
└── 8. 加权合并结果
```

---

## 六、执行流程总结

### 6.1 初始化流程

```
1. Python: KTMoEWrapper(params)
   → 选择后端 → 创建具体 Wrapper

2. 初始化 CPUInfer 单例
   → 创建 WorkerPoolConfig
   → 创建 WorkerPool (NUMA 感知)

3. 加载权重
   → Loader 读取文件
   → 配置 MOEConfig
   → C++ 端初始化 MoE 实例
   → load_weights_task() 执行
```

### 6.2 推理流程

```
1. Python: submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
   ├── 获取/分配 CPU 缓冲区
   ├── 复制输入到 pinned memory (non_blocking)
   ├── 可选: 选择延迟专家 (流水线优化)
   └── cpu_infer.submit_with_cuda_stream(stream, moe.forward_task(...))

2. C++: forward_task 在 TaskQueue 执行
   ├── 分发到各 NUMA 节点
   ├── 每个节点的 TP 实例执行 forward
   └── 合并结果

3. Python: sync_forward(hidden_states, cuda_stream)
   ├── cpu_infer.sync_with_cuda_stream(stream)
   └── output_gpu.copy_(output_cpu, non_blocking=True)
```

---

## 七、关键文件列表

| 文件路径 | 说明 |
|----------|------|
| `python/experts.py` | KTMoEWrapper 工厂类 |
| `python/experts_base.py` | BaseMoEWrapper 基类 |
| `python/utils/amx.py` | AMX/Native Wrapper |
| `python/utils/llamafile.py` | Llamafile Wrapper |
| `python/utils/moe_kernel.py` | General Wrapper |
| `python/utils/loader.py` | 权重加载器 |
| `ext_bindings.cpp` | pybind11 绑定 |
| `cpu_backend/cpuinfer.h` | CPUInfer 类 |
| `cpu_backend/worker_pool.h` | WorkerPool 类 |
| `cpu_backend/task_queue.h` | TaskQueue 类 |
| `operators/moe-tp.hpp` | TP_MOE 模板基类 |
| `operators/llamafile/moe.hpp` | Llamafile MoE 实现 |
| `operators/amx/moe.hpp` | AMX MoE 实现 |
| `operators/moe_kernel/moe.hpp` | 通用 kernel MoE 实现 |

---

## 八、扩展与定制

### 8.1 添加新的量化后端

1. 在 `ext_bindings.cpp` 中注册新的 MoE 类型
2. 创建新的 `*_MOE_TP` 模板实例
3. 在 `python/utils/` 下创建对应的 Wrapper
4. 在 `KTMoEWrapper.__new__()` 中添加选择逻辑

### 8.2 调整并行策略

修改 `WorkerPoolConfig`:
- `subpool_count`: NUMA 子池数量
- `subpool_numa_map`: NUMA 节点映射
- `subpool_thread_count`: 每个子池的线程数

---

## 九、代码示例

### 9.1 基本使用示例

```python
import torch
from kt_kernel import KTMoEWrapper

# 初始化 MoE 层
moe = KTMoEWrapper(
    layer_idx=0,
    num_experts=160,
    num_experts_per_tok=6,
    hidden_size=7168,
    moe_intermediate_size=2048,
    num_gpu_experts=8,  # 前8个专家在 GPU 上
    cpuinfer_threads=80,
    threadpool_count=2,  # 2 个 NUMA 节点
    weight_path="/path/to/weights",
    method="LLAMAFILE"  # 使用 Llamafile 后端
)

# 前向推理
hidden_states = torch.randn(1, 4096, 7168, dtype=torch.bfloat16, device="cuda")
topk_ids = torch.tensor([[0, 1, 2, 3, 8, 9]], dtype=torch.int64, device="cuda")
topk_weights = torch.ones(1, 6, dtype=torch.float32, device="cuda") / 6

# 异步提交
cuda_stream = torch.cuda.current_stream().cuda_stream
moe.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

# 同步获取结果 (原地更新 hidden_states)
moe.sync_forward(hidden_states, cuda_stream)
```

### 9.2 流水线优化示例 (Deferred Experts)

```python
# 第一层: 选择延迟执行的专家
deferred_ids = moe_layer0.select_deferred_experts(hidden_states, topk_ids, topk_weights)
# deferred_ids 可能包含通信/计算开销大的专家

# 提交第一层 (只执行即时专家)
moe_layer0.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)

# 同时准备第二层输入...
# hidden_states_2 = attention_layer(hidden_states)

# 同步第一层 (包含延迟专家的计算)
moe_layer0.sync_forward(hidden_states, cuda_stream)
```

### 9.3 权重加载示例

```python
from kt_kernel.utils.loader import GGUFLoader, SafeTensorLoader

# GGUF 格式加载
loader = GGUFLoader(
    expert_count=160,
    hidden_size=7168,
    intermediate_size=2048
)
weights = loader.load("/path/to/model-experts.gguf", layer_idx=0)

# SafeTensor NUMA 分片加载
loader = SafeTensorLoader(
    num_experts=160,
    numa_ids=[0, 1]
)
weights = loader.load("/path/to/weights/", layer_idx=0)
# weights = {up, gate, down, up_scale, gate_scale, down_scale}
```

---

## 十、类图与时序图

### 10.1 Python 层类图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KTMoEWrapper                                 │
│─────────────────────────────────────────────────────────────────────│
│ + __new__(cls, method, ...) → Wrapper                               │
│   «factory method»                                                  │
└────────────────────────────────────┬────────────────────────────────┘
                                     │ creates
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BaseMoEWrapper «abstract»                        │
│─────────────────────────────────────────────────────────────────────│
│ - _cpu_infer_instance: CPUInfer «class var»                        │
│ - layer_idx: int                                                    │
│ - hidden_size: int                                                  │
│ - moe_intermediate_size: int                                        │
│ - num_experts_per_tok: int                                          │
│ - num_gpu_experts: int                                              │
│─────────────────────────────────────────────────────────────────────│
│ + __init__(layer_idx, hidden_size, ...)                            │
│ + forward(hidden_states, topk_ids, topk_weights, ...)              │
│ + submit_forward(hidden_states, topk_ids, topk_weights, stream)    │
│ + sync_forward(hidden_states, stream)                              │
│ + select_deferred_experts(hidden_states, topk_ids, topk_weights)   │
│ # _create_cpu_infer(threadpool_count, cpuinfer_threads)            │
│ # _create_moe_instance() «abstract»                                │
│ # _load_weights_task() «abstract»                                  │
└────────────────────────────────────┬────────────────────────────────┘
                                     │ extends
        ┌────────────────┬───────────┴───────────┬───────────────────┐
        ▼                ▼                       ▼                   ▼
┌───────────────┐ ┌───────────────┐ ┌─────────────────────┐ ┌────────────────┐
│ AMXMoEWrapper │ │NativeMoEWrapper│ │LlamafileMoEWrapper │ │GeneralMoEWrapper│
│───────────────│ │───────────────│ │─────────────────────│ │────────────────│
│ - moe_config  │ │ - moe_config  │ │ - moe_config        │ │ - moe_config   │
│ - loader      │ │ - loader      │ │ - loader            │ │ - loader       │
│───────────────│ │───────────────│ │─────────────────────│ │────────────────│
│ # _create_moe │ │ # _create_moe │ │ # _create_moe       │ │ # _create_moe  │
│ # _load_wgt   │ │ # _load_wgt   │ │ # _load_wgt         │ │ # _load_wgt    │
└───────────────┘ └───────────────┘ └─────────────────────┘ └────────────────┘
        │                │                    │                      │
        │ uses           │ uses               │ uses                 │ uses
        ▼                ▼                    ▼                      ▼
┌───────────────┐ ┌───────────────┐ ┌─────────────────────┐ ┌────────────────┐
│SafeTensorLoader│ │Compressed/FP8│ │    GGUFLoader       │ │SafeTensorLoader│
└───────────────┘ │   Loader      │ └─────────────────────┘ └────────────────┘
                  └───────────────┘
```

### 10.2 C++ 层类图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CPUInfer                                   │
│─────────────────────────────────────────────────────────────────────│
│ - backend_: WorkerPool*                                             │
│ - task_queue_: TaskQueue*                                           │
│─────────────────────────────────────────────────────────────────────│
│ + CPUInfer(thread_num)                                              │
│ + CPUInfer(thread_num, numa_id)                                     │
│ + CPUInfer(WorkerPoolConfig)                                        │
│ + submit(params: pair<intptr_t, intptr_t>)                         │
│ + submit_with_cuda_stream(stream, params)                           │
│ + sync(allow_n_pending)                                             │
│ + sync_with_cuda_stream(stream, allow_n_pending)                    │
└────────────────────────────────────┬────────────────────────────────┘
                                     │ owns
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
┌───────────────────────────────────┐ ┌───────────────────────────────┐
│          WorkerPool               │ │          TaskQueue             │
│───────────────────────────────────│ │───────────────────────────────│
│ - numa_worker_pools: InNumaPool[] │ │ - head: atomic<Node*>         │
│ - distributor: NumaJobDistributor │ │ - tail: atomic<Node*>         │
│ - config: WorkerPoolConfig        │ │ - pending: atomic<size_t>     │
│───────────────────────────────────│ │ - workerThread: thread        │
│ + get_thread_num()                │ │───────────────────────────────│
│ + dispense_backend()              │ │ + enqueue(task: function)     │
│ + get_subpool(numa_id)            │ │ + sync(allow_n_pending)       │
│ + do_work_stealing_job(...)       │ └───────────────────────────────┘
└────────────────────────────────────┘
         │ contains
         ▼
┌───────────────────────────────────┐
│         InNumaPool                │
│───────────────────────────────────│
│ - worker_count: int               │
│ - workers_: vector<thread>        │
│ - thread_state_: ThreadState[]    │
│───────────────────────────────────│
│ + do_work_stealing_job(n, init,   │
│     compute, finalize)            │
│ + wait()                          │
└───────────────────────────────────┘
```

### 10.3 MoE 模板类层次

```
┌─────────────────────────────────────────────────────────────────────┐
│               MoE_Interface «interface»                             │
│─────────────────────────────────────────────────────────────────────│
│ + forward(qlen, k, expert_ids, weights, input, output)             │
│ + load_weights()                                                    │
│ + warm_up()                                                         │
└────────────────────────────────────┬────────────────────────────────┘
                                     │ implements
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│              TP_MOE_Common<T> «template»                            │
│─────────────────────────────────────────────────────────────────────│
│ # tp_configs: vector<GeneralMOEConfig>                              │
│ # tp_count: int                                                     │
│ # tps: vector<unique_ptr<T>>                                        │
│ # local_output_numa: vector<output_t*>                              │
│ + config: GeneralMOEConfig                                          │
│─────────────────────────────────────────────────────────────────────│
│ + forward(qlen, k, expert_ids, weights, input, output)             │
│ + warm_up()                                                         │
│ # merge_results(qlen, output) «abstract»                           │
│ # load_weights() «abstract»                                        │
└────────────────────────────────────┬────────────────────────────────┘
                                     │ extends
        ┌────────────────────────────┴────────────────────────────────┐
        ▼                                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│ TP_MOE<LLAMA_MOE_TP>          │             │ TP_MOE<MOE_KERNEL_TP<K,T>>    │
│───────────────────────────────│             │───────────────────────────────│
│ + load_weights()              │             │ + load_weights()              │
│ + merge_results(qlen, output) │             │ + merge_results(qlen, output) │
└───────────────────────────────┘             └───────────────────────────────┘
         │ uses                                        │ uses
         ▼                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│      LLAMA_MOE_TP             │             │   MOE_KERNEL_TP<Kernel,Plain> │
│───────────────────────────────│             │───────────────────────────────│
│ - m_local_gate_proj_          │             │ - gate_bb_: BufferB[]         │
│ - m_local_up_proj_            │             │ - up_bb_: BufferB[]           │
│ - m_local_down_proj_          │             │ - down_bb_: BufferB[]         │
│───────────────────────────────│             │───────────────────────────────│
│ + forward(qlen, k, ...)       │             │ + forward(qlen, k, ...)       │
│ + forward_one(k, ...)         │             │ + forward_unified(mode, ...)  │
│ + forward_many(qlen, k, ...)  │             │ + load_weights()              │
│ + load_weights(offset)        │             └───────────────────────────────┘
└───────────────────────────────┘
```

### 10.4 前向推理时序图

```
┌────────┐     ┌─────────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐
│ Python │     │BaseMoEWrapper│    │ CPUInfer │     │ TaskQueue │     │WorkerPool│
└───┬────┘     └──────┬──────┘     └────┬─────┘     └─────┬─────┘     └────┬─────┘
    │                 │                 │                 │                │
    │ submit_forward  │                 │                 │                │
    │────────────────>│                 │                 │                │
    │                 │                 │                 │                │
    │                 │ 分配/获取缓冲区   │                 │                │
    │                 │◄───────────────>│                 │                │
    │                 │                 │                 │                │
    │                 │ 复制输入到CPU    │                 │                │
    │                 │ (cudaMemcpyAsync)│                 │                │
    │                 │─────────────────│                 │                │
    │                 │                 │                 │                │
    │                 │ submit_with_cuda_stream           │                │
    │                 │─────────────────>│                 │                │
    │                 │                 │                 │                │
    │                 │                 │ cudaLaunchHostFunc               │
    │                 │                 │────────────────>│                │
    │                 │                 │                 │                │
    │ return          │                 │                 │ enqueue(task)  │
    │<────────────────│                 │                 │───────────────>│
    │                 │                 │                 │                │
    │ (GPU继续执行)    │                 │                 │ worker执行task │
    │                 │                 │                 │<───────────────│
    │                 │                 │                 │                │
    │                 │                 │                 │ dispense_backend
    │                 │                 │                 │───────────────>│
    │                 │                 │                 │                │
    │                 │                 │                 │ do_numa_job    │
    │                 │                 │                 │       ┌────────┤
    │                 │                 │                 │       │ NUMA 0 │
    │                 │                 │                 │       │forward │
    │                 │                 │                 │       ├────────┤
    │                 │                 │                 │       │ NUMA 1 │
    │                 │                 │                 │       │forward │
    │                 │                 │                 │       └────────┤
    │                 │                 │                 │                │
    │                 │                 │                 │ merge_results  │
    │                 │                 │                 │<───────────────│
    │                 │                 │                 │                │
    │ sync_forward    │                 │                 │ pending--      │
    │────────────────>│                 │                 │                │
    │                 │                 │                 │                │
    │                 │ sync_with_cuda_stream             │                │
    │                 │─────────────────>│                 │                │
    │                 │                 │                 │                │
    │                 │                 │ sync(0)         │                │
    │                 │                 │────────────────>│                │
    │                 │                 │                 │                │
    │                 │                 │ wait for pending==0              │
    │                 │                 │<────────────────│                │
    │                 │                 │                 │                │
    │                 │ 复制输出到GPU    │                 │                │
    │                 │ (cudaMemcpyAsync)│                 │                │
    │                 │─────────────────│                 │                │
    │                 │                 │                 │                │
    │ return          │                 │                 │                │
    │<────────────────│                 │                 │                │
    │                 │                 │                 │                │
```

### 10.5 MoE 计算时序图 (单个 NUMA 节点)

```
┌──────────┐     ┌───────────┐     ┌─────────────┐     ┌─────────────┐
│InNumaPool│     │TP_MOE<T>  │     │ T (具体实现) │     │ GEMM Kernel │
└────┬─────┘     └─────┬─────┘     └──────┬──────┘     └──────┬──────┘
     │                 │                  │                   │
     │ do_work_steal   │                  │                   │
     │◄────────────────│                  │                   │
     │                 │                  │                   │
     │                 │ forward(qlen,k,ids,wts,in,out)       │
     │                 │─────────────────>│                   │
     │                 │                  │                   │
     │                 │                  │ 1. 准备专家映射    │
     │                 │                  │    m_local_num_   │
     │                 │                  │    m_local_pos_   │
     │                 │                  │                   │
     │ work_steal_job  │                  │ 2. 复制输入       │
     │<───────────────────────────────────│                   │
     │ (并行复制各token)│                  │                   │
     │                 │                  │                   │
     │ work_steal_job  │                  │ 3. 量化输入       │
     │<───────────────────────────────────│                   │
     │ (并行量化各expert)                  │                   │
     │                 │                  │                   │
     │ work_steal_job  │                  │ 4. Up/Gate GEMM   │
     │<───────────────────────────────────│──────────────────>│
     │ (nth×mth×expert×2)                 │  cblas_gemm_s8s8  │
     │                 │                  │<──────────────────│
     │                 │                  │                   │
     │ work_steal_job  │                  │ 5. 激活函数       │
     │<───────────────────────────────────│  silu(gate)×up    │
     │                 │                  │                   │
     │ work_steal_job  │                  │ 6. 量化中间结果   │
     │<───────────────────────────────────│                   │
     │                 │                  │                   │
     │ work_steal_job  │                  │ 7. Down GEMM      │
     │<───────────────────────────────────│──────────────────>│
     │ (nth×mth×expert)│                  │  cblas_gemm_s8s8  │
     │                 │                  │<──────────────────│
     │                 │                  │                   │
     │ work_steal_job  │                  │ 8. 加权合并       │
     │<───────────────────────────────────│  Σ(wt_i×out_i)    │
     │ (qlen×block_num)│                  │                   │
     │                 │                  │                   │
     │                 │                  │ return            │
     │                 │<─────────────────│                   │
     │                 │                  │                   │
```

---

## 十一、数据流详解

### 11.1 输入数据流

```
GPU Tensor (hidden_states)
    │
    ▼ cudaMemcpyAsync (GPU→CPU)
Pinned Memory (input_tensor_cpu)
    │
    ▼ memcpy to expert buffers
Expert Local Buffers (m_local_input_)
    │
    ▼ quantize (BF16→INT8)
Quantized Input (BufferA)
```

### 11.2 权重数据流

```
SafeTensor/GGUF File
    │
    ▼ Loader.load()
Numpy Arrays (per expert, per NUMA)
    │
    ▼ MOEConfig assignment
C++ Memory (gate_proj_, up_proj_, down_proj_)
    │
    ▼ quantize/repack (if needed)
Optimized Weight Buffers (BufferB)
```

### 11.3 输出数据流

```
Expert Outputs (m_local_down_output_)
    │
    ▼ weighted sum (Σ weight_i × output_i)
Merged Output (local_output_numa[])
    │
    ▼ TP merge across NUMA nodes
Final Output (output_cpu)
    │
    ▼ cudaMemcpyAsync (CPU→GPU)
GPU Tensor (hidden_states, in-place update)
```

---

## 十二、性能优化要点

### 12.1 NUMA 优化

- 权重按 NUMA 节点分片存储
- 每个 NUMA 节点有独立的线程池 (InNumaPool)
- 内存分配使用 `numa_alloc_onnode()` 确保本地访问

### 12.2 并行策略

- **Expert 级并行**: 多个专家同时计算
- **矩阵分块并行**: Up/Gate/Down GEMM 按 M/N 维度分块
- **Work Stealing**: 动态负载均衡

### 12.3 内存优化

- Pinned Memory 减少 GPU-CPU 拷贝开销
- 缓冲区复用 (KExpertsCPUBuffer 池化)
- 共享内存缓冲区 (shared_mem_buffer)

### 12.4 异步执行

- `submit_forward()` 非阻塞返回
- GPU 可在 CPU 计算期间执行其他操作
- `cudaLaunchHostFunc` 实现 GPU→CPU 任务触发

---

## 十三、故障排查

### 13.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| NUMA 绑定失败 | 权限不足 | 使用 `numactl` 或提升权限 |
| 权重加载失败 | 文件路径/格式错误 | 检查权重文件和命名格式 |
| 性能低于预期 | 线程数配置不当 | 调整 `cpuinfer_threads` |
| 内存不足 | 缓冲区分配过大 | 减少 `max_len` 或专家数 |

### 13.2 调试选项

编译时定义以下宏启用调试输出:
- `FORWARD_TIME_PROFILE`: 输出各阶段耗时
- `FORWARD_TIME_REPORT`: 输出带宽/GFLOPS 报告
- `CHECK`: 启用权重加载校验
