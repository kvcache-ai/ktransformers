# SFT-MOE no-TP 模式下 GPU-CPU 异构实现分析

本文档针对 `最终流程&存储情况.md` 中的**同步点**和**内存管理**两个问题，在 **no-TP（单 NUMA 节点）** 模式下进行深入分析。

---

## 一、架构概述

### 1.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Python 层                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         BaseMoEWrapper                                   │ │
│  │  • submit_forward(): GPU→CPU 异步复制 + 提交任务                         │ │
│  │  • sync_forward(): 等待任务完成 + CPU→GPU 异步复制                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ pybind11
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              C++ 层                                           │
│  ┌──────────────┐    ┌───────────────┐    ┌───────────────────────────────┐ │
│  │   CPUInfer   │───>│   TaskQueue   │───>│        WorkerPool             │ │
│  │              │    │ (无锁队列)     │    │  (NUMA感知线程池)              │ │
│  │ • submit()   │    │ • enqueue()   │    │  • InNumaPool                 │ │
│  │ • sync()     │    │ • sync()      │    │  • do_work_stealing_job()     │ │
│  └──────────────┘    └───────────────┘    └───────────────────────────────┘ │
│                                                        │                     │
│                                                        ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     AMX_SFT_MOE_TP (no-TP: tp_count=1)                   │ │
│  │  • forward_sft(): 前向计算 + 可选缓存                                    │ │
│  │  • backward(): 反向传播计算梯度                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 no-TP 模式特点

| 特性 | no-TP 实现 |
|------|-----------|
| NUMA 节点数 | 1 |
| `tp_part_idx` | 固定为 0 |
| 权重存储 | 零拷贝（直接使用 Python tensor 的 data_ptr） |
| LoRA 权重 | 全部零拷贝 |
| 结果合并 | 无需跨 NUMA 合并 |

---

## 二、同步点详解

### 2.1 完整同步时序图

```
┌─────────┐    ┌───────────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────────┐
│  GPU    │    │ Python        │    │ CPUInfer │    │ TaskQueue │    │ AMX_SFT_MOE_TP   │
└────┬────┘    └───────┬───────┘    └─────┬────┘    └─────┬─────┘    └────────┬─────────┘
     │                 │                  │               │                   │
     │  submit_forward │                  │               │                   │
     │<────────────────│                  │               │                   │
     │                 │                  │               │                   │
     │ ══════════════════════════════════════════════════════════════════════│
     │ ║ 同步点 #1: GPU→CPU 异步复制 (cudaMemcpyAsync implicit)              ║│
     │ ══════════════════════════════════════════════════════════════════════│
     │                 │                  │               │                   │
     │  copy_(non_blocking=True)          │               │                   │
     │  input → pinned_memory             │               │                   │
     │─────────────────│                  │               │                   │
     │                 │                  │               │                   │
     │ ══════════════════════════════════════════════════════════════════════│
     │ ║ 同步点 #2: cudaLaunchHostFunc (GPU stream 触发 CPU 任务)            ║│
     │ ══════════════════════════════════════════════════════════════════════│
     │                 │                  │               │                   │
     │ cudaLaunchHostFunc                 │               │                   │
     │─────────────────│─────────────────>│               │                   │
     │                 │                  │               │                   │
     │                 │                  │ forward_task  │                   │
     │                 │                  │──────────────>│                   │
     │                 │                  │               │ pending++         │
     │                 │                  │               │──────────────────>│
     │                 │                  │               │                   │
     │  (GPU 继续执行其他操作)             │               │ worker 执行        │
     │                 │                  │               │<──────────────────│
     │                 │                  │               │                   │
     │                 │                  │               │  forward_sft()    │
     │                 │                  │               │──────────────────>│
     │                 │                  │               │                   │
     │                 │                  │               │   GEMM 计算        │
     │                 │                  │               │<──────────────────│
     │                 │                  │               │                   │
     │                 │                  │               │ pending--         │
     │                 │                  │               │<──────────────────│
     │                 │                  │               │                   │
     │  sync_forward   │                  │               │                   │
     │<────────────────│                  │               │                   │
     │                 │                  │               │                   │
     │ ══════════════════════════════════════════════════════════════════════│
     │ ║ 同步点 #3: sync_with_cuda_stream (等待 CPU 任务完成)                 ║│
     │ ══════════════════════════════════════════════════════════════════════│
     │                 │                  │               │                   │
     │ cudaLaunchHostFunc(sync_)          │               │                   │
     │─────────────────│─────────────────>│               │                   │
     │                 │                  │  sync()       │                   │
     │                 │                  │──────────────>│                   │
     │                 │                  │               │ spin-wait         │
     │                 │                  │               │ (pending <= 0)    │
     │                 │                  │<──────────────│                   │
     │                 │                  │               │                   │
     │ ══════════════════════════════════════════════════════════════════════│
     │ ║ 同步点 #4: CPU→GPU 异步复制 (cudaMemcpyAsync implicit)              ║│
     │ ══════════════════════════════════════════════════════════════════════│
     │                 │                  │               │                   │
     │  copy_(non_blocking=True)          │               │                   │
     │  output_cpu → output_gpu           │               │                   │
     │<────────────────│                  │               │                   │
     │                 │                  │               │                   │
```

### 2.2 同步点代码位置

| 同步点 | 位置 | 代码 | 作用 |
|--------|------|------|------|
| **#1** | `experts_base.py:272` | `input_tensor_cpu[].copy_(flat_hidden_states, non_blocking=True)` | GPU→CPU 异步复制输入到 pinned memory |
| **#2** | `cpuinfer.h:85-91` | `cudaLaunchHostFunc(stream, func, args)` | 从 CUDA stream 触发 CPU 任务 |
| **#3** | `cpuinfer.h:110-114` | `cudaLaunchHostFunc(stream, sync_, args)` | 等待 CPU 任务完成 |
| **#4** | `experts_base.py:332` | `output_gpu[].copy_(output_cpu[], non_blocking=True)` | CPU→GPU 异步复制结果 |

### 2.3 同步机制实现细节

#### TaskQueue 同步 (`task_queue.cpp:45-48`)
```cpp
void TaskQueue::sync(size_t allow_n_pending) {
  // Spin until the pending task count drops to the allowed threshold.
  while (pending.load(std::memory_order_acquire) > allow_n_pending);
}
```

**关键点**:
- 使用 `atomic<size_t> pending` 计数器
- `enqueue()` 时 `pending++`，任务完成后 `pending--`
- `sync()` spin-wait 直到 `pending <= allow_n_pending`
- no-TP 模式下 `allow_n_pending=0` 意味着等待所有任务完成

#### cudaLaunchHostFunc 机制
```cpp
// cpuinfer.h:85-91
void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
  void (*func)(void*) = (void (*)(void*))params.first;
  void* args = (void*)params.second;
  *((CPUInfer**)args) = this;
  cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
}
```

**关键点**:
- `cudaLaunchHostFunc` 在 GPU stream 上调度一个 host 函数
- 当 stream 执行到此函数时，会在 CPU 上调用 `func(args)`
- 保证 GPU 数据传输完成后才触发 CPU 计算

---

## 三、内存管理详解

### 3.1 内存层次结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GPU 内存                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  hidden_states (bf16)    output_gpu (bf16)                              │ │
│  │  [batch_size, hidden_size]                                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                              │ cudaMemcpyAsync
                              │ (implicit via copy_)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Pinned Memory (CPU)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  KExpertsCPUBuffer (Python 端管理)                                       │ │
│  │  • input_tensor_cpu[2]     (双缓冲)                                      │ │
│  │  • immediate_experts_ids_cpu[2]                                          │ │
│  │  • deferred_experts_ids_cpu[2]                                           │ │
│  │  • weights_cpu[2]                                                        │ │
│  │  • output_cpu[2]                                                         │ │
│  │  • bsz_tensor_cpu[2]                                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                              │ memcpy / 指针传递
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Regular Memory (CPU)                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  SharedMemBufferNuma (C++ 端管理)                                        │ │
│  │  • lora_intermediate_pool_                                               │ │
│  │  • cache_input_pool_ / cache_gate_output_pool_ / ...                     │ │
│  │  • grad_intermediate_pool_ / grad_gate_output_pool_ / ...                │ │
│  │                                                                          │ │
│  │  Python Tensor 数据 (零拷贝)                                              │ │
│  │  • gate_lora_a_, gate_lora_b_, ...  (LoRA 权重)                          │ │
│  │  • gate_proj, up_proj, down_proj  (基础权重)                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Pinned Memory 管理 (Python 端)

**KExpertsCPUBuffer (`experts_base.py:21-86`)**:

```python
class KExpertsCPUBuffer:
    buffer_depth: int = 2  # 双缓冲

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok):
        # 创建 pinned memory 缓冲区
        input_tensor_cpu = [
            torch.zeros(..., pin_memory=True, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        # ... 其他缓冲区类似
```

**关键设计**:
1. **双缓冲机制** (`buffer_depth=2`): 允许当前层计算时，准备下一层的数据
2. **Pinned Memory**: 使用 `pin_memory=True` 创建页锁定内存，加速 GPU-CPU 传输
3. **缓存复用**: 按 `batch_size` 缓存缓冲区，避免重复分配

### 3.3 SharedMemBuffer 管理 (C++ 端)

**单次分配所有缓冲区 (`sft_moe.hpp:489-544`)**:

```cpp
void init_all_buffers() {
  // ★ 单个 alloc() 调用 - 所有缓冲区获得连续、非重叠的地址 ★
  MemoryRequest mem_requests;

  // LoRA 缓冲区
  mem_requests.append_pointer(&lora_intermediate_pool_, lora_intermediate_pool_bytes_);

  // Cache 缓冲区 (4 pools × max_cache_depth)
  mem_requests.append_pointer(&cache_input_pool_, cache_slot_bytes_input_ * max_cache_depth_);
  mem_requests.append_pointer(&cache_gate_output_pool_, ...);
  mem_requests.append_pointer(&cache_up_output_pool_, ...);
  mem_requests.append_pointer(&cache_intermediate_pool_, ...);

  // Gradient 缓冲区 (3 pools)
  mem_requests.append_pointer(&grad_intermediate_pool_, grad_buffer_bytes);
  mem_requests.append_pointer(&grad_gate_output_pool_, grad_buffer_bytes);
  mem_requests.append_pointer(&grad_up_output_pool_, grad_buffer_bytes);

  // 单次分配
  shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
}
```

**SharedMemBuffer 分配 (`shared_mem_buffer.cpp:49-73`)**:

```cpp
void SharedMemBuffer::alloc(void* object, MemoryRequest requests) {
  size_t total_size = requests.total_size();
  object_requests.push_back(requests);

  if (total_size > size) {
    // 重新分配更大的缓冲区
    void* newbuf = nullptr;
    int rc = posix_memalign(&newbuf, 64, total_size);  // 64字节对齐
    buffer = newbuf;
    size = total_size;
    // 更新所有已注册的指针
    for (auto& req : object_requests) {
      req.update_base_ptr(buffer);
    }
  } else {
    requests.update_base_ptr(buffer);
  }
}
```

**关键设计**:
1. **64 字节对齐**: 使用 `posix_memalign(&buf, 64, size)` 优化 SIMD 访问
2. **单次分配**: 避免多次 `alloc()` 导致的内存重叠问题 (Bug #15)
3. **自动增长**: 当请求超过当前大小时自动重新分配

### 3.4 no-TP 模式的零拷贝

**LoRA 权重零拷贝 (`sft_moe.hpp:134-139`)**:

```cpp
// 直接使用 Python tensor 的 data_ptr
gate_lora_a_ = (ggml_bf16_t*)config.gate_lora_a;
gate_lora_b_ = (ggml_bf16_t*)config.gate_lora_b;
up_lora_a_ = (ggml_bf16_t*)config.up_lora_a;
up_lora_b_ = (ggml_bf16_t*)config.up_lora_b;
down_lora_a_ = (ggml_bf16_t*)config.down_lora_a;
down_lora_b_ = (ggml_bf16_t*)config.down_lora_b;
```

**优势**:
- 无需复制权重数据
- Python 修改 tensor 后，C++ 自动可见
- `optimizer.step()` 后无需调用 `update_lora_weights_task()`

---

## 四、no-TP 模式完整数据流

### 4.1 Forward 数据流

```
1. Python: submit_forward()
   │
   ├─► 分配/获取 pinned memory buffer (KExpertsCPUBuffer)
   │
   ├─► GPU→CPU 异步复制: input → input_tensor_cpu (copy_, non_blocking)
   │
   ├─► cudaLaunchHostFunc: 调度 forward_task
   │
   └─► 立即返回 (非阻塞)

2. CPU Worker Thread (TaskQueue)
   │
   ├─► forward_task() 被执行
   │   │
   │   ├─► 专家路由: 统计每个专家的 token 数量
   │   │
   │   ├─► 输入复制: memcpy 到专家本地缓冲区 (m_local_input_ptr_)
   │   │
   │   ├─► 输入量化: BF16 → INT8 (gate_up_ba_[].from_mat())
   │   │
   │   ├─► Gate/Up GEMM: 使用 AMX 指令集计算
   │   │
   │   ├─► Gate/Up LoRA: 零拷贝使用 Python LoRA 权重计算
   │   │
   │   ├─► [可选] 保存 Cache: save_to_cache() (如果 save_for_backward=true)
   │   │
   │   ├─► 激活函数: silu(gate) * up
   │   │
   │   ├─► Down GEMM: 计算下投影
   │   │
   │   ├─► Down LoRA: 零拷贝计算
   │   │
   │   └─► 加权合并: Σ weights[i] * output[i]
   │
   └─► pending--

3. Python: sync_forward()
   │
   ├─► sync_with_cuda_stream: spin-wait 等待 pending==0
   │
   ├─► CPU→GPU 异步复制: output_cpu → output_gpu (copy_, non_blocking)
   │
   └─► 返回 output_gpu
```

### 4.2 Backward 数据流 (SFT 训练)

```
1. Python: backward_task()
   │
   ├─► 从 cache_stack_ 弹出 ForwardCache
   │
   ├─► backward_down():
   │   ├─► 散播 grad_output 到专家缓冲区
   │   ├─► 计算 grad_intermediate = grad_output @ down_proj
   │   ├─► 计算 grad_down_lora_a, grad_down_lora_b
   │
   ├─► backward_activation():
   │   ├─► 使用 cache 的 gate_output, up_output
   │   ├─► 计算 grad_gate, grad_up
   │
   └─► backward_gate_up():
       ├─► 计算 grad_input
       ├─► 计算 grad_gate_lora_a, grad_gate_lora_b
       └─► 计算 grad_up_lora_a, grad_up_lora_b
```

---

## 五、关键代码文件

| 文件 | 作用 |
|------|------|
| `kt-kernel/python/experts_base.py` | Pinned memory 管理，submit/sync 接口 |
| `kt-kernel/cpu_backend/cpuinfer.h` | CPUInfer 类，cudaLaunchHostFunc 接口 |
| `kt-kernel/cpu_backend/task_queue.cpp` | 无锁任务队列，sync 同步机制 |
| `kt-kernel/cpu_backend/shared_mem_buffer.cpp` | SharedMemBuffer 内存池 |
| `kt-kernel/operators/amx/sft_moe.hpp` | AMX_SFT_MOE_TP 实现 |

---

## 六、性能优化要点

### 6.1 异步流水线

- `submit_forward()` 非阻塞返回，允许 GPU 继续执行其他操作
- 双缓冲机制允许前一层同步时准备后一层数据

### 6.2 内存访问优化

- Pinned memory 加速 GPU-CPU 传输
- 64 字节对齐优化 SIMD/AVX-512 访问
- NUMA 感知分配确保本地内存访问

### 6.3 零拷贝

- no-TP 模式下 LoRA 权重完全零拷贝
- Python tensor 修改后 C++ 自动可见

---

## 七、与 TP 模式对比

| 方面 | no-TP | TP |
|------|-------|-----|
| **同步点** | 4 个同步点 | 同样 4 个同步点 + NUMA 间合并 |
| **内存分配** | 单个 SharedMemBuffer | 每个 NUMA 一个 SharedMemBufferNuma |
| **权重复制** | 零拷贝 | 部分权重需要分区复制 |
| **LoRA 更新** | 无需 update_lora_weights | 需要 update_lora_weights_task |
| **结果合并** | 无需合并 | 需要跨 NUMA 合并结果 |

---

此分析文档已完成，涵盖了 no-TP 模式下 GPU-CPU 异构实现中的同步点和内存管理机制。
