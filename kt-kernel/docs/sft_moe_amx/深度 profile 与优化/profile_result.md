# AMX-SFT-MOE 深度 Profile 与优化分析

## 1. 测试配置

```python
# 模型配置 (基于 DeepSeek-V3 架构)
expert_num = 256              # 专家数量
hidden_size = 7168            # 隐藏维度
intermediate_size = 2048      # MLP 中间维度
max_len = 25600               # 最大序列长度
num_experts_per_tok = 8       # Top-k 专家数
qlen = 4                      # 测试序列长度
layer_num = 1                 # 测试层数

# LoRA 配置
lora_rank = 16                # LoRA 秩
lora_alpha = 32.0             # LoRA 缩放因子
lora_scaling = lora_alpha / lora_rank  # 有效缩放: 2.0

# 性能测试配置
perf_warmup_iter = 5          # 预热迭代次数
perf_test_iter = 20           # 性能测试迭代次数
perf_qlen = 128               # 性能测试序列长度
num_threads = 60              # CPU 线程数
```

## 2. NVTX 标记位置

在 `test_moe_sft_amx_no_tp.py` 中，NVTX 标记在 `step == 2` 时触发：

| NVTX Range | 位置 | 包含操作 |
|------------|------|----------|
| `forward_only` | line 1816-1831 | forward_sft_task (save_for_backward=False) |
| `backward_only` | line 1857-1874 | forward_sft_task (save_for_backward=True) + backward_task |
| `full_train_loop` | line 1887-1917 | forward_sft_task + backward_task |

## 3. Nsys Profile 结果

### 3.1 NVTX Push/Pop 统计

```
nsys stats /mnt/data/lpl/nsys/run1.nsys-rep --report nvtx_pushpop_sum
```

| Range | 时间 (ms) | 占比 | 说明 |
|-------|----------|------|------|
| `forward_only` | 131.5 | 13.4% | 仅前向传播 |
| `backward_only` | 359.2 | 36.7% | 仅反向传播 |
| `full_train_loop` | 487.7 | 49.8% | 完整训练循环 |

### 3.2 时间分析

- `full_train_loop` (487.7 ms) ≈ `forward_only` (131.5 ms) + `backward_only` (359.2 ms) = 490.7 ms
- Backward 约为 Forward 的 **2.73 倍**
- 这是合理的，因为 backward 需要计算三组 LoRA 梯度 + activation 反向

### 3.3 OS Runtime 热点

```
nsys stats /mnt/data/lpl/nsys/run1.nsys-rep --report osrt_sum --timeunit=ms
```

| 调用 | 总时间 (ms) | 次数 | 平均 (ms) | 说明 |
|------|------------|------|-----------|------|
| `nanosleep` | 192,038 | 179,993 | 1.07 | **主要热点**: 线程池 idle 等待 |
| `poll` | 52,140 | 531 | 98.19 | I/O 等待 |
| `pthread_cond_timedwait` | 28,006 | 56 | 500.11 | 条件变量等待 |
| `openat` | 27,794 | 290,719 | 0.10 | 文件打开操作 |

**关键发现**: `nanosleep` 占用了大量时间 (192 秒)，来自 work-stealing 线程池的 idle sleep。

## 4. Backward 细粒度计时

### 4.1 计时代码位置

在 `kt-kernel/operators/amx/sft_moe.hpp` 中添加了计时代码：

#### 主函数 `backward()` (line 541-605)
```cpp
BACKWARD_TIMER_START();
// ... Step 1: backward_down_amx
BACKWARD_TIMER_CHECKPOINT("backward_down");
// ... Step 2: backward_activation
BACKWARD_TIMER_CHECKPOINT("backward_activation");
// ... Step 3: backward_gate_up_amx
BACKWARD_TIMER_CHECKPOINT("backward_gate_up");
BACKWARD_TIMER_END();
```

#### `backward_down_amx()` (line 1919-2150)

| 子步骤 | 宏名 | 对应代码 | 说明 |
|--------|------|----------|------|
| D0 | `D0_prepare+memset` | `prepare_backward_weights()` + `memset(grad_intermediate_)` | 准备反向权重 + 清零中间梯度 |
| D1 | `D1_scatter` | `pool->do_work_stealing_job(activated_expert, ...)` | 将 grad_output 分散到各 expert 缓冲区 |
| D2 | `D2_quantize` | `pool->do_work_stealing_job(activated_expert, ...)` | 量化到 BufferA |
| D3 | `D3_gemm` | `pool->do_work_stealing_job(nth * activated_expert, ...)` | AMX GEMM: grad_output @ down_proj^T |
| D4 | `D4_lora_grad` | `pool->do_work_stealing_job(activated_expert, ...)` | LoRA 梯度计算 (for-loop) |

#### `backward_activation()` (line 2152-2232)

| 子步骤 | 宏名 | 对应代码 | 说明 |
|--------|------|----------|------|
| A1 | `silu_backward` | `pool->do_work_stealing_job(activated_expert, ...)` | SiLU 反向: sigmoid(gate) * (1 + gate * (1 - sigmoid(gate))) * up |

#### `backward_gate_up_amx()` (line 2424-2748)

| 子步骤 | 宏名 | 对应代码 | 说明 |
|--------|------|----------|------|
| GU0 | `GU0_prepare+memset` | `prepare_backward_weights()` + `prepare_lora_weights()` + `memset(grad_input)` | 准备权重 + 清零输入梯度 |
| GU1-gate | `base_pass(gate)` | 3x `do_work_stealing_job` | Gate GEMM: grad_gate @ gate_proj^T |
| GU1-up | `base_pass(up)` | 3x `do_work_stealing_job` | Up GEMM: grad_up @ up_proj^T |
| GU1 | `GU1_base_passes_total` | - | Base passes 总计 |
| GU2 | `GU2_requantize_for_lora` | `pool->do_work_stealing_job(activated_expert, ...)` | 重新量化输入用于 LoRA |
| GU3-gate | `lora_pass(gate)` | 6x `do_work_stealing_job` | Gate LoRA 梯度 (Step 1-6) |
| GU3-up | `lora_pass(up)` | 6x `do_work_stealing_job` | Up LoRA 梯度 (Step 1-6) |
| GU3 | `GU3_lora_passes_total` | - | LoRA passes 总计 |

### 4.2 稳定阶段计时输出 (性能测试阶段，step==2)

```
[DOWN] D0_prepare+memset: 69.506 ms
[DOWN] D1_scatter: 5.728 ms
[DOWN] D2_quantize: 0.374 ms
[DOWN] D3_gemm: 41.123 ms
[DOWN] D4_lora_grad: 72.438 ms
[BWD TIMER] backward_down: 193.885 ms (total: 193.885 ms)

[ACT] silu_backward: 1.568 ms
[BWD TIMER] backward_activation: 1.577 ms (total: 195.461 ms)

[GU] GU0_prepare+memset: 0.143 ms
[GU] base_pass(gate): 12.532 ms
[GU] base_pass(up): 11.741 ms
[GU] GU1_base_passes_total: 24.303 ms
[GU] GU2_requantize_for_lora: 0.350 ms
[GU] lora_pass(gate): 49.401 ms
[GU] lora_pass(up): 50.215 ms
[GU] GU3_lora_passes_total: 99.643 ms
[BWD TIMER] backward_gate_up: 124.446 ms (total: 319.908 ms)
```

### 4.3 计时结果分析

| 阶段 | 时间 (ms) | 占比 | 说明 |
|------|----------|------|------|
| **backward_down** | **193.9** | **60.6%** | 主要耗时阶段 |
| ├─ D0_prepare+memset | 69.5 | 21.7% | 权重准备 + 清零 |
| ├─ D1_scatter | 5.7 | 1.8% | 分散 grad_output |
| ├─ D2_quantize | 0.4 | 0.1% | 量化 |
| ├─ D3_gemm | 41.1 | 12.9% | AMX GEMM |
| └─ D4_lora_grad | 72.4 | 22.6% | **LoRA 梯度 (for-loop)** |
| **backward_activation** | **1.6** | **0.5%** | 最快阶段 |
| **backward_gate_up** | **124.4** | **38.9%** | 第二耗时阶段 |
| ├─ GU0_prepare+memset | 0.1 | 0.0% | 准备 |
| ├─ GU1_base_passes | 24.3 | 7.6% | Base GEMM (gate+up) |
| ├─ GU2_requantize | 0.4 | 0.1% | 重量化 |
| └─ GU3_lora_passes | 99.6 | 31.1% | **LoRA 梯度 (gate+up)** |
| **总计** | **319.9** | **100%** | 内部计时 |

### 4.4 Warmup vs 稳定阶段对比

| 子步骤 | Warmup (ms) | 稳定 (ms) | 差异原因 |
|--------|-------------|-----------|----------|
| D0_prepare+memset | 3738.7 | 69.5 | 首次初始化开销 |
| base_pass(gate) | 323.2 | 12.5 | JIT 编译 / 缓存预热 |
| base_pass(up) | 310.3 | 11.7 | 同上 |

### 4.5 性能测试汇总

```
Forward Pass:
  Average latency: 129.576 ms (±1.161)
  Min latency:     126.424 ms
  Max latency:     131.525 ms
  Throughput:      987.8 tokens/s

Backward Pass:
  Average latency: 341.572 ms (±6.189)
  Min latency:     335.360 ms
  Max latency:     355.451 ms
  Throughput:      374.7 tokens/s

Combined (Forward + Backward):
  Average latency: 475.802 ms (±7.512)
  Min latency:     468.008 ms
  Max latency:     490.866 ms
  Throughput:      269.0 tokens/s
```

**观察**:
- 内部计时 (319.9 ms) vs 外部测量 (341.6 ms) 差距约 22 ms
- 差距来源: Python/C++ 调用开销 + 线程池同步开销

## 5. Forward 流程分解

在 `sft_moe.hpp:303-503` 的 `forward_sft()` 函数：

| Step | 操作 | 代码位置 | 说明 |
|------|------|----------|------|
| 1 | Expert routing | line 314-329 | 计算路由：哪些 token 去哪些 expert |
| 2 | Buffer allocation | line 331-377 | 内存分配给各 expert |
| 3 | Copy input | line 390-398 | 复制输入到 expert 缓冲区 |
| 4 | Quantize input | line 401-404 | 输入量化 (BF16 → AMX 格式) |
| 5 | Gate + Up GEMM | line 408-422 | 主要计算: `[M, hidden] × [hidden, intermediate]` |
| 5.5 | Gate + Up LoRA | line 425-431 | LoRA 增量: A×B 两次小 GEMM |
| 6 | Activation | line 440 | SiLU(gate) × up |
| 7 | Quantize intermediate | line 449-455 | 量化中间结果 |
| 8 | Down GEMM | line 458-467 | 主要计算: `[M, intermediate] × [intermediate, hidden]` |
| 8.5 | Down LoRA | line 470-476 | LoRA 增量 |
| 9 | Weighted merge | line 479-502 | 按权重合并各 expert 输出 |

## 6. 代码结构映射

### 6.1 Python → C++ 调用链

```
test_moe_sft_amx_no_tp.py
    │
    ├── moe.forward_sft_task(...)
    │       ↓
    │   ext_bindings.cpp: ForwardSFTBindings::cpuinfer_interface()
    │       ↓
    │   moe-sft-tp.hpp: TP_MOE_SFT<T>::forward_sft_binding()
    │       ↓
    │   sft_moe.hpp: AMX_SFT_MOE_TP<T>::forward_sft()
    │
    └── moe.backward_task(...)
            ↓
        ext_bindings.cpp: BackwardBindings::cpuinfer_interface()
            ↓
        moe-sft-tp.hpp: TP_MOE_SFT<T>::backward_binding()
            ↓
        sft_moe.hpp: AMX_SFT_MOE_TP<T>::backward()
            ├── backward_down_amx()
            ├── backward_activation()
            └── backward_gate_up_amx()
```

### 6.2 关键文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 测试脚本 | `kt-kernel/examples/test_moe_sft_amx_no_tp.py` | 单 NUMA 节点测试 |
| SFT MOE 实现 | `kt-kernel/operators/amx/sft_moe.hpp` | AMX 加速的 SFT MoE |
| TP 封装 | `kt-kernel/operators/moe-sft-tp.hpp` | Tensor Parallel 封装 |
| Python 绑定 | `kt-kernel/ext_bindings.cpp` | pybind11 绑定 |
| 基础 MOE | `kt-kernel/operators/amx/moe.hpp` | 基础 AMX MoE |

## 7. Nanosleep 深度分析

### 7.1 Nanosleep 发生位置

`nanosleep` 来自 work-stealing 线程池的 idle 等待。根据计时结果，nanosleep 主要发生在以下阶段：

| 阶段 | 时间 (ms) | `do_work_stealing_job` 次数 | nanosleep 可能性 | 原因 |
|------|----------|---------------------------|-----------------|------|
| D0_prepare+memset | 69.5 | 0 | ⭐⭐⭐ 高 | 大量 memset，60 线程竞争内存带宽 |
| D1_scatter | 5.7 | 1 | ⭐ 低 | 单次调用，任务均匀 |
| D2_quantize | 0.4 | 1 | ⭐ 低 | 快速完成 |
| D3_gemm | 41.1 | 1 (nth*experts) | ⭐⭐ 中 | GEMM 计算密集，但任务大小可能不均 |
| **D4_lora_grad** | **72.4** | **1** | ⭐⭐⭐ **高** | **for-loop 实现，只有少量线程工作** |
| silu_backward | 1.6 | 1 | ⭐ 低 | 快速完成 |
| GU1_base_passes | 24.3 | 6 (3*2) | ⭐⭐ 中 | 多次同步点 |
| GU2_requantize | 0.4 | 1 | ⭐ 低 | 快速完成 |
| **GU3_lora_passes** | **99.6** | **12 (6*2)** | ⭐⭐⭐ **高** | **大量 for-loop + 频繁同步** |

### 7.2 主要热点分析

#### 热点 1: D0_prepare+memset (69.5 ms)

```cpp
// sft_moe.hpp backward_down_amx()
prepare_backward_weights();  // 准备转置权重
memset(grad_intermediate_, 0, ...);  // 清零大块内存
```

**问题**: 60 线程同时访问内存，造成带宽竞争，部分线程 idle 等待。

**优化方向**:
- 延迟 memset，在使用时按需清零
- 或使用多线程并行 memset

#### 热点 2: D4_lora_grad (72.4 ms, 22.6%)

```cpp
// sft_moe.hpp:2099-2145 (for-loop 实现)
pool->do_work_stealing_job(activated_expert, nullptr,
    [&](int task_id) {
        // 每个 expert 独立计算 LoRA 梯度
        for (int i = 0; i < intermediate_size; i++) {
            for (int r = 0; r < lora_rank; r++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                    sum += grad[t*inter + i] * inter_ptr[t*rank + r];
                }
                grad_lora_b[...] = current + sum * scaling;
            }
        }
    }, nullptr);
```

**问题**:
- 只有 `activated_expert` (约 8-16) 个任务，但有 60 线程
- 大量线程 idle 等待，导致 nanosleep
- for-loop 实现无法利用 AMX 加速

**优化方向**:
- 将 LoRA 梯度计算转为矩阵乘法: `grad^T @ intermediate`
- 使用 AMX GEMM 替代 for-loop

#### 热点 3: GU3_lora_passes (99.6 ms, 31.1%)

```cpp
// sft_moe.hpp:2594-2757 lora_pass lambda
// 每个 lora_pass 包含 6 个 do_work_stealing_job:
// Step 1: input @ lora_A^T -> U (AMX GEMM)
// Step 2: grad_B = grad^T @ U (for-loop)
// Step 3: grad @ lora_B -> G_B (AMX GEMM)
// Step 4: Quantize G_B
// Step 5: G_B @ lora_A -> grad_input (AMX GEMM)
// Step 6: grad_A = input^T @ G_B (for-loop)
```

**问题**:
- 每个 lora_pass 有 6 次 `do_work_stealing_job` 同步
- gate + up 共 12 次同步，每次同步都有 nanosleep 开销
- Step 2 和 Step 6 使用 for-loop，与 D4_lora_grad 相同的问题

**优化方向**:
- 合并 gate 和 up 的 lora_pass，减少同步次数
- 将 Step 2/6 的 for-loop 转为 GEMM

### 7.3 Nanosleep 数量估算

根据 nsys 报告: `nanosleep` 总次数约 180,000 次，总时间 192 秒。

对于单次 backward (60 线程, ~320 ms):
- 估计 nanosleep 次数: 180,000 / (总迭代数) ≈ 几千次
- 平均每次 nanosleep: 1.07 ms

**关键发现**: nanosleep 时间远超计算时间，说明大量线程处于 idle 状态。

## 8. 待优化点

### 8.1 LoRA 梯度优化 (最高优先级)

`D4_lora_grad` + `GU3_lora_passes` 共占 **53.7%** 时间 (172 ms)。

优化方案:
```cpp
// 当前: for-loop O(inter * rank * tokens)
for (int i = 0; i < inter; i++)
    for (int r = 0; r < rank; r++)
        for (int t = 0; t < tokens; t++)
            sum += grad[t,i] * U[t,r];

// 优化: GEMM - grad^T @ U
// grad: [tokens, inter] -> grad^T: [inter, tokens]
// U: [tokens, rank]
// result: [inter, rank]
amx::mat_mul(inter, rank, tokens, grad_T, U, result, ...);
```

### 8.2 减少同步次数

当前 backward 中 `do_work_stealing_job` 调用次数:
- backward_down_amx: 4 次
- backward_activation: 1 次
- backward_gate_up_amx: ~15 次

优化方向: 合并相邻的 work_stealing_job

### 8.3 D0_prepare+memset 优化

当前 69.5 ms 用于 prepare + memset:
- 考虑延迟清零 (lazy zeroing)
- 或在前一次计算完成时顺便清零

## 9. 附录：nsys 命令参考

```bash
# 生成 profile
TMPDIR=/mnt/data/lpl/nsys_tmp nsys profile -o /mnt/data/lpl/nsys/run1 \
  --force-overwrite=true \
  --trace=nvtx,osrt \
  python /home/lpl/ktransformers/kt-kernel/examples/test_moe_sft_amx_no_tp.py

# 查看 NVTX 统计
nsys stats /mnt/data/lpl/nsys/run1.nsys-rep --report nvtx_pushpop_sum

# 查看 NVTX trace
nsys stats /mnt/data/lpl/nsys/run1.nsys-rep --report nvtx_pushpop_trace

# 查看 OS runtime 统计
nsys stats /mnt/data/lpl/nsys/run1.nsys-rep --report osrt_sum --timeunit=ms
```

## 10. Forward vs Backward 计算量分析

### 10.1 为什么 Backward 是 Forward 的 ~2 倍？

从 profile 数据：
- Forward: 131.5 ms
- Backward: 359.2 ms → 比例 **2.73x**

这是合理的，因为 **backward 需要的 GEMM 数量本身就比 forward 多**。

### 10.2 Forward GEMM 数量 (~9 个)

| 操作 | GEMM 数量 | 说明 |
|------|----------|------|
| x @ gate_proj | 1 | Base weight |
| x @ up_proj | 1 | Base weight |
| x @ gate_lora_A → @ gate_lora_B | 2 | LoRA |
| x @ up_lora_A → @ up_lora_B | 2 | LoRA |
| intermediate @ down_proj | 1 | Base weight |
| intermediate @ down_lora_A → @ down_lora_B | 2 | LoRA |
| **总计** | **9** | |

### 10.3 Backward GEMM 数量 (~17 个)

#### backward_down (5 个 GEMM)

| 操作 | 目的 | GEMM |
|------|------|------|
| grad_output @ down_proj^T | 计算 grad_intermediate | 1 |
| cached_inter @ down_lora_A^T | 准备计算 grad_B | 1 |
| grad_output^T @ inter_proj | grad_down_lora_B | 1 |
| grad_output @ down_lora_B | 准备计算 grad_A | 1 |
| cached_inter^T @ grad_times_b | grad_down_lora_A | 1 |

#### backward_gate_up base (2 个 GEMM)

| 操作 | 目的 | GEMM |
|------|------|------|
| grad_gate @ gate_proj^T | 传回 grad_input | 1 |
| grad_up @ up_proj^T | 传回 grad_input | 1 |

#### backward_gate_up LoRA (10 个 GEMM，gate 和 up 各 5 个)

以 gate 为例（up 完全相同）：

| Step | 操作 | 目的 | GEMM |
|------|------|------|------|
| 1 | input @ gate_lora_A^T | 准备 U 用于 grad_B | 1 |
| 2 | grad^T @ U | grad_gate_lora_B | 1 |
| 3 | grad @ gate_lora_B | 准备 G_B | 1 |
| 5 | G_B @ gate_lora_A | LoRA 部分的 grad_input | 1 |
| 6 | input^T @ G_B | grad_gate_lora_A | 1 |

### 10.4 理论 vs 实际对比

| | Forward | Backward | 比例 |
|---|---------|----------|------|
| Base weights (冻结) | 3 GEMM | 3 GEMM | 1:1 |
| LoRA weights (训练) | 6 GEMM | ~14 GEMM | ~2.3:1 |
| **总计** | **9 GEMM** | **~17 GEMM** | **~1.9:1** |

**关键结论**: 即使保存了所有激活值，backward 计算量仍然是 forward 的约 2 倍，因为：
1. 对于需要训练的权重，需要额外计算 grad_weight
2. `Forward: y = W @ x` → 1 GEMM
3. `Backward: grad_x = W^T @ grad_y` → 1 GEMM + `grad_W = grad_y @ x^T` → 1 GEMM

## 11. For-loop 实现的 LoRA 梯度计算

### 11.1 具体代码位置

#### D4_lora_grad (`backward_down_amx`)

**文件**: `sft_moe.hpp:2106-2157`

```cpp
// 计算 inter_proj = cached_intermediate @ lora_A^T (for-loop)
for (int t = 0; t < num_tokens; t++) {
  for (int r = 0; r < lora_rank_; r++) {
    float sum = 0.0f;
    for (int i = 0; i < config_.intermediate_size; i++) {
      float inp = GGML_BF16_TO_FP32(cached_intermediate[t * config_.intermediate_size + i]);
      float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.intermediate_size + i]);
      sum += inp * w;
    }
    inter_proj[t * lora_rank_ + r] = sum;
  }
}

// 计算 grad_B = grad_output^T @ inter_proj (for-loop)
for (int h = 0; h < config_.hidden_size; h++) {
  for (int r = 0; r < lora_rank_; r++) {
    float sum = 0.0f;
    for (int t = 0; t < num_tokens; t++) {
      sum += expert_grad_out[t * config_.hidden_size + h] * inter_proj[t * lora_rank_ + r];
    }
    // ... 累加到 grad_down_b
  }
}
```

#### lora_pass Step 2 (`backward_gate_up_amx`)

**文件**: `sft_moe.hpp:2632-2643`

```cpp
// Step 2: grad_B = grad^T @ U (for-loop)
for (int i = 0; i < config_.intermediate_size; i++) {
  for (int r = 0; r < lora_rank_; r++) {
    float sum = 0.0f;
    for (int t = 0; t < num_tokens; t++) {
      float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
      float u = GGML_BF16_TO_FP32(u_ptr[t * padded_lora_rank_ + r]);
      sum += g * u;
    }
    // ... 累加到 grad_lora_b
  }
}
```

#### lora_pass Step 6 (`backward_gate_up_amx`)

**文件**: `sft_moe.hpp:2734-2746`

```cpp
// Step 6: grad_A = input^T @ G_B (for-loop)
for (int r = 0; r < lora_rank_; r++) {
  for (int h = 0; h < config_.hidden_size; h++) {
    float sum = 0.0f;
    for (int t = 0; t < num_tokens; t++) {
      float gb = GGML_BF16_TO_FP32(g_ptr[t * padded_lora_rank_ + r]);
      float inp = GGML_BF16_TO_FP32(expert_input[t * config_.hidden_size + h]);
      sum += inp * gb;
    }
    // ... 累加到 grad_lora_a
  }
}
```

### 11.2 For-loop 耗时汇总

| 位置 | 代码行 | 操作 | 耗时 |
|------|--------|------|------|
| `backward_down_amx` D4 | 2106-2157 | LoRA 梯度 (4个 for-loop) | 72.4 ms |
| `lora_pass` Step 2 | 2632-2643 | grad_B = grad^T @ U | 99.6 ms (gate+up 共享) |
| `lora_pass` Step 6 | 2734-2746 | grad_A = input^T @ G_B | |

**总计**: 172 ms (**53.7%** 的 backward 时间)

### 11.3 For-loop 中的隐式转置

以 `grad_B = grad^T @ U` 为例：

```cpp
// grad: [num_tokens, intermediate_size]
// U:    [num_tokens, lora_rank]
// 结果: [intermediate_size, lora_rank]

for (int i = 0; i < intermediate_size; i++) {        // 输出行 (grad 的列)
  for (int r = 0; r < lora_rank_; r++) {              // 输出列
    for (int t = 0; t < num_tokens; t++) {            // reduction
      sum += grad[t * intermediate_size + i] * u[t * lora_rank + r];
      //          ↑ 按列访问 grad，等效于 grad^T
    }
  }
}
```

**隐式转置的本质**: 通过改变循环访问模式 `grad[t, i]` → 按 `i` 迭代外层，等效于访问 `grad^T[i, t]`。

## 12. AMX 权重预转置机制

### 12.1 为什么需要预转置？

AMX `mat_mul` 的设计：
```cpp
// amx_kernels.hpp:1933
inline void mat_mul(int m, int n, int k,
                    std::shared_ptr<BufferA> ba,  // 输入矩阵 [m, k]
                    std::shared_ptr<BufferB> bb,  // 权重矩阵 [n, k] (已转置存储)
                    std::shared_ptr<BufferC> bc,  // 输出矩阵 [m, n]
                    int ith, int nth);
```

**关键**: AMX mat_mul 没有转置参数，它假设 BufferB 存储的是已转置的权重，计算的是 `A @ B^T`。

### 12.2 Base Weights 预转置

**函数**: `prepare_backward_weights()` (sft_moe.hpp:738-815)

```cpp
// gate_proj: [intermediate_size, hidden_size] → 转置为 [hidden_size, intermediate_size]
std::vector<ggml_bf16_t> transposed(config_.hidden_size * config_.intermediate_size);
for (int i = 0; i < config_.intermediate_size; i++) {
  for (int h = 0; h < config_.hidden_size; h++) {
    transposed[h * config_.intermediate_size + i] =
        gate_proj[expert_offset + i * config_.hidden_size + h];
  }
}
gate_backward_bb_[expert_idx]->from_mat(transposed.data(), 0, 1);
```

**特点**: 权重是冻结的，只需转置一次。

### 12.3 LoRA Weights 预转置

**函数**: `prepare_lora_weights()` (sft_moe.hpp:654-728)

每个 LoRA 权重准备**两个版本**（原始 + 转置）：

| 函数 | 输出 BufferB | 是否转置 |
|------|-------------|---------|
| `convert_lora_a_to_buffer_b` | `gate_lora_a_bb_` | ❌ 不转置 |
| `convert_lora_b_to_buffer_b` | `gate_lora_b_bb_` | ❌ 不转置 |
| `convert_lora_a_transposed_to_buffer_b` | `gate_lora_a_t_bb_` | ✅ **转置** |
| `convert_lora_b_transposed_to_buffer_b` | `gate_lora_b_t_bb_` | ✅ **转置** |

### 12.4 转置函数实现

**不转置版本** (`convert_lora_a_to_buffer_b`, sft_moe.hpp:1235-1250):
```cpp
// 直接复制 + padding
for (int r = 0; r < src_n; r++) {
  for (int c = 0; c < src_k; c++) {
    padded[r * dst_k + c] = expert_src[r * src_k + c];
  }
}
dst_bb->from_mat(padded.data(), 0, 1);
```

**转置版本** (`convert_lora_a_transposed_to_buffer_b`, sft_moe.hpp:1284-1296):
```cpp
// LoRA A: [lora_rank, hidden_size] → A^T: [hidden_size, padded_lora_rank]
for (int h = 0; h < src_k; h++) {        // hidden_size
  for (int r = 0; r < src_n; r++) {      // lora_rank
    padded[h * dst_k + r] = expert_src[r * src_k + h];  // 转置: [h,r] ← [r,h]
  }
}
dst_bb->from_mat(padded.data(), 0, 1);
```

### 12.5 优化 For-loop 需要转置什么？

| 矩阵 | 类型 | 是否需要转置 | 说明 |
|------|------|-------------|------|
| `grad` | 激活值 | **需要** | 每次 backward 不同 |
| `U` (lora_intermediate) | 激活值 | **需要** | 每次 backward 不同 |
| `lora_A`, `lora_B` | 权重 | **不需要** | 已在 `prepare_lora_weights()` 准备好 |

**结论**: 将 for-loop 换成 AMX GEMM 时，需要每次转置激活值（grad, U, input），权重已经预转置好了。
