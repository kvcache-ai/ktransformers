# SFT MOE AMX LoRA GEMM 优化 Bug 记录

## Bug #1: Gate/Up 中间缓冲区竞争条件 【待验证】

**日期**: 2026-01-07

**提交**: 在 AMX LoRA GEMM 优化后出现

**状态**: 已实施 Buffer 分离修复，添加调试打印，待验证

---

### 1. 现象描述

单测 `test_moe_sft_amx.py` 失败：

```
================================================================================
Forward iteration 0: FAILED (AMX: 0.04..., Torch: 6.0..., relative diff: 1.0)
================================================================================
Forward iteration 1: PASSED
================================================================================
Backward: FAILED (grad_input diff: 0.97)
================================================================================
```

特点：
- Forward iteration 0 **总是**失败，但 iteration 1 **总是**通过
- 这是**确定性行为**，不是随机的竞争条件
- Backward 也失败，可能是同样的原因

---

### 2. 深入分析

#### 2.1 关键发现：warm_up 已经初始化了权重

`moe_base.hpp:148-159` 中的 `warm_up()` 函数在验证迭代之前就已经运行：

```cpp
void warm_up() {
  int qlen = config_.max_len;  // 25600
  // ... 创建测试数据 ...
  forward(qlen, ...);  // 触发 prepare_lora_weights()
}
```

这意味着在 iteration 0 时：
- `lora_weights_prepared_` **已经**是 true（从 warm_up 设置）
- LoRA 权重**已经**转换为 BufferB 格式
- **这排除了 "首次权重转换" 作为失败原因**

#### 2.2 可能的真正原因

| 假设 | 可能性 | 说明 |
|------|--------|------|
| ~~典型竞争条件~~ | **排除** | 应该是非确定性的，但现象是确定性的（总是 iter0 失败，iter1 通过） |
| ~~首次权重转换~~ | **排除** | warm_up 已经转换了权重 |
| Buffer 状态残留 | **高** | warm_up 使用 qlen=25600，validation 使用 qlen=4，buffer 状态可能有残留数据 |
| BufferC 未正确初始化 | **高** | `amx::mat_mul` 可能期望 BufferC 为零，但未显式清零 |
| max_m 不匹配 | 中 | 主 buffer 动态设置 max_m，LoRA buffer 使用静态 max_m |

#### 2.3 为什么之前 for-loop 版本能过？

**for-loop 版本** (`compute_lora_gate_up`，lines 1439-1498)：

```cpp
void compute_lora_gate_up(int qlen, int activated_expert) {
  pool->do_work_stealing_job(
      activated_expert * 2, nullptr,
      [this](int task_id) {
        // 每个 task 创建线程私有的中间缓冲区
        std::vector<float> local_intermediate(num_tokens * lora_rank_);  // 线程私有！

        // Step 1: intermediate = input @ lora_A^T
        // 写入私有缓冲区，不会干扰其他 task

        // Step 2: output += intermediate @ lora_B^T * scaling
        // 直接写入输出
      },
      nullptr);
}
```

关键点：`local_intermediate` 是**线程私有的栈变量**，每次调用都重新分配和初始化。

**AMX 版本** (`compute_lora_gate_up_amx`，lines 1191-1325)：
- 使用预分配的共享 BufferA/BufferC
- 这些 buffer 在 warm_up 时被使用，可能残留数据

---

### 3. 已实施的修复方案

#### 3.1 Buffer 分离（已实施）

为 gate 和 up 创建独立的中间缓冲区：

```cpp
// 新增成员变量 (lines 162-178)
std::vector<std::shared_ptr<typename T::BufferA>> lora_gate_intermediate_ba_;
std::vector<std::shared_ptr<typename T::BufferA>> lora_up_intermediate_ba_;
std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_intermediate_bc_;
std::vector<std::shared_ptr<typename T::BufferC>> lora_up_intermediate_bc_;
std::vector<ggml_bf16_t*> lora_gate_intermediate_ptr_;
std::vector<ggml_bf16_t*> lora_up_intermediate_ptr_;
```

修改位置：
- lines 162-178: 成员变量
- lines 848-865: 缓冲区大小 × 2
- lines 995-1075: 初始化分离的缓冲区
- lines 1199-1325: `compute_lora_gate_up_amx` 使用分离缓冲区

#### 3.2 调试打印（已添加）

在以下位置添加了调试打印：

**位置 1: forward_sft Step 5.5 之前** (lines 418-437)
```cpp
// DEBUG: Print main GEMM output BEFORE LoRA
printf("\n=== forward_sft call #%d: BEFORE LoRA (expert %d, m=%d) ===\n", ...);
printf("  gate_output[0:8] = ...\n");
printf("  up_output[0:8] = ...\n");
```

**位置 2: compute_lora_gate_up_amx 入口** (lines 1196-1203)
```cpp
// DEBUG: Print entry info
printf("\n=== compute_lora_gate_up_amx call #%d (qlen=%d, activated_expert=%d) ===\n", ...);
```

**位置 3: Step 1 之后** (lines 1239-1256)
```cpp
// DEBUG: Print Step 1 results
printf("Step 1 done - expert %d, m=%d\n", ...);
printf("  gate_intermediate_ptr[0:8] = ...\n");
printf("  up_intermediate_ptr[0:8] = ...\n");
```

**位置 4: Step 3 之后** (lines 1306-1324)
```cpp
// DEBUG: Print Step 3 results (final gate/up output after LoRA)
printf("Step 3 done - expert %d\n", ...);
printf("  gate_output[0:8] (after LoRA) = ...\n");
printf("  up_output[0:8] (after LoRA) = ...\n");
```

**位置 5: add_lora_output_to_main** (lines 1415-1428)
```cpp
// DEBUG: Print BufferC values on first call
printf("add_lora_output_to_main call #%d: m=%d, n=%d, scaling=%.4f\n", ...);
printf("  bc[0:8] = ...\n");
printf("  main_output[0:8] (before) = ...\n");
```

---

### 4. 验证检查清单

- [ ] 编译通过
- [ ] 运行测试查看调试输出
- [ ] 分析 Step 1 中间结果是否正确
- [ ] 分析 Step 3 最终输出是否正确
- [ ] 对比 main GEMM 输出（LoRA 前后）
- [ ] 如果 Step 1 输出接近 0 → 问题在 mat_mul 或 BufferB 转换
- [ ] 如果 Step 1 正确但 Step 3 错误 → 问题在第二步 GEMM
- [ ] Forward 单测通过
- [ ] Backward 单测通过
- [ ] TP 模式验证

---

### 5. 相关文件

- Bug 代码位置: `/home/lpl/ktransformers/kt-kernel/operators/amx/sft_moe.hpp`
- 单测文件: `/home/lpl/ktransformers/kt-kernel/examples/test_moe_sft_amx.py`
- 优化文档: `/home/lpl/ktransformers/kt-kernel/docs/sft_moe_amx/AMX_LoRA_GEMM优化文档.md`
- 计划文档: `/home/lpl/.claude/plans/melodic-hopping-matsumoto.md`

---

### 6. 调试输出分析指南

运行测试后，查看以下输出：

1. **warm_up 阶段的输出** (call #1):
   - 应该看到 qlen=25600 的调用
   - 这是 iteration 0 之前的状态

2. **validation iteration 0 的输出** (call #2):
   - 这是失败的迭代
   - 检查 gate_intermediate_ptr 和 up_intermediate_ptr 的值
   - 检查 gate_output 和 up_output 的值

3. **validation iteration 1 的输出** (call #3):
   - 这是通过的迭代
   - 对比与 iteration 0 的差异

关键对比点：
- Step 1 输出应该是 `input @ lora_A^T` 的结果，应该是非零的浮点数
- Step 3 输出应该是 `main_output + lora_contribution * scaling`
- 如果 Step 1 输出接近 0 → 问题在 mat_mul 或 BufferB 转换
- 如果 Step 1 正确但 Step 3 错误 → 问题在第二步 GEMM 或 add_lora_output_to_main

---

### 7. 额外内存开销

Buffer 分离后，每个 expert 额外需要：
- BufferA: `max_m × padded_lora_rank` × 1 (gate 和 up 各一个)
- BufferC: `max_m × padded_lora_rank` × 1
- BF16 buffer: `max_m × padded_lora_rank × sizeof(BF16)` × 1

总计：约 `max_m × padded_lora_rank × (BufferA大小 + BufferC大小 + 2字节)` 额外内存。

---

## Bug #2: Backward 失败 【待调试】

**日期**: 2026-01-07

**状态**: 已添加调试打印，待运行测试分析

---

### 1. 现象描述

**非 TP 模式测试结果**:
```
Forward iteration 0: PASSED (diff: 0.037109)
Forward iteration 1: PASSED (diff: 0.036133)
Backward: FAILED (grad_input diff: 0.941406)
```

**TP 模式测试结果**:
```
Forward iteration 0: FAILED (diff: 1.0)
Forward iteration 1: PASSED
Backward: FAILED (grad_input diff: 0.97)
```

特点：
- **Forward 在非 TP 模式下通过**（说明 Forward 的 buffer 分离修复有效）
- **Backward 在 TP 和非 TP 模式下都失败**（grad_input diff ≈ 0.94-0.97）
- 这是一个更基础的问题，独立于 TP 模式

---

### 2. 代码分析

#### 2.1 backward_down_amx (lines 1882-2079) - 使用 AMX

```
Step 1: Scatter grad_output to per-expert BF16 buffers
Step 2: Quantize to BufferA (grad_output_ba_)
Step 3: AMX GEMM: grad_intermediate = grad_output @ down_proj (使用 down_backward_bb_)
Step 4: Convert BufferC to BF16 (grad_intermediate_)
Step 5: LoRA gradients (for-loop)
```

AMX GEMM 调用（line 1969）:
```cpp
amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
```

#### 2.2 backward_gate_up_amx (lines 2393-2617) - 仍然是 for-loop！

**注意：尽管名字叫 "_amx"，主 GEMM 仍然使用 for-loop！**

Line 2446 注释:
```cpp
// For now, we still use for-loop but with optimized memory access
// Full AMX version would require additional BufferA for grad
```

处理逻辑：
1. 计算 `token_grad_input = grad @ W^T` (for-loop，有优化)
2. Scatter back to grad_input
3. LoRA gradients (for-loop)
4. 计算 grad_input from LoRA

#### 2.3 发现的潜在问题

**问题 1: backward_gate_up_amx 中的梯度跳过优化**
```cpp
// Line 2461
if (std::abs(g) < 1e-10f) continue;  // Skip near-zero gradients
```
- 原始 `backward_gate_up` 没有这个优化
- 但 `1e-10f` 阈值不太可能导致 0.94 的误差

**问题 2: 循环顺序改变**
- 原版: `for t -> for h -> for i`
- AMX版: `for t -> for i -> for h`（为了更好的缓存利用）
- 这应该不会影响结果，但可能有数值稳定性差异

**问题 3: backward_down_amx 中的 AMX GEMM**
- 使用 `down_backward_bb_[expert_idx]` 存储转置后的权重
- 需要验证 `prepare_backward_weights()` 中的转置逻辑是否正确

---

### 3. 已添加的调试打印

#### 3.1 backward_down_amx (line ~1996-2018)
```cpp
// DEBUG: Print grad_intermediate after AMX GEMM (Step 4)
printf("\n=== backward_down_amx call #%d (qlen=%d, activated_expert=%d) ===\n", ...);
printf("  expert %d (m=%d): grad_intermediate_[0:8] = ...\n", ...);
```

#### 3.2 backward_down (for-loop, line ~1782-1793)
```cpp
// DEBUG: Print grad_intermediate for this expert (for-loop version)
printf("\n=== backward_down (for-loop) call #%d, expert %d (m=%d) ===\n", ...);
printf("  grad_intermediate_[0:8] = ...\n");
```

#### 3.3 backward_gate_up_amx (line ~2607-2617)
```cpp
// DEBUG: Print final grad_input after backward_gate_up_amx
printf("\n=== backward_gate_up_amx call #%d (qlen=%d, activated_expert=%d) ===\n", ...);
printf("  final grad_input[0:8] = ...\n");
```

#### 3.4 backward_gate_up (for-loop, line ~2383-2393)
```cpp
// DEBUG: Print final grad_input after backward_gate_up (for-loop version)
printf("\n=== backward_gate_up (for-loop) call #%d (qlen=%d, activated_expert=%d) ===\n", ...);
printf("  final grad_input[0:8] = ...\n");
```

---

### 4. 调试策略

由于 0.94 的误差非常大，可能的原因：

1. **down_backward_bb_ 转置错误** - `prepare_backward_weights()` 中的转置逻辑
2. **AMX GEMM 维度错误** - 传入 `amx::mat_mul` 的 m/n/k 参数
3. **BufferC 输出错误** - `to_mat()` 的调用方式

### 5. 运行测试命令

编译：
```bash
pip install -e .
```

运行非 TP 测试（推荐，排除 TP 影响）：
```bash
cd /home/lpl/ktransformers/kt-kernel
python examples/test_moe_sft_amx_no_tp.py 2>&1 | grep -A2 "backward"
```

运行 TP 测试：
```bash
python examples/test_moe_sft_amx.py 2>&1 | grep -A2 "backward"
```

### 6. 分析要点

1. **对比 AMX 版本和 for-loop 版本的输出**
   - 如果两者输出相同 → 问题不在 AMX GEMM
   - 如果两者输出不同 → 问题在 backward_down_amx 的 AMX GEMM

2. **检查 grad_intermediate_ 的值**
   - AMX 版本和 for-loop 版本应该产生相同的 grad_intermediate_
   - 如果不同，问题在 `down_backward_bb_` 的转置或 AMX GEMM

3. **检查 final grad_input 的值**
   - 这是最终返回的梯度
   - 对比 Torch 的参考值

---

### 7. 对比调试方案（已实施）

**日期**: 2026-01-07

**状态**: 已添加对比调试代码到 `backward()` 函数，待运行测试

#### 7.1 方案描述

由于 BF16 模式下 `supports_standard_mat_mul_v<T> = true`，只有 AMX 版本会执行，
无法直接看到 for-loop 版本的输出作为参考。

解决方案：在 `backward()` 函数中同时运行两个版本，直接对比输出。

#### 7.2 修改位置

**文件**: `sft_moe.hpp` 的 `backward()` 函数 (lines 573-660)

#### 7.3 对比调试代码 - backward_down

```cpp
// Step 1: Down projection backward
if constexpr (supports_standard_mat_mul_v<T>) {
  // ===== 对比调试：backward_down AMX vs for-loop =====
  // 先运行 AMX 版本
  backward_down_amx(cache, grad_output, grad_down_lora_a, grad_down_lora_b);

  // 备份 AMX 结果
  size_t grad_inter_size = config_.max_len * config_.num_experts_per_tok * config_.intermediate_size;
  std::vector<ggml_bf16_t> grad_inter_amx_backup(grad_inter_size);
  memcpy(grad_inter_amx_backup.data(), grad_intermediate_, grad_inter_size * sizeof(ggml_bf16_t));

  // 运行 for-loop 版本（会覆盖 grad_intermediate_）
  backward_down(cache, grad_output, grad_down_lora_a, grad_down_lora_b);

  // 对比输出
  printf("\n=== COMPARISON: backward_down AMX vs for-loop ===\n");
  printf("AMX grad_intermediate_[0:8] = ");
  for (int j = 0; j < 8; j++) printf("%.4f ", GGML_BF16_TO_FP32(grad_inter_amx_backup[j]));
  printf("\n");
  printf("for-loop grad_intermediate_[0:8] = ");
  for (int j = 0; j < 8; j++) printf("%.4f ", GGML_BF16_TO_FP32(grad_intermediate_[j]));
  printf("\n");

  // 计算差异
  float max_diff = 0.0f;
  for (size_t j = 0; j < grad_inter_size; j++) {
    float amx_val = GGML_BF16_TO_FP32(grad_inter_amx_backup[j]);
    float loop_val = GGML_BF16_TO_FP32(grad_intermediate_[j]);
    float diff = std::abs(amx_val - loop_val);
    if (diff > max_diff) max_diff = diff;
  }
  printf("Max diff (AMX vs for-loop): %.6f\n", max_diff);

  // 继续使用 for-loop 结果（假设 for-loop 是正确的）
}
```

#### 7.4 对比调试代码 - backward_gate_up

```cpp
// Step 3: Gate + Up projection backward
if constexpr (supports_standard_mat_mul_v<T>) {
  // ===== 对比调试：backward_gate_up AMX vs for-loop =====
  size_t grad_input_size = qlen * config_.hidden_size;
  std::vector<ggml_bf16_t> grad_input_amx(grad_input_size);

  // 先运行 AMX 版本（写入 grad_input）
  backward_gate_up_amx(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b,
                       grad_up_lora_a, grad_up_lora_b);

  // 备份 AMX 结果
  memcpy(grad_input_amx.data(), grad_input, grad_input_size * sizeof(ggml_bf16_t));

  // 清零 grad_input 并运行 for-loop 版本
  memset(grad_input, 0, grad_input_size * sizeof(ggml_bf16_t));
  backward_gate_up(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b,
                   grad_up_lora_a, grad_up_lora_b);

  // 对比输出
  printf("\n=== COMPARISON: backward_gate_up AMX vs for-loop ===\n");
  ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
  printf("AMX grad_input[0:8] = ");
  for (int j = 0; j < 8; j++) printf("%.4f ", GGML_BF16_TO_FP32(grad_input_amx[j]));
  printf("\n");
  printf("for-loop grad_input[0:8] = ");
  for (int j = 0; j < 8; j++) printf("%.4f ", GGML_BF16_TO_FP32(grad_input_bf16[j]));
  printf("\n");

  // 计算差异
  float max_diff = 0.0f;
  for (size_t j = 0; j < grad_input_size; j++) {
    float amx_val = GGML_BF16_TO_FP32(grad_input_amx[j]);
    float loop_val = GGML_BF16_TO_FP32(grad_input_bf16[j]);
    float diff = std::abs(amx_val - loop_val);
    if (diff > max_diff) max_diff = diff;
  }
  printf("Max diff (AMX vs for-loop): %.6f\n", max_diff);

  // 继续使用 for-loop 结果
}
```

#### 7.5 运行测试命令

```bash
cd /home/lpl/ktransformers/kt-kernel
pip install -e .
python examples/test_moe_sft_amx_no_tp.py 2>&1 | grep -A5 "COMPARISON"
```

#### 7.6 预期输出分析

**场景 1: backward_down 差异大**
```
=== COMPARISON: backward_down AMX vs for-loop ===
AMX grad_intermediate_[0:8] = -0.2129 0.2871 ...
for-loop grad_intermediate_[0:8] = 0.5432 -0.1234 ...   ← 完全不同
Max diff (AMX vs for-loop): 1.234567
```
→ 问题在 `backward_down_amx` 的 AMX GEMM
→ 检查 `down_backward_bb_` 转置逻辑和 `mat_mul` 参数

**场景 2: backward_down 差异小，backward_gate_up 差异大**
```
=== COMPARISON: backward_down AMX vs for-loop ===
AMX grad_intermediate_[0:8] = -0.2129 0.2871 ...
for-loop grad_intermediate_[0:8] = -0.2130 0.2870 ...   ← 基本一致
Max diff (AMX vs for-loop): 0.001234

=== COMPARISON: backward_gate_up AMX vs for-loop ===
AMX grad_input[0:8] = -2.1562 5.3125 ...
for-loop grad_input[0:8] = 0.1234 -0.5678 ...   ← 完全不同
Max diff (AMX vs for-loop): 5.678901
```
→ 问题在 `backward_gate_up_amx` 的循环优化
→ 检查梯度跳过优化 `if (std::abs(g) < 1e-10f) continue;`
→ 检查循环顺序变化的影响

**场景 3: 两者差异都小**
```
Max diff (AMX vs for-loop): 0.000123  (backward_down)
Max diff (AMX vs for-loop): 0.000456  (backward_gate_up)
```
→ AMX 和 for-loop 版本都是正确的（数值精度差异可接受）
→ 问题可能在测试本身或 Torch 参考实现

#### 7.7 注意事项

1. **对比调试代码是临时的**，修复 bug 后应移除
2. **for-loop 版本假设是正确的**，因为之前单测能过
3. **最终使用 for-loop 结果**，确保测试能继续执行（即使 AMX 版本有问题）

---

### 8. Per-Expert Diff 分析（已实施）

**日期**: 2026-01-07

**状态**: 已添加代码，待运行测试

#### 8.1 背景

上一轮测试结果显示：
- `grad_intermediate_[0:8]` AMX 和 for-loop 几乎一致（diff < 0.001）
- 但 Max diff = **0.976562**（数组中某处有很大差异）

**推测**：Expert 0 正确，但其他某个 expert 的结果有问题。

#### 8.2 实施的代码

**位置**: `sft_moe.hpp` backward() 函数，lines 614-653

```cpp
// ===== 按 Expert 分析 diff =====
printf("\n=== PER-EXPERT DIFF ANALYSIS (backward_down) ===\n");
int activated_expert = cache.activated_expert_cache;
for (int i = 0; i < activated_expert; i++) {
  int expert_idx = m_expert_id_map_[i];
  int m = m_local_num_[expert_idx];

  // 计算这个 expert 在 grad_intermediate_ 中的偏移
  size_t offset = 0;
  for (int e = 0; e < i; e++) {
    offset += m_local_num_[m_expert_id_map_[e]];
  }
  offset *= config_.intermediate_size;

  // 计算这个 expert 的 max_diff
  float expert_max_diff = 0.0f;
  int max_diff_pos = -1;
  for (int t = 0; t < m; t++) {
    for (int j = 0; j < config_.intermediate_size; j++) {
      size_t idx = offset + t * config_.intermediate_size + j;
      float amx_val = GGML_BF16_TO_FP32(grad_inter_amx_backup[idx]);
      float loop_val = GGML_BF16_TO_FP32(grad_intermediate_[idx]);
      float diff = std::abs(amx_val - loop_val);
      if (diff > expert_max_diff) {
        expert_max_diff = diff;
        max_diff_pos = t * config_.intermediate_size + j;
      }
    }
  }

  // 只打印有显著 diff 的 expert
  if (expert_max_diff > 0.01f) {
    printf("Expert %d (task %d, m=%d): max_diff = %.6f at local_pos %d\n",
           expert_idx, i, m, expert_max_diff, max_diff_pos);
    printf("  AMX value = %.6f, for-loop value = %.6f\n",
           GGML_BF16_TO_FP32(grad_inter_amx_backup[offset + max_diff_pos]),
           GGML_BF16_TO_FP32(grad_intermediate_[offset + max_diff_pos]));
  }
}
printf("=== END PER-EXPERT ANALYSIS ===\n");
```

#### 8.3 LoRA 梯度 buffer 重置（已修复）

**问题**: 对比调试代码运行两个版本，但 LoRA 梯度会累加到同一个 buffer，导致 2x 问题。

**修复位置 1**: backward_down LoRA 重置 (lines 584-590)
```cpp
// 重置 LoRA 梯度 buffer，防止 for-loop 版本累加到 AMX 结果上
if (down_lora_a_ != nullptr) {
  size_t lora_a_size = config_.expert_num * lora_rank_ * config_.intermediate_size;
  size_t lora_b_size = config_.expert_num * config_.hidden_size * lora_rank_;
  memset(grad_down_lora_a, 0, lora_a_size * sizeof(ggml_bf16_t));
  memset(grad_down_lora_b, 0, lora_b_size * sizeof(ggml_bf16_t));
}
```

**修复位置 2**: backward_gate_up LoRA 重置 (lines 681-689)
```cpp
// 重置 gate/up LoRA 梯度 buffer，防止 for-loop 版本累加到 AMX 结果上
if (gate_lora_a_ != nullptr) {
  size_t gate_up_lora_a_size = config_.expert_num * lora_rank_ * config_.hidden_size;
  size_t gate_up_lora_b_size = config_.expert_num * config_.intermediate_size * lora_rank_;
  memset(grad_gate_lora_a, 0, gate_up_lora_a_size * sizeof(ggml_bf16_t));
  memset(grad_gate_lora_b, 0, gate_up_lora_b_size * sizeof(ggml_bf16_t));
  memset(grad_up_lora_a, 0, gate_up_lora_a_size * sizeof(ggml_bf16_t));
  memset(grad_up_lora_b, 0, gate_up_lora_b_size * sizeof(ggml_bf16_t));
}
```

#### 8.4 运行测试命令

```bash
cd /home/lpl/ktransformers/kt-kernel
pip install -e .
python examples/test_moe_sft_amx_no_tp.py 2>&1 | grep -A30 "PER-EXPERT"
```

#### 8.5 预期输出

**场景 1: 单个 expert 有问题**
```
=== PER-EXPERT DIFF ANALYSIS (backward_down) ===
Expert 14 (task 7, m=1): max_diff = 0.976562 at local_pos 1234
  AMX value = 0.123456, for-loop value = 1.100018
=== END PER-EXPERT ANALYSIS ===
```
→ 定位到 Expert 14 有问题
→ 检查该 expert 的 `down_backward_bb_[14]` 转置是否正确

**场景 2: 多个 expert 有问题**
```
=== PER-EXPERT DIFF ANALYSIS (backward_down) ===
Expert 5 (task 2, m=1): max_diff = 0.456789 at local_pos 567
Expert 14 (task 7, m=1): max_diff = 0.976562 at local_pos 1234
Expert 23 (task 12, m=1): max_diff = 0.654321 at local_pos 890
=== END PER-EXPERT ANALYSIS ===
```
→ 多个 expert 有问题
→ 可能是 `prepare_backward_weights()` 的系统性问题
→ 或者是某些特定条件下的 AMX GEMM bug

**场景 3: 无显著 diff**
```
=== PER-EXPERT DIFF ANALYSIS (backward_down) ===
=== END PER-EXPERT ANALYSIS ===
```
→ 所有 expert diff < 0.01
→ Max diff 0.976562 可能来自数组中未使用的区域（padding）
→ 需要检查 offset 计算逻辑

---

## Bug #3: backward_down_amx to_mat 参数错误 【已修复】

**日期**: 2026-01-07

**状态**: ✅ **已修复**

### 1. 根本原因

`backward_down_amx` 的 Step 4 `to_mat()` 使用了错误的参数！

**BF16 kernel 配置**：
- `N_BLOCK = 256`（每个线程处理 256 列）
- `nth = (intermediate_size + 255) / 256`
- 对于 `intermediate_size = 2048`，`nth = 8`（8 个线程并行）

**问题代码（修复前）**：
```cpp
// Step 3: mat_mul（多线程，正确）
int nth = T::recommended_nth(config_.intermediate_size);  // nth = 8
pool->do_work_stealing_job(
    nth * activated_expert, [](int _) { T::config(); },
    [this, nth](int task_id) {
      int ith = task_id % nth;  // ith = 0,1,2,3,4,5,6,7
      amx::mat_mul(..., ith, nth);  // 每个线程计算 256 列
    }, nullptr);

// Step 4: to_mat（单独的任务，参数错误！）
pool->do_work_stealing_job(
    activated_expert, nullptr,
    [this](int task_id) {
      // ❌ 使用 (0, 1) 只输出第一个线程的 256 列！
      grad_intermediate_bc_[expert_idx]->to_mat(m, ptr, 0, 1);
    }, nullptr);
```

**结果**：
- 列 0-255 正确输出（线程 0 的结果）
- 列 256-2047 全为 0！（线程 1-7 的结果丢失）

### 2. 测试输出证据

Per-expert 分析显示所有 expert 在 local_pos >= 256 的位置 AMX 值为 0：
```
Expert 0 (task 0, m=1): max_diff = 0.785156 at local_pos 371
  AMX value = 0.000000, for-loop value = -0.785156
Expert 7 (task 1, m=1): max_diff = 0.427734 at local_pos 1705
  AMX value = 0.000000, for-loop value = -0.427734
...
```

### 3. 修复方案

合并 Step 3 和 Step 4，让 `to_mat` 使用与 `mat_mul` 相同的 `ith, nth`：

**修复后代码** (lines 2086-2127):
```cpp
// Step 3+4: AMX GEMM + to_mat (merged to use same ith/nth)
int nth = T::recommended_nth(config_.intermediate_size);

// Pre-compute offsets for each expert
std::vector<size_t> expert_offsets(activated_expert);
{
  size_t offset = 0;
  for (int i = 0; i < activated_expert; i++) {
    expert_offsets[i] = offset * config_.intermediate_size;
    offset += m_local_num_[m_expert_id_map_[i]];
  }
}

pool->do_work_stealing_job(
    nth * activated_expert, [](int _) { T::config(); },
    [this, nth, &expert_offsets](int task_id) {
      int task_idx = task_id / nth;  // Which expert
      int expert_idx = m_expert_id_map_[task_idx];
      int ith = task_id % nth;
      int m = m_local_num_[expert_idx];

      if (m == 0) return;

      auto& ba = grad_output_ba_[expert_idx];
      auto& bb = down_backward_bb_[expert_idx];
      auto& bc = grad_intermediate_bc_[expert_idx];

      // mat_mul
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);

      // to_mat - use same ith, nth as mat_mul!
      bc->to_mat(m, grad_intermediate_ + expert_offsets[task_idx], ith, nth);
    },
    nullptr);
```

### 4. 修改的文件和行号

| 文件 | 行号 | 修改内容 |
|------|------|----------|
| `sft_moe.hpp` | 2086-2127 | 合并 Step 3 和 Step 4，修复 to_mat 参数 |

### 5. 预期结果

修复后：
- 所有列（0-2047）都会正确输出
- Per-expert diff 应该接近 0（只有数值精度差异）
- backward_down_amx 测试应该通过

---

## Bug #4: gate_backward_bb_ 和 up_backward_bb_ 的 from_mat 参数错误 【已修复】

**日期**: 2026-01-11

**状态**: ✅ **已修复**

### 1. 根本原因

在 `prepare_backward_weights()` 中，`gate_backward_bb_` 和 `up_backward_bb_` 的 `from_mat` 调用使用了错误的参数！

这与 Bug #3 完全相同的问题，只是发生在不同的 BufferB 上。

**问题代码（修复前）**：
```cpp
// case 0: gate_proj
gate_backward_bb_[expert_idx]->from_mat(transposed.data(), 0, 1);  // ❌ 只填充第一个 N_BLOCK

// case 1: up_proj
up_backward_bb_[expert_idx]->from_mat(transposed.data(), 0, 1);  // ❌ 只填充第一个 N_BLOCK

// case 2: down_proj (已修复)
int nth = T::recommended_nth(config_.intermediate_size);
for (int ith = 0; ith < nth; ith++) {
  down_backward_bb_[expert_idx]->from_mat(transposed.data(), ith, nth);  // ✅ 正确
}
```

**在 `backward_gate_up_amx` 中的使用**：
```cpp
int nth = T::recommended_nth(config_.hidden_size);  // hidden_size=7168 → nth=28
amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
// ↑ mat_mul 使用 (ith, 28)，但 BufferB 只用 (0, 1) 填充
```

**结果**：
- 只有前 256 列有数据
- 其他 6912 列全为 0
- 导致 grad_input diff: 0.972656

### 2. 测试输出

accuracy 模式测试失败：
```
============================================================
Testing MOE SFT Backward Pass - BF16 mode (NO TP)
============================================================
--- Iteration 0 ---
grad_input diff: 0.972656
[FAILED] Test failed with error: grad_input accuracy failed: 0.972656
```

### 3. 修复方案

与 Bug #3 的修复相同，为 `gate_backward_bb_` 和 `up_backward_bb_` 使用循环调用 `from_mat`：

**修复后代码** (sft_moe.hpp:829-834, 847-852):
```cpp
// case 0: gate_proj
int nth = T::recommended_nth(config_.hidden_size);  // 使用 hidden_size
for (int ith = 0; ith < nth; ith++) {
  gate_backward_bb_[expert_idx]->from_mat(transposed.data(), ith, nth);
}

// case 1: up_proj
int nth = T::recommended_nth(config_.hidden_size);
for (int ith = 0; ith < nth; ith++) {
  up_backward_bb_[expert_idx]->from_mat(transposed.data(), ith, nth);
}
```

### 4. 关键点

不同 BufferB 使用不同的 nth 计算：

| 矩阵 | 输出维度 N | nth 计算 |
|------|-----------|----------|
| `down_backward_bb_` | intermediate_size (2048) | `recommended_nth(2048)` = 8 |
| `gate_backward_bb_` | hidden_size (7168) | `recommended_nth(7168)` = 28 |
| `up_backward_bb_` | hidden_size (7168) | `recommended_nth(7168)` = 28 |

### 5. 修改的文件和行号

| 文件 | 行号 | 修改内容 |
|------|------|----------|
| `sft_moe.hpp` | 829-834 | `gate_backward_bb_` 使用循环 from_mat |
| `sft_moe.hpp` | 847-852 | `up_backward_bb_` 使用循环 from_mat |

### 6. 预期结果

修复后：
- 所有 7168 列都会正确填充
- grad_input diff 应该接近 0（与 for-loop 版本一致）
- Backward Pass 测试应该通过
