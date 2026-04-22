# SFT-MOE-AMX Real Data NaN Bug 记录

## 问题概述

| 属性 | 值 |
|------|-----|
| 测试文件 | `/home/lpl/ktransformers-llama/kt-kernel/examples/test_moe_sft_amx_no_tp.py` |
| 数据文件 | `/mnt/data/lpl/kt_nan_debug_data.pt` |
| 问题模式 | `--mode real_data` |
| 正常模式 | `--mode accuracy`, `--mode perf` |

### 问题表现

- `mode=accuracy` (随机数据): forward 正确, backward 有小数值差异但无 NaN
- `mode=real_data` (真实训练数据): 产生 47104 个 NaN

### 模型配置

| 参数 | 值 |
|------|-----|
| expert_num | 64 |
| hidden_size | 2048 |
| intermediate_size | 1408 |
| num_experts_per_tok | 6 |
| lora_rank | 8 |
| padded_lora_rank | 32 (对齐到 K_STEP=32) |
| lora_alpha | 16.0 |
| qlen | 48 |

---

## 关键发现

### 1. NaN 只出现在 Expert 17-24

```
[NaN TRACE Step5.5] Expert 17 GATE+LoRA: nan=28, inf=0, first_idx=274
[NaN TRACE Step5.5] Expert 18 GATE+LoRA: nan=15, inf=0, first_idx=583
[NaN TRACE Step5.5] Expert 19 GATE+LoRA: nan=24, inf=0, first_idx=269
[NaN TRACE Step5.5] Expert 20 GATE+LoRA: nan=29, inf=0, first_idx=260
[NaN TRACE Step5.5] Expert 21 GATE+LoRA: nan=22, inf=0, first_idx=283
[NaN TRACE Step5.5] Expert 22 GATE+LoRA: nan=15, inf=0, first_idx=575
[NaN TRACE Step5.5] Expert 23 GATE+LoRA: nan=27, inf=0, first_idx=265
[NaN TRACE Step5.5] Expert 24 GATE+LoRA: nan=27, inf=0, first_idx=263
```

- NaN 首次出现在 **Step 5.5** (base GEMM + LoRA 计算后)
- **只有 Expert 17-24** 这 8 个连续的 expert 有问题
- NaN 位置 (first_idx) 在 260-650 范围内

### 2. PyTorch 参考实现正常

| 实现 | NaN 数量 |
|------|----------|
| AMX C++ | 47104 |
| PyTorch | 0 |

**结论**: 问题 100% 在 C++ 代码中，不在数据本身。

### 3. for-loop 版本也有 NaN

| 版本 | NaN 数量 |
|------|----------|
| AMX 优化版本 | 47104 |
| for-loop 版本 (git:2119584) | 47104 |

**结论**: 问题不在 AMX GEMM 优化本身，而是在公共的 LoRA 数据准备逻辑中。

---

## 排除的原因

### 1. PT 文件数据格式问题 - ❌ 已排除

权重形状验证结果:

| 张量 | 期望形状 | 实际形状 | 状态 |
|------|----------|----------|------|
| gate_proj | (64, 1408, 2048) | (64, 1408, 2048) | ✅ |
| up_proj | (64, 1408, 2048) | (64, 1408, 2048) | ✅ |
| down_proj | (64, 2048, 1408) | (64, 2048, 1408) | ✅ |
| gate_lora_a | (64, 8, 2048) | (64, 8, 2048) | ✅ |
| gate_lora_b | (64, 1408, 8) | (64, 1408, 8) | ✅ |
| up_lora_a | (64, 8, 2048) | (64, 8, 2048) | ✅ |
| up_lora_b | (64, 1408, 8) | (64, 1408, 8) | ✅ |
| down_lora_a | (64, 8, 1408) | (64, 8, 1408) | ✅ |
| down_lora_b | (64, 2048, 8) | (64, 2048, 8) | ✅ |

### 2. LoRA B 全零问题 - ❌ 已排除

测试脚本: `test_lora_b_zero_issue.py`

| 测试 | 结果 |
|------|------|
| AMX (LoRA B = 0) | 47104 NaN |
| AMX (LoRA B = 非零) | 47104 NaN |
| PyTorch (LoRA B = 0) | 0 NaN |
| PyTorch (LoRA B = 非零) | 0 NaN |

**结论**: 问题与 LoRA B 的值无关。

### 3. TP 分区复制逻辑问题 - ❌ 已排除

测试脚本: `test_partition_data.py`

Python 模拟 TP 分区复制后，所有 Expert 的分区数据与原始数据完全一致。

Expert 17-24 的内存偏移分析:
```
Expert 17: offset = 191488 to 202752 (size = 11264)
Expert 18: offset = 202752 to 214016 (size = 11264)
...
Expert 24: offset = 270336 to 281600 (size = 11264)
总数据大小: 720896
Expert 24 结束位置: 281600
是否越界: False
```

### 4. Expert 17-24 原始数据问题 - ❌ 已排除

测试脚本: `debug_expert_17_24.py`

Expert 17-24 的原始数据检查:
- 无 NaN
- 无 Inf
- 数值范围正常

手动 Python 计算 Expert 17-24 的 forward:
- 所有 Expert 输出均无 NaN

### 5. 配置参数问题 - ❌ 已排除

accuracy 模式使用相同的 real_data 配置 (2048/1408)，测试通过。

---

## 调试进展详情

### 第一轮调试：验证源数据 [已排除]

**日期**: 2026-01-10

在 `convert_lora_b_to_buffer_b` 函数中添加调试输出，验证源数据。

**结果**:
```
[BUG-A Debug] Expert 17: src_offset=191488, nan_in_src=0, nan_in_padded=0
[BUG-A Debug] Expert 24: src_offset=270336, nan_in_src=0, nan_in_padded=0
[BUG-A Debug] Expert 25: src_offset=281600, nan_in_src=0, nan_in_padded=0
```

**结论**: 所有 Expert 的源数据和 padded 数据都无 NaN，问题不在原始数据。

---

### 第二轮调试：定位 NaN 引入位置 [重大发现]

**日期**: 2026-01-10

在 `compute_lora_gate_up_amx` 的 Step 1 和 Step 3 添加调试输出。

**Step 1 输出 (input @ lora_A^T) - 全部正常**:
```
[BUG-A Debug Step1] Expert 17 GATE intermediate: m=1, padded_rank=32, nan_count=0
[BUG-A Debug Step1] Expert 18 GATE intermediate: m=3, padded_rank=32, nan_count=0
[BUG-A Debug Step1] Expert 25 GATE intermediate: m=6, padded_rank=32, nan_count=0
```

**Step 3 BufferC (intermediate @ lora_B^T) - NaN 在此出现**:
```
[BUG-A Debug GEMM] Expert 17 GATE BufferC after GEMM: m=1, nan_count=23
[BUG-A Debug GEMM] Expert 18 GATE BufferC after GEMM: m=3, nan_count=15
[BUG-A Debug GEMM] Expert 19 GATE BufferC after GEMM: m=2, nan_count=4
[BUG-A Debug GEMM] Expert 20 GATE BufferC after GEMM: m=3, nan_count=27
[BUG-A Debug GEMM] Expert 21 GATE BufferC after GEMM: m=2, nan_count=4
[BUG-A Debug GEMM] Expert 22 GATE BufferC after GEMM: m=7, nan_count=80
[BUG-A Debug GEMM] Expert 23 GATE BufferC after GEMM: m=3, nan_count=15
[BUG-A Debug GEMM] Expert 24 GATE BufferC after GEMM: m=7, nan_count=40
[BUG-A Debug GEMM] Expert 25 GATE BufferC after GEMM: m=6, nan_count=0  ← 正常！
```

**关键发现**:
1. ✅ Step 1 (input @ lora_A^T) 输出正常 → lora_A GEMM 无问题
2. ❌ Step 3 (intermediate @ lora_B^T) 输出异常 → **问题在 lora_B 相关的 GEMM**
3. ✅ **Expert 25 完全正常** (nan_count=0)
4. ❌ **Expert 17-24 都有 NaN**
5. Expert 16 未激活 (不在本次测试的 token 分配中)

---

## 问题定位

### 已确认的问题范围

问题出现在 **Step 3: intermediate @ lora_B^T** 的 GEMM 计算中:

```cpp
// Step 3 in compute_lora_gate_up_amx
amx::mat_mul(m, config_.intermediate_size, padded_lora_rank_, ba, bb, bc, ith, nth);
// C[m,1408] = A[m,32] @ B[1408,32]^T
```

涉及的数据结构:
- `ba`: `lora_gate_intermediate_ba_[expert_idx]` - 已验证正常 (Step 1 输出)
- `bb`: `gate_lora_b_bb_[expert_idx]` - **可疑！**
- `bc`: `lora_gate_out_bc_[expert_idx]` - 输出有 NaN

### 待验证假设

#### 假设 1: 矩阵转置/存储布局问题 [待验证]

用户怀疑矩阵存储方式与 AMX 计算方式不匹配。

**分析**:
- LoRA B 原始形状: `[expert_num=64, intermediate_size=1408, lora_rank=8]`
- Padded 形状: `[1408, 32]` (row-major)
- BufferB 期望: GEMM 中作为 `B[1408, 32]^T` 使用

**需要验证**:
- `BufferB::from_mat()` 如何解释输入数据的行/列
- 转换后的 BufferB 内部布局是否正确

#### 假设 2: Expert 索引特殊性 [待验证]

Expert 17-24 正好是 8 个连续 expert:
- 17 = 0x11, 24 = 0x18
- 8 个 expert 可能与某种分块大小 (如 AMX tile 16x16) 相关

#### 假设 3: BufferB 内存问题 [待验证]

可能 Expert 17-24 的 BufferB:
- 未正确分配
- 被其他数据覆盖
- 初始化不完整

---

### 第三轮调试：深入检查 BufferB 和 GEMM 输入 [进行中]

**日期**: 2026-01-10

#### 代码分析

**BufferB::from_mat 分析** (`amx_raw_buffers.hpp:136-157`):
```cpp
void from_mat(ggml_bf16_t* src, int ith, int nth) {
  // 遍历 n_begin (0 到 n_block_size, 步长 N_STEP=32)
  // 遍历 k_block (0 到 k, 步长 K_BLOCK)
  // 对每行复制 K_STEP=32 个 BF16 值，然后进行 16x16 transpose
}
```
- 源偏移: `(n_begin + i) * k`
- 目标偏移: `n_begin * k_block_size + i * K_STEP`
- 对于 n=1408, k=32: 写入 1408 * 32 * 2 = 90112 bytes
- **结论**: from_mat 逻辑正确，完整覆盖所有内存位置 ✅

**convert_lora_b_to_buffer_b 分析** (`sft_moe.hpp:1491-1530`):
- 创建 padded 临时数组，初始化为 0
- 处理 k 维度 padding (8 -> 32)
- **结论**: 转换逻辑正确 ✅

**BufferB 内存分配分析** (`sft_moe.hpp:1343-1345`):
- 每个 expert 独立分配 90112 bytes
- **结论**: 无内存重叠 ✅

#### 新增调试代码

**位置 1**: `convert_lora_b_to_buffer_b` 中 from_mat 后 (sft_moe.hpp:1531-1543)
```cpp
// BUG-A Debug: Check BufferB data AFTER from_mat
if (expert_idx >= 16 && expert_idx <= 25) {
  int nan_count = 0;
  size_t total_elements = (size_t)dst_n * dst_k;
  for (size_t i = 0; i < total_elements; i++) {
    float val = GGML_BF16_TO_FP32(dst_bb->b[i]);
    if (std::isnan(val) || std::isinf(val)) nan_count++;
  }
  printf("[BUG-A Debug] Expert %d BufferB after from_mat: total_elements=%zu, nan_count=%d\n",
         expert_idx, total_elements, nan_count);
}
```

**位置 2**: Step 3 GEMM 前 (sft_moe.hpp:1693-1715)
```cpp
// BUG-A Debug: Check inputs BEFORE GEMM for Expert 16-25
if (ith == 0 && !do_up && expert_idx >= 16 && expert_idx <= 25) {
  // Check BufferB (gate_lora_b_bb_)
  int bb_nan = 0;
  for (size_t i = 0; i < bb_total; i++) {
    float val = GGML_BF16_TO_FP32(bb->b[i]);
    if (std::isnan(val) || std::isinf(val)) bb_nan++;
  }
  // Check BufferA (lora_intermediate_ba_)
  int ba_nan = 0;
  // ... check through get_submat ...
  printf("[BUG-A Debug Step3 Input] Expert %d GATE: m=%d, ba_nan=%d, bb_nan=%d\n",
         expert_idx, m, ba_nan, bb_nan);
}
```

#### 期望输出

运行测试后应显示:
```
[BUG-A Debug] Expert 17 BufferB after from_mat: total_elements=45056, nan_count=?
[BUG-A Debug Step3 Input] Expert 17 GATE: m=1, ba_nan=?, bb_nan=?
[BUG-A Debug GEMM] Expert 17 GATE BufferC after GEMM: m=1, nan_count=?
```

#### 分析逻辑

| BufferB after from_mat | Step3 Input (ba, bb) | GEMM Output | 结论 |
|------------------------|----------------------|-------------|------|
| nan_count=0 | ba_nan=0, bb_nan=0 | nan_count>0 | GEMM 内部 bug |
| nan_count>0 | - | - | from_mat bug |
| nan_count=0 | bb_nan>0 | - | 内存污染 |
| nan_count=0 | ba_nan>0 | - | Step 2 量化 bug |

---

## 相关文件

| 文件 | 描述 |
|------|------|
| `sft_moe.hpp` | AMX SFT MOE 核心实现 |
| `moe-sft-tp.hpp` | TP 包装器 |
| `amx_raw_buffers.hpp` | BufferA/B/C 定义 |
| `debug_expert_17_24.py` | Expert 17-24 数据分析 |
| `test_lora_b_zero_issue.py` | LoRA B 全零测试 |
| `test_partition_data.py` | TP 分区逻辑验证 |

---

## 时间线

| 日期 | 进展 |
|------|------|
| 2026-01-10 | 初步定位 NaN 出现在 Step 5.5，只有 Expert 17-24 |
| 2026-01-10 | 排除 PT 文件格式、LoRA B 全零、TP 分区逻辑等原因 |
| 2026-01-10 | 确认问题在 C++ 代码的 LoRA 计算路径中 |
| 2026-01-10 | 第一轮调试: 验证源数据和 padded 数据 → 全部正常 |
| 2026-01-10 | 第二轮调试: 定位 NaN 在 Step 3 GEMM (lora_B) 引入 |
| 2026-01-10 | 第三轮调试: 添加 BufferB after from_mat 和 GEMM 输入检查 |
| 2026-01-10 | 发现 if constexpr 类型检查错误 → Expert 18 行为不一致 |
| 2026-01-10 | 修复类型检查 + 增强调试输出 |

---

## 第三轮调试结果 [2026-01-10] ⚠️ 重大发现

### 3.1 调试输出分析

**所有输入数据都是干净的**:
```
[BUG-A Debug] Expert XX: nan_in_src=0, nan_in_padded=0  ← 全部干净
[BUG-A Debug Step1] Expert XX: nan_count=0              ← 全部干净
```

**GEMM 输出**:
```
Expert 17: nan_count=26
Expert 18: nan_count=0   ← 新发现！之前有 NaN，现在干净
Expert 19: nan_count=6
Expert 20: nan_count=48
Expert 21: nan_count=7
Expert 22: nan_count=17
Expert 23: nan_count=22
Expert 24: nan_count=21
Expert 25: nan_count=0   ← 依然干净
```

### 3.2 关键发现

1. ✅ 所有输入数据都是干净的（源数据、padded 数据、Step 1 输出）
2. ❌ NaN 在 **Step 3 GEMM (`amx::mat_mul`)** 计算后首次出现
3. ⚠️ **Expert 18 行为不稳定**：之前有 NaN，现在干净
4. ⚠️ BufferB/BufferA 检查没有输出 → `if constexpr` 类型检查失败

### 3.3 类型检查修复

**问题**: 原来的类型检查使用了错误的类型
```cpp
// 错误的检查 - 永远返回 false
if constexpr (std::is_same_v<typename T::BufferB, amx::BufferBBF16Impl<T>>)
```

**原因分析**:
- `T` = `amx::GemmKernel224BF`
- `T::BufferB` = `GemmKernel224BF::BufferB` (嵌套结构体)
- `amx::BufferBBF16Impl<T>` 是独立的模板类
- 两者永远不相等！

**修复**:
```cpp
// 正确的检查 - 判断是否为 BF16 kernel
if constexpr (std::is_same_v<T, amx::GemmKernel224BF>)
```

**修改位置**:
1. `sft_moe.hpp:1532` - convert_lora_b_to_buffer_b 后
2. `sft_moe.hpp:1696` - Step 3 GEMM 前

### 3.4 增强的调试输出

除了 NaN 检查，现在还输出:
- 零值数量 (zero_count)
- 数值范围 (min, max)
- 第一个 NaN 的位置 (m, n 坐标)

### 3.5 问题结论

**Expert 18 不稳定行为强烈暗示**:
- **BufferC 未初始化** - GEMM 输出缓冲区可能包含垃圾数据
- **竞态条件** - 多线程并行执行时的竞争

### 3.6 待执行的修复方案 ← 已过时，见第四轮调试

**方案 A**: 在 GEMM 前初始化 BufferC 为 0 ❌ 不是根本原因
**方案 B**: 单线程执行排除竞态 ❌ 不是根本原因
**方案 C**: 检查 mat_mul 内部累加逻辑 ❌ 不是根本原因

---

## 第四轮调试结果 [2026-01-10] 🔴 根本原因确认

### 4.1 关键调试输出

在 `convert_lora_b_to_buffer_b` 的 `from_mat` 调用**前**添加调试输出：

```
[BEFORE from_mat] Expert 17: zeros=21632, nan=38, range=[-3.35e+38, 3.39e+38], total=45056
[BEFORE from_mat] Expert 17 padded[0:8]: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000

[BEFORE from_mat] Expert 25: zeros=45056, nan=0, range=[0.00e+00, 0.00e+00], total=45056
[BEFORE from_mat] Expert 25 padded[0:8]: 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
```

### 4.2 🔴 关键发现

| Expert | BEFORE from_mat | padded 源数据 | 问题 |
|--------|-----------------|---------------|------|
| 17 | zeros=21632, nan=38 | 全零 | **BufferB 内存已被污染！** |
| 25 | zeros=45056, nan=0 | 全零 | ✓ 干净 |

**结论**: Expert 17 的 BufferB 在 `from_mat` 调用**之前**就已经包含垃圾数据（包括 NaN）！

这说明:
1. ✅ 源数据 (padded) 没有问题 - 全是零
2. ✅ `from_mat` 函数本身没有 bug
3. ❌ **BufferB 的内存区域被其他代码污染**

### 4.3 根本原因：`shared_mem_buffer.alloc` 内存共享问题

用户回忆起之前遇到过类似问题：

> "多次调用 `shared_mem_buffer.alloc` 实际是给不同的指针分配同一片内存空间（例如两个不会同时调用的函数可以共用同一块空间），这样可以节省内存。但如果两个缓冲区实际上会同时使用，就会产生数据污染。"

**问题机制**:
1. `shared_mem_buffer.alloc` 是一种内存优化机制
2. 它会给**不会同时使用的缓冲区**分配**同一片物理内存**
3. 但如果这些缓冲区实际上**会同时使用**，就会互相覆盖

**本次问题的具体表现**:
- `lora_bb_pool_` (LoRA BufferB 内存池) 通过 `mem_requests.append_pointer()` 分配
- 这导致它与其他缓冲区共享了内存空间
- 当其他代码写入这片共享内存时，Expert 17-24 的 BufferB 区域被污染
- Expert 25 的区域恰好未被覆盖（可能是内存布局的偶然）

### 4.4 修复方案

**修改文件**: `operators/amx/sft_moe.hpp`

**原来的分配方式** (通过 mem_requests，会导致内存共享):
```cpp
// 在 MOE_Base::compute_mem_requests() 中
mem_requests.append_pointer(&lora_bb_pool_, lora_bb_pool_bytes_);
mem_requests.append_pointer(&lora_ba_pool_, lora_ba_pool_bytes_);
mem_requests.append_pointer(&lora_bc_inter_pool_, lora_bc_inter_pool_bytes_);
mem_requests.append_pointer(&lora_bc_out_pool_, lora_bc_out_pool_bytes_);
mem_requests.append_pointer(&lora_intermediate_bf16_pool_, lora_intermediate_bf16_pool_bytes_);
```

**修复后的分配方式** (独立分配，避免内存共享):
```cpp
// 在 init() 中使用 aligned_alloc 独立分配
if (lora_bb_pool_bytes_ > 0) {
  lora_bb_pool_ = aligned_alloc(64, lora_bb_pool_bytes_);
  memset(lora_bb_pool_, 0, lora_bb_pool_bytes_);
}
if (lora_ba_pool_bytes_ > 0) {
  lora_ba_pool_ = aligned_alloc(64, lora_ba_pool_bytes_);
  memset(lora_ba_pool_, 0, lora_ba_pool_bytes_);
}
if (lora_bc_inter_pool_bytes_ > 0) {
  lora_bc_inter_pool_ = aligned_alloc(64, lora_bc_inter_pool_bytes_);
  memset(lora_bc_inter_pool_, 0, lora_bc_inter_pool_bytes_);
}
if (lora_bc_out_pool_bytes_ > 0) {
  lora_bc_out_pool_ = aligned_alloc(64, lora_bc_out_pool_bytes_);
  memset(lora_bc_out_pool_, 0, lora_bc_out_pool_bytes_);
}
if (lora_intermediate_bf16_pool_bytes_ > 0) {
  lora_intermediate_bf16_pool_ = aligned_alloc(64, lora_intermediate_bf16_pool_bytes_);
  memset(lora_intermediate_bf16_pool_, 0, lora_intermediate_bf16_pool_bytes_);
}
```

**析构函数更新**:
```cpp
~AMX_SFT_MOE_TP() {
  // Bug-A Fix: 释放使用 aligned_alloc 分配的 LoRA 缓冲区
  if (lora_bb_pool_) free(lora_bb_pool_);
  if (lora_ba_pool_) free(lora_ba_pool_);
  if (lora_bc_inter_pool_) free(lora_bc_inter_pool_);
  if (lora_bc_out_pool_) free(lora_bc_out_pool_);
  if (lora_intermediate_bf16_pool_) free(lora_intermediate_bf16_pool_);
}
```

### 4.5 修复预期效果

修复后:
1. Expert 17-24 的 BufferB 将在 `from_mat` 前是干净的（全零）
2. `from_mat` 将正确复制 padded 数据到 BufferB
3. GEMM 计算将产生正确结果，无 NaN

### 4.6 为什么是 Expert 17-24？

8 个连续 expert (17-24) 受影响的原因推测：
- 内存池按 expert 顺序分配
- 共享内存的"其他用户"写入的数据大小恰好覆盖了 Expert 17-24 的区域
- Expert 0-16 和 25-63 的区域可能未被覆盖，或被覆盖但恰好是合法值

---

## 总结

### Bug-A 根本原因

**根本原因**: `shared_mem_buffer.alloc` 内存共享机制导致 LoRA BufferB 内存池与其他缓冲区共享了物理内存，其他代码写入时污染了 Expert 17-24 的 BufferB 数据。

**表现**: Expert 17-24 的 BufferB 在数据复制 (`from_mat`) 前就已包含垃圾数据（包括 NaN），导致后续 GEMM 计算产生 NaN 输出。

**修复**: 将 LoRA 相关的内存池从 `mem_requests.append_pointer()` 改为 `aligned_alloc()` 独立分配，确保 LoRA 缓冲区拥有专属的内存空间。

### 关键教训

1. `shared_mem_buffer.alloc` 是一种内存优化机制，**只适用于不会同时使用的缓冲区**
2. 如果缓冲区会在 forward/backward 过程中同时存在，必须使用独立分配
3. 调试时检查 **写入前** 的内存状态很重要，可以区分是"写入逻辑错误"还是"内存被污染"

---

## 时间线 (更新)

| 日期 | 进展 |
|------|------|
| 2026-01-10 | 初步定位 NaN 出现在 Step 5.5，只有 Expert 17-24 |
| 2026-01-10 | 排除 PT 文件格式、LoRA B 全零、TP 分区逻辑等原因 |
| 2026-01-10 | 确认问题在 C++ 代码的 LoRA 计算路径中 |
| 2026-01-10 | 第一轮调试: 验证源数据和 padded 数据 → 全部正常 |
| 2026-01-10 | 第二轮调试: 定位 NaN 在 Step 3 GEMM (lora_B) 引入 |
| 2026-01-10 | 第三轮调试: 添加 BufferB after from_mat 和 GEMM 输入检查 |
| 2026-01-10 | 第四轮调试: **发现 BufferB 在 from_mat 前就有垃圾！** |
| 2026-01-10 | **🔴 根本原因确认: shared_mem_buffer 内存共享问题** |
| 2026-01-10 | **修复: 将 lora pool 改为 aligned_alloc 独立分配** |
| 2026-01-11 | **✅ Bug-A 修复验证通过** |

---

## 第五轮调试结果 [2026-01-11] ✅ Bug-A 修复验证

### 5.1 测试结果

修复后运行 `test_moe_sft_amx_no_tp.py --mode real_data`：

| 指标 | 结果 |
|------|------|
| PyTorch Reference NaN | 0 ✅ |
| AMX Implementation NaN | 0 ✅ |
| Max diff | 0.500000 |
| Mean diff | 0.004038 |

**结论**: Bug-A (NaN 问题) 已完全修复。

### 5.2 精度验证

采用与 accuracy mode 相同的验证方式：

```python
# 相对误差计算
threshold = BF16_FORWARD_THRESHOLD  # 0.05
diff = mean(abs(amx - torch)) / (mean(abs(torch)) + 1e-8)
assert diff < threshold
```

### 5.3 调试代码清理

修复验证后，已清理所有 Bug-A 相关调试代码：
- C++ 调试打印 (`[Bug-A Debug]`, `[BEFORE from_mat]`, etc.)
- 测试文件中的 original 对比代码

### 5.4 修复总结

| 问题 | 根本原因 | 修复方案 |
|------|----------|----------|
| Expert 17-24 产生 NaN | `shared_mem_buffer.alloc` 内存共享导致 BufferB 被污染 | 将 LoRA 缓冲区改为 `aligned_alloc` 独立分配 |

---

## Bug-A 状态: ✅ 已解决

---

# Bug-C: accuracy 模式内存问题

## 问题概述

| 属性 | 值 |
|------|-----|
| 触发条件 | 运行 `python test_moe_sft_amx_no_tp.py --mode accuracy` |
| 问题表现 | 首次创建 MOE 对象时内存持续增长到 300+ GB |
| 关联 | Bug-A 修复的副作用 |

---

## 🔴 为什么 Bug-A 修复导致内存增加

### Bug-A 修复内容回顾

为了解决 Expert 17-24 的 NaN 问题，将 LoRA 缓冲区从 `shared_mem_buffer` 共享池改为 `aligned_alloc` 独立分配：

```cpp
// 修复前 (447dd6b): 通过 shared_mem_buffer 分配
mem_requests.append_pointer(&lora_bb_pool_, lora_bb_pool_bytes_);
mem_requests.append_pointer(&lora_ba_pool_, lora_ba_pool_bytes_);
// ... 所有缓冲区都用 mem_requests
shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

// 修复后: LoRA 缓冲区独立分配
lora_bb_pool_ = aligned_alloc(64, lora_bb_pool_bytes_);
lora_ba_pool_ = aligned_alloc(64, lora_ba_pool_bytes_);
// ... 其他缓冲区仍用 mem_requests
shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
```

### shared_mem_buffer 的内存复用机制

`shared_mem_buffer` 是一种内存优化机制：

```cpp
// shared_mem_buffer.cpp:49-72
void SharedMemBuffer::alloc(void* object, MemoryRequest requests) {
  size_t total_size = requests.total_size();
  object_requests.push_back(requests);

  if (total_size > size) {
    // 只有当请求大于当前缓冲区时才重新分配
    if (buffer) free(buffer);
    posix_memalign(&newbuf, 64, total_size);
    buffer = newbuf;
    size = total_size;
    // 更新所有已注册对象的指针
    for (auto& req : object_requests) {
      req.update_base_ptr(buffer);
    }
  } else {
    // 复用现有缓冲区！
    requests.update_base_ptr(buffer);
  }
}
```

**关键点**：
1. 多个对象的 `append_pointer` 请求会**共享同一片物理内存**
2. 只要总大小不超过已分配的大小，就会复用内存
3. 这在**缓冲区不会同时使用**时是安全的内存优化
4. 但如果缓冲区**会同时使用**，就会产生数据污染（Bug-A 的根因）

### 内存增加的原因

| 方面 | 修复前 (shared_mem_buffer) | 修复后 (aligned_alloc) |
|------|--------------------------|------------------------|
| LoRA 缓冲区分配 | 与其他缓冲区共享内存 | 独立内存空间 |
| 内存复用 | ✅ 高效（多个缓冲区共用） | ❌ 无复用（独立分配） |
| NaN 问题 | ❌ 内存污染导致 NaN | ✅ 无污染 |
| 内存占用 | 低（复用） | 高（独立） |

**结论**：Bug-A 修复是必要的（否则有 NaN），但它暴露了原本被"隐藏"的内存需求问题。

---

## accuracy 模式配置分析

```python
# test_moe_sft_amx_no_tp.py:40-44
expert_num = 256          # 专家数量 (vs real_data: 64)
hidden_size = 7168        # 隐藏维度 (vs real_data: 2048)
intermediate_size = 2048  # MLP 中间维度 (vs real_data: 1408)
max_len = 25600          # 最大序列长度
num_experts_per_tok = 8   # 每 token 激活的专家数
```

### 问题 1: max_m 计算错误

```cpp
// sft_moe.hpp:935 (修复前)
size_t max_m = ((config_.max_len * config_.num_experts_per_tok + M_STEP - 1) / M_STEP) * M_STEP;
            = ((25600 * 8 + 63) / 64) * 64 = 204,800  // 错误！

// 正确计算: 每个 expert 最多处理 max_len 个 token
size_t max_m = ((config_.max_len + M_STEP - 1) / M_STEP) * M_STEP;
            = ((25600 + 63) / 64) * 64 = 25,600  // 正确
```

**影响**: 内存需求差 8 倍

### 问题 2: 每个 expert 独立分配大缓冲区

原始代码为每个 expert 都分配 max_m 大小的缓冲区：

```cpp
// 每个 expert 都分配 max_m × output_dim 的 BufferC
lora_bc_out_pool_bytes_ = config_.expert_num * (lora_gate_up_out_bc_size * 2 + lora_down_out_bc_size);
//                      = 256 × (大尺寸) = 巨大内存
```

而实际上，所有 256 个 expert **共享**同一组 token（最多 max_len 个），应该用**共享池**而不是独立分配。

---

## 修复完成 [2026-01-11] ✅ 成功

### 已实现的修改

#### Step 1: 修正 max_m 计算 ✅

```cpp
// sft_moe.hpp:935
// 修改前: max_m = max_len * num_experts_per_tok = 25600 × 8 = 204800 (错误!)
// 修改后: max_m = max_len = 25600 (正确: 每个 expert 最多处理 max_len 个 token)
size_t max_m = ((config_.max_len + M_STEP - 1) / M_STEP) * M_STEP;
```

#### Step 2: 使用共享缓冲区池 ✅

修改了以下部分：
1. `init_all_buffers` 中的池大小计算 (sft_moe.hpp:980-1021)
2. `init_lora_amx_buffers` 使用 nullptr 初始化 BufferA/BufferC (sft_moe.hpp:1219-1253)
3. `compute_lora_gate_up_amx` / `compute_lora_down_amx` 动态分配
4. `backward_down_amx` / `backward_gate_up_amx` 动态分配

### 测试结果 ✅

```
========== Memory Allocation Summary ==========
Config: expert_num=256, hidden_size=7168, intermediate_size=2048
Config: max_len=25600, num_experts_per_tok=8, lora_rank=16, padded_lora_rank=32
Calculated max_m=25600, max_total_tokens=204800

--- LoRA Buffers (aligned_alloc) ---
  lora_bb_pool_bytes_:              754,974,720 bytes (720.00 MB)
  lora_ba_pool_bytes_:               26,214,400 bytes ( 25.00 MB)
  lora_bc_inter_pool_bytes_:         52,428,800 bytes ( 50.00 MB)
  lora_bc_out_pool_bytes_:        9,227,468,800 bytes (  8.59 GB)
  lora_intermediate_bf16_pool_bytes_: 26,214,400 bytes ( 25.00 MB)

--- Backward Buffers (shared_mem_buffer) ---
  backward_ba_pool_bytes_:        2,936,012,800 bytes (  2.73 GB)
  backward_bc_pool_bytes_:        7,549,747,200 bytes (  7.03 GB)
  grad_output_bf16_pool_bytes_:   2,936,012,800 bytes (  2.73 GB)
  backward_bb_pool_bytes_:       22,548,578,304 bytes ( 21.00 GB)

--- Other Buffers (shared_mem_buffer) ---
  lora_intermediate_pool_bytes_:      6,553,600 bytes (  0.01 GB)
  grad_buffer_bytes (×3):         2,516,582,400 bytes (  2.34 GB)
  cache_total (depth=1):          2,883,584,000 bytes (  2.69 GB)

--- Summary ---
  Total aligned_alloc:           10,087,301,120 bytes (  9.39 GB)
  Total shared_mem_buffer:       41,377,071,104 bytes ( 38.54 GB)
  GRAND TOTAL:                   51,464,372,224 bytes ( 47.93 GB)
===============================================
```

内存需求约 **48 GB**，与理论计算一致。

---

## 内存计算公式

### 配置参数

| 参数 | 符号 | accuracy 模式值 |
|------|------|-----------------|
| 专家数量 | E | 256 |
| 隐藏维度 | H | 7168 |
| MLP 中间维度 | I | 2048 |
| 最大序列长度 | L | 25600 |
| 每 token 激活专家数 | K | 8 |
| LoRA rank | R | 16 |
| Padded LoRA rank | R' | 32 (对齐到 K_STEP=32) |

### 计算公式

```
max_m = align64(L) = 25600
max_total_tokens = L × K = 204800

--- LoRA 缓冲区 (aligned_alloc) ---
lora_bb_pool = E × (BufferB(R', H) × 2 + BufferB(I, R') × 2 +
                    BufferB(H, R') × 2 + BufferB(R', I) × 2 +
                    BufferB(R', I) + BufferB(H, R'))
             ≈ 720 MB

lora_ba_pool = BufferA(max_total_tokens, R') × 2
             = 204800 × 32 × 2 × 2 = 26 MB

lora_bc_inter_pool = BufferC(max_total_tokens, R') × 2
                   = 204800 × 32 × 4 × 2 = 52 MB

lora_bc_out_pool = BufferC(max_total_tokens, I) × 2 + BufferC(max_total_tokens, H)
                 = (204800 × 2048 × 4 × 2) + (204800 × 7168 × 4)
                 = 3.35 GB + 5.87 GB = 8.59 GB (实测)

lora_intermediate_bf16_pool = max_total_tokens × R' × 2 × 2 = 26 MB

--- Backward 缓冲区 (shared_mem_buffer) ---
backward_ba_pool = BufferA(max_total_tokens, H)
                 = 204800 × 7168 × 2 = 2.73 GB

backward_bc_pool = BufferC(max_total_tokens, I) + BufferC(max_total_tokens, H)
                 = (204800 × 2048 × 4) + (204800 × 7168 × 4)
                 = 1.67 GB + 5.87 GB = 7.03 GB (实测)

grad_output_bf16_pool = max_total_tokens × H × 2 = 2.73 GB

backward_bb_pool = E × (BufferB(H, I) × 2 + BufferB(I, H))
                 ≈ 21 GB

--- 其他缓冲区 ---
grad_buffer × 3 = L × K × I × 2 × 3 = 2.34 GB
cache_total = (L × H × 2 + L × K × I × 2 × 3) × depth
            = (367 MB + 2.52 GB) × 1 = 2.69 GB
```

### 总计

| 类别 | 大小 |
|------|------|
| LoRA (aligned_alloc) | ~9.4 GB |
| Backward (shared_mem_buffer) | ~38.5 GB |
| **总计** | **~47.9 GB** |

---

## Bug-C 状态: ✅ 已解决

### 修复总结

| 问题 | 原因 | 修复方案 | 效果 |
|------|------|----------|------|
| max_m 计算错误 | 错误地乘以 num_experts_per_tok | 改为 max_len 直接对齐 | 内存从 ~4 TB 降到 ~500 GB |
| 每个 expert 独立分配 | 为每个 expert 分配 max_m 大小缓冲区 | 使用共享池，forward/backward 时动态分配 | 内存从 ~500 GB 降到 ~48 GB |

### 关键代码位置

| 文件 | 位置 | 修改内容 |
|------|------|----------|
| sft_moe.hpp:935 | init_all_buffers | max_m 计算修正 |
| sft_moe.hpp:980-1021 | init_all_buffers | 池大小计算 |
| sft_moe.hpp:1219-1253 | init_lora_amx_buffers | Buffer 初始化为 nullptr |
| sft_moe.hpp:1392-1444 | compute_lora_gate_up_amx | 动态分配 |
| sft_moe.hpp:1538-1573 | compute_lora_down_amx | 动态分配 |
| sft_moe.hpp:2111-2141 | backward_down_amx | 动态分配 |
| sft_moe.hpp:2653-2723 | backward_gate_up_amx | 动态分配 |
