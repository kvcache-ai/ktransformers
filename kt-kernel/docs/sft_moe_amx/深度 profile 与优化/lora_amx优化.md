# LoRA 梯度计算 AMX 优化

## 1. 优化目标

根据 `profile_result.md` 的分析，LoRA 梯度计算是 backward pass 的主要瓶颈：

| 位置 | 耗时 | 占比 | 问题 |
|------|------|------|------|
| D4_lora_grad | 72.4 ms | 22.6% | for-loop 实现 |
| GU3_lora_passes | 99.6 ms | 31.1% | for-loop + 频繁同步 |
| **总计** | **172 ms** | **53.7%** | 超过一半的 backward 时间 |

## 2. 优化策略

### 2.1 For-loop 分析

原始实现中的 6 个 for-loop：

| 位置 | 操作 | 形式 | 数学表示 |
|------|------|------|----------|
| D4 Loop 1 | `inter_proj = cached_intermediate @ lora_A^T` | A @ B^T | 可用 AMX |
| D4 Loop 2 | `grad_B = grad_output^T @ inter_proj` | A^T @ B | 需向量化 |
| D4 Loop 3 | `grad_times_b = grad_output @ lora_B` | A @ B | 可转为 A @ B^T |
| D4 Loop 4 | `grad_A = cached_intermediate^T @ grad_times_b` | A^T @ B | 需向量化 |
| lora_pass Step 2 | `grad_B = grad^T @ U` | A^T @ B | 需向量化 |
| lora_pass Step 6 | `grad_A = input^T @ G_B` | A^T @ B | 需向量化 |

### 2.2 AMX mat_mul 约束

```cpp
// AMX mat_mul 签名
void mat_mul(int m, int n, int k,
             std::shared_ptr<BufferA> ba,  // [m, k]
             std::shared_ptr<BufferB> bb,  // [n, k] (内部存储为转置)
             std::shared_ptr<BufferC> bc,  // [m, n]
             int ith, int nth);
// 计算: C[m,n] = A[m,k] @ B[n,k]^T
```

**关键约束**: AMX mat_mul 只能计算 `A @ B^T`，没有转置参数。

### 2.3 优化方案

| 形式 | 优化方案 | 说明 |
|------|----------|------|
| A @ B^T | 直接使用 AMX GEMM | 匹配 mat_mul 计算模式 |
| A @ B | 预转置 B → B^T，然后 AMX GEMM | 权重可预转置 |
| A^T @ B | **AVX-512 向量化** | 激活值每次不同，转置开销大 |

## 3. 实现细节

### 3.1 Phase 1: 添加转置权重 BufferB

#### 3.1.1 新增成员变量

**文件**: `sft_moe.hpp` (约 line 205-209)

```cpp
// Down LoRA transposed BufferB for backward GEMM
std::vector<std::shared_ptr<typename T::BufferB>>
    down_lora_a_t_bb_;  // [expert_num] [intermediate_size, padded_lora_rank]
std::vector<std::shared_ptr<typename T::BufferB>>
    down_lora_b_t_bb_;  // [expert_num] [padded_lora_rank, hidden_size]
```

#### 3.1.2 Buffer 大小计算

**文件**: `sft_moe.hpp` (约 line 893-995)

```cpp
// 新增 size 变量
size_t lora_a_down_t_bb_size = 0;  // down_lora_a^T: [intermediate_size, padded_lora_rank]
size_t lora_b_down_t_bb_size = 0;  // down_lora_b^T: [padded_lora_rank, hidden_size]

// 计算大小
if constexpr (supports_standard_mat_mul_v<T>) {
  // ... existing calculations ...
  lora_a_down_t_bb_size = T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_);
  lora_b_down_t_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.hidden_size);
}

// 更新 lora_bb_pool_bytes_，增加 2 个 BufferB
lora_bb_pool_bytes_ = config_.expert_num * (
    lora_a_gate_bb_size + lora_b_gate_bb_size +
    lora_a_up_bb_size + lora_b_up_bb_size +
    lora_a_down_bb_size + lora_b_down_bb_size +
    lora_a_gate_t_bb_size + lora_b_gate_t_bb_size +
    lora_a_up_t_bb_size + lora_b_up_t_bb_size +
    lora_a_down_t_bb_size +   // NEW
    lora_b_down_t_bb_size     // NEW
);
```

#### 3.1.3 初始化 BufferB

**文件**: `sft_moe.hpp` `init_lora_amx_buffers()` (约 line 1139-1265)

```cpp
// Resize vectors
down_lora_a_t_bb_.resize(config_.expert_num);
down_lora_b_t_bb_.resize(config_.expert_num);

// 在 per-expert loop 中初始化
for (int i = 0; i < config_.expert_num; i++) {
  // ... existing initializations ...

  down_lora_a_t_bb_[i] = std::make_shared<typename T::BufferB>(
      config_.intermediate_size, padded_lora_rank_, (void*)bb_ptr);
  bb_ptr += lora_a_down_t_bb_size;

  down_lora_b_t_bb_[i] = std::make_shared<typename T::BufferB>(
      padded_lora_rank_, config_.hidden_size, (void*)bb_ptr);
  bb_ptr += lora_b_down_t_bb_size;
}
```

#### 3.1.4 权重转换

**文件**: `sft_moe.hpp` `prepare_lora_weights()` (约 line 732-741)

```cpp
// 更新 job count: 10 → 12
pool->do_work_stealing_job(
    config_.expert_num * 12, nullptr,  // 从 10 增加到 12
    [this](int task_id) {
      int expert_idx = task_id / 12;
      int lora_type = task_id % 12;

      switch (lora_type) {
        // ... cases 0-9 ...

        case 10:  // down_lora_a^T [lora_rank, intermediate_size] -> [intermediate_size, padded_lora_rank]
          convert_lora_a_transposed_to_buffer_b(down_lora_a_, down_lora_a_t_bb_[expert_idx], expert_idx,
                                                lora_rank_, config_.intermediate_size,
                                                config_.intermediate_size, padded_lora_rank_);
          break;
        case 11:  // down_lora_b^T [hidden_size, lora_rank] -> [padded_lora_rank, hidden_size]
          convert_lora_b_transposed_to_buffer_b(down_lora_b_, down_lora_b_t_bb_[expert_idx], expert_idx,
                                                config_.hidden_size, lora_rank_,
                                                padded_lora_rank_, config_.hidden_size);
          break;
      }
    }, nullptr);
```

### 3.2 Phase 2: D4_lora_grad 优化

#### 3.2.1 添加中间缓冲区

**文件**: `sft_moe.hpp` (约 line 267-274)

```cpp
// D4_lora_grad intermediate buffers (for AMX GEMM optimization)
std::vector<ggml_bf16_t*> d4_inter_proj_ptr_;     // [expert_num], [max_m, padded_lora_rank]
std::vector<ggml_bf16_t*> d4_grad_times_b_ptr_;   // [expert_num], [max_m, padded_lora_rank]
void* d4_intermediate_pool_ = nullptr;
size_t d4_intermediate_pool_bytes_ = 0;
```

**内存分配** (约 line 992-1050):

```cpp
// D4_lora_grad intermediate buffers (inter_proj and grad_times_b)
// Each buffer: [max_m, padded_lora_rank] in BF16
d4_intermediate_pool_bytes_ = config_.expert_num * max_m * padded_lora_rank_ * sizeof(ggml_bf16_t) * 2;

// 分配并初始化指针
d4_intermediate_pool_ = std::aligned_alloc(64, d4_intermediate_pool_bytes_);
char* d4_ptr = (char*)d4_intermediate_pool_;
size_t d4_buffer_size = max_m * padded_lora_rank_ * sizeof(ggml_bf16_t);
for (int i = 0; i < config_.expert_num; i++) {
  d4_inter_proj_ptr_[i] = (ggml_bf16_t*)d4_ptr;
  d4_ptr += d4_buffer_size;
  d4_grad_times_b_ptr_[i] = (ggml_bf16_t*)d4_ptr;
  d4_ptr += d4_buffer_size;
}
```

#### 3.2.2 向量化辅助函数

**文件**: `sft_moe.hpp` (约 line 1395-1455)

```cpp
/**
 * @brief Vectorized computation: grad_B = grad_output^T @ inter_proj
 *
 * Computes: grad_B[H, R] += sum_t grad_output[t, H] * inter_proj[t, R] * scaling
 * Where H = hidden_size, R = lora_rank, T = num_tokens
 */
void compute_grad_B_down_vectorized(int num_tokens, const float* grad_output, const ggml_bf16_t* inter_proj,
                                    ggml_bf16_t* grad_B, size_t lora_b_offset, float scaling) {
  for (int h = 0; h < config_.hidden_size; h++) {
    for (int r = 0; r < lora_rank_; r++) {
      float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
      for (int t = 0; t < num_tokens; t++) {
        sum += grad_output[t * config_.hidden_size + h] * GGML_BF16_TO_FP32(inter_proj[t * padded_lora_rank_ + r]);
      }
      size_t idx = lora_b_offset + h * lora_rank_ + r;
      float current = GGML_BF16_TO_FP32(grad_B[idx]);
      grad_B[idx] = GGML_FP32_TO_BF16(current + sum * scaling);
    }
  }
}

/**
 * @brief Vectorized computation: grad_A = intermediate^T @ grad_times_b
 *
 * Computes: grad_A[R, I] += sum_t intermediate[t, I] * grad_times_b[t, R] * scaling
 * Where I = intermediate_size, R = lora_rank, T = num_tokens
 */
void compute_grad_A_down_vectorized(int num_tokens, const ggml_bf16_t* intermediate, const ggml_bf16_t* grad_times_b,
                                    ggml_bf16_t* grad_A, size_t lora_a_offset, float scaling) {
  for (int r = 0; r < lora_rank_; r++) {
    for (int i = 0; i < config_.intermediate_size; i++) {
      float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
      for (int t = 0; t < num_tokens; t++) {
        sum += GGML_BF16_TO_FP32(intermediate[t * config_.intermediate_size + i]) *
               GGML_BF16_TO_FP32(grad_times_b[t * padded_lora_rank_ + r]);
      }
      size_t idx = lora_a_offset + r * config_.intermediate_size + i;
      float current = GGML_BF16_TO_FP32(grad_A[idx]);
      grad_A[idx] = GGML_FP32_TO_BF16(current + sum * scaling);
    }
  }
}
```

#### 3.2.3 D4_lora_grad 5 阶段并行结构

**文件**: `sft_moe.hpp` (约 line 2267-2377)

原始结构：单个 `do_work_stealing_job`，4 个 for-loop 串行执行。

**新结构**: 5 个独立阶段，每个阶段一个 `do_work_stealing_job`:

```cpp
// =====================================================
// Step 5: LoRA gradient computation (AMX + vectorized optimization)
// =====================================================
// Optimized 5-stage parallel implementation:
// Stage 1: Quantize cached_intermediate → BufferA
// Stage 2: GEMM: inter_proj = cached_intermediate @ lora_A^T
// Stage 3: GEMM: grad_times_b = grad_output @ lora_B
// Stage 4: Vectorized: grad_B = grad_output^T @ inter_proj
// Stage 5: Vectorized: grad_A = intermediate^T @ grad_times_b
// =====================================================

if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
  // Pre-compute cache offsets
  std::vector<size_t> d4_cache_offsets(activated_expert);
  {
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      d4_cache_offsets[i] = offset;
      offset += m_local_num_[m_expert_id_map_[i]];
    }
  }

  // Stage 1: Quantize cached_intermediate to BufferA (reuse down_ba_)
  pool->do_work_stealing_job(
      activated_expert, nullptr,
      [this, &cache, &d4_cache_offsets](int task_id) {
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        const ggml_bf16_t* cached_intermediate =
            cache.intermediate_cache + d4_cache_offsets[task_id] * config_.intermediate_size;
        down_ba_[expert_idx]->from_mat(num_tokens, (ggml_bf16_t*)cached_intermediate, 0, 1);
      },
      nullptr);

  // Stage 2: GEMM inter_proj = cached_intermediate @ lora_A^T
  // mat_mul: [num_tokens, intermediate_size] @ [padded_lora_rank, intermediate_size]^T = [num_tokens, padded_lora_rank]
  int nth_lora = T::recommended_nth(padded_lora_rank_);
  pool->do_work_stealing_job(
      nth_lora * activated_expert, [](int _) { T::config(); },
      [this, nth_lora](int task_id) {
        int task_idx = task_id / nth_lora;
        int expert_idx = m_expert_id_map_[task_idx];
        int ith = task_id % nth_lora;
        int m = m_local_num_[expert_idx];
        if (m == 0) return;

        amx::mat_mul(m, padded_lora_rank_, config_.intermediate_size, down_ba_[expert_idx], down_lora_a_bb_[expert_idx],
                     lora_gate_intermediate_bc_[expert_idx], ith, nth_lora);
        lora_gate_intermediate_bc_[expert_idx]->to_mat(m, d4_inter_proj_ptr_[expert_idx], ith, nth_lora);
      },
      nullptr);

  // Stage 3: GEMM grad_times_b = grad_output @ lora_B
  // mat_mul: [num_tokens, hidden_size] @ [padded_lora_rank, hidden_size]^T = [num_tokens, padded_lora_rank]
  // Note: down_lora_b_t_bb_ stores lora_B^T [padded_lora_rank, hidden_size]
  pool->do_work_stealing_job(
      nth_lora * activated_expert, [](int _) { T::config(); },
      [this, nth_lora](int task_id) {
        int task_idx = task_id / nth_lora;
        int expert_idx = m_expert_id_map_[task_idx];
        int ith = task_id % nth_lora;
        int m = m_local_num_[expert_idx];
        if (m == 0) return;

        // grad_output already quantized in grad_output_ba_ from D2_quantize
        amx::mat_mul(m, padded_lora_rank_, config_.hidden_size, grad_output_ba_[expert_idx],
                     down_lora_b_t_bb_[expert_idx], lora_up_intermediate_bc_[expert_idx], ith, nth_lora);
        lora_up_intermediate_bc_[expert_idx]->to_mat(m, d4_grad_times_b_ptr_[expert_idx], ith, nth_lora);
      },
      nullptr);

  // Stage 4: Vectorized grad_B = grad_output^T @ inter_proj
  pool->do_work_stealing_job(
      activated_expert, nullptr,
      [this, grad_down_b](int task_id) {
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Convert grad_output to float32 for vectorized computation
        ggml_bf16_t* expert_grad_bf16 = grad_output_bf16_ptr_[expert_idx];
        std::vector<float> expert_grad_out(num_tokens * config_.hidden_size);
        for (int i = 0; i < num_tokens * config_.hidden_size; i++) {
          expert_grad_out[i] = GGML_BF16_TO_FP32(expert_grad_bf16[i]);
        }

        size_t lora_b_offset = expert_idx * config_.hidden_size * lora_rank_;
        compute_grad_B_down_vectorized(num_tokens, expert_grad_out.data(), d4_inter_proj_ptr_[expert_idx], grad_down_b,
                                       lora_b_offset, lora_scaling_);
      },
      nullptr);

  // Stage 5: Vectorized grad_A = intermediate^T @ grad_times_b
  pool->do_work_stealing_job(
      activated_expert, nullptr,
      [this, &cache, &d4_cache_offsets, grad_down_a](int task_id) {
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        const ggml_bf16_t* cached_intermediate =
            cache.intermediate_cache + d4_cache_offsets[task_id] * config_.intermediate_size;
        size_t lora_a_offset = expert_idx * lora_rank_ * config_.intermediate_size;

        compute_grad_A_down_vectorized(num_tokens, cached_intermediate, d4_grad_times_b_ptr_[expert_idx], grad_down_a,
                                       lora_a_offset, lora_scaling_);
      },
      nullptr);

  DOWN_CHECKPOINT("D4_lora_grad");
}
```

### 3.3 Phase 3: lora_pass 优化

#### 3.3.1 向量化辅助函数

**文件**: `sft_moe.hpp` (约 line 1457-1515)

```cpp
/**
 * @brief Vectorized computation: grad_B = grad^T @ U (for gate/up lora_pass)
 *
 * Computes: grad_B[I, R] += sum_t grad[t, I] * U[t, R] * scaling
 * Where I = intermediate_size, R = lora_rank, T = num_tokens
 */
void compute_grad_B_gate_up_vectorized(int num_tokens, const ggml_bf16_t* grad, const ggml_bf16_t* u_ptr,
                                       ggml_bf16_t* grad_B, size_t lora_b_offset, float scaling) {
  for (int i = 0; i < config_.intermediate_size; i++) {
    for (int r = 0; r < lora_rank_; r++) {
      float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
      for (int t = 0; t < num_tokens; t++) {
        sum += GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]) *
               GGML_BF16_TO_FP32(u_ptr[t * padded_lora_rank_ + r]);
      }
      size_t idx = lora_b_offset + i * lora_rank_ + r;
      float current = GGML_BF16_TO_FP32(grad_B[idx]);
      grad_B[idx] = GGML_FP32_TO_BF16(current + sum * scaling);
    }
  }
}

/**
 * @brief Vectorized computation: grad_A = input^T @ G_B (for gate/up lora_pass)
 *
 * Computes: grad_A[R, H] += sum_t input[t, H] * G_B[t, R] * scaling
 * Where R = lora_rank, H = hidden_size, T = num_tokens
 */
void compute_grad_A_gate_up_vectorized(int num_tokens, const ggml_bf16_t* expert_input, const ggml_bf16_t* g_ptr,
                                       ggml_bf16_t* grad_A, size_t lora_a_offset, float scaling) {
  for (int r = 0; r < lora_rank_; r++) {
    for (int h = 0; h < config_.hidden_size; h++) {
      float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
      for (int t = 0; t < num_tokens; t++) {
        sum += GGML_BF16_TO_FP32(expert_input[t * config_.hidden_size + h]) *
               GGML_BF16_TO_FP32(g_ptr[t * padded_lora_rank_ + r]);
      }
      size_t idx = lora_a_offset + r * config_.hidden_size + h;
      float current = GGML_BF16_TO_FP32(grad_A[idx]);
      grad_A[idx] = GGML_FP32_TO_BF16(current + sum * scaling);
    }
  }
}
```

#### 3.3.2 Step 2 优化

**文件**: `sft_moe.hpp` (约 line 2845-2851)

**原始代码**:
```cpp
for (int i = 0; i < config_.intermediate_size; i++) {
  for (int r = 0; r < lora_rank_; r++) {
    float sum = 0.0f;
    for (int t = 0; t < num_tokens; t++) {
      float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
      float u = GGML_BF16_TO_FP32(u_ptr[t * padded_lora_rank_ + r]);
      sum += g * u;
    }
    float current = GGML_BF16_TO_FP32(grad_lora_b[lora_b_offset + i * lora_rank_ + r]);
    grad_lora_b[lora_b_offset + i * lora_rank_ + r] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
  }
}
```

**优化后**:
```cpp
// Vectorized: grad_B[I,R] = sum_t grad[t,I] * U[t,R]
compute_grad_B_gate_up_vectorized(num_tokens, grad, u_ptr, grad_lora_b, lora_b_offset, lora_scaling_);
```

#### 3.3.3 Step 6 优化

**文件**: `sft_moe.hpp` (约 line 2940-2944)

**原始代码**:
```cpp
for (int r = 0; r < lora_rank_; r++) {
  for (int h = 0; h < config_.hidden_size; h++) {
    float sum = 0.0f;
    for (int t = 0; t < num_tokens; t++) {
      float gb = GGML_BF16_TO_FP32(g_ptr[t * padded_lora_rank_ + r]);
      float inp = GGML_BF16_TO_FP32(expert_input[t * config_.hidden_size + h]);
      sum += inp * gb;
    }
    float current = GGML_BF16_TO_FP32(grad_lora_a[lora_a_offset + r * config_.hidden_size + h]);
    grad_lora_a[lora_a_offset + r * config_.hidden_size + h] =
        GGML_FP32_TO_BF16(current + sum * lora_scaling_);
  }
}
```

**优化后**:
```cpp
// Vectorized: grad_A[R,H] = sum_t input[t,H] * G_B[t,R]
compute_grad_A_gate_up_vectorized(num_tokens, expert_input, g_ptr, grad_lora_a, lora_a_offset,
                                  lora_scaling_);
```

## 4. 优化效果分析

### 4.1 理论性能提升

| 阶段 | 原始方式 | 优化方式 | 预期提升 |
|------|----------|----------|----------|
| D4 Loop 1 | for-loop | AMX GEMM | ~4x |
| D4 Loop 2 | for-loop | OpenMP SIMD | ~2x |
| D4 Loop 3 | for-loop | AMX GEMM | ~4x |
| D4 Loop 4 | for-loop | OpenMP SIMD | ~2x |
| lora_pass Step 2 | for-loop | OpenMP SIMD | ~2x |
| lora_pass Step 6 | for-loop | OpenMP SIMD | ~2x |

### 4.2 为什么 A^T @ B 使用向量化而非 AMX？

对于 `A^T @ B` 形式的矩阵乘法：

1. **AMX mat_mul 只支持 A @ B^T**
2. **要使用 AMX 需要显式转置**:
   - 将 A 转置为 A^T 存储到新 Buffer
   - 或将 B 转置为 B^T 存储到 BufferB
3. **激活值每次不同**: A (grad, input) 是激活值，每次 backward 都不同
4. **转置开销**: 显式转置 `[num_tokens, dim]` 的矩阵需要 O(num_tokens * dim) 次内存访问
5. **向量化优势**: OpenMP SIMD 可以直接在原始数据布局上工作，避免转置开销

**结论**: 对于 A^T @ B 形式，OpenMP SIMD 向量化是更好的选择。

## 5. 修改文件汇总

| 文件 | 修改内容 |
|------|----------|
| `sft_moe.hpp` | 新增成员变量、Buffer 分配、权重转换、D4_lora_grad 重构、lora_pass 向量化 |

### 5.1 关键代码位置

| 功能 | 大约行号 |
|------|----------|
| 新成员变量 (down_lora_*_t_bb_) | 205-209 |
| D4 中间缓冲区变量 | 267-274 |
| Buffer 大小计算 | 893-995 |
| init_lora_amx_buffers | 1139-1265 |
| 向量化辅助函数 (down) | 1395-1455 |
| 向量化辅助函数 (gate/up) | 1457-1515 |
| prepare_lora_weights cases 10-11 | 732-741 |
| D4_lora_grad 5 阶段实现 | 2267-2377 |
| lora_pass Step 2 向量化调用 | 2845-2851 |
| lora_pass Step 6 向量化调用 | 2940-2944 |

## 6. 后续优化方向

### 6.1 进一步减少同步

当前 D4_lora_grad 有 5 个 `do_work_stealing_job` 调用。可以考虑：
- 合并 Stage 2 和 Stage 3（两个独立的 GEMM）
- 合并 Stage 4 和 Stage 5（两个独立的向量化计算）

### 6.2 更高级的向量化

当前使用 `#pragma omp simd` 依赖编译器自动向量化。可以考虑：
- 手写 AVX-512 intrinsics
- 使用 `_mm512_fmadd_ps` 等指令显式向量化

### 6.3 内存预取

对于大矩阵运算，添加预取指令可能进一步提升性能：
```cpp
_mm_prefetch((const char*)&grad[...], _MM_HINT_T0);
```

## 7. 测试验证

运行测试验证优化后的正确性：

```bash
cd /home/lpl/ktransformers/kt-kernel
source /mnt/data/lpl/anaconda3/bin/activate ref-llama
python examples/test_moe_sft_amx.py
```

测试项包括：
- Forward Pass 精度测试
- Backward Pass 精度测试 (grad_input, grad_lora_a/b)
- 性能对比测试
