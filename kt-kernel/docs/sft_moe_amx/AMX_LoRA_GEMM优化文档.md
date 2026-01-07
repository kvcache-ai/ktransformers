# SFT MOE LoRA GEMM AMX 优化文档

## 1. 问题背景

### 1.1 性能问题
原始实现中，LoRA 计算使用朴素的三重嵌套 for 循环实现 GEMM：

```cpp
// compute_lora_gate_up() 中的原始实现 (lines 600-627)
for (int t = 0; t < num_tokens; t++) {
  for (int r = 0; r < lora_rank_; r++) {
    float sum = 0.0f;
    for (int h = 0; h < config_.hidden_size; h++) {  // hidden_size = 7168!
      float inp = GGML_BF16_TO_FP32(input[t * config_.hidden_size + h]);
      float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.hidden_size + h]);
      sum += inp * w;
    }
    local_intermediate[t * lora_rank_ + r] = sum;
  }
}
```

这种实现的问题：
- 无法利用 AMX 的 16×16 tile 并行
- 内存访问模式不是 VNNI 友好的
- 与推理算子（`moe.hpp`）使用 AMX GEMM 形成性能差距

### 1.2 推理算子的正确模式

推理算子（`moe.hpp`）使用 BufferA/BufferB/BufferC + amx::mat_mul：

```cpp
// 推理算子的 GEMM 模式
gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
```

## 2. 优化方案

### 2.1 Padding 策略

AMX 的硬件约束要求维度对齐：
- K_STEP = 32 (BF16 模式)
- N_STEP = 32
- M_STEP = 32

LoRA rank 通常较小（如 16），不满足 K_STEP 约束。

**解决方案：Padding**
```cpp
int padded_lora_rank = ((lora_rank + 31) / 32) * 32;  // 16 → 32
```

### 2.2 LoRA GEMM 两步计算

LoRA 计算公式：
```
output = input @ W^T + (input @ A^T @ B^T) * (alpha / rank)
```

分解为两步 GEMM：
1. **Step 1**: `intermediate = input @ lora_A^T`
   - M = num_tokens, N = padded_lora_rank, K = hidden_size
   - 复用 `gate_up_ba_`（已量化的 input）

2. **Step 2**: `lora_output = intermediate @ lora_B^T`
   - M = num_tokens, N = output_dim, K = padded_lora_rank
   - 输出累加到主输出

## 3. 代码实现

### 3.1 新增成员变量

位置：`sft_moe.hpp` private section

```cpp
// Padded lora_rank for AMX alignment
int padded_lora_rank_;

// LoRA weight BufferB
std::vector<std::shared_ptr<typename T::BufferB>> gate_lora_a_bb_;  // [expert_num]
std::vector<std::shared_ptr<typename T::BufferB>> up_lora_a_bb_;    // [expert_num]
std::vector<std::shared_ptr<typename T::BufferB>> down_lora_a_bb_;  // [expert_num]
std::vector<std::shared_ptr<typename T::BufferB>> gate_lora_b_bb_;  // [expert_num]
std::vector<std::shared_ptr<typename T::BufferB>> up_lora_b_bb_;    // [expert_num]
std::vector<std::shared_ptr<typename T::BufferB>> down_lora_b_bb_;  // [expert_num]

// LoRA intermediate BufferA and BufferC
std::vector<std::shared_ptr<typename T::BufferA>> lora_intermediate_ba_;  // [expert_num]
std::vector<std::shared_ptr<typename T::BufferC>> lora_intermediate_bc_;  // [expert_num]

// LoRA step 2 output BufferC
std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_out_bc_;  // [expert_num]
std::vector<std::shared_ptr<typename T::BufferC>> lora_up_out_bc_;    // [expert_num]
std::vector<std::shared_ptr<typename T::BufferC>> lora_down_out_bc_;  // [expert_num]

// LoRA intermediate BF16 pointers (for step 1 → step 2)
std::vector<ggml_bf16_t*> lora_intermediate_ptr_;  // [expert_num]

// Buffer pools
void* lora_bb_pool_;
void* lora_ba_pool_;
void* lora_bc_inter_pool_;
void* lora_bc_out_pool_;
void* lora_intermediate_bf16_pool_;

// Backward pass buffers
std::vector<std::shared_ptr<typename T::BufferA>> grad_output_ba_;
std::vector<std::shared_ptr<typename T::BufferC>> grad_intermediate_bc_;
std::vector<std::shared_ptr<typename T::BufferC>> grad_gate_up_bc_;
std::vector<ggml_bf16_t*> grad_output_bf16_ptr_;
```

### 3.2 `init_all_buffers()` 修改

1. 计算 padded_lora_rank：
```cpp
constexpr int K_STEP = T::K_STEP;
constexpr int N_STEP = T::N_STEP;
padded_lora_rank_ = ((lora_rank_ + K_STEP - 1) / K_STEP) * K_STEP;
padded_lora_rank_ = std::max(padded_lora_rank_, ((lora_rank_ + N_STEP - 1) / N_STEP) * N_STEP);
```

2. 计算 buffer 大小：
```cpp
size_t lora_a_gate_up_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.hidden_size);
size_t lora_b_gate_up_bb_size = T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_);
// ...
```

3. 添加到 MemoryRequest 并分配

### 3.3 `prepare_lora_weights()` 方法

将 BF16 LoRA 权重转换为 AMX BufferB 格式：

```cpp
void prepare_lora_weights() {
  if (lora_weights_prepared_) return;
  if (gate_lora_a_ == nullptr) return;

  auto pool = config_.pool->get_subpool(tp_part_idx);

  pool->do_work_stealing_job(
      config_.expert_num * 6, nullptr,
      [this](int task_id) {
        int expert_idx = task_id / 6;
        int lora_type = task_id % 6;

        switch (lora_type) {
          case 0:  // gate_lora_a
            convert_lora_a_to_buffer_b(gate_lora_a_, gate_lora_a_bb_[expert_idx],
                                       expert_idx, lora_rank_, config_.hidden_size,
                                       padded_lora_rank_, config_.hidden_size);
            break;
          // ... 其他 5 个矩阵
        }
      },
      nullptr);

  lora_weights_prepared_ = true;
}
```

### 3.4 `compute_lora_gate_up_amx()` 方法

AMX 优化版本的 LoRA gate/up 计算：

```cpp
void compute_lora_gate_up_amx(int qlen, int activated_expert) {
  if (gate_lora_a_ == nullptr) return;

  auto pool = config_.pool->get_subpool(tp_part_idx);
  prepare_lora_weights();

  // Step 1: input @ lora_A^T → lora_intermediate
  int nth = T::recommended_nth(padded_lora_rank_);
  pool->do_work_stealing_job(
      nth * activated_expert * 2, [](int _) { T::config(); },
      [this, nth](int task_id2) {
        int expert_idx = m_expert_id_map_[(task_id2 / 2) / nth];
        bool do_up = task_id2 % 2;
        int ith = (task_id2 / 2) % nth;
        int m = m_local_num_[expert_idx];

        if (m == 0) return;

        auto& ba = gate_up_ba_[expert_idx];  // 复用已量化的 input
        auto& bb = do_up ? up_lora_a_bb_[expert_idx] : gate_lora_a_bb_[expert_idx];
        auto& bc = lora_intermediate_bc_[expert_idx];

        amx::mat_mul(m, padded_lora_rank_, config_.hidden_size, ba, bb, bc, ith, nth);
        bc->to_mat(m, lora_intermediate_ptr_[expert_idx], ith, nth);
      },
      nullptr);

  // Step 2: Quantize lora_intermediate to BufferA
  pool->do_work_stealing_job(
      activated_expert, nullptr,
      [this](int task_id) {
        int expert_idx = m_expert_id_map_[task_id];
        int m = m_local_num_[expert_idx];
        if (m == 0) return;
        lora_intermediate_ba_[expert_idx]->from_mat(m, lora_intermediate_ptr_[expert_idx], 0, 1);
      },
      nullptr);

  // Step 3: lora_intermediate @ lora_B^T → add to main output
  nth = T::recommended_nth(config_.intermediate_size);
  pool->do_work_stealing_job(
      nth * activated_expert * 2, [](int _) { T::config(); },
      [this, nth](int task_id2) {
        int expert_idx = m_expert_id_map_[(task_id2 / 2) / nth];
        bool do_up = task_id2 % 2;
        int ith = (task_id2 / 2) % nth;
        int m = m_local_num_[expert_idx];

        if (m == 0) return;

        auto& ba = lora_intermediate_ba_[expert_idx];
        auto& bb = do_up ? up_lora_b_bb_[expert_idx] : gate_lora_b_bb_[expert_idx];
        auto& bc = do_up ? lora_up_out_bc_[expert_idx] : lora_gate_out_bc_[expert_idx];

        amx::mat_mul(m, config_.intermediate_size, padded_lora_rank_, ba, bb, bc, ith, nth);

        ggml_bf16_t* main_output = do_up ? m_local_up_output_ptr_[expert_idx]
                                         : m_local_gate_output_ptr_[expert_idx];
        add_lora_output_to_main(bc.get(), main_output, m, config_.intermediate_size,
                                lora_scaling_, ith, nth);
      },
      nullptr);
}
```

### 3.5 `compute_lora_down_amx()` 方法

类似于 gate/up，但使用 down 相关的权重和 buffer。

### 3.6 `add_lora_output_to_main()` 辅助方法

使用 AVX-512 将 LoRA BufferC 输出加到主输出：

```cpp
void add_lora_output_to_main(typename T::BufferC* bc, ggml_bf16_t* main_output,
                              int m, int n, float scaling, int ith, int nth) {
  auto [n_start, n_end] = T::split_range_n(n, ith, nth);

  for (int m_i = 0; m_i < m; m_i++) {
    for (int n_i = n_start; n_i < n_end; n_i += 32) {
      float* c_ptr = bc->get_submat(m, n, m_i, n_i);

      __m512 main0, main1;
      avx512_32xbf16_to_32xfp32((__m512i*)(main_output + m_i * n + n_i), &main0, &main1);

      __m512 scale = _mm512_set1_ps(scaling);
      __m512 lora0 = _mm512_load_ps(c_ptr);
      __m512 lora1 = _mm512_load_ps(c_ptr + 16);
      main0 = _mm512_fmadd_ps(lora0, scale, main0);
      main1 = _mm512_fmadd_ps(lora1, scale, main1);

      avx512_32xfp32_to_32xbf16(&main0, &main1, (__m512i*)(main_output + m_i * n + n_i));
    }
  }
}
```

## 4. 调用流程

### 4.1 Forward Pass

```
forward_sft()
  ├── Step 1-4: Expert routing, buffer allocation, input scatter, input quantization
  ├── Step 5: Gate + Up GEMM (base weights)
  ├── Step 5.5: Gate + Up LoRA (AMX-optimized)
  │     └── compute_lora_gate_up_amx()
  │           ├── prepare_lora_weights()  // 首次调用时转换权重
  │           ├── Step 1: input @ lora_A^T (AMX GEMM)
  │           ├── Step 2: Quantize intermediate
  │           └── Step 3: intermediate @ lora_B^T + accumulate (AMX GEMM)
  ├── Step 6: Activation (silu(gate) * up)
  ├── Step 7-8: Quantize intermediate, Down GEMM
  ├── Step 8.5: Down LoRA (AMX-optimized)
  │     └── compute_lora_down_amx()
  └── Step 9: Weighted merge
```

### 4.2 LoRA 权重更新

当调用 `update_lora_weights()` 时：
```cpp
void update_lora_weights(...) {
  // 更新权重指针
  gate_lora_a_ = (ggml_bf16_t*)gate_lora_a;
  // ...

  // 标记需要重新转换
  lora_weights_prepared_ = false;
}
```

下次 forward 时会自动调用 `prepare_lora_weights()` 重新转换。

## 5. 内存布局

### 5.1 BufferB 格式 (LoRA 权重)

原始 LoRA 权重：`[expert_num, lora_rank, hidden_size]`

转换为 BufferB 格式：
1. Padding 到 `[expert_num, padded_lora_rank, hidden_size]`
2. 内部使用 VNNI 格式重排

### 5.2 BufferA 格式 (输入)

对于 BF16 模式（GemmKernel224BF），BufferA 直接使用 BF16 数据，按 K_BLOCK 分块存储。

### 5.3 BufferC 格式 (输出)

FP32 累加，按 M_STEP × N_STEP 分块存储。

## 6. 性能预期

| 操作 | 原始实现 | AMX 优化 | 提升 |
|------|----------|----------|------|
| compute_lora_gate_up | O(tokens × rank × hidden_size) 标量 | AMX 16×16 tile | ~10-50x |
| compute_lora_down | O(tokens × rank × intermediate_size) 标量 | AMX 16×16 tile | ~10-50x |

## 7. Backward Pass 优化

### 7.1 Backward BufferB 成员变量

反向传播 GEMM 需要转置版本的基础权重：

```cpp
// Forward: input @ W^T 使用 gate_bb_[intermediate_size, hidden_size]
// Backward: grad @ W 需要 BufferB[hidden_size, intermediate_size]

std::vector<std::shared_ptr<typename T::BufferB>> gate_backward_bb_;  // [hidden_size, intermediate_size]
std::vector<std::shared_ptr<typename T::BufferB>> up_backward_bb_;    // [hidden_size, intermediate_size]
std::vector<std::shared_ptr<typename T::BufferB>> down_backward_bb_;  // [intermediate_size, hidden_size]
```

### 7.2 `prepare_backward_weights()` 方法

将基础权重转换为转置的 BufferB 格式：

```cpp
void prepare_backward_weights() {
  if (backward_weights_prepared_) return;

  // 并行转换 gate_proj^T, up_proj^T, down_proj^T
  // 每个 expert 3 个矩阵
  pool->do_work_stealing_job(
      config_.expert_num * 3, nullptr,
      [this](int task_id) {
        // 对每个矩阵进行转置并转换为 BufferB 格式
      },
      nullptr);

  backward_weights_prepared_ = true;
}
```

### 7.3 `backward_down_amx()` 方法

AMX 优化的 backward_down：

```
backward_down_amx()
  ├── prepare_backward_weights()  // 首次调用时转换权重
  ├── Step 1: Scatter grad_output to per-expert BF16 buffers
  ├── Step 2: Quantize grad_output to BufferA
  ├── Step 3: AMX GEMM: grad_intermediate = grad_output @ down_proj
  │     └── mat_mul(m, intermediate_size, hidden_size, ba, down_backward_bb_, bc)
  ├── Step 4: Convert BufferC to grad_intermediate_ BF16
  └── Step 5: LoRA gradient computation (for-loop, small matrices)
```

### 7.4 `backward_gate_up_amx()` 方法

AMX 优化的 backward_gate_up：

```
backward_gate_up_amx()
  ├── prepare_backward_weights()
  └── For each expert (gate and up in parallel):
        ├── AMX GEMM: token_grad_input = grad @ W
        │     └── mat_mul with gate_backward_bb_/up_backward_bb_
        ├── Scatter to grad_input
        └── LoRA gradient computation (for-loop)
```

## 8. 已知限制

1. **Padding 开销**：当 lora_rank 不是 32 的倍数时，有额外的零填充计算
2. **LoRA 梯度计算**：backward 中的 LoRA 梯度仍使用 for 循环（矩阵较小，AMX 优化收益有限）
3. **TP 模式**：需要在 `update_lora_weights()` 后调用 `prepare_lora_weights()`
4. **内存开销**：backward BufferB 需要额外存储转置权重

## 9. 测试

使用 `examples/test_moe_sft_amx.py` 运行测试：

```bash
cd /home/lpl/ktransformers/kt-kernel
pip install -e . --no-build-isolation
python examples/test_moe_sft_amx.py
```

测试会验证：
1. 前向传播精度（与 PyTorch 参考实现对比）
2. 反向传播精度
3. 不同量化模式（bf16, int8 等）

## 10. API 概览

### 10.1 Forward Pass

| 方法 | 描述 | AMX 优化 |
|------|------|----------|
| `forward_sft()` | 完整前向传播 | ✓ 基础 GEMM |
| `compute_lora_gate_up_amx()` | LoRA gate/up 计算 | ✓ |
| `compute_lora_down_amx()` | LoRA down 计算 | ✓ |

### 10.2 Backward Pass

| 方法 | 描述 | AMX 优化 |
|------|------|----------|
| `backward_down()` | 原始实现 | ✗ for 循环 |
| `backward_down_amx()` | AMX 优化版 | ✓ 主 GEMM |
| `backward_activation()` | 激活函数梯度 | N/A (element-wise) |
| `backward_gate_up()` | 原始实现 | ✗ for 循环 |
| `backward_gate_up_amx()` | AMX 优化版 | ✓ 主 GEMM |

### 10.3 权重准备

| 方法 | 描述 | 自动调用 |
|------|------|----------|
| `prepare_lora_weights()` | 转换 LoRA 权重到 BufferB | forward 时 |
| `prepare_backward_weights()` | 转换基础权重到转置 BufferB | backward 时 |

## 11. Kernel 类型适配

### 11.1 问题背景

不同的 GemmKernel 类型有不同的 API：

**支持 `amx::mat_mul()` 的 Kernel**:
- `GemmKernel224BF`
- `GemmKernel224Int8`
- `GemmKernel224Int4`
- `GemmKernel224Int4_1`

**不支持的 Kernel（使用 `mat_mul_kgroup()`）**:
- `GemmKernel224Int4KGroup`
- `GemmKernel224Int4_1KGroup`
- `GemmKernel224Int4_1_LowKGroup`
- `GemmKernel224Int4SmallKGroup`

KGroup kernel 的差异：
1. `mat_mul_kgroup()` 需要额外的 `k_group_size` 参数
2. `BufferB::required_size(n, k, k_group_size)` 需要 3 个参数
3. 部分类型使用 `from_raw_mat()` 而非 `from_mat()`

### 11.2 解决方案：类型特征 + `if constexpr`

在 `sft_moe.hpp` 开头定义类型特征：

```cpp
// Type trait to detect if kernel supports standard mat_mul API
template <typename T>
struct supports_standard_mat_mul : std::false_type {};

template <>
struct supports_standard_mat_mul<amx::GemmKernel224BF> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int8> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int4> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int4_1> : std::true_type {};

template <typename T>
inline constexpr bool supports_standard_mat_mul_v = supports_standard_mat_mul<T>::value;
```

### 11.3 调度逻辑

使用 C++17 `if constexpr` 在编译时选择实现路径：

```cpp
// Forward LoRA 计算
if constexpr (supports_standard_mat_mul_v<T>) {
  compute_lora_gate_up_amx(qlen, activated_expert);  // AMX 路径
} else {
  compute_lora_gate_up(qlen, activated_expert);  // For-loop 回退
}

// Backward 计算
if constexpr (supports_standard_mat_mul_v<T>) {
  backward_down_amx(cache, grad_output, ...);  // AMX 路径
} else {
  backward_down(cache, grad_output, ...);  // For-loop 回退
}
```

### 11.4 Buffer 分配

对于不支持的 kernel，跳过 AMX buffer 分配：

```cpp
if constexpr (supports_standard_mat_mul_v<T>) {
  // 分配 LoRA AMX buffers
  lora_bb_pool_bytes_ = config_.expert_num * ...;
  // ...
} else {
  // KGroup kernels 不需要这些 buffer
  lora_bb_pool_bytes_ = 0;
  // ...
}
```

### 11.5 权重准备

对于不支持的 kernel，`prepare_lora_weights()` 和 `prepare_backward_weights()` 会提前返回：

```cpp
void prepare_lora_weights() {
  if constexpr (!supports_standard_mat_mul_v<T>) {
    return;  // KGroup kernels 使用 for-loop 实现
  }
  // ...正常的权重准备逻辑...
}
```

### 11.6 性能影响

| Kernel 类型 | Forward LoRA | Backward GEMM | 性能 |
|-------------|--------------|---------------|------|
| GemmKernel224BF | AMX | AMX | 最优 |
| GemmKernel224Int8 | AMX | AMX | 最优 |
| GemmKernel224Int4 | AMX | AMX | 最优 |
| GemmKernel224Int4_1 | AMX | AMX | 最优 |
| GemmKernel224Int4KGroup | for-loop | for-loop | 较慢 |
| GemmKernel224Int4_1KGroup | for-loop | for-loop | 较慢 |
| GemmKernel224Int4_1_LowKGroup | for-loop | for-loop | 较慢 |
| GemmKernel224Int4SmallKGroup | for-loop | for-loop | 较慢 |

## 12. 待完成工作

### 12.1 进一步优化

- 考虑合并 Step 1 和 Step 2 的 GEMM 以减少内存带宽
- 探索使用 FP8 量化进一步提升性能
- backward_gate_up 中 BufferA 的复用优化

### 12.2 LoRA 梯度 AMX 优化

当前 LoRA 梯度计算使用 for 循环，可考虑：
- 对于 lora_rank >= 32 的情况使用 AMX
- 批量处理多个 expert 的梯度计算

### 12.3 KGroup Kernel 优化

为 KGroup kernel 添加 AMX 优化支持：
- 实现 `mat_mul_kgroup` 版本的 LoRA 计算
- 需要处理 `BufferB::required_size(n, k, k_group_size)` 参数差异
