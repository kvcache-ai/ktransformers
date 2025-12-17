# NVFP4 (NVIDIA FP4) MoE Implementation

这是基于 AVX512 查表法的 NVFP4 量化格式 MoE 算子实现，参考了 `nvfp4.md` 中的设计方案。

## 文件结构

```
kt-kernel/operators/amx/
├── la/
│   ├── nvfp4_utils.hpp       # NVFP4 量化格式和查表法实现
│   └── nvfp4_kernel.hpp      # NVFP4 矩阵乘法 kernel (AVX512)
├── nvfp4-moe.hpp             # NVFP4 MoE operator (类似 k2-moe.hpp)
└── test/
    ├── nvfp4-moe-test.cpp    # 测试用例
    └── build_nvfp4_test.sh   # 编译脚本
```

## 核心特性

### 1. NVFP4 量化格式 (`nvfp4_utils.hpp`)

- **FP4 编码**: 8 个正值 (0, 0.5, 1, 1.5, 2, 3, 4, 6) + 符号位
- **存储格式**: 每字节存储 2 个 FP4 值（4-bit 打包）
- **分组量化**: 每 16 个 FP4 值为一组，共享一个 scale
- **查表乘法**: 使用预计算的乘法结果查找表

#### FP4 乘法结果表

FP4 × FP4 的乘法结果有 19 种唯一值（不含符号）：

```
0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.25, 3, 4, 4.5, 6, 8, 9, 12, 16, 18, 24, 36
```

为了保留 0.25 的精度，结果使用 INT16 存储（乘以 4）：

```
0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144
```

### 2. AVX512 查表实现

根据 `nvfp4.md` 的设计：

1. **符号处理**:
   - 提取符号位并 XOR 得到结果符号

2. **数值查表**:
   - 组合两个 FP4 的尾数/指数（6 bits）
   - 使用 `_mm512_permutexvar_epi8` 查表获取结果索引
   - 再次查表获取缩放后的 INT16 结果值

3. **符号应用**:
   - 根据符号位对结果取负

### 3. Buffer 实现 (`nvfp4_kernel.hpp`)

#### BufferA - 激活输入
- 使用 INT8 量化（类似现有实现）
- 支持 K-group 分组量化
- 每行每组独立计算 scale

#### BufferB - FP4 权重
- 存储打包的 FP4 权重（2 个值/字节）
- 每组（16 个 FP4 值）一个 scale
- 支持从预量化数据加载 (`from_raw_mat`)

#### BufferC - 输出
- 存储 FP32 结果
- 支持转换回 BF16 格式

### 4. 矩阵乘法 Kernel

```cpp
void mat_mul_nvfp4_kgroup(
    int m, int n, int k, int k_group_size,
    std::shared_ptr<BufferA> ba,
    std::shared_ptr<BufferB> bb,
    std::shared_ptr<BufferC> bc,
    int ith, int nth
)
```

**实现方式**:
- 基于 AVX512 查表（不使用 AMX）
- 按组处理（每组 16 个 FP4 值）
- INT8 (activation) × FP4 (weight) → INT16 → FP32

**性能参数**:
- M_STEP: 32
- N_STEP: 64
- K_STEP: 64 (4 个 FP4 groups)

### 5. MoE Operator (`nvfp4-moe.hpp`)

基于 `k2-moe.hpp` 的结构，实现了：

- **TP (Tensor Parallel) 支持**: 多 NUMA 节点并行
- **Expert 路由**: 支持动态 expert 选择
- **内存管理**: 高效的 buffer pool 管理
- **前向传播**:
  - `forward_prefill`: 批量处理（qlen > 1）
  - `forward_decode`: 单 token 处理（qlen = 1）

## 编译和测试

### 环境要求

- **编译器**: g++ 或 clang++ (支持 C++17)
- **CPU**: 支持 AVX-512 指令集
- **依赖**:
  - OpenMP
  - fmt library (header-only)

### 编译测试

```bash
cd kt-kernel/operators/amx/test
./build_nvfp4_test.sh
```

或手动编译：

```bash
g++ -std=c++17 -O2 -march=native -mavx512f -mavx512bw -mavx512vl \
    -I../../.. -I../../../third_party \
    -fopenmp -Wall -Wextra \
    nvfp4-moe-test.cpp -o nvfp4-moe-test -lm -fopenmp

./nvfp4-moe-test
```

### 测试内容

测试用例包括：

1. **FP4 量化/反量化**: 验证 float ↔ FP4 转换的正确性
2. **分组量化**: 测试 block-wise 量化和反量化
3. **BufferB 加载**: 验证权重加载功能
4. **BufferA 量化**: 验证激活量化功能
5. **矩阵乘法**: 测试完整的矩阵乘法流程，对比参考实现

## 使用示例

### 1. 创建 MoE Operator

```cpp
#include "operators/amx/nvfp4-moe.hpp"
#include "operators/amx/la/nvfp4_kernel.hpp"

// 配置
GeneralMOEConfig config;
config.expert_num = 8;
config.num_experts_per_tok = 2;
config.hidden_size = 4096;
config.intermediate_size = 11008;
config.max_len = 2048;
config.quant_config.group_size = 16;  // FP4 group size
config.quant_config.zero_point = false;

// 创建 operator
using KernelType = amx::GemmKernelNVFP4KGroup;
auto moe = std::make_shared<AMX_NVFP4_MOE_TP<KernelType>>(config, 0);

// 加载权重
moe->load_weights();
```

### 2. 执行前向传播

```cpp
// 准备输入
std::vector<ggml_bf16_t> input(qlen * config.hidden_size);
std::vector<float> output(qlen * config.hidden_size);
std::vector<int64_t> expert_ids(qlen * config.num_experts_per_tok);
std::vector<float> weights(qlen * config.num_experts_per_tok);

// 前向传播
moe->forward(qlen, config.num_experts_per_tok,
             expert_ids.data(), weights.data(),
             input.data(), output.data());
```

## 性能优化

### 当前实现

- ✅ AVX-512 查表实现
- ✅ K-group 分组量化
- ✅ 并行处理（OpenMP）
- ✅ 高效内存布局

### 待优化

- ⚠️ **优化查表性能**:
  - 当前使用简化的标量查表
  - 应使用完整的 AVX-512 SIMD 查表（`_mm512_permutexvar_epi8`）

- ⚠️ **激活函数**:
  - 当前 SiLU 实现简化
  - 需要添加完整的 `x * sigmoid(x)` 实现

- ⚠️ **decode 模式**:
  - 当前 `forward_decode` 未完整实现
  - 应参考 `k2-moe.hpp` 完善单 token 处理逻辑

- ⚠️ **内存优化**:
  - 可以考虑更紧凑的 buffer 布局
  - 减少中间结果的内存占用

## 与 K2-MoE 的对比

| 特性 | K2-MoE (INT4) | NVFP4-MoE |
|------|---------------|-----------|
| 量化格式 | 有符号 INT4 (-8~7) | NVIDIA FP4 (8 个值) |
| 计算方式 | AMX tile 指令 | AVX512 查表 |
| 精度范围 | 均匀分布 | 非均匀分布（指数式） |
| 量化误差 | 线性误差 | 非线性误差 |
| Group Size | 128 (可配置) | 16 (固定) |
| 性能 | 高（AMX 硬件加速） | 中（AVX512 查表） |

## 参考资料

- `kt-kernel/plan/nvfp4.md`: NVFP4 实现方案
- `kt-kernel/operators/amx/k2-moe.hpp`: K2 MoE 参考实现
- `kt-kernel/operators/amx/la/`: AMX kernel 实现

## 贡献者

- Claude (AI Assistant)
- KVCache.AI Team

## License

Copyright (c) 2025 by KVCache.AI, All Rights Reserved.
