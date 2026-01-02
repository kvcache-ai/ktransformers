# SFT MoE AMX Bug 调试记录

本文档记录 SFT MoE AMX 实现过程中遇到的 bug 及其修复方案。

---

# 预备知识：Cache 机制与公式推导

本板块介绍 SFT MoE 的 ForwardCache 机制及 backward pass 的数学推导，为理解后续 bug 提供理论基础。

---

## 1. MoE SFT Forward Cache 设计

### 1.1 Cache 的目的

在训练场景中，需要保存 forward pass 的中间结果用于 backward pass 计算梯度。由于 MoE 层的特殊性（routing、多专家并行），需要保存：

1. **Routing 信息**：哪些 token 被路由到哪些 expert
2. **中间激活值**：gate/up projection 的输出（activation 之前）
3. **Expert 映射**：activated expert 的顺序

### 1.2 ForwardCache 结构

```cpp
struct ForwardCache {
  // 中间值指针 (指向预分配的 buffer pool)
  ggml_bf16_t* input_cache;          // [qlen, hidden_size]
  ggml_bf16_t* gate_output_cache;    // [tokens_total, intermediate_size]
  ggml_bf16_t* up_output_cache;      // [tokens_total, intermediate_size]
  ggml_bf16_t* intermediate_cache;   // [tokens_total, intermediate_size]

  // Routing 信息
  std::vector<int64_t> expert_ids_cache;        // [qlen * k] 每个 token 选择的专家
  std::vector<float> weights_cache;             // [qlen * k] 路由权重
  std::vector<int> m_local_num_cache;           // [expert_num] 每个专家处理的 token 数
  std::vector<std::vector<int>> m_local_pos_cache; // [qlen][k] 每个 token 在专家内的位置
  std::vector<int> m_expert_id_map_cache;       // [activated_expert] 激活专家的顺序

  int qlen_cache, k_cache, activated_expert_cache;
  bool valid = false;
};
```

### 1.3 Cache Buffer 的内存布局

**关键概念**：`gate_output_cache` 和 `up_output_cache` 存储数据的顺序由 `m_expert_id_map_` 决定！

```
假设 forward 时激活了 3 个专家，顺序为 [Expert 5, Expert 10, Expert 0]：
- m_expert_id_map_[0] = 5   (2 tokens)
- m_expert_id_map_[1] = 10  (1 token)
- m_expert_id_map_[2] = 0   (1 token)

gate_output_cache 内存布局：
+--------------------------------------------------+
| Expert 5 的 2 个 token | Expert 10 的 1 个 token | Expert 0 的 1 个 token |
| [2 * intermediate_size] | [1 * intermediate_size] | [1 * intermediate_size]|
+--------------------------------------------------+
offset=0                   offset=2                  offset=3
```

### 1.4 save_to_cache 流程

```cpp
void save_to_cache(ForwardCache& cache, ...) {
  // 1. 保存 routing 信息
  cache.m_local_num_cache = m_local_num_;
  cache.m_expert_id_map_cache = m_expert_id_map_;

  // 2. 按 m_expert_id_map_ 的顺序复制 gate/up 输出
  size_t offset = 0;
  for (int i = 0; i < activated_expert; i++) {
    int expert_idx = m_expert_id_map_[i];           // 第 i 个激活的专家 ID
    int num_tokens = m_local_num_[expert_idx];      // 这个专家处理的 token 数

    // 从 m_local_gate_output_ptr_[expert_idx] 复制到 cache
    memcpy(cache.gate_output_cache + offset * intermediate_size,
           m_local_gate_output_ptr_[expert_idx],
           num_tokens * intermediate_size * sizeof(bf16));

    offset += num_tokens;
  }
}
```

---

## 2. Backward Pass 公式推导

### 2.1 MoE FFN Forward 公式

对于单个专家的 FFN：
```
y = down_proj(activation(gate_proj(x) * up_proj(x)))
  = W_down @ (silu(W_gate @ x) * (W_up @ x))
```

其中 `silu(x) = x * sigmoid(x)`。

设：
- `g = W_gate @ x`（gate projection 输出）
- `u = W_up @ x`（up projection 输出）
- `intermediate = silu(g) * u = g * sigmoid(g) * u`
- `y = W_down @ intermediate`

### 2.2 Backward Pass 链式法则

设 loss 为 L，反向传播需要计算：
- `∂L/∂x` (grad_input) - 用于继续反向传播
- `∂L/∂W_gate`, `∂L/∂W_up`, `∂L/∂W_down` (LoRA 梯度)

#### Step 1: backward_down

```
给定：∂L/∂y (grad_output)
计算：∂L/∂intermediate = ∂L/∂y @ W_down^T

其中 intermediate = silu(gate_out) * up_out
```

#### Step 2: backward_activation (SiLU backward)

```
设 g = gate_out, u = up_out
intermediate = silu(g) * u = g * sigmoid(g) * u

∂L/∂g = ∂L/∂intermediate * u * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
∂L/∂u = ∂L/∂intermediate * silu(g) = ∂L/∂intermediate * g * sigmoid(g)
```

**关键观察**：如果 `g ≈ 0`，那么 `silu(g) = g * sigmoid(g) ≈ 0`，导致 `∂L/∂u ≈ 0`！

这就是 Bug #15 中 `grad_up = 0` 的数学原因。

#### Step 3: backward_gate_up

```
∂L/∂x = ∂L/∂g @ W_gate^T + ∂L/∂u @ W_up^T
```

### 2.3 完整的 LoRA 梯度公式

对于 LoRA 层：`y = x @ W^T + (x @ A^T @ B^T) * scaling`

Backward:
```
grad_x = grad_y @ W + (grad_y @ B @ A) * scaling
grad_A = (x^T @ (grad_y @ B)) * scaling
grad_B = (grad_y^T @ (x @ A^T)) * scaling
```

---

## 3. Cache 在 Backward 中的使用

### 3.1 正确的 backward 流程

```cpp
void backward(...) {
  ForwardCache cache = pop_cache();  // 获取对应的 forward cache

  // ★ 恢复 routing 信息 ★
  m_local_num_ = cache.m_local_num_cache;
  m_expert_id_map_ = cache.m_expert_id_map_cache;

  // 调用各个 backward 函数
  backward_down(cache, grad_output, ...);      // 计算 grad_intermediate
  backward_activation(cache);                   // 计算 grad_gate, grad_up
  backward_gate_up(cache, grad_input, ...);    // 计算 grad_input
}
```

### 3.2 backward_activation 如何读取 cache

```cpp
void backward_activation(const ForwardCache& cache) {
  for (int task_id = 0; task_id < activated_expert; task_id++) {
    // 使用 ★当前★ m_expert_id_map_（已在 backward() 中恢复）
    int expert_idx = m_expert_id_map_[task_id];
    int num_tokens = m_local_num_[expert_idx];

    // 计算在 cache 中的 offset（按 m_expert_id_map_ 顺序）
    size_t offset = 0;
    for (int i = 0; i < task_id; i++) {
      offset += m_local_num_[m_expert_id_map_[i]];
    }

    // 读取 cache 数据
    ggml_bf16_t* gate_output = cache.gate_output_cache + offset * intermediate_size;
    ggml_bf16_t* up_output = cache.up_output_cache + offset * intermediate_size;

    // 计算梯度（使用上面的 SiLU backward 公式）
    for (int i = 0; i < num_tokens * intermediate_size; i++) {
      float g = GGML_BF16_TO_FP32(gate_output[i]);
      float u = GGML_BF16_TO_FP32(up_output[i]);
      float sigmoid_g = 1.0f / (1.0f + expf(-g));
      float silu_g = g * sigmoid_g;
      float grad_i = GGML_BF16_TO_FP32(grad_intermediate_[offset + i]);

      float grad_gate_val = grad_i * u * sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));
      float grad_up_val = grad_i * silu_g;  // 如果 g ≈ 0，这里 ≈ 0！
      // ...
    }
  }
}
```

---

## 4. 关键调试技巧

### 4.1 使用 Norm 追踪数据流

在 backward 各阶段打印 norm 值可以快速定位问题：

```cpp
printf("[DEBUG] grad_intermediate norm: %f\n", compute_bf16_norm(...));
printf("[DEBUG] grad_gate norm: %f, grad_up norm: %f\n", ...);
printf("[DEBUG] grad_input norm: %f\n", ...);
```

如果某个 norm 突然变成 0，说明该阶段出了问题。

### 4.2 检查内存地址避免 Buffer 重叠

当多个 buffer 分配时，需要检查它们的地址是否重叠：

```cpp
printf("[DEBUG ADDR] buffer1 = %p, buffer2 = %p\n", (void*)buf1, (void*)buf2);
printf("[DEBUG BEFORE memset] buf2[0..3] = %.4f %.4f %.4f %.4f\n", ...);
memset(buf1, 0, size);
printf("[DEBUG AFTER memset] buf2[0..3] = %.4f %.4f %.4f %.4f\n", ...);
```

如果 BEFORE 有值而 AFTER 变成 0，说明 buf1 和 buf2 有内存重叠！

---

# 第一板块：语法 Bug

本板块记录编译期和运行时的语法、类型、继承相关问题。

---

## Bug #1: C++ 继承链中的私有成员访问问题

### 问题现象

编译时出现大量错误，提示基类成员在派生类中不可访问：

```
sft_moe.hpp:57:15: error: 'GeneralMOEConfig AMX_MOE_BASE<...>::config_' is private within this context
   57 |   using Base::config_;
      |               ^~~~~~~
moe.hpp:23:15: note: declared private here
   23 |   using Base::config_;
```

涉及的成员变量包括：`config_`, `tp_part_idx`, `down_ba_`, `down_bb_`, `down_bc_`, `gate_bb_`, `gate_bc_`, `gate_up_ba_`, `up_bb_`, `up_bc_`, `m_local_num_` 等。

### 问题原因

在 C++ 中，`using Base::member` 声明的访问级别取决于它在派生类中所处的 section (public/protected/private)。

**继承链结构：**
```
AMX_MOE_BASE<T, Derived>  (所有成员为 public)
      ↓ 继承
AMX_MOE_TP<T>             (private section 中使用 using Base::*)
      ↓ 继承
AMX_SFT_MOE_TP<T>         (尝试访问父类的这些成员)
```

问题出在 `moe.hpp` 中的 `AMX_MOE_TP` 类：

```cpp
template <class T>
class AMX_MOE_TP : public AMX_MOE_BASE<T, AMX_MOE_TP<T>> {
 private:  // <-- 问题所在：这些 using 声明在 private section
  using Base = AMX_MOE_BASE<T, AMX_MOE_TP<T>>;
  using Base::config_;
  using Base::tp_part_idx;
  // ... 其他成员
```

虽然这些成员在 `AMX_MOE_BASE` 中是 public 的，但在 `AMX_MOE_TP` 中通过 `using` 声明后变成了 private。当 `AMX_SFT_MOE_TP` 继承 `AMX_MOE_TP` 时，无法访问这些私有成员。

### 解决方案

将 `moe.hpp` 中的 using 声明从 `private` section 移到 `protected` section：

```cpp
template <class T>
class AMX_MOE_TP : public AMX_MOE_BASE<T, AMX_MOE_TP<T>> {
 protected:  // 改为 protected
  using Base = AMX_MOE_BASE<T, AMX_MOE_TP<T>>;
  using Base::config_;
  using Base::tp_part_idx;
  using Base::gate_bb_;
  using Base::up_bb_;
  using Base::down_bb_;
  using Base::gate_up_ba_;
  using Base::gate_bc_;
  using Base::up_bc_;
  using Base::down_ba_;
  using Base::down_bc_;
  using Base::m_local_num_;

 private:  // 实际的私有成员放在这里
  std::filesystem::path prefix;
  void* gate_proj_;
  void* up_proj_;
  void* down_proj_;
```

同时，在 `sft_moe.hpp` 中也做相同的修改，将 using 声明移到 protected section。

---

## Bug #2: MOE_TP_PART Concept 不满足

### 问题现象

编译时出现 concept 约束失败错误：

```
moe-tp.hpp:20:5: note: the required expression 'new T' is invalid
   20 |   { new T(config, tp_idx) } -> std::same_as<T*>;
      |     ^~~~~~~~~~~~~~~~~~~~~
```

```
moe-sft-tp.hpp:27:7: error: template constraint failure for 'template<class T>  requires  MOE_TP_PART<T> class TP_MOE'
   27 | class TP_MOE_SFT : public TP_MOE<T> {
      |       ^~~~~~~~~~
```

### 问题原因

`moe-tp.hpp` 中定义的 `MOE_TP_PART` concept 要求类型 T 必须有一个接受 `GeneralMOEConfig` 参数的构造函数：

```cpp
template <typename T>
concept MOE_TP_PART = requires(T t, ..., GeneralMOEConfig config, int tp_idx) {
  typename T::output_t;
  { new T(config, tp_idx) } -> std::same_as<T*>;  // 要求 GeneralMOEConfig 构造函数
  { t.forward(...) } -> std::same_as<void>;
};
```

但是 `AMX_SFT_MOE_TP<T>` 只有接受 `MOESFTConfig` 的构造函数：

```cpp
AMX_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
```

由于没有 `GeneralMOEConfig` 构造函数，concept 检查失败，导致 `TP_MOE<AMX_SFT_MOE_TP<T>>` 无法实例化，进而导致 `TP_MOE_SFT<AMX_SFT_MOE_TP<T>>` 编译失败。

### 解决方案

1. **在 `common.hpp` 中为 `MOESFTConfig` 添加转换构造函数：**

```cpp
struct MOESFTConfig : public GeneralMOEConfig {
  // ... 现有字段 ...

  MOESFTConfig() : GeneralMOEConfig() {}

  MOESFTConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size)
      : GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size) {}

  // 新增：从 GeneralMOEConfig 转换的构造函数
  explicit MOESFTConfig(const GeneralMOEConfig& base) : GeneralMOEConfig(base) {
    // LoRA 字段使用默认值（已在结构体定义中初始化）
  }
};
```

2. **在 `sft_moe.hpp` 中为 `AMX_SFT_MOE_TP` 添加接受 `GeneralMOEConfig` 的构造函数：**

```cpp
public:
  // 主构造函数（现有）
  AMX_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
      : Base(static_cast<GeneralMOEConfig>(config), tp_part_idx), sft_config_(config) {
    // ... 初始化代码 ...
  }

  // 新增：满足 MOE_TP_PART concept 的构造函数
  AMX_SFT_MOE_TP(GeneralMOEConfig config, int tp_part_idx)
      : AMX_SFT_MOE_TP(MOESFTConfig(config), tp_part_idx) {}
```

这个新构造函数使用委托构造，将 `GeneralMOEConfig` 转换为 `MOESFTConfig`（使用默认的 LoRA 配置），然后调用主构造函数。

---

## Bug #3: 缺失的成员方法

### 问题现象

编译时提示 `TP_MOE_SFT` 类没有 `warm_up` 和 `load_weights` 方法：

```
ext_bindings.cpp:365:23: error: 'warm_up' is not a member of 'MoeClass' {aka 'TP_MOE_SFT<AMX_SFT_MOE_TP<amx::GemmKernel224BF> >'}
ext_bindings.cpp:366:28: error: 'load_weights' is not a member of 'MoeClass'
```

### 问题原因

这是 Bug #2 的连锁反应。由于 `MOE_TP_PART` concept 检查失败：
1. `TP_MOE<AMX_SFT_MOE_TP<T>>` 无法正确实例化
2. `TP_MOE_SFT<T>` 继承自 `TP_MOE<T>` 失败
3. 原本从 `TP_MOE_Common` 继承的 `warm_up` 和 `load_weights` 方法不可用

### 解决方案

修复 Bug #1 和 Bug #2 后，继承链恢复正常，这些方法将自动可用。

---

## Bug #4: TP_MOE_SFT 是抽象类

### 问题现象

修复 Bug #1-3 后，编译时出现新的错误：

```
error: invalid new-expression of abstract class type 'TP_MOE_SFT<AMX_SFT_MOE_TP<amx::GemmKernel224BF> >'

note: because the following virtual functions are pure within 'TP_MOE_SFT<...>':
  'void TP_MOE_Common<T>::load_weights()'
  'void TP_MOE_Common<T>::merge_results(int qlen, void* output)'
```

### 问题原因

`TP_MOE_Common<T>` 定义了两个纯虚函数 (moe-tp.hpp:215-217):

```cpp
virtual void load_weights() = 0;
virtual void merge_results(int qlen, void* output) = 0;
```

存在一个模板特化 `TP_MOE<AMX_MOE_BASE<T, Derived>>` (moe_base.hpp:700-761) 实现了这两个函数。

**继承链分析：**
```
TP_MOE_Common<T>                     (定义纯虚函数 load_weights, merge_results)
      ↓ 继承
TP_MOE<T>                            (通用模板，没有实现纯虚函数)
      ↓ 继承
TP_MOE_SFT<T>                        (也没有实现，仍然是抽象类)
```

模板特化 `TP_MOE<AMX_MOE_BASE<T, Derived>>` 实现了这些函数，但是：
- `TP_MOE_SFT<T>` 继承自 `TP_MOE<T>`，其中 `T = AMX_SFT_MOE_TP<Kernel>`
- `AMX_SFT_MOE_TP<Kernel>` 不是 `AMX_MOE_BASE<T, Derived>` 类型，而是其派生类
- **C++ 模板特化不会匹配派生类**，所以 `TP_MOE<AMX_SFT_MOE_TP<Kernel>>` 使用的是通用模板而非特化版本
- 通用模板没有实现纯虚函数，因此 `TP_MOE_SFT` 仍然是抽象类

### 解决方案

在 `moe-sft-tp.hpp` 的 `TP_MOE_SFT` 类中直接实现这两个纯虚函数：

```cpp
// 实现纯虚函数 load_weights
void load_weights() override {
  auto pool = config.pool;
  pool->dispense_backend()->do_numa_job([this](int numa_id) {
    tps[numa_id]->load_weights();
  });
  weights_loaded = true;
}

// 实现纯虚函数 merge_results
void merge_results(int qlen, void* output) override {
  merge_results(qlen, output, false);
}

void merge_results(int qlen, void* output, bool incremental) override {
  // 复用 moe_base.hpp 中的 AVX-512 优化逻辑
  // 合并各 NUMA 节点的输出结果
  auto merge_fn = [this, output, incremental](int token_nth) {
    float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
    // ... AVX-512 SIMD 合并逻辑
  };

  if (qlen < 10) {
    for (int i = 0; i < qlen; i++) merge_fn(i);
  } else {
    pool->do_work_stealing_job(qlen, nullptr, merge_fn, nullptr);
  }
}
```

### 关键知识点

**C++ 模板特化不匹配派生类**：当定义 `TP_MOE<AMX_MOE_BASE<T, Derived>>` 特化时，它只能精确匹配 `AMX_MOE_BASE<T, Derived>` 类型，不会匹配其派生类如 `AMX_MOE_TP<T>` 或 `AMX_SFT_MOE_TP<T>`。

这是 C++ 模板特化的标准行为。如果需要让特化也匹配派生类，可以：
1. 为每个派生类创建单独的特化（不推荐，维护困难）
2. 在派生类的包装器中直接实现虚函数（本文采用的方案）
3. 使用 SFINAE 或 concepts 进行更灵活的匹配

---

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/amx/moe.hpp` | 将 `using Base::*` 声明从 private 移到 protected section |
| `operators/common.hpp` | 为 `MOESFTConfig` 添加 `explicit MOESFTConfig(const GeneralMOEConfig&)` 构造函数 |
| `operators/amx/sft_moe.hpp` | 1. 将 `using Base::*` 声明从 private 移到 protected section<br>2. 添加 `AMX_SFT_MOE_TP(GeneralMOEConfig, int)` 构造函数 |

---

## 总结

这次编译错误的根本原因是 C++ 中访问控制和模板 concept 的交互问题：

1. **继承中的访问控制**：在派生类中使用 `using Base::member` 时，成员的最终访问级别由 using 声明所在的 section 决定，而非原始基类中的访问级别。

2. **C++20 Concept 约束**：Concept 要求必须严格满足，包括构造函数签名。即使 `MOESFTConfig` 是 `GeneralMOEConfig` 的派生类，也需要显式提供接受 `GeneralMOEConfig` 的构造函数来满足 concept 要求。

修复建议：在设计继承层次时，如果期望派生类能够访问基类成员，应将 using 声明放在 protected section 而非 private section。

---

## Bug #5: 错误的 Include 路径

### 问题现象

编译时出现头文件找不到错误：

```
moe-sft-tp.hpp:15:10: fatal error: amx/llama.cpp/ggml.h: No such file or directory
   15 | #include "amx/llama.cpp/ggml.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~
```

### 问题原因

在 `moe-sft-tp.hpp` 中添加 `merge_results` 实现时，错误地添加了 `#include "amx/llama.cpp/ggml.h"`。

实际上：
1. `llama.cpp/` 目录不在 `amx/` 目录下，正确路径应该是 `"llama.cpp/ggml.h"`
2. 更重要的是，这个 include 完全不需要，因为：
   - `moe-sft-tp.hpp` → `moe-tp.hpp` → `common.hpp` → `ggml.h`
   - `ggml_bf16_t` 类型已经通过这个 include 链可用

### 解决方案

删除多余的错误 include 行：

```cpp
// 修改前
#include <immintrin.h>

#include "moe-tp.hpp"
#include "amx/la/amx.hpp"
#include "amx/llama.cpp/ggml.h"  // 删除这行

// 修改后
#include <immintrin.h>

#include "moe-tp.hpp"
#include "amx/la/amx.hpp"
```

### 关键知识点

在添加 include 时，应该：
1. 检查头文件的实际路径是否正确
2. 检查所需的类型/函数是否已经通过现有 include 链可用，避免重复 include

---

## Bug #6: Python 绑定缺失核心配置字段

### 问题现象

运行 `test_moe_sft_amx.py` 时出现 AttributeError：

```
test_moe_sft_amx.py:628: AttributeError: 'kt_kernel_ext.moe.MOESFTConfig' object has no attribute 'expert_num'
```

测试代码尝试设置配置字段：

```python
config = kt_kernel_ext.moe.MOESFTConfig()
config.expert_num = expert_num  # <-- 报错
config.num_experts_per_tok = num_experts_per_tok
config.hidden_size = hidden_size
config.intermediate_size = intermediate_size
```

### 问题原因

`ext_bindings.cpp` 中的 pybind11 绑定没有暴露 `GeneralMOEConfig` 的核心字段。

**问题代码 (ext_bindings.cpp:691-747)：**

```cpp
py::class_<GeneralMOEConfig>(moe_module, "MOEConfig")
    .def(py::init([](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size) {
      return GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size);
    }))
    // 构造函数接受这些参数...
    .def_readwrite("layer_idx", &GeneralMOEConfig::layer_idx)  // 直接跳到其他字段
    .def_readwrite("pool", &GeneralMOEConfig::pool)
    // ... 没有 expert_num, num_experts_per_tok, hidden_size, intermediate_size 的 def_readwrite！
```

虽然构造函数可以接受这些参数进行初始化，但由于没有 `.def_readwrite()` 声明，Python 端无法在构造后读取或修改这些属性。

`MOESFTConfig` 继承自 `GeneralMOEConfig`（通过 `py::class_<MOESFTConfig, GeneralMOEConfig>`），因此也缺失这些属性。

### 解决方案

在 `ext_bindings.cpp` 的 `GeneralMOEConfig` 绑定中添加缺失的字段声明：

```cpp
py::class_<GeneralMOEConfig>(moe_module, "MOEConfig")
    .def(py::init([](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size) {
      return GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size);
    }))
    // ... 其他 init ...
    // 新增：核心配置字段
    .def_readwrite("expert_num", &GeneralMOEConfig::expert_num)
    .def_readwrite("num_experts_per_tok", &GeneralMOEConfig::num_experts_per_tok)
    .def_readwrite("hidden_size", &GeneralMOEConfig::hidden_size)
    .def_readwrite("intermediate_size", &GeneralMOEConfig::intermediate_size)
    .def_readwrite("layer_idx", &GeneralMOEConfig::layer_idx)
    // ... 其余绑定 ...
```

### 关键知识点

**pybind11 继承与属性暴露**：
1. 当派生类通过 `py::class_<Derived, Base>` 声明继承关系时，基类中通过 `.def_readwrite()` 暴露的属性会自动被派生类继承
2. 但构造函数参数不会自动变成可访问的属性——必须显式声明 `.def_readwrite()`
3. 如果基类没有暴露某个字段，所有派生类都无法访问该字段

### 修改文件清单

| 文件 | 修改内容 |
|------|---------||
| `ext_bindings.cpp` | 在 `GeneralMOEConfig` 绑定中添加 `expert_num`, `num_experts_per_tok`, `hidden_size`, `intermediate_size` 的 `.def_readwrite()` 声明 |

---

# 第二板块：数值 Bug

本板块记录计算结果不正确的数值问题。

---

## Bug #7: 测试文件输出 Buffer 数据类型错误

### 问题现象

运行 `test_moe_sft_amx.py` 时，forward 测试出现极大的相对误差：

```
Relative difference: 1.359375
[FAILED] Test failed with error: Forward pass accuracy test failed: diff=1.359375 >= 0.05
```

相比之下，推理测试 `test_moe_amx.py` 的误差约为 0.046，在可接受范围内。

### 问题原因

C++ 实现中 `TP_MOE_SFT::merge_results()` 将最终输出转换为 **bf16** 格式：

```cpp
// moe-sft-tp.hpp:81-84
for (int e = 0; e < config.hidden_size; e += 32) {
  __m512 x0 = *(__m512*)(merge_to + e);
  __m512 x1 = *(__m512*)(merge_to + e + 16);
  avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
}
```

但测试文件 `test_moe_sft_amx.py` 分配的输出 buffer 是 **float32**：

```python
# test_moe_sft_amx.py:698
output = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()  # 错误！
```

**数据类型不匹配的后果：**
- bf16 每个元素 2 字节，float32 每个元素 4 字节
- C++ 向 float32 buffer 写入 bf16 数据，只填充了 buffer 的一半
- Python 将这些 bf16 字节解释为 float32 → 得到完全错误的数值

### 解决方案

修改 `test_moe_sft_amx.py`，将所有 SFT forward 输出 buffer 的 dtype 从 `float32` 改为 `bfloat16`：

**修改点：**

| 函数 | 行号 | 修改内容 |
|------|------|---------|
| `test_moe_sft_forward()` | 698 | `dtype=torch.float32` → `dtype=torch.bfloat16` |
| `test_moe_sft_forward()` | 712-716 | 删除 `.to(torch.bfloat16)` 转换 |
| `test_moe_sft_backward()` | 854 | `dtype=torch.float32` → `dtype=torch.bfloat16` |
| `test_moe_sft_lora_weight_sync()` | 998, 1026, 1068 | `dtype=torch.float32` → `dtype=torch.bfloat16` |
| `test_moe_sft_training_loop()` | 1205 | `dtype=torch.float32` → `dtype=torch.bfloat16` |

**修改示例：**

```python
# 修改前
output = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()
# ...
output_bf16 = output.to(torch.bfloat16)
diff = torch.mean(torch.abs(output_bf16 - torch_output)) / ...

# 修改后
output = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
# ...
diff = torch.mean(torch.abs(output - torch_output)) / ...
```

### 关键知识点

1. **数据类型必须匹配**：C++ 和 Python 之间通过指针传递数据时，双方必须使用相同的数据类型解释内存
2. **SFT forward 输出为 bf16**：与推理模式一致，SFT 的 forward 输出也是 bf16 格式

---

## Bug #8: TP 模式下基础权重未正确分区

### 问题现象

修复 Bug #7 后，输出不再是垃圾值，但仍有较大误差（约 1.71）：

```
[AMX SFT DEBUG] AMX output[:8] = tensor([ 4.2021e-06,  1.1086e-05, ...])
[MOE SFT DEBUG] Final output[:8] = tensor([-1.2457e-05, -5.0366e-06, ...])
Relative difference: 1.710938
```

AMX 输出和 PyTorch 参考输出数值范围相近（都是 1e-5 到 1e-6），但具体值明显不同。

### 问题原因

**TP（Tensor Parallel）模式的工作原理：**
- intermediate_size 被分割到多个 NUMA 节点
- 每个 NUMA 节点处理 intermediate_size / tp_count 的权重
- 各 NUMA 节点的输出结果相加得到最终输出

**推理模式 `TP_MOE<AMX_MOE_TP<K>>::load_weights()`** (moe.hpp:370-430) 正确处理了权重分区：

```cpp
for (auto i = 0; i < tp_count; i++) {
  auto& tpc = tps[i]->config_;
  size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;

  // 分配临时分区 buffer
  tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];

  // 复制对应分区的权重（注意 i * gate_up_elcount 偏移）
  memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
         (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
             i * gate_up_elcount,  // <-- 关键：按 NUMA 节点偏移
         sizeof(ggml_bf16_t) * gate_up_elcount);
}
```

**但 SFT 模式 `TP_MOE_SFT::load_weights()`** (moe-sft-tp.hpp:49-53) 没有做分区：

```cpp
void load_weights() override {
  auto pool = config.pool;
  // 直接调用各 NUMA 的 load_weights，没有先分区！
  pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });
  weights_loaded = true;
}
```

**导致的问题：**
1. 测试设置 `config.gate_proj` 指向完整权重张量
2. TP_MOE_SFT 构造时，各 NUMA 的 `tp_configs[i].intermediate_size` 被除以 `tp_count`
3. `load_weights()` 调用时，各 NUMA 的 `AMX_MOE_TP::load_weights()` 使用缩小后的 `intermediate_size` 计算偏移
4. 但源指针仍指向完整权重，导致各 NUMA 读取了错误的权重分区
5. NUMA 0 和 NUMA 1 读取相同或重叠的数据，而非正确的分区

### 解决方案

修改 `TP_MOE_SFT::load_weights()`，在加载前正确分区基础权重：

```cpp
void load_weights() override {
  auto pool = config.pool;

  // 如果 gate_proj 直接设置（非预量化），需要分区权重
  if (config.gate_proj != nullptr) {
    // 为每个 NUMA 节点分配临时分区 buffer
    for (int i = 0; i < tp_count; i++) {
      auto& tpc = tps[i]->config_;
      size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;

      tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
      tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
      tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];

      // 复制分区后的权重
      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, i](int expert_id) {
            // gate 和 up: [expert_num, intermediate_size, hidden_size]
            // 每个 NUMA 获取 intermediate_size 的一个切片
            memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
                   (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                       i * gate_up_elcount,
                   sizeof(ggml_bf16_t) * gate_up_elcount);

            memcpy((ggml_bf16_t*)tpc.up_proj + expert_id * gate_up_elcount,
                   (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                       i * gate_up_elcount,
                   sizeof(ggml_bf16_t) * gate_up_elcount);

            // down: [expert_num, hidden_size, intermediate_size]
            // 每个 NUMA 获取 intermediate_size 的一个切片（列）
            for (size_t row = 0; row < config.hidden_size; row++) {
              memcpy((ggml_bf16_t*)tpc.down_proj + expert_id * tpc.hidden_size * tpc.intermediate_size +
                         row * tpc.intermediate_size,
                     (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                         row * config.intermediate_size + i * tpc.intermediate_size,
                     sizeof(ggml_bf16_t) * tpc.intermediate_size);
            }
          },
          nullptr);
    }

    // 在各 NUMA 节点加载权重
    pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });

    // 清理临时 buffer
    for (int i = 0; i < tp_count; i++) {
      auto& tpc = tps[i]->config_;
      delete[] (ggml_bf16_t*)tpc.gate_proj;
      delete[] (ggml_bf16_t*)tpc.up_proj;
      delete[] (ggml_bf16_t*)tpc.down_proj;
    }
  } else {
    // 无需分区（预量化或无权重）
    pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });
  }

  weights_loaded = true;
}
```

### 关键知识点

1. **TP 模式权重分区**：当使用 Tensor Parallel 时，每个 NUMA 节点只处理 intermediate_size 的一部分。必须在加载前将完整权重按正确偏移分区到各节点。

2. **gate/up vs down 的分区方式不同**：
   - gate_proj, up_proj: 形状为 `[expert_num, intermediate_size, hidden_size]`，按 intermediate_size 维度切片（连续块）
   - down_proj: 形状为 `[expert_num, hidden_size, intermediate_size]`，按 intermediate_size 维度切片（需逐行复制）

3. **SFT 继承推理逻辑**：SFT 模式应尽量复用推理模式的基础设施，包括权重分区逻辑。

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/moe-sft-tp.hpp` | 重写 `load_weights()` 方法，添加 TP 权重分区逻辑 |
| `examples/test_moe_sft_amx.py` | 将输出 buffer dtype 从 `float32` 改为 `bfloat16` |

---

## Bug #9: Forward Cache Stack Overflow 【已修复】

### 问题现象

运行 `test_moe_sft_amx_no_tp.py` 时，第二次迭代崩溃：

```
--- Iteration 1 ---
terminate called after throwing an instance of 'std::runtime_error'
  what():  Forward cache stack overflow
Aborted (core dumped)
```

### 问题原因

测试配置 `max_cache_depth = 1`，但测试循环调用 `forward_sft` 两次且都设置 `save_for_backward=True`：

**sft_moe.hpp:604-609:**
```cpp
ForwardCache& push_cache() {
  if (cache_stack_top_ >= max_cache_depth_) {
    throw std::runtime_error("Forward cache stack overflow");
  }
  return cache_stack_[cache_stack_top_++];
}
```

**执行流程：**
1. 第一次 `forward_sft(save_for_backward=True)`: `cache_stack_top_` 从 0 变为 1
2. 第二次 `forward_sft(save_for_backward=True)`: `cache_stack_top_` = 1 >= `max_cache_depth_` = 1 → 抛出异常

### 解决方案

**方案 A：测试文件修改（临时解决）**

将 `save_for_backward` 设为 False（仅测试 forward 时不需要保存 cache）：

```python
# test_moe_sft_amx_no_tp.py
CPUInfer.submit(
    moe.forward_sft_task(
        ...
        False,  # save_for_backward = False
    )
)
```

**方案 B：增加 cache 深度**

```python
config.max_cache_depth = validation_iter  # 至少等于迭代次数
```

**方案 C：每次 forward 后调用 backward（pop cache）**

在训练场景中，每次 forward 后都应该有 backward 来消费 cache。

### 关键知识点

`ForwardCache` 是一个栈结构，用于梯度检查点（gradient checkpointing）。每次 `forward_sft(save_for_backward=True)` 会 push，每次 `backward()` 会 pop。如果只有 forward 没有 backward，栈会溢出。

---

## Bug #10: SFT Forward 数值差异分析（无 LoRA 相关）【已修复】

### 问题现象

非 TP 模式下 SFT forward 测试失败，但推理测试通过：

| 测试 | 相对误差 | 输出量级 | 结果 |
|------|---------|---------|------|
| 推理 test_moe_amx.py | 0.048 | ~80 | PASS |
| SFT test_moe_sft_amx_no_tp.py | 0.14 | ~1e-5 | FAIL |

**测试输出对比：**
```
# 推理测试
torch output: [-81.5000, -21.8750, ...]
amx output:   [-83.0000, -21.6250, ...]
diff = 0.048

# SFT 测试
torch output: [-1.2457e-05, -5.0366e-06, ...]
amx output:   [-1.2636e-05, -4.5598e-06, ...]
diff = 0.14
```

### 关键发现：权重初始化差异

**推理测试 (test_moe_amx.py:115-128, 187-189):**
```python
gate_proj = torch.randn(..., dtype=torch.bfloat16)  # ~1.0
up_proj = torch.randn(...)    # ~1.0
down_proj = torch.randn(...)  # ~1.0
input = torch.randn(...) / 100  # ~0.01
```

**SFT 测试 (test_moe_sft_amx.py:553-560, 676):**
```python
gate_proj = torch.randn(...) / 100  # ~0.01
up_proj = torch.randn(...) / 100    # ~0.01
down_proj = torch.randn(...) / 100  # ~0.01
input_data = torch.randn(...) / 100  # ~0.01
```

**输出量级计算：**
- 推理：output ≈ (0.01 × 1.0) × 1.0 × 1.0 × √(7168 × 2048) ≈ 数十
- SFT：output ≈ (0.01 × 0.01) × 0.01 × 0.01 × √(7168 × 2048) ≈ 1e-5

### 问题分析

当输出值很小时（~1e-5），相同的绝对误差会导致更大的相对误差：

```
# 假设绝对误差都是 1e-6
推理：relative_diff = 1e-6 / 80 = 1.25e-8
SFT： relative_diff = 1e-6 / 1e-5 = 0.1
```

**但这不能完全解释问题。** 0.14 的相对误差意味着 AMX 输出和 PyTorch 参考之间存在系统性差异。

### 潜在原因（待验证）

1. **LoRA 计算使用标量循环 vs AMX 使用矩阵分块**
   - LoRA 路径：逐元素 bf16→fp32→计算→fp32→bf16 转换
   - AMX 路径：批量处理，更少的精度损失

2. **中间结果 bf16 截断**
   ```cpp
   // sft_moe.hpp:521
   lora_intermediate_[t * lora_rank_ + r] = GGML_FP32_TO_BF16(sum);
   ```
   每次存储都损失精度

3. **小数值放大误差**
   - 当值接近 0 时，bf16 的有效精度下降
   - 1e-5 在 bf16 中只有约 2-3 位有效数字

### 验证方案

1. **禁用 LoRA 测试基础 GEMM 路径**
   - 将 `gate_lora_a`, `gate_lora_b` 等设为 nullptr
   - 预期：diff 应该接近推理测试的 0.048

2. **使用推理测试的权重初始化**
   - 不除以 100，使用正常量级权重
   - 预期：diff 应该显著下降

3. **对比单专家输出**
   - 在 C++ 和 Python 中打印同一个专家的中间结果
   - 定位具体哪一步引入了误差

### 问题解决

验证结果表明，问题根源是权重初始化除以 100 导致输出值过小（~1e-5），在 bf16 精度下相对误差放大。

**修复方案：** 移除权重初始化中的 `/100`，与推理测试保持一致。

**修复后结果：** 非 TP 模式 forward 测试通过（diff < 0.05）。

---

## Bug #11: PyTorch 参考实现中的 Dtype 不匹配（Backward 测试）【已修复】

### 问题现象

运行 `test_moe_sft_amx_no_tp.py` 的 backward 测试时崩溃：

```
[OK] MOE SFT Forward Pass Test - BF16 mode (NO TP) PASSED
...
--- Iteration 0 ---

[FAILED] Test failed with error: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16
Traceback (most recent call last):
  File ".../test_moe_sft_amx_no_tp.py", line 1326, in run_all_tests
    test_moe_sft_backward_no_tp()
  File ".../test_moe_sft_amx_no_tp.py", line 806, in test_moe_sft_backward_no_tp
    torch_grads = moe_sft_torch_backward(...)
  File ".../test_moe_sft_amx_no_tp.py", line 450, in moe_sft_torch_backward
    grads = mlp_lora_backward(...)
  File ".../test_moe_sft_amx_no_tp.py", line 234, in mlp_lora_backward
    grad_intermediate, ... = lora_linear_backward(...)
  File ".../test_moe_sft_amx_no_tp.py", line 123, in lora_linear_backward
    grad_input = torch.mm(grad_output, weight)
RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16
```

### 问题原因

**这是 PyTorch 参考实现的 bug，不是 C++ MoE 算子的问题。** 错误发生在 C++ backward 被调用之前。

**代码分析 (moe_sft_torch_backward 函数)：**

```python
# test_moe_sft_amx_no_tp.py:420-423
grad_output_expanded = grad_output.unsqueeze(1) * weights.unsqueeze(-1)
grad_output_expanded = grad_output_expanded.view(-1, grad_output.shape[-1])
```

数据类型转换：
- `grad_output`: `BFloat16` (来自上游梯度)
- `weights`: `Float32` (routing weights，由 `torch.rand()` 生成)
- `grad_output * weights` → **`Float32`** (PyTorch 自动向上转型)

后续调用链：
```
moe_sft_torch_backward()  → grad_output_expanded (float32)
    ↓
mlp_lora_backward()       → grad_output (float32)
    ↓
lora_linear_backward()    → torch.mm(grad_output, weight)
                                    ↓          ↓
                                float32     bf16  → TypeError!
```

### 解决方案

在 `moe_sft_torch_backward()` 中将 `grad_output_expanded` 转回 bf16：

```python
# 修改前
grad_output_expanded = grad_output.unsqueeze(1) * weights.unsqueeze(-1)
grad_output_expanded = grad_output_expanded.view(-1, grad_output.shape[-1])

# 修改后
grad_output_expanded = grad_output.unsqueeze(1) * weights.unsqueeze(-1)
grad_output_expanded = grad_output_expanded.view(-1, grad_output.shape[-1]).to(grad_output.dtype)
```

**需要修改的文件：**
- `examples/test_moe_sft_amx_no_tp.py`: `moe_sft_torch_backward()` 函数
- `examples/test_moe_sft_amx.py`: 同样的 `moe_sft_torch_backward()` 函数（如果存在同样问题）

### 关键知识点

1. **PyTorch 自动类型提升**：当 bf16 和 float32 张量进行运算时，结果自动提升为 float32。
2. **矩阵乘法要求类型匹配**：`torch.mm()` 要求两个输入张量类型相同。
3. **梯度类型应与激活类型一致**：在混合精度训练中，梯度应保持与对应激活相同的数据类型。

---

# 第三板块：对话历史摘要

本板块记录重要的调试对话和进展。

---

## 2024-12-31 ~ 2025-01-02: 非 TP 模式测试修复

### 进展摘要

1. **Bug #9 修复**：将 forward-only 测试的 `save_for_backward` 设为 `False`，避免 cache overflow。

2. **Bug #10 修复**：移除权重初始化中的 `/100`，使输出值保持正常量级（~80 而非 ~1e-5），降低 bf16 精度损失导致的相对误差。

3. **任务完成情况**：
   - ✓ 任务 1：同步 TP 测试文件修改（权重初始化、save_for_backward）
   - ✓ 任务 2：优化权重生成（CUDA → CPU）
   - ✓ 任务 3：添加 backward/LoRA 测试到非 TP 文件
     - 已添加 `lora_linear_backward()`, `mlp_lora_backward()`, `moe_sft_torch_backward()`
     - 已添加 `test_moe_sft_backward_no_tp()`, `test_moe_sft_lora_weight_sync_no_tp()`, `test_moe_sft_training_loop_no_tp()`

4. **Bug #11 修复**：PyTorch 参考实现中 dtype 不匹配已修复（添加 `.to(grad_output.dtype)`）。

### 当前状态

| 测试 | 状态 | 备注 |
|------|------|------|
| 非 TP forward | ✓ PASSED | 已修复 |
| 非 TP backward | ? 待验证 | Bug #12, #13, #14 已修复 |
| 非 TP weight sync | ? 待验证 | Bug #11 已修复 |
| 非 TP training loop | ? 待验证 | Bug #11 已修复 |
| TP forward | ? 待验证 | 已同步修改 |
| TP backward | ? 待验证 | Bug #11 已修复 |

---

## Bug #12: Backward pass 中 grad_intermediate 未被计算 【已修复】

### 问题现象

运行 `test_moe_sft_amx_no_tp.py` 的 backward 测试时：

```
[BACKWARD DEBUG] qlen=4, k=8, activated_expert=30, total_tokens=32
[BACKWARD DEBUG] grad_output norm: 1.680211          ← 有值
[BACKWARD DEBUG] After backward_down - grad_intermediate norm: 0.000000   ← 0！
[BACKWARD DEBUG] After backward_activation - grad_gate norm: 0.000000, grad_up norm: 0.000000
[BACKWARD DEBUG] After backward_gate_up - grad_input norm: 0.000000
```

`grad_input diff = 1.0`，backward 计算完全不正确。

### 问题原因

**文件**: `operators/amx/sft_moe.hpp`，`backward_down()` 函数

`backward_down()` 只计算了 LoRA 权重梯度，但**没有计算 `grad_intermediate = grad_output @ down_proj^T`**。

**正确的反向传播流程**:
```
grad_output [qlen, hidden_size]
    ↓ backward_down: grad_intermediate = grad_output @ down_proj^T  ← 缺失！
grad_intermediate [tokens, intermediate_size]
    ↓ backward_activation: SiLU backward
grad_gate, grad_up [tokens, intermediate_size]
    ↓ backward_gate_up: grad_input = grad_gate @ gate_W^T + grad_up @ up_W^T  ← Bug #14
grad_input [qlen, hidden_size]
```

原代码只有：
```cpp
// Line 713-714: 只是初始化为零，从未填充实际值！
memset(grad_intermediate_, 0, ...);
```

### 解决方案

在 `backward_down()` 中添加 `grad_intermediate = grad_output @ down_proj` 的计算：

```cpp
// Compute grad w.r.t. intermediate: grad_intermediate = grad_output @ down_proj
// down_proj layout: [expert_num, hidden_size, intermediate_size]
// grad_output: [num_tokens, hidden_size], grad_intermediate: [num_tokens, intermediate_size]
// grad_intermediate[t, i] = sum_h grad_output[t, h] * down_proj[h, i]
{
  const ggml_bf16_t* down_proj = (const ggml_bf16_t*)config_.down_proj;
  size_t expert_offset = (size_t)expert_idx * config_.hidden_size * config_.intermediate_size;

  // Compute offset into grad_intermediate_ for this expert
  size_t grad_inter_offset = 0;
  for (int e = 0; e < task_id; e++) {
    grad_inter_offset += m_local_num_[m_expert_id_map_[e]];
  }
  grad_inter_offset *= config_.intermediate_size;

  for (int t = 0; t < num_tokens; t++) {
    for (int i = 0; i < config_.intermediate_size; i++) {
      float sum = 0.0f;
      for (int h = 0; h < config_.hidden_size; h++) {
        float grad_out_val = expert_grad_out[t * config_.hidden_size + h];
        float down_val = GGML_BF16_TO_FP32(down_proj[expert_offset + h * config_.intermediate_size + i]);
        sum += grad_out_val * down_val;
      }
      grad_intermediate_[grad_inter_offset + t * config_.intermediate_size + i] = GGML_FP32_TO_BF16(sum);
    }
  }
}
```

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/amx/sft_moe.hpp` | 在 `backward_down()` 中添加 grad_intermediate 计算 |

---

## Bug #13: grad_input 数据类型错误导致内存损坏 【已修复】

### 问题现象

运行 backward 测试时程序崩溃：

```
*** Error in `python': double free or corruption (!prev): 0x00007f8d6c000010 ***
Aborted (core dumped)
```

GDB backtrace 显示问题在 `backward_gate_up()` 函数。

### 问题原因

**文件**: `operators/amx/sft_moe.hpp`，`backward_gate_up()` 函数

C++ 代码将 `grad_input` 当作 `float` (4 bytes) 处理：

```cpp
// 原代码 Line 855: 用 float (4 bytes) 初始化
memset(grad_input, 0, qlen * config_.hidden_size * sizeof(float));

// 原代码 Line 973: 当作 float* 写入
((float*)grad_input)[i * config_.hidden_size + h] += sum * lora_scaling_;
```

但 Python 传入的是 `torch.bfloat16` (2 bytes)！

**导致的问题：**
1. `memset` 清零了两倍的内存（越界）
2. 写入时错误地将 bf16 buffer 解释为 float，导致写入位置错误
3. 最终导致内存损坏和 double free

### 解决方案

将 `grad_input` 处理改为 bf16：

```cpp
// 修改后：用 bf16 (2 bytes) 初始化
memset(grad_input, 0, qlen * config_.hidden_size * sizeof(ggml_bf16_t));

// 修改后：用 bf16 累加
ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
// ...
float current = GGML_BF16_TO_FP32(grad_input_bf16[i * config_.hidden_size + h]);
grad_input_bf16[i * config_.hidden_size + h] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
```

同时修复 `backward_down()` 中 `grad_output` 的读取：

```cpp
// 修改后：从 bf16 读取
const ggml_bf16_t* grad_out_bf16 = (const ggml_bf16_t*)grad_output;
// ...
expert_grad_out[pos * config_.hidden_size + h] +=
    GGML_BF16_TO_FP32(grad_out_bf16[i * config_.hidden_size + h]) * w;
```

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/amx/sft_moe.hpp` | `backward_gate_up()` 和 `backward_down()` 中将 float 处理改为 bf16 |

---

## Bug #14: grad_input 缺少 base weight 贡献 【已修复】

### 问题现象

即使修复 Bug #12 和 #13 后，`grad_input` 计算仍然不完整。

### 问题原因

**文件**: `operators/amx/sft_moe.hpp`，`backward_gate_up()` 函数

原代码只计算了 LoRA 的贡献：
```cpp
// grad_input += grad @ lora_B @ lora_A * scaling
```

但缺少 base weight 的贡献：
```cpp
// 缺失：grad_input += grad_gate @ gate_proj^T + grad_up @ up_proj^T
```

### 解决方案

在 `backward_gate_up()` 中添加 base weight 贡献，并将其移到 LoRA 条件检查之前（确保即使没有 LoRA 也会计算）：

```cpp
// First, compute base weight contribution to grad_input (always, regardless of LoRA)
// grad_input += grad @ W^T (for gate or up, depending on do_up)
// W layout: [expert_num, intermediate_size, hidden_size]
// grad: [num_tokens, intermediate_size]
// grad_input[t, h] += sum_i grad[t, i] * W[i, h]
{
  ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
  const ggml_bf16_t* base_proj =
      do_up ? (const ggml_bf16_t*)config_.up_proj : (const ggml_bf16_t*)config_.gate_proj;
  size_t expert_offset = (size_t)expert_idx * config_.intermediate_size * config_.hidden_size;

  // Pre-compute grad_input contribution per token, then scatter
  std::vector<float> token_grad_input(num_tokens * config_.hidden_size, 0.0f);
  for (int t = 0; t < num_tokens; t++) {
    for (int h = 0; h < config_.hidden_size; h++) {
      float sum = 0.0f;
      for (int i = 0; i < config_.intermediate_size; i++) {
        float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
        float w = GGML_BF16_TO_FP32(base_proj[expert_offset + i * config_.hidden_size + h]);
        sum += g * w;
      }
      token_grad_input[t * config_.hidden_size + h] = sum;
    }
  }

  // Scatter back to grad_input
  for (int i = 0; i < qlen; i++) {
    for (int j = 0; j < k; j++) {
      if (cache.expert_ids_cache[i * k + j] == expert_idx) {
        int pos = cache.m_local_pos_cache[i][j];
        for (int h = 0; h < config_.hidden_size; h++) {
          float current = GGML_BF16_TO_FP32(grad_input_bf16[i * config_.hidden_size + h]);
          grad_input_bf16[i * config_.hidden_size + h] =
              GGML_FP32_TO_BF16(current + token_grad_input[pos * config_.hidden_size + h]);
        }
      }
    }
  }
}

// LoRA gradients and contribution - only if LoRA is enabled
if (lora_a == nullptr || lora_b == nullptr) return;
// ... LoRA computation continues ...
```

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/amx/sft_moe.hpp` | `backward_gate_up()` 中添加 base weight grad_input 贡献 |

### 关键知识点

**完整的 MoE 层反向传播公式：**

```
Forward:  y = (silu(x @ gate_W^T) * (x @ up_W^T)) @ down_W^T
        + LoRA 贡献（如果启用）

Backward:
  grad_intermediate = grad_output @ down_W
  grad_gate, grad_up = silu_backward(grad_intermediate, gate_out, up_out)
  grad_input = grad_gate @ gate_W + grad_up @ up_W
             + LoRA 贡献（如果启用）
```

---

## Bug #15: backward_activation 中 grad_up = 0 【已修复 - SharedMemBuffer 内存重叠】

### 第一轮调试输出

Bug #12, #13, #14 修复后，运行测试显示：

```
[BACKWARD DEBUG] qlen=4, k=8, activated_expert=30, total_tokens=32
[BACKWARD DEBUG] grad_output norm: 1.680211          ✓ 有值
[BACKWARD DEBUG] After backward_down - grad_intermediate norm: 32.011452   ✓ Bug #12 修复成功！
[BACKWARD DEBUG] After backward_activation - grad_gate norm: 13.238412, grad_up norm: 0.000000  ← Bug #15！
[BACKWARD DEBUG] After backward_gate_up - grad_input norm: 1116.860474
grad_input diff: 0.804688
```

**关键问题**：`grad_gate` 有值（13.238412），但 `grad_up` 是 0！

### 公式分析

**SiLU backward 公式**（在 `backward_activation()` 中）：

```cpp
float g = GGML_BF16_TO_FP32(gate_output[i]);         // 从 cache 读取
float u = GGML_BF16_TO_FP32(up_output[i]);           // 从 cache 读取
float sigmoid_g = 1.0f / (1.0f + expf(-g));
float silu_g = g * sigmoid_g;                         // silu(g) = g * sigmoid(g)

float grad_i = GGML_BF16_TO_FP32(grad_inter[i]);     // 从 backward_down 计算得到

// Compute gradients
float grad_gate_val = grad_i * u * sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));  // ≈ 13.24
float grad_up_val = grad_i * silu_g;                                               // = 0 ！
```

**推论**：
- 如果 `grad_gate_val ≠ 0`，说明 `grad_i`, `u`, `sigmoid_g` 都有值
- 如果 `grad_up_val = 0`，那么 `silu_g = g * sigmoid_g ≈ 0`
- 由于 `sigmoid_g ∈ (0, 1)` 且不可能为 0，所以必然是 **`g (gate_output) ≈ 0`**

### 第二轮调试输出（关键发现）

```
[DEBUG save_to_cache] total_tokens=32, gate_output_cache[0..7] = 0.1689 -1.0078 0.0410 -0.2109 1.3203 -1.3203 0.0077 -0.1904

[BACKWARD DEBUG] qlen=4, k=8, activated_expert=30, total_tokens=32
[DEBUG backward_activation] task_id=0, expert_idx=0, num_tokens=1, offset=0
[DEBUG] gate_output[0..7] = 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000   ← 全零！
[DEBUG] up_output[0..7] = -0.0244 1.0156 -0.1816 -0.0011 1.2500 0.0889 -0.5117 -0.0796  ← 有值！
[DEBUG] grad_inter[0..7] = -0.2129 0.2871 -0.3789 -0.1934 0.3945 0.3203 -0.3008 0.1079
```

### 关键发现：内存覆盖问题！

**奇怪现象**：
1. `save_to_cache` 时 `gate_output_cache[0..7]` 有正常值（0.1689, -1.0078, ...）
2. `backward_activation` 时读取同一个 offset=0，但 `gate_output` 全是 0
3. **同一个 offset 的 `up_output` 却有值！**

**这是不可能的**——两者使用相同的 offset 读取，但结果不同。唯一的解释是：

**`cache.gate_output_cache` 指向的内存被覆盖了，而 `cache.up_output_cache` 没有。**

### 可能的内存覆盖来源

`gate_output_cache` 和 `up_output_cache` 使用**不同的内存池**：

```cpp
// init_cache_buffers() 中
cache_stack_[i].gate_output_cache = (ggml_bf16_t*)cache_gate_output_pool_ + ...;
cache_stack_[i].up_output_cache = (ggml_bf16_t*)cache_up_output_pool_ + ...;
```

**嫌疑最大**：`backward_down()` 中的 memset

```cpp
// backward_down() 第 720-721 行
memset(grad_intermediate_, 0,
       config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t));
```

如果 `shared_mem_buffer_numa.alloc()` 在多次调用时复用了相同的内存区域，那么 `grad_intermediate_` 可能与 `cache.gate_output_cache` 指向相同（或重叠）的内存！

### 已添加的内存地址调试代码

```cpp
// save_to_cache 中添加
printf("[DEBUG ADDR] cache.gate_output_cache = %p, cache.up_output_cache = %p\n", ...);

// backward_down 中添加
printf("[DEBUG ADDR backward_down] grad_intermediate_ = %p\n", ...);
printf("[DEBUG ADDR backward_down] cache.gate_output_cache = %p\n", ...);
printf("[DEBUG BEFORE memset] gate_cache[0..3] = ...\n");
memset(grad_intermediate_, 0, ...);
printf("[DEBUG AFTER memset] gate_cache[0..3] = ...\n");
```

### 预期调试结果

运行后，如果看到：
1. `grad_intermediate_` 和 `cache.gate_output_cache` 地址相同或接近
2. `BEFORE memset` 有值，`AFTER memset` 变成 0

→ 确认内存覆盖问题。

### 修复方案

**合并所有 buffer 分配到一个 `alloc()` 调用**：

当前代码分三次调用 `alloc()`：
1. `init_lora_buffers()` - 分配 LoRA 中间 buffer
2. `init_cache_buffers()` - 分配 cache buffer
3. `init_grad_buffers()` - 分配梯度 buffer

**修复后**：合并到一个 `init_buffers()` 函数：

```cpp
void init_buffers() {
  MemoryRequest mem_requests;

  // LoRA buffers
  mem_requests.append_pointer(&lora_intermediate_pool_, lora_intermediate_pool_bytes_);

  // Cache buffers
  mem_requests.append_pointer(&cache_input_pool_, cache_slot_bytes_input_ * max_cache_depth_);
  mem_requests.append_pointer(&cache_gate_output_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);
  mem_requests.append_pointer(&cache_up_output_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);
  mem_requests.append_pointer(&cache_intermediate_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);

  // Gradient buffers
  mem_requests.append_pointer(&grad_intermediate_pool_, grad_buffer_bytes);
  mem_requests.append_pointer(&grad_gate_output_pool_, grad_buffer_bytes);
  mem_requests.append_pointer(&grad_up_output_pool_, grad_buffer_bytes);

  // Single allocation for all buffers
  shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

  // Initialize pointers after allocation
  // ...
}
```

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/amx/sft_moe.hpp` | 合并 buffer 分配，修复内存重叠问题 |

### 第三轮调试输出（确认根因）

根据调试代码输出：

```
[DEBUG ADDR] cache.gate_output_cache = 0x7f0703aa7040
[DEBUG ADDR] cache.up_output_cache   = 0x7f0735aa7040
[DEBUG ADDR backward_down] grad_intermediate_ = 0x7f06edca7040
[DEBUG BEFORE memset] gate_cache[0..3] = 0.1689 -1.0078 0.0410 -0.2109
[DEBUG AFTER memset] gate_cache[0..3] = 0.0000 0.0000 0.0000 0.0000  ← 被清零！
```

**确认内存覆盖！** `grad_intermediate_` 的 memset（800 MB）覆盖了 `cache.gate_output_cache`。

### SharedMemBuffer 工作原理（根因分析）

查看 `/home/lpl/ktransformers/kt-kernel/cpu_backend/shared_mem_buffer.cpp:49-73`：

```cpp
void SharedMemBuffer::alloc(void* object, MemoryRequest requests) {
  size_t total_size = requests.total_size();
  object_requests.push_back(requests);

  if (total_size > size) {
    buffer = posix_memalign(..., total_size);
    size = total_size;
    // ★ 关键：所有请求都从同一个 base 开始！
    for (auto& req : object_requests) {
      req.update_base_ptr(buffer);
    }
  } else {
    requests.update_base_ptr(buffer);
  }
}
```

**设计意图**：SharedMemBuffer 是一个**共享内存池**，让多个临时 buffer 可以复用同一块内存。

**问题**：SFT 的 cache buffer 和 grad buffer **不是临时的**——它们需要**同时存在**！

### 内存分配顺序（问题所在）

原代码在构造函数中分三次调用 `alloc()`：

```cpp
init_lora_buffers();   // alloc #1: ~1 MB
init_cache_buffers();  // alloc #2: ~800 MB
init_grad_buffers();   // alloc #3: ~800 MB
```

由于 SharedMemBuffer 让所有请求从同一个 base 开始：

```
SharedMemBuffer (size = 800 MB):
+---------------------------------------------------------------------+
| 0                                                        800 MB     |
+---------------------------------------------------------------------+
                              ↑
          cache_gate_output_pool_ 从某个 offset 开始
          grad_intermediate_pool_ 从 0 开始 ← 覆盖 cache!
```

memset 大小 = 25600 × 8 × 2048 × 2 = 800 MB，覆盖了 `cache.gate_output_cache`。

### 最终修复方案

**合并所有 buffer 到单次 `alloc()` 调用**，确保所有 buffer 获得连续、不重叠的地址。

**新函数 `init_all_buffers()`**：

```cpp
void init_all_buffers() {
  // 计算所有 buffer 大小
  lora_intermediate_pool_bytes_ = sizeof(ggml_bf16_t) * config_.max_len *
                                  config_.num_experts_per_tok * lora_rank_;
  cache_slot_bytes_input_ = config_.max_len * config_.hidden_size * sizeof(ggml_bf16_t);
  cache_slot_bytes_intermediate_ =
      config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);
  size_t grad_buffer_bytes =
      config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);

  // ★ 单次 alloc() 调用，所有 buffer 获得连续地址 ★
  MemoryRequest mem_requests;

  // LoRA buffers
  mem_requests.append_pointer(&lora_intermediate_pool_, lora_intermediate_pool_bytes_);

  // Cache buffers (4 个 pool × max_cache_depth)
  mem_requests.append_pointer(&cache_input_pool_, cache_slot_bytes_input_ * max_cache_depth_);
  mem_requests.append_pointer(&cache_gate_output_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);
  mem_requests.append_pointer(&cache_up_output_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);
  mem_requests.append_pointer(&cache_intermediate_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);

  // Gradient buffers (3 个 pool)
  mem_requests.append_pointer(&grad_intermediate_pool_, grad_buffer_bytes);
  mem_requests.append_pointer(&grad_gate_output_pool_, grad_buffer_bytes);
  mem_requests.append_pointer(&grad_up_output_pool_, grad_buffer_bytes);

  // 单次分配
  shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

  // 初始化指针和 cache stack...
}
```

**构造函数修改**：

```cpp
// 原代码（删除）
init_lora_buffers();
init_cache_buffers();
init_grad_buffers();

// 新代码
init_all_buffers();
```

---

## Bug #16: LoRA 指针 Object Slicing 导致 LoRA 梯度全零 【已修复】

### 问题现象

Bug #15 修复后，运行测试显示：

```
[DEBUG BEFORE memset] gate_cache[0..3] = 0.1689 -1.0078 0.0410 -0.2109
[DEBUG AFTER memset] gate_cache[0..3] = 0.1689 -1.0078 0.0410 -0.2109  ← Bug #15 已修复！
grad_input diff: 0.006775     ← 正确！
gate_lora_a diff: 1.000000    ← 完全错误！
```

`diff = 1.0` 意味着 C++ 输出全零，而 PyTorch 有非零值。LoRA 梯度没有被计算。

### 问题原因

**根因：C++ Object Slicing**

**继承链：**
```
TP_MOE_SFT<T>
  ↓ 继承
TP_MOE<T>
  ↓ 存储
GeneralMOEConfig config;  // 不是 MOESFTConfig！
```

**关键代码 (moe-tp.hpp:115-123)：**

```cpp
for (auto i = 0; i < tp_count; i++) {
  tps.push_back(nullptr);
  GeneralMOEConfig tp_config = config;  // ★ Object Slicing！★
  tp_config.intermediate_size /= tp_count;
  tp_configs.push_back(tp_config);
}

config.pool->dispense_backend()->do_numa_job(
    [this, config](int i) {
      tps[i] = std::move(std::unique_ptr<T>(new T(tp_configs[i], i)));  // ★ LoRA 指针丢失！★
    });
```

当 `config` 是 `MOESFTConfig` 时，`GeneralMOEConfig tp_config = config` 会切片掉所有 SFT 特有字段：
- `gate_lora_a`, `gate_lora_b` → nullptr
- `up_lora_a`, `up_lora_b` → nullptr
- `down_lora_a`, `down_lora_b` → nullptr

**AMX_SFT_MOE_TP 构造函数：**

```cpp
// sft_moe.hpp:142-162
AMX_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
    : Base(static_cast<GeneralMOEConfig>(config), tp_part_idx), sft_config_(config) {
  // ...
  gate_lora_a_ = (ggml_bf16_t*)config.gate_lora_a;  // config.gate_lora_a = nullptr!
  gate_lora_b_ = (ggml_bf16_t*)config.gate_lora_b;  // config.gate_lora_b = nullptr!
  // ...
}

// 满足 concept 的构造函数 (被 TP_MOE 调用)
AMX_SFT_MOE_TP(GeneralMOEConfig config, int tp_part_idx)
    : AMX_SFT_MOE_TP(MOESFTConfig(config), tp_part_idx) {}  // ★ 使用默认 LoRA 值 (nullptr)！★
```

**结果：**

1. `TP_MOE_SFT` 构造时，基类 `TP_MOE<T>` 创建 `tps[i]` 使用 `GeneralMOEConfig`
2. `AMX_SFT_MOE_TP` 被调用 `GeneralMOEConfig` 构造函数，转换为 `MOESFTConfig` 时 LoRA 指针为 nullptr
3. `backward_gate_up` 中检查 `if (lora_a == nullptr || lora_b == nullptr) return;` → 早期返回，不计算 LoRA 梯度
4. LoRA 梯度 buffer 保持全零

### 解决方案

在 `TP_MOE_SFT` 构造函数中调用 `update_lora_weights()` 将 LoRA 指针传递给所有 NUMA 节点的实例。

**文件**: `/home/lpl/ktransformers/kt-kernel/operators/moe-sft-tp.hpp`

**修改后的构造函数：**

```cpp
TP_MOE_SFT(MOESFTConfig config) : Base(static_cast<GeneralMOEConfig>(config)), sft_config(config) {
  printf("Creating TP_MOE_SFT layer %d\n", config.layer_idx);

  // ★ Bug #16 fix: 将 LoRA 指针传递给所有 NUMA 节点的实例 ★
  if (config.gate_lora_a != nullptr) {
    update_lora_weights(
        config.gate_lora_a, config.gate_lora_b,
        config.up_lora_a, config.up_lora_b,
        config.down_lora_a, config.down_lora_b);
  }
}
```

这会调用 `AMX_SFT_MOE_TP::update_lora_weights()` 为每个实例设置正确的 LoRA 指针。

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/moe-sft-tp.hpp` | 在 `TP_MOE_SFT` 构造函数中调用 `update_lora_weights()` |

### 关键知识点

**C++ Object Slicing**：当派生类对象赋值给基类对象时，派生类特有的成员会被"切掉"。这在模板继承中尤其危险，因为基类可能存储的是基类类型而非派生类类型。

**解决方案选择**：
1. ✗ 修改 `TP_MOE` 基类存储 `MOESFTConfig` — 会破坏非 SFT 的 MoE 使用
2. ✗ 为每个派生类创建模板特化 — 维护困难
3. ✓ 在派生类构造函数中手动传递丢失的字段 — 简单有效

---

## 2025-01-02: Backward Pass 完整修复

### 进展摘要

1. **调试验证**：添加了 `compute_bf16_norm()` 和 `compute_f32_norm()` 辅助函数，在 `backward()` 各阶段打印 norm 值，确认问题分析正确。

2. **Bug #12 修复**：在 `backward_down()` 中添加了 `grad_intermediate = grad_output @ down_proj` 的计算。

3. **Bug #13 修复**：将 `grad_input` 和 `grad_output` 的处理从 float 改为 bf16，修复内存损坏问题。

4. **Bug #14 修复**：在 `backward_gate_up()` 中添加了 base weight 对 grad_input 的贡献，并将其移到 LoRA 条件检查之前。

5. **Bug #15 修复**：发现 `grad_up = 0` 问题，根因是 SharedMemBuffer 多次 `alloc()` 调用导致内存重叠。修复方案：合并所有 buffer 分配到单个 `init_all_buffers()` 函数。

6. **Bug #16 修复**：发现 `gate_lora_a diff = 1.0` 问题，根因是 C++ Object Slicing 导致 LoRA 指针丢失。修复方案：在 `TP_MOE_SFT` 构造函数中调用 `update_lora_weights()` 传递 LoRA 指针。

### 当前状态

| 测试 | 状态 | 备注 |
|------|------|------|
| 非 TP forward | ✓ PASSED | 已修复 |
| 非 TP backward | ✓ 代码已修复 | Bug #12, #13, #14, #15, #16 已修复，待验证 |
| 非 TP weight sync | ? 待验证 | 依赖 backward |
| 非 TP training loop | ? 待验证 | 依赖 backward |

### Bug 修复总结

| Bug | 问题 | 状态 |
|-----|------|------|
| Bug #12 | grad_intermediate 未计算 | ✓ 已修复 |
| Bug #13 | grad_input/grad_output 数据类型错误 | ✓ 已修复 |
| Bug #14 | grad_input 缺少 base weight 贡献 | ✓ 已修复 |
| Bug #15 | SharedMemBuffer 内存重叠 | ✓ 已修复 |
| Bug #16 | LoRA 指针 Object Slicing | ✓ 已修复 |
| Bug #17a | save_to_cache 存储 m_local_input_ (expert-sorted) | ✓ 已修复 |
| Bug #17b | backward_gate_up 需要原始 token order 的 input | ✓ 已修复 |
| Bug #17c | backward_down 使用 gate_output_cache (激活前) | ✓ 已修复 |

---

## 2026-01-02: Bug #17 系列修复

### Bug #17a & #17b: input_cache 与原始输入不一致

**现象**：
```
[TORCH DEBUG] x[0, 0:8] = [-1.7700e-03,  1.8921e-03, ...]
[DEBUG] expert_input[0..7] = 0.0156 -0.0084 ...
```
值完全不同，甚至符号都不同。

**根因分析**：
- `save_to_cache` 之前将 `m_local_input_` 复制到 cache
- `m_local_input_` 是 **expert-sorted layout**（按专家排序）
- 但 `backward_gate_up` 从 cache 读取时假设是 **原始 token order**

**修复方案**：
1. 修改 `save_to_cache` 函数签名，添加 `const void* input` 参数
2. 复制原始 `input` 而非 `m_local_input_`
3. 修改 `forward_sft` 调用时传入 `input` 参数

**代码修改** (sft_moe.hpp):
```cpp
// 修改前
void save_to_cache(ForwardCache& cache, int qlen, int k, const int64_t* expert_ids,
                   const float* weights, int activated_expert) {
  // ...
  memcpy(cache.input_cache, m_local_input_, qlen * config_.hidden_size * sizeof(ggml_bf16_t));
}

// 修改后
void save_to_cache(ForwardCache& cache, int qlen, int k, const int64_t* expert_ids,
                   const float* weights, int activated_expert, const void* input) {
  // ...
  // Bug #17b fix: 存储原始 input (token order)，而非 m_local_input_ (expert-sorted)
  memcpy(cache.input_cache, input, qlen * config_.hidden_size * sizeof(ggml_bf16_t));
}
```

### Bug #17c: backward_down 使用错误的 intermediate

**现象**：
```
gate_lora_a diff: 0.005066  ✓ 正确
up_lora_a diff: 0.004456   ✓ 正确
down_lora_a diff: 3.031250  ✗ 失败
```

**根因分析**：

Forward 流程：
```cpp
// Save gate/up outputs before activation
if (save_for_backward) {
  save_to_cache(cache, ...);  // ★ 保存激活前的 gate/up
}
// Step 6: Activation (silu(gate) * up)
Base::apply_activation(activated_expert, nth, qlen);  // ★ m_local_gate_output_ 变为 intermediate
```

`cache.gate_output_cache` = gate 输出 (**激活前**)
`cache.intermediate_cache` = **未保存！**

Backward 代码：
```cpp
const ggml_bf16_t* cached_intermediate = cache.gate_output_cache + cache_offset * ...;
// ★ 错误：使用激活前的 gate_output，而非激活后的 intermediate！
```

Down LoRA 梯度公式需要的是 `intermediate = silu(gate) * up`（激活后），不是 `gate`（激活前）！

**修复方案**：

1. 添加 `save_intermediate_to_cache` 函数：
```cpp
void save_intermediate_to_cache(ForwardCache& cache, int activated_expert) {
  size_t offset = 0;
  for (int i = 0; i < activated_expert; i++) {
    int expert_idx = m_expert_id_map_[i];
    int num_tokens = m_local_num_[expert_idx];
    // m_local_gate_output_ptr_ 现在包含 intermediate (激活后: silu(gate) * up)
    memcpy(cache.intermediate_cache + offset * config_.intermediate_size,
           m_local_gate_output_ptr_[expert_idx],
           num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
    offset += num_tokens;
  }
}
```

2. 在 `apply_activation` **之后**调用：
```cpp
// Step 6: Activation (silu(gate) * up)
Base::apply_activation(activated_expert, nth, qlen);

// Bug #17c fix: 保存激活后的 intermediate
if (save_for_backward) {
  ForwardCache& cache = cache_stack_[cache_stack_top_ - 1];
  save_intermediate_to_cache(cache, activated_expert);
}
```

3. 修改 `backward_down` 使用 `cache.intermediate_cache`：
```cpp
// 修改前（错误）
const ggml_bf16_t* cached_intermediate = cache.gate_output_cache + cache_offset * ...;

// 修改后（正确）
const ggml_bf16_t* cached_intermediate = cache.intermediate_cache + cache_offset * ...;
```

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/amx/sft_moe.hpp` | Bug #17a/b/c: save_to_cache, save_intermediate_to_cache, backward_down |

---
