# SFT MoE AMX Bug 调试记录

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
