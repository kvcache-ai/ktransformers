/**
 * @file mesh_hook.hpp
 * @brief MESH inline hook 函数
 *
 * 所有 hook 函数：mesh_mgr == nullptr 时编译期优化为空操作。
 * 原版 KT 代码通过调用这些 hook 接入 MESH，无需 include mesh.hpp。
 *
 * 使用方式（在 amx/moe_base.hpp 等 KT 原版文件中）：
 *   void* gate_bb = mesh::hook::get_gate_bb(mesh_mgr, layer, tp, expert_id);
 *   if (gate_bb) { /* MESH 路径 *\/ } else { /* 原版路径 *\/ }
 */
#pragma once

#include <cstdint>

namespace mesh {
namespace hook {

// 前向声明，避免强制 include mesh_residency.hpp
class MeshResidencyManager;

// ===== 注册函数指针（由 ext_bindings.cpp 在模块初始化时调用）=====

// 这些函数指针用于解耦 mesh_hook.hpp 和 mesh_residency.hpp
// 避免原版 KT 文件 include mesh_residency.hpp 带来的编译依赖
struct HookRegistry {
  void* (*get_gate_ptr)(void* mgr, int layer, int tp, int expert_id) = nullptr;
  void* (*get_up_ptr)(void* mgr, int layer, int tp, int expert_id) = nullptr;
  void* (*get_down_ptr)(void* mgr, int layer, int tp, int expert_id) = nullptr;
  // 同步加载版本：slot 未命中时阻塞从 SSD 加载到 slot
  void* (*load_gate_ptr)(void* mgr, int layer, int tp, int expert_id) = nullptr;
  void* (*load_up_ptr)(void* mgr, int layer, int tp, int expert_id) = nullptr;
  void* (*load_down_ptr)(void* mgr, int layer, int tp, int expert_id) = nullptr;
  // 引用计数：防止 AMX 计算期间 slot 被驱逐
  void (*acquire_reader)(void* mgr, int layer, int tp, int expert_id) = nullptr;
  void (*release_reader)(void* mgr, int layer, int tp, int expert_id) = nullptr;
  // Bug 1 fix: decode 层级回调 — 提交异步预取 + 更新 Heat/Markov
  void (*on_decode_layer)(void* mgr, int layer, int tp, const int* topk, int k, const float* weights, int expert_num) = nullptr;
};

inline HookRegistry& get_registry() {
  static HookRegistry registry;
  return registry;
}

// 注册 hook 函数（ext_bindings.cpp 调用）
inline void register_hooks(HookRegistry r) {
  get_registry() = r;
}

// ===== 权重指针重定向 hook =====

// 获取 gate 矩阵指针（MESH 启用时返回 slot buffer，否则返回 nullptr 走原版）
// mesh_mgr 为 nullptr 时直接返回 nullptr，编译期可优化
inline void* get_gate_bb(void* mesh_mgr, int layer, int tp, int expert_id) {
  if (!mesh_mgr) return nullptr;
  auto fn = get_registry().get_gate_ptr;
  if (!fn) return nullptr;
  return fn(mesh_mgr, layer, tp, expert_id);
}

inline void* get_up_bb(void* mesh_mgr, int layer, int tp, int expert_id) {
  if (!mesh_mgr) return nullptr;
  auto fn = get_registry().get_up_ptr;
  if (!fn) return nullptr;
  return fn(mesh_mgr, layer, tp, expert_id);
}

inline void* get_down_bb(void* mesh_mgr, int layer, int tp, int expert_id) {
  if (!mesh_mgr) return nullptr;
  auto fn = get_registry().get_down_ptr;
  if (!fn) return nullptr;
  return fn(mesh_mgr, layer, tp, expert_id);
}

// ===== 同步加载 hook（slot 未命中时阻塞加载）=====
// MESH 模式下 gate_bb_ 为空，slot 未命中不能回退，必须同步从 SSD 加载到 slot

inline void* load_gate_bb(void* mesh_mgr, int layer, int tp, int expert_id) {
  if (!mesh_mgr) return nullptr;
  auto fn = get_registry().load_gate_ptr;
  if (!fn) return nullptr;
  return fn(mesh_mgr, layer, tp, expert_id);
}

inline void* load_up_bb(void* mesh_mgr, int layer, int tp, int expert_id) {
  if (!mesh_mgr) return nullptr;
  auto fn = get_registry().load_up_ptr;
  if (!fn) return nullptr;
  return fn(mesh_mgr, layer, tp, expert_id);
}

inline void* load_down_bb(void* mesh_mgr, int layer, int tp, int expert_id) {
  if (!mesh_mgr) return nullptr;
  auto fn = get_registry().load_down_ptr;
  if (!fn) return nullptr;
  return fn(mesh_mgr, layer, tp, expert_id);
}

// ===== 引用计数 hook =====
// 防止 AMX 计算期间 slot 被驱逐（overwrite 会 spin-wait active_readers 归零）

inline void acquire_reader(void* mesh_mgr, int layer, int tp, int expert_id) {
  if (!mesh_mgr) return;
  auto fn = get_registry().acquire_reader;
  if (!fn) return;
  fn(mesh_mgr, layer, tp, expert_id);
}

inline void release_reader(void* mesh_mgr, int layer, int tp, int expert_id) {
  if (!mesh_mgr) return;
  auto fn = get_registry().release_reader;
  if (!fn) return;
  fn(mesh_mgr, layer, tp, expert_id);
}

// ===== MESH 启用判断 hook =====

inline bool is_mesh_enabled(void* mesh_mgr) {
  return mesh_mgr != nullptr;
}

// ===== Bug 1 fix: decode 层级回调 hook =====
// 在 forward_decode 的 GEMM 之前调用，提交异步 io_uring 预取 + 更新 Heat/Markov
inline void on_decode_layer(void* mesh_mgr, int layer, int tp,
                            const int* topk, int k, const float* weights, int expert_num) {
  if (!mesh_mgr) return;
  auto fn = get_registry().on_decode_layer;
  if (!fn) return;
  fn(mesh_mgr, layer, tp, topk, k, weights, expert_num);
}

}  // namespace hook
}  // namespace mesh
