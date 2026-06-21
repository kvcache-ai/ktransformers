/**
 * @file mesh_slot_pool.hpp
 * @brief MESH Slot 池 + 专家状态机
 *
 * 每个 MoE 层、每个 TP/NUMA 分片各自维护一个固定大小的 slot 数组。
 * slot 不存在释放语义，全部都是专家权重的内存覆盖。
 *
 * 状态机：BASELINE → LOADING → CACHED →（驱逐）→ DEMOTING → BASELINE
 * slot_active_readers_ 保证驱逐时无 reader（AMX 计算 / GPU 搬运 / 内部访问）
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <numa.h>
#include <numaif.h>
#include <stdexcept>
#include <vector>

namespace mesh {

// 专家生命周期状态
enum class ExpertState : uint8_t {
  BASELINE,   // 不在缓存，权重在磁盘上
  LOADING,    // 正在从磁盘读取，io_uring CQE 还没到
  CACHED,     // 已在 slot 中，AMX kernel 可直接消费
  DEMOTING,   // 正在被驱逐，等待活跃 reader 归零后解绑
};

// 单个 slot 的元数据
struct Slot {
  std::atomic<uint8_t> state{static_cast<uint8_t>(ExpertState::BASELINE)};
  std::atomic<int> active_readers{0};  // 引用计数：AMX 计算 / GPU 搬运 / 内部访问
  int bound_expert_id{-1};             // 当前 slot 装的是哪个专家，-1 = 空

  // std::atomic 不可拷贝/移动，需自定义 move 以支持 std::vector
  Slot() = default;
  Slot(Slot&& other) noexcept
      : state(other.state.load(std::memory_order_relaxed)),
        active_readers(other.active_readers.load(std::memory_order_relaxed)),
        bound_expert_id(other.bound_expert_id) {}
  Slot& operator=(Slot&& other) noexcept {
    if (this != &other) {
      state.store(other.state.load(std::memory_order_relaxed), std::memory_order_relaxed);
      active_readers.store(other.active_readers.load(std::memory_order_relaxed), std::memory_order_relaxed);
      bound_expert_id = other.bound_expert_id;
    }
    return *this;
  }
  Slot(const Slot&) = delete;
  Slot& operator=(const Slot&) = delete;

  ExpertState get_state() const {
    return static_cast<ExpertState>(state.load(std::memory_order_acquire));
  }
  void set_state(ExpertState s) {
    state.store(static_cast<uint8_t>(s), std::memory_order_release);
  }
};

/**
 * @brief Slot 池：每层每 TP 一个实例
 *
 * 内存布局：cap 个 slot 连续分配在 NUMA 本地内存上。
 * 每个 slot 包含 gate + up + down 三个 buffer，大小相同。
 *
 * 两种索引方式：
 * - slots_[i]：按 slot 位置索引，用于驱逐扫描
 * - expert_to_slot_[expert_id]：按专家 ID 索引，用于 KT 计算取指针
 */
class MeshSlotPool {
 public:
  MeshSlotPool(int layer_idx, int tp_part_idx, int numa_node, int cap,
               size_t slot_bytes)
      : layer_idx_(layer_idx),
        tp_part_idx_(tp_part_idx),
        numa_node_(numa_node),
        cap_(cap),
        slot_bytes_(slot_bytes) {
    if (cap <= 0) {
      throw std::runtime_error("MeshSlotPool: cap must be positive");
    }
    // 一次性 numa_alloc_onnode 分配全部 slot 内存，不在关键路径 alloc/free
    total_bytes_ = static_cast<size_t>(cap) * slot_bytes_;
    memory_ = numa_alloc_onnode(total_bytes_, numa_node_);
    if (!memory_) {
      throw std::runtime_error("MeshSlotPool: numa_alloc_onnode failed");
    }
    std::memset(memory_, 0, total_bytes_);

    slots_.resize(cap);
    slot_to_expert_.assign(cap, -1);
  }

  ~MeshSlotPool() {
    if (memory_) {
      numa_free(memory_, total_bytes_);
    }
  }

  // 禁止拷贝
  MeshSlotPool(const MeshSlotPool&) = delete;
  MeshSlotPool& operator=(const MeshSlotPool&) = delete;

  // 允许移动（std::vector 需要）
  MeshSlotPool(MeshSlotPool&& other) noexcept
      : slots_(std::move(other.slots_)),
        expert_to_slot_(std::move(other.expert_to_slot_)),
        slot_to_expert_(std::move(other.slot_to_expert_)),
        layer_idx_(other.layer_idx_),
        tp_part_idx_(other.tp_part_idx_),
        numa_node_(other.numa_node_),
        cap_(other.cap_),
        slot_bytes_(other.slot_bytes_),
        gate_up_bytes_(other.gate_up_bytes_),
        memory_(other.memory_),
        total_bytes_(other.total_bytes_) {
    other.memory_ = nullptr;
    other.total_bytes_ = 0;
  }
  MeshSlotPool& operator=(MeshSlotPool&& other) noexcept {
    if (this != &other) {
      if (memory_) numa_free(memory_, total_bytes_);
      slots_ = std::move(other.slots_);
      expert_to_slot_ = std::move(other.expert_to_slot_);
      slot_to_expert_ = std::move(other.slot_to_expert_);
      layer_idx_ = other.layer_idx_;
      tp_part_idx_ = other.tp_part_idx_;
      numa_node_ = other.numa_node_;
      cap_ = other.cap_;
      slot_bytes_ = other.slot_bytes_;
      gate_up_bytes_ = other.gate_up_bytes_;
      memory_ = other.memory_;
      total_bytes_ = other.total_bytes_;
      other.memory_ = nullptr;
      other.total_bytes_ = 0;
    }
    return *this;
  }

  // ===== 指针访问 =====

  // 获取 slot 中 gate 矩阵的地址
  void* gate_ptr(int slot_idx) {
    return static_cast<char*>(memory_) + static_cast<size_t>(slot_idx) * slot_bytes_ + 0;
  }
  // 获取 slot 中 up 矩阵的地址
  void* up_ptr(int slot_idx) {
    return static_cast<char*>(memory_) + static_cast<size_t>(slot_idx) * slot_bytes_ + gate_up_bytes_;
  }
  // 获取 slot 中 down 矩阵的地址
  void* down_ptr(int slot_idx) {
    return static_cast<char*>(memory_) + static_cast<size_t>(slot_idx) * slot_bytes_ + gate_up_bytes_ * 2;
  }

  // 根据专家 ID 获取 gate 指针（KT 计算用），未缓存返回 nullptr
  void* expert_gate_ptr(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return nullptr;
    if (slots_[slot_idx].get_state() != ExpertState::CACHED) return nullptr;
    return gate_ptr(slot_idx);
  }
  void* expert_up_ptr(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return nullptr;
    if (slots_[slot_idx].get_state() != ExpertState::CACHED) return nullptr;
    return up_ptr(slot_idx);
  }
  void* expert_down_ptr(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return nullptr;
    if (slots_[slot_idx].get_state() != ExpertState::CACHED) return nullptr;
    return down_ptr(slot_idx);
  }

  // ===== 状态查询 =====

  bool is_cached(int expert_id) const {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return false;
    return slots_[slot_idx].get_state() == ExpertState::CACHED;
  }

  ExpertState get_expert_state(int expert_id) const {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return ExpertState::BASELINE;
    return slots_[slot_idx].get_state();
  }

  // B1 fix: 查询专家绑定的 slot_idx（驱逐用），未绑定返回 -1
  int expert_to_slot_idx(int expert_id) const {
    return expert_to_slot_.lookup(expert_id);
  }

  // B1 fix: 查询 slot 绑定的专家 ID（驱逐用），未绑定返回 -1
  int slot_to_expert_id(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= cap_) return -1;
    return slot_to_expert_[slot_idx];
  }

  // ===== 绑定 / 覆盖 =====

  // 绑定：将 expert 读入空闲 slot，设置指针指向
  // 调用者需保证 slot_idx 处于 BASELINE 或 DEMOTING(reader==0) 状态
  void bind(int slot_idx, int expert_id) {
    Slot& s = slots_[slot_idx];
    s.set_state(ExpertState::LOADING);
    // io_uring 读取完成后调用 mark_cached
    s.bound_expert_id = expert_id;
    slot_to_expert_[slot_idx] = expert_id;
    expert_to_slot_.insert(expert_id, slot_idx);
  }

  // 标记 slot 已缓存完毕（io_uring CQE 到达后调用）
  void mark_cached(int slot_idx) {
    slots_[slot_idx].set_state(ExpertState::CACHED);
  }

  // 覆盖：解绑旧 expert，将新 expert 读入同一个 slot
  // 必须等 active_readers 归零后才能执行
  void overwrite(int slot_idx, int new_expert_id) {
    Slot& s = slots_[slot_idx];
    // 等待活跃 reader 归零
    while (s.active_readers.load(std::memory_order_acquire) > 0) {
      // spin wait，实际实现可用 futex
    }
    int old_expert_id = s.bound_expert_id;
    if (old_expert_id >= 0) {
      expert_to_slot_.erase(old_expert_id);
    }
    s.set_state(ExpertState::DEMOTING);
    // 解绑完成，进入 LOADING 状态等待新数据
    s.set_state(ExpertState::LOADING);
    s.bound_expert_id = new_expert_id;
    slot_to_expert_[slot_idx] = new_expert_id;
    expert_to_slot_.insert(new_expert_id, slot_idx);
  }

  // ===== 引用计数 =====

  // AMX 计算 / GPU 搬运前递增 reader
  void acquire_reader(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return;
    slots_[slot_idx].active_readers.fetch_add(1, std::memory_order_acq_rel);
  }

  // 计算完毕递减 reader
  void release_reader(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return;
    slots_[slot_idx].active_readers.fetch_sub(1, std::memory_order_acq_rel);
  }

  // ===== 驱逐扫描 =====

  // 找一个可驱逐的 slot：状态为 CACHED 且 active_readers==0
  // 返回 slot_idx，找不到返回 -1
  int find_evictable() const {
    for (int i = 0; i < cap_; i++) {
      if (slots_[i].get_state() == ExpertState::CACHED &&
          slots_[i].active_readers.load(std::memory_order_acquire) == 0) {
        return i;
      }
    }
    return -1;
  }

  // 获取所有 CACHED 状态的专家 ID（驱逐评分用）
  std::vector<int> cached_experts() const {
    std::vector<int> result;
    for (int i = 0; i < cap_; i++) {
      if (slots_[i].get_state() == ExpertState::CACHED &&
          slots_[i].bound_expert_id >= 0) {
        result.push_back(slots_[i].bound_expert_id);
      }
    }
    return result;
  }

  // ===== 基本访问器 =====
  int layer_idx() const { return layer_idx_; }
  int tp_part_idx() const { return tp_part_idx_; }
  int numa_node() const { return numa_node_; }
  int cap() const { return cap_; }
  size_t slot_bytes() const { return slot_bytes_; }

  // 设置 gate/up 的单块字节数（用于指针偏移计算）
  void set_gate_up_bytes(size_t bytes) { gate_up_bytes_ = bytes; }

 private:
  int layer_idx_;
  int tp_part_idx_;
  int numa_node_;
  int cap_;
  size_t slot_bytes_;
  size_t gate_up_bytes_ = 0;  // gate 或 up 单块字节数
  void* memory_ = nullptr;
  size_t total_bytes_ = 0;

  std::vector<Slot> slots_;  // [cap_]，驱逐扫描用

  // expert_id -> slot_idx 的双向映射
  // 简单实现用 vector，O(1) 查找
  struct ExpertToSlot {
    std::vector<int> data;  // [expert_num] -> slot_idx or -1
    void resize(int n) { data.assign(n, -1); }
    int lookup(int expert_id) const {
      if (expert_id < 0 || expert_id >= (int)data.size()) return -1;
      return data[expert_id];
    }
    void insert(int expert_id, int slot_idx) {
      if (expert_id >= 0 && expert_id < (int)data.size()) {
        data[expert_id] = slot_idx;
      }
    }
    void erase(int expert_id) {
      if (expert_id >= 0 && expert_id < (int)data.size()) {
        data[expert_id] = -1;
      }
    }
  } expert_to_slot_;

  std::vector<int> slot_to_expert_;  // [cap_] -> expert_id or -1

 public:
  // 初始化 expert_to_slot_ 的大小
  void init_expert_map(int expert_num) {
    expert_to_slot_.resize(expert_num);
  }
};

}  // namespace mesh
