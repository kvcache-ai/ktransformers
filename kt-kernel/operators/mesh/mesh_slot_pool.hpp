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
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
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
    // per-slot mutex + CV：保护 active_readers 的 acquire/release/overwrite 同步
    // 用 unique_ptr 因为 mutex/CV 不可移动
    slot_mtx_.resize(cap);
    slot_cv_.resize(cap);
    for (int i = 0; i < cap; i++) {
      slot_mtx_[i] = std::make_unique<std::mutex>();
      slot_cv_[i] = std::make_unique<std::condition_variable>();
    }
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
        slot_mtx_(std::move(other.slot_mtx_)),
        slot_cv_(std::move(other.slot_cv_)),
        layer_idx_(other.layer_idx_),
        tp_part_idx_(other.tp_part_idx_),
        numa_node_(other.numa_node_),
        cap_(other.cap_),
        slot_bytes_(other.slot_bytes_),
        gate_up_bytes_(other.gate_up_bytes_),
        gate_up_weights_bytes_(other.gate_up_weights_bytes_),
        down_weights_bytes_(other.down_weights_bytes_),
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
      slot_mtx_ = std::move(other.slot_mtx_);
      slot_cv_ = std::move(other.slot_cv_);
      layer_idx_ = other.layer_idx_;
      tp_part_idx_ = other.tp_part_idx_;
      numa_node_ = other.numa_node_;
      cap_ = other.cap_;
      slot_bytes_ = other.slot_bytes_;
      gate_up_bytes_ = other.gate_up_bytes_;
      gate_up_weights_bytes_ = other.gate_up_weights_bytes_;
      down_weights_bytes_ = other.down_weights_bytes_;
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

  // AMXINT4: 获取 slot 中 gate scale 的地址（紧跟 gate 权重之后）
  // BufferB 期望 d = ptr + n*k/2，n=intermediate/tp, k=hidden
  void* gate_scale_ptr(int slot_idx) {
    return static_cast<char*>(gate_ptr(slot_idx)) + gate_up_weights_bytes_;
  }
  // AMXINT4: 获取 slot 中 up scale 的地址
  void* up_scale_ptr(int slot_idx) {
    return static_cast<char*>(up_ptr(slot_idx)) + gate_up_weights_bytes_;
  }
  // AMXINT4: 获取 slot 中 down scale 的地址
  // down 的 BufferB n=hidden, k=intermediate/tp → 权重=n*k/2=hidden*(intermediate/tp)/2
  void* down_scale_ptr(int slot_idx) {
    return static_cast<char*>(down_ptr(slot_idx)) + down_weights_bytes_;
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

  // ===== 原子化 acquire（lookup + state check + reader + pointer 全部在同一锁内）=====
  // 消除 TOCTOU 竞争：原 expert_gate_ptr + acquire_reader 分两次独立 lookup，
  // 中间 expert 可能被驱逐并重新加载到另一个 slot，导致返回的指针（旧 slot）和
  // reader（新 slot）不在同一个 slot 上。GEMM 用无 reader 保护的旧 slot 指针，
  // 旧 slot 被 overwrite 覆盖 → 读到空内存 → 段错误。
  // 此方法保证指针和 reader 在同一个 slot 上，overwrite 持同一把锁无法插入。
  void* acquire_gate_ptr(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return nullptr;
    std::lock_guard<std::mutex> lock(*slot_mtx_[slot_idx]);
    if (slot_to_expert_[slot_idx] != expert_id) return nullptr;
    if (slots_[slot_idx].get_state() != ExpertState::CACHED) return nullptr;
    slots_[slot_idx].active_readers.fetch_add(1, std::memory_order_acq_rel);
    return gate_ptr(slot_idx);
  }
  void* acquire_up_ptr(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return nullptr;
    std::lock_guard<std::mutex> lock(*slot_mtx_[slot_idx]);
    if (slot_to_expert_[slot_idx] != expert_id) return nullptr;
    if (slots_[slot_idx].get_state() != ExpertState::CACHED) return nullptr;
    slots_[slot_idx].active_readers.fetch_add(1, std::memory_order_acq_rel);
    return up_ptr(slot_idx);
  }
  void* acquire_down_ptr(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return nullptr;
    std::lock_guard<std::mutex> lock(*slot_mtx_[slot_idx]);
    if (slot_to_expert_[slot_idx] != expert_id) return nullptr;
    if (slots_[slot_idx].get_state() != ExpertState::CACHED) return nullptr;
    slots_[slot_idx].active_readers.fetch_add(1, std::memory_order_acq_rel);
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

  // Bug 7 fix: 直接按 slot_idx 查状态，避免 expert_id->slot_idx->expert_id 双重映射
  ExpertState slot_state(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= cap_) return ExpertState::BASELINE;
    return slots_[slot_idx].get_state();
  }

  // Bug 7 fix: 找空闲 slot（BASELINE 或未绑定），返回 slot_idx，找不到返回 -1
  // 替代 drain_and_submit 中的线性扫描 + 双重映射
  int find_free_slot() const {
    for (int i = 0; i < cap_; i++) {
      if (slot_to_expert_[i] < 0 ||
          slots_[i].get_state() == ExpertState::BASELINE) {
        return i;
      }
    }
    return -1;
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

  // 解绑：将 slot 恢复为 BASELINE（加载失败时调用）
  void unbind(int slot_idx) {
    Slot& s = slots_[slot_idx];
    int old_expert = s.bound_expert_id;
    s.set_state(ExpertState::BASELINE);
    s.bound_expert_id = -1;
    slot_to_expert_[slot_idx] = -1;
    if (old_expert >= 0) expert_to_slot_.erase(old_expert);
  }

  // 标记 slot 已缓存完毕（io_uring CQE 到达后调用）
  void mark_cached(int slot_idx) {
    slots_[slot_idx].set_state(ExpertState::CACHED);
  }

  // 覆盖：解绑旧 expert，将新 expert 读入同一个 slot
  // 用 condition_variable 等待 active_readers 归零（30秒超时，超时后强制覆盖）
  // 持有 slot_mtx_ 期间修改绑定关系，防止 acquire_reader 的 TOCTOU 竞争
  void overwrite(int slot_idx, int new_expert_id) {
    Slot& s = slots_[slot_idx];
    {
      std::unique_lock<std::mutex> lock(*slot_mtx_[slot_idx]);
      // CV 等待 active_readers 归零：GEMM 完成后 release_reader 会 notify
      // 30秒超时诊断：如果 reader 未释放，说明有 reader leak 或死锁
      // 注意：wait_for 之前不能调用 fprintf，否则 stderr 缓冲区满时 fprintf 阻塞，
      // 锁被永久持有，wait_for 永远不被调用，30秒超时永远不触发 → 永久死锁
      if (s.active_readers.load(std::memory_order_acquire) > 0) {
        auto status = slot_cv_[slot_idx]->wait_for(lock, std::chrono::seconds(30), [&] {
          return s.active_readers.load(std::memory_order_acquire) == 0;
        });
        if (!status) {
          // 超时：reader 未释放，强制覆盖以避免永久死锁
          // 此时 fprintf 在 wait_for 之后，锁仍持有但 wait_for 已执行过，不会阻止超时
          int still = s.active_readers.load(std::memory_order_acquire);
          fprintf(stderr, "[MESH OVERWRITE TIMEOUT] layer=%d tp=%d slot=%d old_expert=%d "
                  "new_expert=%d active_readers=%d — FORCING overwrite after 30s timeout! "
                  "Reader leak detected.\n",
                  layer_idx_, tp_part_idx_, slot_idx, s.bound_expert_id,
                  new_expert_id, still);
        }
      }
      // 仍在锁内：修改绑定关系，acquire_reader 会看到新状态
      int old_expert_id = s.bound_expert_id;
      if (old_expert_id >= 0) {
        expert_to_slot_.erase(old_expert_id);
      }
      s.set_state(ExpertState::LOADING);
      s.bound_expert_id = new_expert_id;
      slot_to_expert_[slot_idx] = new_expert_id;
      expert_to_slot_.insert(new_expert_id, slot_idx);
    }
  }

  // ===== 引用计数 =====

  // AMX 计算 / GPU 搬运前递增 reader
  // 返回 true 表示成功获取 reader（slot 仍绑定该 expert 且 CACHED）
  // 返回 false 表示 slot 已被驱逐（调用者应走 load 路径）
  // 持有 slot_mtx_ 防止与 overwrite 的 TOCTOU 竞争
  bool acquire_reader(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) return false;
    std::lock_guard<std::mutex> lock(*slot_mtx_[slot_idx]);
    // 双重检查：持锁后确认 expert 仍绑定且 CACHED
    if (slot_to_expert_[slot_idx] != expert_id) return false;
    if (slots_[slot_idx].get_state() != ExpertState::CACHED) return false;
    // 不在此处打印日志：acquire_reader 是超高频路径（每 token 每 expert 每 GEMM），
    // 持锁 fprintf 会在 stderr 缓冲区满时阻塞，导致 slot_mtx_ 被永久持有 → 死锁
    slots_[slot_idx].active_readers.fetch_add(1, std::memory_order_acq_rel);
    return true;
  }

  // 计算完毕递减 reader，并 notify overwrite 的 CV 等待
  // 死锁修复：不持有 slot_mtx_ 锁。active_readers 是 std::atomic，fetch_sub 不需要锁。
  // 原实现持锁 fetch_sub 会导致死锁：overwrite 持 slot_mtx_ CV wait 等 active_readers==0，
  // release_reader 需要同一个 slot_mtx_ 才能 fetch_sub → 永久死锁。
  // 不持锁的安全性：lookup 可能和 overwrite 的 erase/insert 竞争，但 int 读写通常原子。
  // 若 lookup 返回 -1（expert 已被 evict），说明 overwrite 已 CV wait 完成，迟到 return 安全。
  // 若 lookup 返回正确 slot_idx，overwrite 还在 CV wait（active_readers>0），fetch_sub 安全。
  void release_reader(int expert_id) {
    int slot_idx = expert_to_slot_.lookup(expert_id);
    if (slot_idx < 0) {
      // expert 已被 evict，reader 无法释放。
      // 这是潜在 reader leak 的来源之一，但此处不打印日志（超高频路径）。
      // 若需诊断，可在 overwrite 超时日志中观察到 active_readers>0。
      return;
    }
    slots_[slot_idx].active_readers.fetch_sub(1, std::memory_order_acq_rel);
    slot_cv_[slot_idx]->notify_all();
  }

  // Bug 11 fix: 按 slot_idx 递增/递减 reader，不依赖 expert 绑定关系
  // 用于 handoff 等场景，expert 可能未绑定但 slot 需要保护
  void acquire_slot_reader(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= cap_) return;
    slots_[slot_idx].active_readers.fetch_add(1, std::memory_order_acq_rel);
  }
  void release_slot_reader(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= cap_) return;
    slots_[slot_idx].active_readers.fetch_sub(1, std::memory_order_acq_rel);
    slot_cv_[slot_idx]->notify_all();
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

  // 获取所有可驱逐的专家 ID（CACHED 且 active_readers==0）
  // SKILL.md 第 62 行：不驱逐读取中/使用中的专家
  // active_readers>0 表示有 GEMM 正在读取，强制驱逐会导致读到空内存
  std::vector<int> cached_experts() const {
    std::vector<int> result;
    for (int i = 0; i < cap_; i++) {
      if (slots_[i].get_state() == ExpertState::CACHED &&
          slots_[i].bound_expert_id >= 0 &&
          slots_[i].active_readers.load(std::memory_order_acquire) == 0) {
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

  // 设置 gate/up 的单块字节数（含 scale，用于指针偏移计算）
  void set_gate_up_bytes(size_t bytes) { gate_up_bytes_ = bytes; }

  // AMXINT4: 设置 gate/up 的纯权重大小（不含 scale，用于 scale 指针偏移）
  void set_gate_up_weights_bytes(size_t bytes) { gate_up_weights_bytes_ = bytes; }

  // AMXINT4: 设置 down 的纯权重大小（不含 scale，用于 down scale 指针偏移）
  void set_down_weights_bytes(size_t bytes) { down_weights_bytes_ = bytes; }

 private:
  int layer_idx_;
  int tp_part_idx_;
  int numa_node_;
  int cap_;
  size_t slot_bytes_;
  size_t gate_up_bytes_ = 0;  // gate 或 up 单块字节数（含 scale）
  size_t gate_up_weights_bytes_ = 0;  // AMXINT4: gate/up 纯权重大小（不含 scale）
  size_t down_weights_bytes_ = 0;     // AMXINT4: down 纯权重大小（不含 scale）
  void* memory_ = nullptr;
  size_t total_bytes_ = 0;

  std::vector<Slot> slots_;  // [cap_]，驱逐扫描用

  // per-slot mutex + condition_variable
  // 保护 acquire_reader/release_reader/overwrite 之间的同步
  // acquire_reader 持锁检查 state + 增计数；overwrite 持锁 CV 等待计数归零
  std::vector<std::unique_ptr<std::mutex>> slot_mtx_;
  std::vector<std::unique_ptr<std::condition_variable>> slot_cv_;

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
