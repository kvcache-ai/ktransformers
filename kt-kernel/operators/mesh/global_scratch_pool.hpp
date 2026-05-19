/**
 * @Description  : Cross-layer shared scratch buffer pool for MESH.
 *
 * Design intent (matches user spec):
 *   - 40 layers share ONE temp/scratch buffer pool per NUMA node.
 *   - Each block holds one expert's {gate, up, down} weight buffers.
 *   - Lifecycle states: IDLE -> LOADING -> IN_USE -> IDLE (covered on next read).
 *   - Pool capacity is small (~per-layer scratch peak) regardless of layer count;
 *     total memory becomes "40 * stable_per_layer + pool_capacity" instead of
 *     "40 * cache_capacity_". This is the "low-memory" goal.
 *   - Cross-stage (prefill + decode) shared: pool persists across the prefill
 *     -> decode transition.
 *   - "Read in parallel": a layer requests K blocks in one shot, submits all
 *     io_uring reads as a single batch, computes the layer's stable set while
 *     those reads fly, then waits once for the whole batch.
 *
 * Concurrency assumption: MoE forward is layer-serial today (one layer's
 * forward call returns before the next starts), so block acquire/release does
 * not need a hot lock. The state atomics still protect against io_uring
 * completion happening on the io thread while main thread is mid-acquire.
 */

#ifndef CPUINFER_OPERATOR_MESH_GLOBAL_SCRATCH_POOL_HPP
#define CPUINFER_OPERATOR_MESH_GLOBAL_SCRATCH_POOL_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __linux__
#include <numa.h>
#define KT_GSP_HAVE_NUMA 1
#endif

namespace mesh {

enum class GlobalSlotState : uint8_t {
  IDLE = 0,    // Empty or stale; safe to overwrite
  LOADING = 1, // io_uring read submitted, not yet complete
  IN_USE = 2,  // A layer's compute is reading this block's buffer
};

struct GlobalScratchBlock {
  void* gate_owner = nullptr;
  void* up_owner = nullptr;
  void* down_owner = nullptr;

  std::atomic<uint8_t> state{static_cast<uint8_t>(GlobalSlotState::IDLE)};

  // Metadata: which (layer, expert) was last loaded here. Used by find_cached()
  // for the warm-restore path (0-IO recovery if a recently evicted expert is
  // requested again before its block is overwritten).
  int last_layer_id = -1;
  int last_expert_id = -1;

  // Monotonic counter used only to break ties when two IDLE blocks share the
  // same (layer, expert) key in idle_by_key_; higher = more recently touched.
  uint64_t use_tick = 0;

  // Intrusive FIFO links for the IDLE-list (oldest at head, newest at tail).
  // -1 means "not linked"; in_idle_list disambiguates whether a node with
  // prev=-1 is the head or simply detached.
  int prev_idle = -1;
  int next_idle = -1;
  bool in_idle_list = false;
};

class GlobalScratchPool {
 public:
  GlobalScratchPool() = default;
  ~GlobalScratchPool() { shutdown(); }

  GlobalScratchPool(const GlobalScratchPool&) = delete;
  GlobalScratchPool& operator=(const GlobalScratchPool&) = delete;

  // Allocate N blocks of (gate_bytes, up_bytes, down_bytes) on the given NUMA
  // node. Must be called once before any acquire(); subsequent calls verify
  // the size matches (so different layers can register without re-allocating).
  bool initialize(int capacity,
                  size_t gate_bytes,
                  size_t up_bytes,
                  size_t down_bytes,
                  int numa_node) {
    std::lock_guard<std::mutex> guard(mu_);
    if (initialized_) {
      // Sanity-check that registering layers agree on block geometry.
      if (gate_bytes != gate_bytes_ || up_bytes != up_bytes_ || down_bytes != down_bytes_) {
        return false;
      }
      if (numa_node != numa_node_) {
        return false;
      }
      return true;
    }
    if (capacity <= 0) return false;

    blocks_ = std::make_unique<GlobalScratchBlock[]>(static_cast<size_t>(capacity));
    idle_by_key_.clear();
    idle_head_ = -1;
    idle_tail_ = -1;
    idle_size_ = 0;
    capacity_ = capacity;
    gate_bytes_ = gate_bytes;
    up_bytes_ = up_bytes;
    down_bytes_ = down_bytes;
    numa_node_ = numa_node;

    for (int i = 0; i < capacity_; ++i) {
#ifdef KT_GSP_HAVE_NUMA
      blocks_[i].gate_owner = numa_alloc_onnode(gate_bytes_, numa_node_);
      blocks_[i].up_owner = numa_alloc_onnode(up_bytes_, numa_node_);
      blocks_[i].down_owner = numa_alloc_onnode(down_bytes_, numa_node_);
#else
      blocks_[i].gate_owner = std::malloc(gate_bytes_);
      blocks_[i].up_owner = std::malloc(up_bytes_);
      blocks_[i].down_owner = std::malloc(down_bytes_);
#endif
      if (blocks_[i].gate_owner == nullptr || blocks_[i].up_owner == nullptr ||
          blocks_[i].down_owner == nullptr) {
        shutdown_locked();
        return false;
      }
      blocks_[i].state.store(static_cast<uint8_t>(GlobalSlotState::IDLE),
                             std::memory_order_release);
      idle_list_push_back_locked(i);
    }
    initialized_ = true;
    return true;
  }

  void shutdown() {
    std::lock_guard<std::mutex> guard(mu_);
    shutdown_locked();
  }

  int capacity() const { return capacity_; }
  bool initialized() const { return initialized_; }

  // Try to acquire `count` blocks for loading. Returns block indices in
  // out_indices on success; returns false (with out_indices cleared) if the
  // pool cannot satisfy the request (too many LOADING/IN_USE blocks).
  //
  // Preference order:
  //   1. IDLE blocks (oldest use_tick first — these are the safest to overwrite)
  //   2. Fail (caller must retry or split into chunks)
  //
  // The acquired blocks have last_layer_id/last_expert_id reset to -1 so a
  // try_warm_restore probe cannot match them while their buffer is still
  // LOADING or after an io_uring read failure leaves the buffer dirty. The
  // real (layer, expert) tag is published only by set_block_metadata() after
  // a successful read.
  bool acquire_for_load(int count, std::vector<int>& out_indices) {
    out_indices.clear();
    if (count <= 0) return true;
    if (!initialized_ || count > capacity_) return false;

    std::lock_guard<std::mutex> guard(mu_);

    if (idle_size_ < count) {
      // Not enough IDLE — caller should wait/retry or chunk.
      return false;
    }

    // Pop the `count` oldest IDLE blocks (intrusive list head = oldest).
    // State transitions all happen under mu_, so a plain store from IDLE→
    // LOADING is sufficient — no CAS rollback path is reachable.
    out_indices.reserve(count);
    for (int i = 0; i < count; ++i) {
      const int idx = idle_list_pop_front_locked();
      blocks_[idx].state.store(static_cast<uint8_t>(GlobalSlotState::LOADING),
                               std::memory_order_release);
      erase_idle_index_for_block_locked(idx);
      blocks_[idx].last_layer_id = -1;
      blocks_[idx].last_expert_id = -1;
      blocks_[idx].use_tick = ++tick_counter_;
      out_indices.push_back(idx);
    }
    return true;
  }

  // Look up whether a (layer, expert) is already cached (state == IDLE and
  // metadata matches). On hit, atomically transitions IDLE → IN_USE and
  // returns the block index. The caller can immediately use the buffer (no
  // io_uring read needed — the prior read's data is still resident).
  // Returns -1 on miss.
  //
  // This is the "warm-restore" path: cross-layer / cross-prefill-chunk hits
  // that avoid disk re-reads when the pool block's content has not been
  // covered by a more recent acquire.
  int try_warm_restore(int layer_id, int expert_id) {
    if (!initialized_ || layer_id < 0 || expert_id < 0) return -1;
    std::lock_guard<std::mutex> guard(mu_);
    const uint64_t key = pack_key(layer_id, expert_id);
    auto it = idle_by_key_.find(key);
    if (it == idle_by_key_.end()) return -1;

    const int idx = it->second;
    if (!idle_index_entry_valid_locked(idx, layer_id, expert_id)) {
      idle_by_key_.erase(it);
      return -1;
    }

    // idle_index_entry_valid_locked has already verified state==IDLE under mu_.
    // No other thread can change state without taking mu_, so a plain store
    // is sufficient — CAS would never observe a different value here.
    blocks_[idx].state.store(static_cast<uint8_t>(GlobalSlotState::IN_USE),
                             std::memory_order_release);
    idle_by_key_.erase(it);
    idle_list_erase_locked(idx);
    blocks_[idx].use_tick = ++tick_counter_;
    return idx;
  }

  // Get the buffer pointers (caller has previously acquired this block).
  void get_owners(int block_idx, void** gate, void** up, void** down) const {
    if (block_idx < 0 || block_idx >= capacity_) {
      *gate = *up = *down = nullptr;
      return;
    }
    *gate = blocks_[block_idx].gate_owner;
    *up = blocks_[block_idx].up_owner;
    *down = blocks_[block_idx].down_owner;
  }

  // Update the (last_layer_id, last_expert_id) tag on a block. Called after
  // the io_uring read for this block has filled the buffer with a specific
  // (layer, expert)'s weight, so a subsequent try_warm_restore can find it.
  // This is necessary because acquire_for_load happens BEFORE the slot is
  // bound to a real expert (slot_to_expert_[slot] is -1 at acquire time),
  // so the hint passed to acquire is unreliable.
  void set_block_metadata(int block_idx, int layer_id, int expert_id) {
    if (block_idx < 0 || block_idx >= capacity_) return;
    std::lock_guard<std::mutex> guard(mu_);
    erase_idle_index_for_block_locked(block_idx);
    blocks_[block_idx].last_layer_id = layer_id;
    blocks_[block_idx].last_expert_id = expert_id;
    insert_idle_index_for_block_locked(block_idx);
  }

  // Transition: LOADING -> IN_USE. Call after io_uring read for this block
  // completes successfully and the layer is about to compute against it.
  void mark_in_use(int block_idx) {
    if (block_idx < 0 || block_idx >= capacity_) return;
    std::lock_guard<std::mutex> guard(mu_);
    erase_idle_index_for_block_locked(block_idx);
    idle_list_erase_locked(block_idx);
    blocks_[block_idx].state.store(static_cast<uint8_t>(GlobalSlotState::IN_USE),
                                   std::memory_order_release);
  }

  // Transition: IN_USE -> IDLE (or LOADING if load failed). Call after compute
  // finishes (or on failure rollback).
  void release(int block_idx) {
    if (block_idx < 0 || block_idx >= capacity_) return;
    std::lock_guard<std::mutex> guard(mu_);
    blocks_[block_idx].state.store(static_cast<uint8_t>(GlobalSlotState::IDLE),
                                   std::memory_order_release);
    insert_idle_index_for_block_locked(block_idx);
    idle_list_push_back_locked(block_idx);
  }

  // Bulk release helper.
  void release_all(const std::vector<int>& indices) {
    for (int idx : indices) release(idx);
  }

  // Diagnostics
  void counts(int* idle, int* loading, int* in_use) const {
    int i = 0, l = 0, u = 0;
    for (int k = 0; k < capacity_; ++k) {
      const uint8_t s = blocks_[k].state.load(std::memory_order_acquire);
      if (s == static_cast<uint8_t>(GlobalSlotState::IDLE)) ++i;
      else if (s == static_cast<uint8_t>(GlobalSlotState::LOADING)) ++l;
      else ++u;
    }
    if (idle) *idle = i;
    if (loading) *loading = l;
    if (in_use) *in_use = u;
  }

 private:
  static uint64_t pack_key(int layer_id, int expert_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer_id)) << 32) |
           static_cast<uint32_t>(expert_id);
  }

  static bool valid_metadata(int layer_id, int expert_id) {
    return layer_id >= 0 && expert_id >= 0;
  }

  bool idle_index_entry_valid_locked(int block_idx, int layer_id, int expert_id) const {
    if (block_idx < 0 || block_idx >= capacity_) return false;
    const auto& blk = blocks_[block_idx];
    return blk.last_layer_id == layer_id && blk.last_expert_id == expert_id &&
           blk.state.load(std::memory_order_acquire) ==
               static_cast<uint8_t>(GlobalSlotState::IDLE);
  }

  void erase_idle_index_for_block_locked(int block_idx) {
    if (block_idx < 0 || block_idx >= capacity_) return;
    const auto& blk = blocks_[block_idx];
    if (!valid_metadata(blk.last_layer_id, blk.last_expert_id)) return;
    const uint64_t key = pack_key(blk.last_layer_id, blk.last_expert_id);
    auto it = idle_by_key_.find(key);
    if (it != idle_by_key_.end() && it->second == block_idx) {
      idle_by_key_.erase(it);
    }
  }

  // Intrusive doubly-linked IDLE list (head=oldest, tail=newest). All ops are
  // O(1) and require no heap allocation.
  void idle_list_push_back_locked(int block_idx) {
    if (block_idx < 0 || block_idx >= capacity_) return;
    auto& blk = blocks_[block_idx];
    if (blk.in_idle_list) return;
    blk.prev_idle = idle_tail_;
    blk.next_idle = -1;
    if (idle_tail_ >= 0) {
      blocks_[idle_tail_].next_idle = block_idx;
    } else {
      idle_head_ = block_idx;
    }
    idle_tail_ = block_idx;
    blk.in_idle_list = true;
    ++idle_size_;
  }

  int idle_list_pop_front_locked() {
    if (idle_head_ < 0) return -1;
    const int idx = idle_head_;
    auto& blk = blocks_[idx];
    idle_head_ = blk.next_idle;
    if (idle_head_ >= 0) {
      blocks_[idle_head_].prev_idle = -1;
    } else {
      idle_tail_ = -1;
    }
    blk.prev_idle = -1;
    blk.next_idle = -1;
    blk.in_idle_list = false;
    --idle_size_;
    return idx;
  }

  void idle_list_erase_locked(int block_idx) {
    if (block_idx < 0 || block_idx >= capacity_) return;
    auto& blk = blocks_[block_idx];
    if (!blk.in_idle_list) return;
    const int prev = blk.prev_idle;
    const int next = blk.next_idle;
    if (prev >= 0) {
      blocks_[prev].next_idle = next;
    } else {
      idle_head_ = next;
    }
    if (next >= 0) {
      blocks_[next].prev_idle = prev;
    } else {
      idle_tail_ = prev;
    }
    blk.prev_idle = -1;
    blk.next_idle = -1;
    blk.in_idle_list = false;
    --idle_size_;
  }

  void insert_idle_index_for_block_locked(int block_idx) {
    if (block_idx < 0 || block_idx >= capacity_) return;
    const auto& blk = blocks_[block_idx];
    if (!valid_metadata(blk.last_layer_id, blk.last_expert_id)) return;
    if (blk.state.load(std::memory_order_acquire) !=
        static_cast<uint8_t>(GlobalSlotState::IDLE)) {
      return;
    }

    const uint64_t key = pack_key(blk.last_layer_id, blk.last_expert_id);
    auto it = idle_by_key_.find(key);
    if (it == idle_by_key_.end()) {
      idle_by_key_.emplace(key, block_idx);
      return;
    }

    const int current = it->second;
    if (current < 0 || current >= capacity_ ||
        !idle_index_entry_valid_locked(current, blk.last_layer_id, blk.last_expert_id) ||
        blocks_[current].use_tick <= blk.use_tick) {
      it->second = block_idx;
    }
  }

  void shutdown_locked() {
    if (!initialized_) return;
    idle_by_key_.clear();
    idle_head_ = -1;
    idle_tail_ = -1;
    idle_size_ = 0;
    for (int i = 0; i < capacity_; ++i) {
      auto& blk = blocks_[i];
#ifdef KT_GSP_HAVE_NUMA
      if (blk.gate_owner) numa_free(blk.gate_owner, gate_bytes_);
      if (blk.up_owner) numa_free(blk.up_owner, up_bytes_);
      if (blk.down_owner) numa_free(blk.down_owner, down_bytes_);
#else
      if (blk.gate_owner) std::free(blk.gate_owner);
      if (blk.up_owner) std::free(blk.up_owner);
      if (blk.down_owner) std::free(blk.down_owner);
#endif
      blk.gate_owner = blk.up_owner = blk.down_owner = nullptr;
    }
    blocks_.reset();
    initialized_ = false;
    capacity_ = 0;
  }

  mutable std::mutex mu_;
  bool initialized_ = false;
  int capacity_ = 0;
  size_t gate_bytes_ = 0;
  size_t up_bytes_ = 0;
  size_t down_bytes_ = 0;
  int numa_node_ = -1;
  // std::atomic<uint8_t> inside the block is non-movable, so vector<> can't
  // resize it. Use a heap-allocated fixed-size array instead — we know
  // capacity at initialize() time and never resize after that.
  std::unique_ptr<GlobalScratchBlock[]> blocks_;
  uint64_t tick_counter_ = 0;
  std::unordered_map<uint64_t, int> idle_by_key_;
  // Intrusive doubly-linked list of IDLE blocks (head=oldest, tail=newest).
  int idle_head_ = -1;
  int idle_tail_ = -1;
  int idle_size_ = 0;
};

// Per-NUMA-node singleton registry. All AMX_MOE_TP layers on the same NUMA
// node share one pool. Pool capacity is read once from the env var
// KT_MESH_GLOBAL_POOL_CAPACITY (default 128) on first access.
class GlobalScratchPoolRegistry {
 public:
  static GlobalScratchPool& get(int numa_node) {
    auto& self = instance();
    std::lock_guard<std::mutex> guard(self.mu_);
    auto it = self.pools_.find(numa_node);
    if (it == self.pools_.end()) {
      auto pool = std::make_unique<GlobalScratchPool>();
      auto& ref = *pool;
      self.pools_.emplace(numa_node, std::move(pool));
      return ref;
    }
    return *it->second;
  }

  static int env_capacity_default() {
    // Default sized for the user's design: every layer keeps a small
    // stable set permanently borrowed, plus a single shared temp region
    // ~scratch peak. For DeepSeek-V3 (40 MoE layers): 40 * 32 + 192 = 1472,
    // 40 * 64 + 256 = 2816. Set conservatively higher; the user should
    // override via KT_MESH_GLOBAL_POOL_CAPACITY based on stable_per_layer.
    const char* raw = std::getenv("KT_MESH_GLOBAL_POOL_CAPACITY");
    if (raw == nullptr || raw[0] == '\0') return 3072;
    char* end = nullptr;
    const long v = std::strtol(raw, &end, 10);
    if (end == raw || v <= 0) return 3072;
    return static_cast<int>(std::min<long>(v, 16384));
  }

  // For testing/shutdown.
  static void clear_all() {
    auto& self = instance();
    std::lock_guard<std::mutex> guard(self.mu_);
    self.pools_.clear();
  }

 private:
  GlobalScratchPoolRegistry() = default;
  static GlobalScratchPoolRegistry& instance() {
    static GlobalScratchPoolRegistry inst;
    return inst;
  }
  std::mutex mu_;
  // Heterogeneous keying not needed; numa_node is a small int. Use map for
  // stable iteration if we ever need to walk all pools.
  std::unordered_map<int, std::unique_ptr<GlobalScratchPool>> pools_;
};

}  // namespace mesh

#endif  // CPUINFER_OPERATOR_MESH_GLOBAL_SCRATCH_POOL_HPP
