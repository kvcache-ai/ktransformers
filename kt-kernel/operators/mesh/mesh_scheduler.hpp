/**
 * @file mesh_scheduler.hpp
 * @brief schedule_key 优先队列调度器
 *
 * 每次 forward 计算一个 schedule_key，决定 io_uring 读请求的优先级排序。
 * 请求入优先队列（越小越优先），drain 时按序提交给 io_uring。
 *
 * schedule_key 计算：
 * - Prefill (qlen>1): schedule_key = layer_idx，按层号线性推进
 * - Decode immediate (qlen=1): schedule_key = timeline_step × total_layers + layer_idx
 * - Decode deferred (qlen=1): schedule_key = timeline_step × total_layers + layer_idx + 1
 *   （跟下一层同优先级，因为 deferred 专家推迟到下一层计算）
 */
#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

namespace mesh {

// 调度请求
struct ScheduledRequest {
  uint64_t schedule_key;   // 优先级键，越小越优先
  int expert_id;           // 要读取的专家 ID
  int tp_part_idx;         // TP 分片索引
  int layer_idx;           // 层索引
  ReadPriority priority;   // Demand / Prefetch
  // 目标 buffer 指针（由 slot pool 提供）
  void* gate_dst = nullptr;
  void* up_dst = nullptr;
  void* down_dst = nullptr;
  void* gate_scale_dst = nullptr;
  void* up_scale_dst = nullptr;
  void* down_scale_dst = nullptr;
  void* gate_mins_dst = nullptr;
  void* up_mins_dst = nullptr;
  void* down_mins_dst = nullptr;
  // 完成回调
  std::function<void()> on_complete;

  // 优先队列比较器：schedule_key 小的优先
  bool operator>(const ScheduledRequest& o) const {
    return schedule_key > o.schedule_key;
  }
};

/**
 * @brief 全局时序调度器
 *
 * 维护 timeline_step（算上 prefill 的第几个 token）和优先队列。
 * prefill 和 decode 共用同一个调度器，通过 is_prefill_ 区分 schedule_key 计算。
 */
class MeshScheduler {
 public:
  MeshScheduler(int total_layers) : total_layers_(total_layers) {}

  // ===== schedule_key 计算 =====

  // Prefill：按层号线性推进
  uint64_t prefill_key(int layer_idx) const {
    return static_cast<uint64_t>(layer_idx);
  }

  // Decode immediate：当前 token 当前层
  uint64_t decode_immediate_key(int layer_idx) const {
    return static_cast<uint64_t>(timeline_step_) * total_layers_ + layer_idx;
  }

  // Decode deferred：跟下一层同优先级
  uint64_t decode_deferred_key(int layer_idx) const {
    return decode_immediate_key(layer_idx) + 1;
  }

  // ===== 提交请求入队 =====

  void submit_prefill(int layer_idx, int expert_id, int tp_part_idx,
                      void* gate_dst, void* up_dst, void* down_dst,
                      ReadPriority priority = ReadPriority::Prefetch) {
    ScheduledRequest req;
    req.schedule_key = prefill_key(layer_idx);
    req.expert_id = expert_id;
    req.tp_part_idx = tp_part_idx;
    req.layer_idx = layer_idx;
    req.priority = priority;
    req.gate_dst = gate_dst;
    req.up_dst = up_dst;
    req.down_dst = down_dst;
    enqueue(std::move(req));
  }

  void submit_decode_immediate(int layer_idx, int expert_id, int tp_part_idx,
                               void* gate_dst, void* up_dst, void* down_dst,
                               std::function<void()> on_complete = nullptr) {
    ScheduledRequest req;
    req.schedule_key = decode_immediate_key(layer_idx);
    req.expert_id = expert_id;
    req.tp_part_idx = tp_part_idx;
    req.layer_idx = layer_idx;
    req.priority = ReadPriority::Demand;
    req.gate_dst = gate_dst;
    req.up_dst = up_dst;
    req.down_dst = down_dst;
    req.on_complete = std::move(on_complete);
    enqueue(std::move(req));
  }

  void submit_decode_deferred(int layer_idx, int expert_id, int tp_part_idx,
                              void* gate_dst, void* up_dst, void* down_dst,
                              std::function<void()> on_complete = nullptr) {
    ScheduledRequest req;
    req.schedule_key = decode_deferred_key(layer_idx);
    req.expert_id = expert_id;
    req.tp_part_idx = tp_part_idx;
    req.layer_idx = layer_idx;
    req.priority = ReadPriority::Demand;
    req.gate_dst = gate_dst;
    req.up_dst = up_dst;
    req.down_dst = down_dst;
    req.on_complete = std::move(on_complete);
    enqueue(std::move(req));
  }

  // ===== 队列管理 =====

  // 从队列取出所有请求，按 schedule_key 排序后提交给 io_uring
  // caller 提供实际提交函数
  std::vector<ScheduledRequest> drain_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ScheduledRequest> result;
    while (!pq_.empty()) {
      result.push_back(std::move(const_cast<ScheduledRequest&>(pq_.top())));
      pq_.pop();
    }
    // result 已按 schedule_key 升序排列（priority_queue 是大顶堆，top 是最大的）
    // 实际上我们需要最小的先出，所以用 greater 比较器
    // 这里 result 是从大到小，需要反转
    std::reverse(result.begin(), result.end());
    return result;
  }

  // 取出 schedule_key 最小的请求（非阻塞）
  bool try_pop(ScheduledRequest& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pq_.empty()) return false;
    out = std::move(const_cast<ScheduledRequest&>(pq_.top()));
    pq_.pop();
    return true;
  }

  // 清空队列（过渡阶段调用）
  void clear_queue() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!pq_.empty()) pq_.pop();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pq_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pq_.size();
  }

  // ===== 时序状态 =====

  void switch_to_decode() { is_prefill_ = false; }
  void switch_to_prefill() { is_prefill_ = true; }
  bool is_prefill() const { return is_prefill_; }

  void inc_timeline_step() { timeline_step_++; }
  int timeline_step() const { return timeline_step_; }
  void set_timeline_step(int step) { timeline_step_ = step; }

 private:
  int total_layers_;
  int timeline_step_ = 0;  // 算上 prefill 的第几个 token
  bool is_prefill_ = true;

  // 最小堆：schedule_key 小的优先
  using PriorityQueue =
      std::priority_queue<ScheduledRequest, std::vector<ScheduledRequest>,
                          std::greater<ScheduledRequest>>;
  PriorityQueue pq_;
  mutable std::mutex mutex_;

  void enqueue(ScheduledRequest req) {
    std::lock_guard<std::mutex> lock(mutex_);
    pq_.push(std::move(req));
  }
};

}  // namespace mesh
