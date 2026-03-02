/**
 * @Description  : Polling-based memcpy worker for batch expert loading
 * @Date         : 2025-01-27
 *
 * Features:
 * - Dedicated worker thread per rank for polling + cudaMemcpyAsync
 * - CPU affinity binding to avoid conflict with cpuinfer cores
 * - Adaptive sleep: busy-poll during prefill, sleep between prefills
 * - Uses DMA for H2D transfers (doesn't occupy GPU SM)
 */

#ifndef POLLING_MEMCPY_WORKER_H
#define POLLING_MEMCPY_WORKER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "polling_sync_batch.cuh"

#ifndef KTRANSFORMERS_CPU_ONLY
#include <cuda_runtime.h>
#endif

namespace polling_memcpy_worker {

/**
 * @brief Worker thread that polls sync slot and performs cudaMemcpyAsync
 *
 * Each rank has one PollingMemcpyWorker. The worker:
 * 1. Polls sync_slot->signal for DATA_READY
 * 2. Reads expert_id and slot_idx
 * 3. Calls cudaMemcpyAsync for all 4 weight/scale buffers
 * 4. Calls cudaStreamSynchronize to wait for DMA completion
 * 5. Sets signal = GPU_DONE
 *
 * Adaptive sleep: If no DATA_READY signal for a threshold time,
 * worker sleeps to reduce CPU usage between layerwise prefills.
 * Once DATA_READY arrives, switches back to busy-polling mode.
 */
class PollingMemcpyWorker {
 public:
  PollingMemcpyWorker() = default;
  ~PollingMemcpyWorker();

  // Non-copyable
  PollingMemcpyWorker(const PollingMemcpyWorker&) = delete;
  PollingMemcpyWorker& operator=(const PollingMemcpyWorker&) = delete;

  /**
   * @brief Setup worker configuration (call before start())
   *
   * @param rank Worker's rank ID
   * @param cuda_device CUDA device ID for this rank
   * @param cpu_core CPU core to bind worker thread (-1 for auto-select)
   * @param sync_slot Pointer to sync slot (pinned memory)
   * @param src_buffers Source buffer pointers [8]: w13_w_s0, w13_w_s1, w13_s_s0, w13_s_s1, w2_w_s0, w2_w_s1, w2_s_s0, w2_s_s1
   * @param dst_w13_weight GPU destination for w13 weight (base pointer)
   * @param dst_w13_scale GPU destination for w13 scale
   * @param dst_w2_weight GPU destination for w2 weight
   * @param dst_w2_scale GPU destination for w2 scale
   * @param w13_weight_size Per-expert w13 weight size in bytes
   * @param w13_scale_size Per-expert w13 scale size in bytes
   * @param w2_weight_size Per-expert w2 weight size in bytes
   * @param w2_scale_size Per-expert w2 scale size in bytes
   */
  void setup(int rank, int cuda_device, int cpu_core,
             polling_sync_batch::BatchSyncSlot* sync_slot,
             void** src_buffers,
             void* dst_w13_weight, void* dst_w13_scale,
             void* dst_w2_weight, void* dst_w2_scale,
             size_t w13_weight_size, size_t w13_scale_size,
             size_t w2_weight_size, size_t w2_scale_size);

  /**
   * @brief Start the worker thread
   */
  void start();

  /**
   * @brief Stop the worker thread (signals shutdown and joins)
   */
  void stop();

  /**
   * @brief Check if worker is running
   */
  bool is_running() const { return running_.load(std::memory_order_relaxed); }

  /**
   * @brief Get number of experts processed
   */
  int get_processed_count() const { return processed_count_.load(std::memory_order_relaxed); }

 private:
  void worker_loop();
  void bind_cpu_core(int core);
  void set_cuda_device();

  // Configuration
  int rank_ = -1;
  int cuda_device_ = 0;
  int cpu_core_ = -1;
  polling_sync_batch::BatchSyncSlot* sync_slot_ = nullptr;

  // Source buffers (double-buffered): [w13_w_s0, w13_w_s1, w13_s_s0, w13_s_s1, w2_w_s0, w2_w_s1, w2_s_s0, w2_s_s1]
  void* src_w13_weight_[2] = {nullptr, nullptr};
  void* src_w13_scale_[2] = {nullptr, nullptr};
  void* src_w2_weight_[2] = {nullptr, nullptr};
  void* src_w2_scale_[2] = {nullptr, nullptr};

  // Destination (GPU memory base pointers)
  void* dst_w13_weight_ = nullptr;
  void* dst_w13_scale_ = nullptr;
  void* dst_w2_weight_ = nullptr;
  void* dst_w2_scale_ = nullptr;

  // Sizes per expert
  size_t w13_weight_size_ = 0;
  size_t w13_scale_size_ = 0;
  size_t w2_weight_size_ = 0;
  size_t w2_scale_size_ = 0;

  // Thread management
  std::unique_ptr<std::thread> worker_thread_;
  std::atomic<bool> should_stop_{false};
  std::atomic<bool> running_{false};
  std::atomic<int> processed_count_{0};

#ifndef KTRANSFORMERS_CPU_ONLY
  cudaStream_t stream_ = nullptr;
#endif

  // Adaptive sleep configuration
  static constexpr int kBusyPollSpins = 1000;           // Spins before considering sleep
  static constexpr int kSleepThresholdUs = 100;         // Sleep after this many us of idle
  static constexpr int kSleepDurationUs = 50;           // Sleep duration when idle
};

/**
 * @brief Manager for all polling memcpy workers (one per process)
 *
 * Each GPU process creates one PollingMemcpyWorkerManager which manages
 * the local rank's PollingMemcpyWorker. The manager provides a simple
 * interface for setup, start, and stop operations.
 */
class PollingMemcpyWorkerManager {
 public:
  static PollingMemcpyWorkerManager& instance();

  /**
   * @brief Create worker for local rank
   */
  void create_worker(int rank, int cuda_device, int cpu_core,
                     polling_sync_batch::BatchSyncSlot* sync_slot,
                     void** src_buffers,
                     void* dst_w13_weight, void* dst_w13_scale,
                     void* dst_w2_weight, void* dst_w2_scale,
                     size_t w13_weight_size, size_t w13_scale_size,
                     size_t w2_weight_size, size_t w2_scale_size);

  /**
   * @brief Start local worker
   */
  void start_worker();

  /**
   * @brief Stop local worker
   */
  void stop_worker();

  /**
   * @brief Check if worker exists and is running
   */
  bool has_worker() const { return worker_ != nullptr; }
  bool is_worker_running() const { return worker_ && worker_->is_running(); }

  // Make destructor public for pybind11 compatibility (singleton won't actually be destroyed)
  ~PollingMemcpyWorkerManager();

 private:
  PollingMemcpyWorkerManager() = default;

  std::unique_ptr<PollingMemcpyWorker> worker_;
};

}  // namespace polling_memcpy_worker

#endif  // POLLING_MEMCPY_WORKER_H
