/**
 * @Description  : Polling-based memcpy worker implementation
 * @Date         : 2025-01-27
 */

#include "polling_memcpy_worker.h"

#include <immintrin.h>  // For _mm_pause
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifndef KTRANSFORMERS_CPU_ONLY
#include <cuda_runtime.h>
#endif

namespace polling_memcpy_worker {

// ============================================================================
// PollingMemcpyWorker Implementation
// ============================================================================

PollingMemcpyWorker::~PollingMemcpyWorker() {
  stop();
#ifndef KTRANSFORMERS_CPU_ONLY
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
#endif
}

void PollingMemcpyWorker::setup(int rank, int cuda_device, int cpu_core,
                                 polling_sync_batch::BatchSyncSlot* sync_slot,
                                 void** src_buffers,
                                 void* dst_w13_weight, void* dst_w13_scale,
                                 void* dst_w2_weight, void* dst_w2_scale,
                                 size_t w13_weight_size, size_t w13_scale_size,
                                 size_t w2_weight_size, size_t w2_scale_size) {
  rank_ = rank;
  cuda_device_ = cuda_device;
  cpu_core_ = cpu_core;
  sync_slot_ = sync_slot;

  // Parse source buffers: [w13_w_s0, w13_w_s1, w13_s_s0, w13_s_s1, w2_w_s0, w2_w_s1, w2_s_s0, w2_s_s1]
  src_w13_weight_[0] = src_buffers[0];
  src_w13_weight_[1] = src_buffers[1];
  src_w13_scale_[0] = src_buffers[2];
  src_w13_scale_[1] = src_buffers[3];
  src_w2_weight_[0] = src_buffers[4];
  src_w2_weight_[1] = src_buffers[5];
  src_w2_scale_[0] = src_buffers[6];
  src_w2_scale_[1] = src_buffers[7];

  dst_w13_weight_ = dst_w13_weight;
  dst_w13_scale_ = dst_w13_scale;
  dst_w2_weight_ = dst_w2_weight;
  dst_w2_scale_ = dst_w2_scale;

  w13_weight_size_ = w13_weight_size;
  w13_scale_size_ = w13_scale_size;
  w2_weight_size_ = w2_weight_size;
  w2_scale_size_ = w2_scale_size;

  printf("[PollingMemcpyWorker] Rank %d setup: device=%d, cpu_core=%d, "
         "w13_weight_size=%zu, w13_scale_size=%zu, w2_weight_size=%zu, w2_scale_size=%zu\n",
         rank_, cuda_device_, cpu_core_,
         w13_weight_size_, w13_scale_size_, w2_weight_size_, w2_scale_size_);
}

void PollingMemcpyWorker::start() {
  if (running_.load(std::memory_order_relaxed)) {
    printf("[PollingMemcpyWorker] Rank %d already running\n", rank_);
    return;
  }

  should_stop_.store(false, std::memory_order_relaxed);
  worker_thread_ = std::make_unique<std::thread>(&PollingMemcpyWorker::worker_loop, this);

  printf("[PollingMemcpyWorker] Rank %d started\n", rank_);
}

void PollingMemcpyWorker::stop() {
  if (!worker_thread_) {
    return;
  }

  // Signal shutdown via sync slot
  if (sync_slot_ != nullptr) {
    polling_sync_batch::cpu_signal_shutdown(sync_slot_);
  }

  should_stop_.store(true, std::memory_order_release);

  if (worker_thread_->joinable()) {
    worker_thread_->join();
  }
  worker_thread_.reset();

  printf("[PollingMemcpyWorker] Rank %d stopped, processed %d experts\n",
         rank_, processed_count_.load(std::memory_order_relaxed));
}

void PollingMemcpyWorker::bind_cpu_core(int core) {
  if (core < 0) {
    return;  // No binding requested
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);

  int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (ret != 0) {
    printf("[PollingMemcpyWorker] Rank %d failed to bind to core %d: %s\n",
           rank_, core, strerror(ret));
  } else {
    printf("[PollingMemcpyWorker] Rank %d bound to core %d\n", rank_, core);
  }
}

void PollingMemcpyWorker::set_cuda_device() {
#ifndef KTRANSFORMERS_CPU_ONLY
  cudaError_t err = cudaSetDevice(cuda_device_);
  if (err != cudaSuccess) {
    printf("[PollingMemcpyWorker] Rank %d failed to set CUDA device %d: %s\n",
           rank_, cuda_device_, cudaGetErrorString(err));
    return;
  }

  // Create stream for this worker
  err = cudaStreamCreate(&stream_);
  if (err != cudaSuccess) {
    printf("[PollingMemcpyWorker] Rank %d failed to create CUDA stream: %s\n",
           rank_, cudaGetErrorString(err));
    stream_ = nullptr;
  }
#endif
}

void PollingMemcpyWorker::worker_loop() {
  running_.store(true, std::memory_order_release);

  // Bind to CPU core
  bind_cpu_core(cpu_core_);

  // Set CUDA device and create stream
  set_cuda_device();

#ifndef KTRANSFORMERS_CPU_ONLY
  if (stream_ == nullptr) {
    printf("[PollingMemcpyWorker] Rank %d: no CUDA stream, cannot proceed\n", rank_);
    running_.store(false, std::memory_order_release);
    return;
  }
#endif

  // Adaptive sleep state
  int idle_spins = 0;
  auto last_work_time = std::chrono::steady_clock::now();
  bool in_prefill_mode = false;  // Start in idle mode

  printf("[PollingMemcpyWorker] Rank %d entering main loop\n", rank_);

  while (!should_stop_.load(std::memory_order_relaxed)) {
    // Memory fence for visibility
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int32_t sig = sync_slot_->signal;

    if (sig == polling_sync_batch::SIGNAL_SHUTDOWN) {
      printf("[PollingMemcpyWorker] Rank %d received shutdown signal\n", rank_);
      break;
    }

    if (sig != polling_sync_batch::SIGNAL_DATA_READY) {
      // No data ready - adaptive sleep logic
      idle_spins++;

      if (in_prefill_mode) {
        // During prefill: busy-poll with minimal pause
        _mm_pause();

        // Check if we've been idle too long - maybe prefill is done
        if (idle_spins > kBusyPollSpins * 100) {
          auto now = std::chrono::steady_clock::now();
          auto idle_us = std::chrono::duration_cast<std::chrono::microseconds>(now - last_work_time).count();
          if (idle_us > 10000) {  // 10ms without work = prefill likely done
            in_prefill_mode = false;
            idle_spins = 0;
          }
        }
      } else {
        // Between prefills: use adaptive sleep
        if (idle_spins < kBusyPollSpins) {
          _mm_pause();
        } else {
          // Sleep to reduce CPU usage
          std::this_thread::sleep_for(std::chrono::microseconds(kSleepDurationUs));
        }
      }
      continue;
    }

    // DATA_READY: switch to prefill mode, reset idle state
    in_prefill_mode = true;
    idle_spins = 0;
    last_work_time = std::chrono::steady_clock::now();

    // Read expert_id and slot_idx
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const int expert_id = sync_slot_->expert_id;
    const int slot_idx = sync_slot_->slot_idx;

#ifndef KTRANSFORMERS_CPU_ONLY
    // Calculate destination offsets
    char* dst_w13_w = (char*)dst_w13_weight_ + (size_t)expert_id * w13_weight_size_;
    char* dst_w13_s = (char*)dst_w13_scale_ + (size_t)expert_id * w13_scale_size_;
    char* dst_w2_w = (char*)dst_w2_weight_ + (size_t)expert_id * w2_weight_size_;
    char* dst_w2_s = (char*)dst_w2_scale_ + (size_t)expert_id * w2_scale_size_;

    // Issue async memcpy (DMA - doesn't use GPU SM)
    cudaMemcpyAsync(dst_w13_w, src_w13_weight_[slot_idx], w13_weight_size_,
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(dst_w13_s, src_w13_scale_[slot_idx], w13_scale_size_,
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(dst_w2_w, src_w2_weight_[slot_idx], w2_weight_size_,
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(dst_w2_s, src_w2_scale_[slot_idx], w2_scale_size_,
                    cudaMemcpyHostToDevice, stream_);

    // Wait for all copies to complete
    cudaStreamSynchronize(stream_);
#endif

    // Signal completion
    std::atomic_thread_fence(std::memory_order_seq_cst);
    sync_slot_->processed_count = processed_count_.load(std::memory_order_relaxed) + 1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    sync_slot_->signal = polling_sync_batch::SIGNAL_GPU_DONE;
    std::atomic_thread_fence(std::memory_order_seq_cst);

    processed_count_.fetch_add(1, std::memory_order_relaxed);
  }

  running_.store(false, std::memory_order_release);
  printf("[PollingMemcpyWorker] Rank %d exited main loop\n", rank_);
}

// ============================================================================
// PollingMemcpyWorkerManager Implementation
// ============================================================================

PollingMemcpyWorkerManager& PollingMemcpyWorkerManager::instance() {
  static PollingMemcpyWorkerManager instance;
  return instance;
}

PollingMemcpyWorkerManager::~PollingMemcpyWorkerManager() {
  stop_worker();
}

void PollingMemcpyWorkerManager::create_worker(
    int rank, int cuda_device, int cpu_core,
    polling_sync_batch::BatchSyncSlot* sync_slot,
    void** src_buffers,
    void* dst_w13_weight, void* dst_w13_scale,
    void* dst_w2_weight, void* dst_w2_scale,
    size_t w13_weight_size, size_t w13_scale_size,
    size_t w2_weight_size, size_t w2_scale_size) {
  if (worker_) {
    stop_worker();
  }

  worker_ = std::make_unique<PollingMemcpyWorker>();
  worker_->setup(rank, cuda_device, cpu_core, sync_slot, src_buffers,
                 dst_w13_weight, dst_w13_scale, dst_w2_weight, dst_w2_scale,
                 w13_weight_size, w13_scale_size, w2_weight_size, w2_scale_size);
}

void PollingMemcpyWorkerManager::start_worker() {
  if (worker_) {
    worker_->start();
  }
}

void PollingMemcpyWorkerManager::stop_worker() {
  if (worker_) {
    worker_->stop();
    worker_.reset();
  }
}

}  // namespace polling_memcpy_worker
