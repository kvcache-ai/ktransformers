/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:38
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_BACKEND_H
#define CPUINFER_BACKEND_H

#include <hwloc.h>
#include <numa.h>

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// #define PROFILE_BALANCE

inline void set_to_numa(int this_numa) {
  struct bitmask* mask = numa_bitmask_alloc(numa_num_configured_nodes());
  numa_bitmask_setbit(mask, this_numa);
  numa_bind(mask);
  numa_bitmask_free(mask);
}

inline void set_memory_to_numa(int this_numa) {
  // printf("Set memory to NUMA %d\n", this_numa);
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, this_numa);
  if (!obj) {
    fprintf(stderr, "NUMA node %d not found.\n", this_numa);
    hwloc_topology_destroy(topology);
    return;
  }

  auto ret = hwloc_set_membind(topology, obj->nodeset, HWLOC_MEMBIND_BIND,
                               HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT | HWLOC_MEMBIND_BYNODESET);
  if (ret != 0) {
    perror("hwloc_set_membind_nodeset");
  }

  hwloc_topology_destroy(topology);
}

enum ThreadStatus {
  WORKING,
  WAITING,
  EXIT,
};

struct alignas(64) ThreadState {
  std::atomic<ThreadStatus> status;
  uint64_t finish_cycles;  // Per-thread timing (always enabled)
  int task_count;          // Per-thread task count
  uint64_t start_ts;       // Absolute start timestamp (RDTSC)
  uint64_t end_ts;         // Absolute end timestamp (RDTSC)
};

class InNumaPool {
 public:
  InNumaPool(int thread_count);
  InNumaPool(int max_thread_num, int numa_id, int threads_id_start);
  ~InNumaPool();
  int get_thread_num();
  void set_restricted_worker_count(int count);

  void do_work_stealing_job_async(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>,
                                  int block_size = 0);
  void wait();

  void do_work_stealing_job(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>,
                            const char* task_name = nullptr, int block_size = 0, bool async = false);
  void do_work_stealing_job(int, std::function<void(int)>, const char* task_name = nullptr, int block_size = 0,
                            bool async = false);

  // Get per-thread timing info
  int get_worker_count() const { return worker_count; }
  int get_numa_id() const { return numa_id_; }
  uint64_t get_thread_cycles(int tid) const { return thread_state_[tid].finish_cycles; }
  int get_thread_task_count(int tid) const { return thread_state_[tid].task_count; }
  uint64_t get_thread_start_ts(int tid) const { return thread_state_[tid].start_ts; }
  uint64_t get_thread_end_ts(int tid) const { return thread_state_[tid].end_ts; }

  // Reset per-thread timing/task counters (call before timing a sequence of operations)
  // NOTE: Only call when all workers are in WAITING state (after wait() returns)
  void reset_counters() {
    for (int i = 0; i < total_worker_count; i++) {
      thread_state_[i].finish_cycles = 0;
      thread_state_[i].task_count = 0;
      thread_state_[i].start_ts = 0;
      thread_state_[i].end_ts = 0;
    }
  }

 private:
  int worker_count;
  int total_worker_count;
  int numa_id_;

  std::unique_ptr<ThreadState[]> thread_state_;  // [thread_num]
  std::vector<std::thread> workers_;

  // changed ever time called do_work_stealing_job_async
  int restricted_worker_count;
  int block_size_;
  std::function<void(int)> init_func_;
  std::function<void(int)> compute_func_;
  std::function<void(int)> finalize_func_;
  std::atomic<int> curr_;
  int end_;

  void process_tasks(int);
  void worker_thread(int, int);
};

class NumaJobDistributor {
 public:
  NumaJobDistributor(int numa_count);
  NumaJobDistributor(std::vector<int> numa_ids);
  NumaJobDistributor(std::vector<int> numa_ids, std::vector<int> thread_count);

  ~NumaJobDistributor();

  void do_numa_job(std::function<void(int)>);

 private:
  void init(std::vector<int> numa_ids);
  void init(std::vector<int> numa_ids, std::vector<int> thread_count);

  std::unique_ptr<std::barrier<>> ready_bar;

  int numa_count;
  std::vector<int> numa_ids;
  std::vector<std::unique_ptr<std::atomic<ThreadStatus>>> status;
  std::function<void(int)> compute_func;
  std::vector<std::thread> workers;

  void worker_thread(int);
};

struct WorkerPoolConfig {
  int subpool_count;
  std::vector<int> subpool_numa_map;
  std::vector<int> subpool_thread_count;
};

class WorkerPool {
 public:
  WorkerPool(int total_thread_count);
  WorkerPool(int total_thread_count, int single_numa_id);
  WorkerPool(WorkerPoolConfig config);
  ~WorkerPool();
  int get_thread_num();
  void set_restricted_worker_count(int count);

  static thread_local int thread_local_id;

  NumaJobDistributor* dispense_backend();

  InNumaPool* get_subpool(int numa_id);

  void do_work_stealing_job(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>,
                            const char* task_name = nullptr, int block_size = 0, bool async = false);
  void do_work_stealing_job(int, std::function<void(int)>, const char* task_name = nullptr, int block_size = 0,
                            bool async = false);

  void wait();

  WorkerPoolConfig config;

 private:
  void init(WorkerPoolConfig config);

  int total_thread_count;
  int numa_count;
  int threads_per_numa;
  std::unique_ptr<NumaJobDistributor> distributor;

  std::vector<std::unique_ptr<InNumaPool>> numa_worker_pools;
};

// =====================================================
// Global per-thread timing for SFT MOE forward/backward
// =====================================================
// Define SFT_TIMER_DISABLED to disable all timing (functions become no-ops)
// #define SFT_TIMER_DISABLED
namespace sft_timer {

#ifdef SFT_TIMER_DISABLED
// Disabled: all functions are no-ops
inline void reset_forward() {}
inline void reset_backward() {}
inline void collect_forward(InNumaPool*) {}
inline void collect_backward(InNumaPool*) {}
inline void print_forward() {}
inline void print_backward(const char* = "backward") {}
inline void print_op_stats(InNumaPool*, const char*) {}
inline uint64_t get_trace_timestamp() { return 0; }
inline void add_kernel_trace(const char*, uint64_t, uint64_t, int, int, const char* = nullptr) {}
#else
// Enabled: declarations only, implementation in worker_pool.cpp
void reset_forward();
void reset_backward();
void collect_forward(InNumaPool* pool);
void collect_backward(InNumaPool* pool);
void print_forward();
void print_backward(const char* name = "backward");

// Print per-thread timing for a single operation
// Call pool->reset_counters() BEFORE the operation, then call this AFTER
void print_op_stats(InNumaPool* pool, const char* op_name);

// =====================================================
// Kernel-level tracing API
// For tracing individual kernels (e.g., AVX matmul) within worker threads
// =====================================================

// Get current RDTSC timestamp (lightweight, ~20 cycles overhead)
uint64_t get_trace_timestamp();

// Add a kernel trace event
// @param name      Kernel name (e.g., "lora_bf16_matmul_t4r4")
// @param start_ts  Start timestamp from get_trace_timestamp()
// @param end_ts    End timestamp from get_trace_timestamp()
// @param numa_id   NUMA node ID (use -1 for auto-detect or 0 if unknown)
// @param thread_id Thread ID within the pool (use WorkerPool::thread_local_id)
// @param args      Optional JSON args string (e.g., "{\"tokens\":128,\"rank\":8}")
void add_kernel_trace(const char* name, uint64_t start_ts, uint64_t end_ts, int numa_id, int thread_id,
                      const char* args = nullptr);
#endif

}  // namespace sft_timer

#endif