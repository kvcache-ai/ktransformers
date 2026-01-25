/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "worker_pool.h"

#include <hwloc/bitmap.h>
#include <numa.h>
#include <numaif.h>
#include <signal.h>
#include <x86intrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hwloc.h"

// RDTSC-based timer for lightweight timing
// Uses CPU timestamp counter instead of system clock for lower overhead
namespace {

// Read CPU timestamp counter (RDTSC)
inline uint64_t rdtsc_now() { return __rdtsc(); }

// Estimate RDTSC cycles for given milliseconds
// This is calculated once at startup
static uint64_t g_rdtsc_cycles_per_ms = 0;

// Initialize RDTSC frequency by measuring against chrono
static uint64_t init_rdtsc_frequency() {
  auto start_chrono = std::chrono::high_resolution_clock::now();
  uint64_t start_rdtsc = rdtsc_now();

  // Busy wait for ~10ms to calibrate
  while (true) {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_chrono).count();
    if (elapsed >= 10) break;
  }

  uint64_t end_rdtsc = rdtsc_now();
  auto end_chrono = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono).count();

  if (elapsed_ms > 0) {
    return (end_rdtsc - start_rdtsc) / elapsed_ms;
  }
  // Fallback: assume 2.5 GHz CPU
  return 2500000;
}

// Get cycles per millisecond (lazy initialization)
inline uint64_t get_rdtsc_cycles_per_ms() {
  if (g_rdtsc_cycles_per_ms == 0) {
    g_rdtsc_cycles_per_ms = init_rdtsc_frequency();
  }
  return g_rdtsc_cycles_per_ms;
}

}  // namespace

// =====================================================
// Global per-thread timing for SFT MOE forward/backward
// Collects timing from InNumaPool worker threads
// =====================================================
namespace sft_timer {

constexpr int MAX_THREADS = 256;
static uint64_t forward_rt[MAX_THREADS] = {0};
static uint64_t backward_rt[MAX_THREADS] = {0};
static int forward_tasks[MAX_THREADS] = {0};
static int backward_tasks[MAX_THREADS] = {0};
static int forward_threads = 0;
static int backward_threads = 0;

inline double ticks_to_ms(uint64_t cycles) { return (double)cycles / get_rdtsc_cycles_per_ms(); }

// =====================================================
// Chrome Trace Event Format support
// =====================================================
struct TraceEvent {
  std::string name;       // event name (op_name)
  std::string cat;        // category
  char ph;                // phase: 'X' for complete event, 'B' for begin, 'E' for end
  double ts;              // timestamp in microseconds (with ns precision via decimals)
  double dur;             // duration in microseconds (with ns precision via decimals)
  int pid;                // process id (numa_id)
  int tid;                // thread id
  int task_count;         // number of tasks processed
  std::string args_json;  // optional custom args JSON (for kernel traces)
};

static std::vector<TraceEvent> g_trace_events;
static std::mutex g_trace_mutex;
static uint64_t g_trace_start_time = 0;  // baseline timestamp
static std::string g_trace_output_path = "sft_trace.json";

// Thread-safe initialization using std::call_once
static std::once_flag g_trace_init_flag;

// Initialize trace start time (thread-safe)
static void init_trace() {
  std::call_once(g_trace_init_flag, []() {
    g_trace_start_time = rdtsc_now();
    // Check for custom output path from environment
    const char* env_path = std::getenv("SFT_TRACE_PATH");
    if (env_path && env_path[0] != '\0') {
      g_trace_output_path = env_path;
    }
  });
}

// Convert RDTSC cycles to microseconds with nanosecond precision (as double)
// Chrome tracing uses microseconds but supports fractional values for sub-us precision
static double cycles_to_us(uint64_t cycles) {
  // cycles_per_ms * 1000 = cycles_per_us
  // cycles / cycles_per_us = microseconds
  // Using 1e6 for cycles_per_ms -> cycles_per_s, then divide to get us with ns precision
  double cycles_per_us = get_rdtsc_cycles_per_ms() / 1000.0;
  return static_cast<double>(cycles) / cycles_per_us;
}

// Add trace events for an operation using absolute timestamps
static void add_trace_events(const char* op_name, int numa_id, int thread_count, const uint64_t* start_ts_arr,
                             const uint64_t* end_ts_arr, const int* tasks) {
  init_trace();

  std::lock_guard<std::mutex> lock(g_trace_mutex);

  for (int i = 0; i < thread_count; i++) {
    // Convert absolute RDTSC timestamps to relative microseconds from trace start
    double start_us = (start_ts_arr[i] > g_trace_start_time) ? cycles_to_us(start_ts_arr[i] - g_trace_start_time) : 0.0;
    double end_us = (end_ts_arr[i] > g_trace_start_time) ? cycles_to_us(end_ts_arr[i] - g_trace_start_time) : 0.0;
    double dur_us = end_us - start_us;
    if (dur_us < 0) dur_us = 0;

    TraceEvent ev;
    ev.name = op_name;
    ev.cat = "sft_op";
    ev.ph = 'X';  // Complete event
    ev.ts = start_us;
    ev.dur = dur_us;
    ev.pid = numa_id;
    ev.tid = i;
    ev.task_count = tasks[i];

    g_trace_events.push_back(ev);
  }
}

// Write trace events to JSON file (Chrome Trace Event Format)
static void write_trace_to_file() {
  std::lock_guard<std::mutex> lock(g_trace_mutex);

  if (g_trace_events.empty()) {
    return;
  }

  // Sort events by (pid, tid, ts) to fix overlap issues in Chrome trace viewer
  // Events from same thread should be ordered by start time
  std::sort(g_trace_events.begin(), g_trace_events.end(), [](const TraceEvent& a, const TraceEvent& b) {
    if (a.pid != b.pid) return a.pid < b.pid;
    if (a.tid != b.tid) return a.tid < b.tid;
    return a.ts < b.ts;
  });

  std::ofstream ofs(g_trace_output_path);
  if (!ofs.is_open()) {
    fprintf(stderr, "sft_timer: Failed to open trace file: %s\n", g_trace_output_path.c_str());
    return;
  }

  // Use fixed precision for nanosecond accuracy (3 decimal places in microseconds = nanoseconds)
  ofs << std::fixed << std::setprecision(3);

  ofs << "{\n";
  ofs << "  \"traceEvents\": [\n";

  for (size_t i = 0; i < g_trace_events.size(); i++) {
    const auto& ev = g_trace_events[i];
    ofs << "    {";
    ofs << "\"name\":\"" << ev.name << "\",";
    ofs << "\"cat\":\"" << ev.cat << "\",";
    ofs << "\"ph\":\"" << ev.ph << "\",";
    ofs << "\"ts\":" << ev.ts << ",";
    ofs << "\"dur\":" << ev.dur << ",";
    ofs << "\"pid\":" << ev.pid << ",";
    ofs << "\"tid\":" << ev.tid << ",";
    if (!ev.args_json.empty()) {
      ofs << "\"args\":" << ev.args_json;
    } else {
      ofs << "\"args\":{\"task_count\":" << ev.task_count << "}";
    }
    ofs << "}";
    if (i < g_trace_events.size() - 1) {
      ofs << ",";
    }
    ofs << "\n";
  }

  ofs << "  ],\n";
  ofs << "  \"displayTimeUnit\": \"ns\"\n";
  ofs << "}\n";

  ofs.close();
  fprintf(stderr, "sft_timer: Trace written to %s (%zu events)\n", g_trace_output_path.c_str(), g_trace_events.size());
}

// Signal handler for SIGTERM
static void sigterm_handler(int sig) {
  fprintf(stderr, "sft_timer: Received signal %d, writing trace...\n", sig);
  write_trace_to_file();
  // Re-raise the signal with default handler to allow normal termination
  signal(sig, SIG_DFL);
  raise(sig);
}

// Register signal handlers
static void register_signal_handlers() {
  static bool registered = false;
  if (!registered) {
    signal(SIGTERM, sigterm_handler);
    signal(SIGINT, sigterm_handler);
    registered = true;
  }
}

// Destructor function - called at program exit
__attribute__((destructor)) static void trace_destructor() { write_trace_to_file(); }

void print_rt(FILE* out, const char* name, uint64_t* rt, int* tasks, int rt_threads) {
  if (rt_threads <= 0) return;
  FILE* output = out ? out : stderr;
  auto max_val = *std::max_element(rt, rt + rt_threads);
  auto min_val = *std::min_element(rt, rt + rt_threads);
  uint64_t sum = std::accumulate(rt, rt + rt_threads, (uint64_t)0);
  int total_tasks = std::accumulate(tasks, tasks + rt_threads, 0);

  // Sort to find 20% and 80% percentile thresholds
  std::vector<uint64_t> sorted(rt, rt + rt_threads);
  std::sort(sorted.begin(), sorted.end());
  int p20_idx = rt_threads * 20 / 100;
  int p80_idx = rt_threads * 80 / 100;
  uint64_t p20_threshold = sorted[p20_idx];  // Fast threshold (top 20%)
  uint64_t p80_threshold = sorted[p80_idx];  // Slow threshold (bottom 20%)

  // ANSI color codes
  const char* GREEN = "\033[32m";
  const char* RED = "\033[31m";
  const char* RESET = "\033[0m";

  // Line 1: time
  fprintf(output, "%30s max %.3f min %.3f avg %.3f : ", name, ticks_to_ms(max_val), ticks_to_ms(min_val),
          ticks_to_ms(sum / rt_threads));
  for (int i = 0; i < rt_threads; i++) {
    if (rt[i] <= p20_threshold) {
      fprintf(output, "%s%.3f%s ", GREEN, ticks_to_ms(rt[i]), RESET);
    } else if (rt[i] >= p80_threshold) {
      fprintf(output, "%s%.3f%s ", RED, ticks_to_ms(rt[i]), RESET);
    } else {
      fprintf(output, "%.3f ", ticks_to_ms(rt[i]));
    }
  }
  fprintf(output, "\n");

  // Line 2: task count
  fprintf(output, "%30s total %d : ", "tasks", total_tasks);
  for (int i = 0; i < rt_threads; i++) {
    if (rt[i] <= p20_threshold) {
      fprintf(output, "%s%d%s ", GREEN, tasks[i], RESET);
    } else if (rt[i] >= p80_threshold) {
      fprintf(output, "%s%d%s ", RED, tasks[i], RESET);
    } else {
      fprintf(output, "%d ", tasks[i]);
    }
  }
  fprintf(output, "\n");
}

void reset_forward() {
  std::fill(forward_rt, forward_rt + MAX_THREADS, 0);
  std::fill(forward_tasks, forward_tasks + MAX_THREADS, 0);
  forward_threads = 0;
}

void reset_backward() {
  std::fill(backward_rt, backward_rt + MAX_THREADS, 0);
  std::fill(backward_tasks, backward_tasks + MAX_THREADS, 0);
  backward_threads = 0;
}

void collect_forward(InNumaPool* pool) {
  int n = pool->get_worker_count();
  for (int i = 0; i < n && forward_threads < MAX_THREADS; i++) {
    forward_rt[forward_threads] = pool->get_thread_cycles(i);
    forward_tasks[forward_threads] = pool->get_thread_task_count(i);
    forward_threads++;
  }
}

void collect_backward(InNumaPool* pool) {
  int n = pool->get_worker_count();
  for (int i = 0; i < n && backward_threads < MAX_THREADS; i++) {
    backward_rt[backward_threads] = pool->get_thread_cycles(i);
    backward_tasks[backward_threads] = pool->get_thread_task_count(i);
    backward_threads++;
  }
}

void print_forward() { print_rt(stderr, "forward", forward_rt, forward_tasks, forward_threads); }
void print_backward(const char* name) { print_rt(stderr, name, backward_rt, backward_tasks, backward_threads); }

void print_op_stats(InNumaPool* pool, const char* op_name) {
  if (pool == nullptr || op_name == nullptr || op_name[0] == '\0') {
    return;
  }
  int n = pool->get_worker_count();
  if (n <= 0) {
    return;
  }

  // Ensure signal handlers are registered on first call
  static bool handlers_registered = false;
  if (!handlers_registered) {
    register_signal_handlers();
    handlers_registered = true;
  }

  FILE* output = stderr;
  int numa_id = pool->get_numa_id();
  // if (numa_id == 0) {
  //   output = stdout;
  // } else if (numa_id == 1) {
  //   output = stderr;
  // }
  std::vector<uint64_t> rt(n);
  std::vector<uint64_t> start_ts(n);
  std::vector<uint64_t> end_ts(n);
  std::vector<int> tasks(n);
  for (int i = 0; i < n; i++) {
    rt[i] = pool->get_thread_cycles(i);
    tasks[i] = pool->get_thread_task_count(i);
    start_ts[i] = pool->get_thread_start_ts(i);
    end_ts[i] = pool->get_thread_end_ts(i);
  }
  // print_rt(output, op_name, rt.data(), tasks.data(), n);

  // Save trace data to memory for later export
  add_trace_events(op_name, numa_id, n, start_ts.data(), end_ts.data(), tasks.data());
}

// =====================================================
// Kernel-level tracing API implementation
// =====================================================

uint64_t get_trace_timestamp() { return rdtsc_now(); }

void add_kernel_trace(const char* name, uint64_t start_ts, uint64_t end_ts, int numa_id, int thread_id,
                      const char* args) {
  init_trace();

  // Convert absolute RDTSC timestamps to relative microseconds from trace start
  double start_us = (start_ts > g_trace_start_time) ? cycles_to_us(start_ts - g_trace_start_time) : 0.0;
  double end_us = (end_ts > g_trace_start_time) ? cycles_to_us(end_ts - g_trace_start_time) : 0.0;
  double dur_us = end_us - start_us;
  if (dur_us < 0) dur_us = 0;

  std::lock_guard<std::mutex> lock(g_trace_mutex);

  TraceEvent ev;
  ev.name = name;
  ev.cat = "kernel";
  ev.ph = 'X';  // Complete event
  ev.ts = start_us;
  ev.dur = dur_us;
  ev.pid = numa_id;
  ev.tid = thread_id;
  ev.task_count = 0;  // Not applicable for kernel traces
  if (args != nullptr && args[0] != '\0') {
    ev.args_json = args;
  }

  g_trace_events.push_back(ev);
}

}  // namespace sft_timer

// Intel ITT API for profiler integration (VTune, etc.)
// Allows profilers to identify spin-wait regions
#ifdef USE_ITT_NOTIFY
#include <ittnotify.h>
static __itt_domain* g_itt_domain = nullptr;
static __itt_string_handle* g_itt_spin_wait = nullptr;

static void init_itt() {
  if (g_itt_domain == nullptr) {
    g_itt_domain = __itt_domain_create("WorkerPool");
    g_itt_spin_wait = __itt_string_handle_create("SpinWait");
  }
}

#define ITT_SYNC_PREPARE(addr) __itt_sync_prepare(addr)
#define ITT_SYNC_CANCEL(addr) __itt_sync_cancel(addr)
#define ITT_SYNC_ACQUIRED(addr) __itt_sync_acquired(addr)
#else
#define ITT_SYNC_PREPARE(addr) ((void)0)
#define ITT_SYNC_CANCEL(addr) ((void)0)
#define ITT_SYNC_ACQUIRED(addr) ((void)0)
static void init_itt() {}
#endif

thread_local int WorkerPool::thread_local_id = -1;

InNumaPool::InNumaPool(int max_thread_num) {
  printf("In Numa Worker Pool at NUMA %d, %d threads\n", numa_node_of_cpu(sched_getcpu()), max_thread_num);
  numa_id_ = numa_node_of_cpu(sched_getcpu());
  total_worker_count = max_thread_num;
  block_size_ = 0;
  set_restricted_worker_count(total_worker_count);
  thread_state_ = std::unique_ptr<ThreadState[]>(new ThreadState[max_thread_num]);
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WAITING, std::memory_order_release);
  }
  workers_.resize(total_worker_count);
  for (int i = 1; i < total_worker_count; i++) {
    workers_[i] = std::thread(&InNumaPool::worker_thread, this, i, -1);
  }
}

InNumaPool::InNumaPool(int max_thread_num, int numa_id, int threads_id_start) {
  printf("===========In NumaPool============\n");
  hwloc_topology_t topology;
  hwloc_obj_t numa_obj, core_obj;
  hwloc_bitmap_t cpuset;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);
  printf("In Numa Worker Pool at NUMA %d, %d threads\n", numa_node_of_cpu(sched_getcpu()), max_thread_num);
  numa_id_ = numa_id;
  total_worker_count = max_thread_num;
  block_size_ = 0;
  set_restricted_worker_count(total_worker_count);
  thread_state_ = std::unique_ptr<ThreadState[]>(new ThreadState[max_thread_num]);
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WAITING, std::memory_order_release);
  }
  workers_.resize(total_worker_count);
  for (int i = 1; i < total_worker_count; i++) {
    workers_[i] = std::thread(&InNumaPool::worker_thread, this, i, numa_id);
    // set the thread name as: "numa_(numa_id)_t_(i+threads_id_start)"
    std::string thread_name = "numa_" + std::to_string(numa_id) + "_t_" + std::to_string(i + threads_id_start);
    pthread_t native_handle = workers_[i].native_handle();
    auto res_set_name = pthread_setname_np(native_handle, thread_name.c_str());
    if (res_set_name != 0) {
      fprintf(stderr, "Failed to set thread name: %s\n", strerror(res_set_name));
    }
    // 检查线程是否成功命名
    char name[16];
    pthread_getname_np(native_handle, name, sizeof(name));
    if (strcmp(name, thread_name.c_str()) == 0) {
      // printf("Thread name set successfully: %s\n", name);
    } else {
      // printf("Failed to set thread name: %s\n", name);
    }
    // Set the thread affinity to the specified NUMA node's CPU
    numa_obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa_id);
    if (!numa_obj) {
      fprintf(stderr, "NUMA node %d not found\n", numa_id);
      // throw std::runtime_error("NUMA node not found");
      continue;
    }
    core_obj = hwloc_get_obj_inside_cpuset_by_type(topology, numa_obj->cpuset, HWLOC_OBJ_CORE, i + threads_id_start);
    if (!core_obj) {
      fprintf(stderr, "Core %d inside NUMA node %d not found\n", i, numa_id);
      // throw std::runtime_error("Core not found inside NUMA node");
      continue;
    }
    cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_copy(cpuset, core_obj->cpuset);
    hwloc_bitmap_singlify(cpuset);
    auto res = hwloc_set_thread_cpubind(topology, native_handle, cpuset, HWLOC_CPUBIND_STRICT);
    if (res != 0) {
      fprintf(stderr, "Failed to set thread CPU binding: %s\n", strerror(errno));
    }
  }
}

InNumaPool::~InNumaPool() {
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::EXIT, std::memory_order_release);
  }
  for (int i = 0; i < total_worker_count; i++) {
    if (workers_[i].joinable()) {
      workers_[i].join();
    }
  }
}

int InNumaPool::get_thread_num() {
  throw std::runtime_error("Deprecated");
  return total_worker_count;
}

void InNumaPool::set_restricted_worker_count(int count) { restricted_worker_count = count; }

void InNumaPool::wait() {
  for (int i = 0; i < worker_count; i++) {
    while (thread_state_[i].status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }

#ifdef PROFILE_BALANCE
  size_t max_time = 0;
  size_t min_time = thread_state_[0].finish_ns;
  size_t sum = 0;
  for (int i = 0; i < worker_count; i++) {
    sum += thread_state_[i].finish_ns;
    max_time = std::max(max_time, thread_state_[i].finish_ns);
    min_time = std::min(min_time, thread_state_[i].finish_ns);
  }
  double balance = 1.0 * sum / (max_time * worker_count);
  printf("max_time: %ld, min_time: %ld, sum_time: %ld, balance: %f\n", max_time, min_time, sum, balance);

#endif
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func, const char* task_name,
                                      int block_size, bool async) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr, task_name, block_size);
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func,
                                      const char* task_name, int block_size, bool async) {
  bool has_name = task_name != nullptr && task_name[0] != '\0';
  if (has_name) {
    reset_counters();
  }
  do_work_stealing_job_async(task_num, init_func, compute_func, finalize_func, block_size);
  if (!async) wait();
  if (has_name) {
    sft_timer::print_op_stats(this, task_name);
  }
}

void InNumaPool::do_work_stealing_job_async(int task_num, std::function<void(int)> init_func,
                                            std::function<void(int)> compute_func,
                                            std::function<void(int)> finalize_func, int block_size) {
  init_func_ = init_func;
  compute_func_ = compute_func;
  finalize_func_ = finalize_func;
  block_size_ = block_size;
  worker_count = std::min(restricted_worker_count, task_num);
  curr_.store(0, std::memory_order_release);
  end_ = task_num;

  for (int i = 0; i < worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WORKING, std::memory_order_release);
  }
  WorkerPool::thread_local_id = 0;
  process_tasks(0);
}

void InNumaPool::process_tasks(int thread_id) {
  uint64_t start_cycles = rdtsc_now();
  auto& s = thread_state_[thread_id];
  int local_task_count = 0;

  // Record absolute start timestamp
  s.start_ts = start_cycles;

  if (init_func_ != nullptr) {
    init_func_(thread_id);
  }

  // omp-guided-style work scheduling
  while (true) {
    int old = curr_.load(std::memory_order_relaxed);
    int rem = end_ - old;
    if (rem <= 0) {
      break;
    }

    int block = 0;
    if (block_size_ > 0) {
      block = std::min(block_size_, rem);
    } else {
      block = (rem + worker_count - 1) / worker_count;
    }
    block = 1;
    int task_id = curr_.fetch_add(block, std::memory_order_acq_rel);
    if (task_id >= end_) {
      break;
    }

    for (int i = 0; i < block; i++) {
      if (task_id + i >= end_) {
        break;
      }
      compute_func_(task_id + i);
      local_task_count++;
    }
  }

  if (finalize_func_ != nullptr) {
    finalize_func_(thread_id);
  }

  // IMPORTANT: Update timing BEFORE setting status to WAITING
  // The release semantics of status.store() ensures all prior writes are visible
  uint64_t end_cycles = rdtsc_now();
  s.finish_cycles = end_cycles - start_cycles;
  s.task_count = local_task_count;
  s.end_ts = end_cycles;

  // Signal completion - release ensures timing writes are visible to wait()
  s.status.store(ThreadStatus::WAITING, std::memory_order_release);
}

void InNumaPool::worker_thread(int thread_id, int numa_id) {
  if (numa_id >= 0) {
    set_memory_to_numa(numa_id);
  }
  init_itt();  // Initialize ITT if enabled
  // Use RDTSC for lightweight timing instead of std::chrono
  const uint64_t sleep_threshold_cycles = get_rdtsc_cycles_per_ms() * 50;  // 50ms in cycles
  uint64_t start = rdtsc_now();
  WorkerPool::thread_local_id = thread_id;  // 设置线程本地变量
  while (true) {
    ITT_SYNC_PREPARE(&thread_state_[thread_id].status);  // Signal profiler: about to spin-wait
    ThreadStatus status = thread_state_[thread_id].status.load(std::memory_order_acquire);
    if (status == ThreadStatus::WORKING) {
      ITT_SYNC_ACQUIRED(&thread_state_[thread_id].status);  // Signal profiler: acquired work
      process_tasks(thread_id);
      start = rdtsc_now();
    } else if (status == ThreadStatus::WAITING) {
      // PAUSE instruction hints to CPU this is a spin-wait loop
      _mm_pause();
      uint64_t now = rdtsc_now();
      uint64_t elapsed_cycles = now - start;
      if (elapsed_cycles > sleep_threshold_cycles) {
        ITT_SYNC_CANCEL(&thread_state_[thread_id].status);  // Signal profiler: going to sleep
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    } else if (status == ThreadStatus::EXIT) {
      ITT_SYNC_CANCEL(&thread_state_[thread_id].status);  // Signal profiler: exiting
      return;
    }
  }
}

NumaJobDistributor::NumaJobDistributor(int numa_count) {
  std::vector<int> numa_ids;
  for (int i = 0; i < numa_count; i++) {
    numa_ids.push_back(i);
  }
  init(numa_ids);
}

NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids) { init(numa_ids); }
NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids, std::vector<int> thread_count) {
  init(numa_ids, thread_count);
}

void NumaJobDistributor::init(std::vector<int> numa_ids) {
  this->numa_count = numa_ids.size();
  this->ready_bar = std::unique_ptr<std::barrier<>>(new std::barrier<>(numa_count + 1));
  this->numa_ids = numa_ids;
  for (size_t i = 0; i < numa_count; i++) {
    status.push_back(nullptr);
  }

  workers.resize(numa_count);
  for (int i = 0; i < numa_count; i++) {
    std::thread([this, i]() { workers[i] = std::thread(&NumaJobDistributor::worker_thread, this, i); }).join();
  }
  ready_bar->arrive_and_wait();
}

void NumaJobDistributor::init(std::vector<int> numa_ids, std::vector<int> thread_count) {
  hwloc_topology_t topology;
  hwloc_obj_t numa_obj, core_obj;
  hwloc_bitmap_t cpuset;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  this->numa_count = numa_ids.size();
  this->ready_bar = std::unique_ptr<std::barrier<>>(new std::barrier<>(numa_count + 1));
  this->numa_ids = numa_ids;
  for (size_t i = 0; i < numa_count; i++) {
    status.push_back(nullptr);
  }

  workers.resize(numa_count);
  std::vector<int> numa_threads_count(numa_count, 0);
  for (int i = 0; i < numa_count; i++) {
    workers[i] = std::thread(&NumaJobDistributor::worker_thread, this, i);
    auto this_numa = numa_ids[i];
    auto start_id = numa_threads_count[this_numa];
    // set the thread name as: "worker_numa_(numa_id)_main_start_id(0)"
    // printf("nuam_id %d, start_id %d\n", this_numa, start_id);
    std::string thread_name = "numa_" + std::to_string(numa_ids[i]) + "_m_" + std::to_string(start_id);
    pthread_t native_handle = workers[i].native_handle();
    pthread_setname_np(native_handle, thread_name.c_str());
    // Set the thread affinity to the specified NUMA node's CPU (0)
    numa_obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, this_numa);
    if (!numa_obj) {
      fprintf(stderr, "NUMA node %d not found\n", this_numa);
      // throw std::runtime_error("NUMA node not found");
      continue;
    }
    core_obj = hwloc_get_obj_inside_cpuset_by_type(topology, numa_obj->cpuset, HWLOC_OBJ_CORE, start_id);
    if (!core_obj) {
      fprintf(stderr, "Core %d inside NUMA node %d not found\n", 0, this_numa);
      // throw std::runtime_error("Core not found inside NUMA node");
      continue;
    }
    // 精简 cpuset
    auto cpuset_simple = hwloc_bitmap_alloc();
    hwloc_bitmap_copy(cpuset_simple, core_obj->cpuset);
    hwloc_bitmap_singlify(cpuset_simple);
    // 打印绑定的具体的 CPU 物理索引
    unsigned long i_in;
    // hwloc_bitmap_foreach_begin(i_in, cpuset_simple) { printf("Thread %d bound to CPU %ld\n", start_id, i_in); }
    // hwloc_bitmap_foreach_end();
    auto res = hwloc_set_thread_cpubind(topology, native_handle, cpuset_simple, HWLOC_CPUBIND_STRICT);
    if (res != 0) {
      fprintf(stderr, "Failed to set thread CPU binding: %s\n", strerror(errno));
    }
    // 检查线程是否绑定到指定的 核上了
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_thread_cpubind(topology, native_handle, cpuset, HWLOC_CPUBIND_THREAD);
    // hwloc_bitmap_foreach_begin(i_in, cpuset) { printf("Thread %d is bound to CPU %ld\n", start_id, i_in); }
    // hwloc_bitmap_foreach_end();

    numa_threads_count[this_numa] += thread_count[i];
  }
  ready_bar->arrive_and_wait();
}

NumaJobDistributor::~NumaJobDistributor() {
  for (int i = 0; i < numa_count; i++) {
    status[i]->store(ThreadStatus::EXIT, std::memory_order_release);
  }
  for (int i = 0; i < numa_count; i++) {
    if (workers[i].joinable()) {
      workers[i].join();
    }
  }
}

#ifdef USE_NUMA_JOB_DIRECT_WORK

void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {
  this->compute_func = compute_func;
  auto me_numa = numa_node_of_cpu(sched_getcpu());
  for (int i = 0; i < numa_count; i++) {
    if (i == me_numa) continue;

    status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
  }
  compute_func(me_numa);
  for (int i = 0; i < numa_count; i++) {
    if (i == me_numa) continue;

    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }
}
#else
void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {
  this->compute_func = compute_func;
  for (int i = 0; i < numa_count; i++) {
    status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
  }
  for (int i = 0; i < numa_count; i++) {
    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }
}
#endif

void NumaJobDistributor::worker_thread(int numa_id) {
  init_itt();  // Initialize ITT if enabled
  // Use RDTSC for lightweight timing instead of std::chrono
  const uint64_t sleep_threshold_cycles = get_rdtsc_cycles_per_ms() * 50;  // 50ms in cycles
  uint64_t start = rdtsc_now();
  set_memory_to_numa(numa_id);
  status[numa_id] =
      std::move(std::unique_ptr<std::atomic<ThreadStatus>>(new std::atomic<ThreadStatus>(ThreadStatus::WAITING)));
  ready_bar->arrive_and_wait();
  while (true) {
    ITT_SYNC_PREPARE(status[numa_id].get());  // Signal profiler: about to spin-wait
    auto stat = status[numa_id]->load(std::memory_order_acquire);
    if (stat == ThreadStatus::WORKING) {
      ITT_SYNC_ACQUIRED(status[numa_id].get());  // Signal profiler: acquired work
      auto me_numa = numa_node_of_cpu(sched_getcpu());
      // printf("numa work on %d, me %d\n", numa_id, me_numa);
      compute_func(numa_id);
      status[numa_id]->store(ThreadStatus::WAITING, std::memory_order_release);
      start = rdtsc_now();
    } else if (stat == ThreadStatus::WAITING) {
      // PAUSE instruction hints to CPU this is a spin-wait loop
      _mm_pause();
      uint64_t now = rdtsc_now();
      uint64_t elapsed_cycles = now - start;
      if (elapsed_cycles > sleep_threshold_cycles) {
        ITT_SYNC_CANCEL(status[numa_id].get());  // Signal profiler: going to sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } else if (stat == ThreadStatus::EXIT) {
      ITT_SYNC_CANCEL(status[numa_id].get());  // Signal profiler: exiting
      return;
    }
  }
}

void WorkerPool::init(WorkerPoolConfig config) {
  printf("WorkerPool[0x%lx] %d subpools, [numa:threads]", (intptr_t)this, config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    printf("[%d:%d] ", config.subpool_numa_map[i], config.subpool_thread_count[i]);
  }
  printf("\n");

  for (int i = 0; i < config.subpool_count; i++) {
    numa_worker_pools.push_back(nullptr);
  }
  std::vector<int> numa_threads_count(config.subpool_count, 0);
  for (int i = 0; i < config.subpool_count; i++) {
    auto this_numa = config.subpool_numa_map[i];
    auto this_thread_count = config.subpool_thread_count[i];
    auto this_thread_id_start = numa_threads_count[this_numa];
    std::thread([this, i, this_numa, this_thread_count, this_thread_id_start]() {
      set_to_numa(this_numa);
      numa_worker_pools[i] =
          std::move(std::unique_ptr<InNumaPool>(new InNumaPool(this_thread_count, this_numa, this_thread_id_start)));
      // numa_worker_pools[i] = std::move(std::unique_ptr<InNumaPool>(new InNumaPool(this_thread_count)));
    }).join();
    numa_threads_count[this_numa] += this_thread_count;
  }

  distributor = std::move(std::unique_ptr<NumaJobDistributor>(
      new NumaJobDistributor(config.subpool_numa_map, config.subpool_thread_count)));
  // distributor = std::move(std::unique_ptr<NumaJobDistributor>(new NumaJobDistributor(config.subpool_numa_map)));
}

WorkerPool::WorkerPool(WorkerPoolConfig config) : config(config) { init(config); }

WorkerPool::WorkerPool(int total_threads) {
  config.subpool_count = numa_num_configured_nodes();
  config.subpool_numa_map.resize(config.subpool_count);
  config.subpool_thread_count.resize(config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    config.subpool_numa_map[i] = i;
    config.subpool_thread_count[i] = total_threads / config.subpool_count;
  }
  init(config);
}

WorkerPool::WorkerPool(int total_threads, int single_numa_id) {
  set_to_numa(single_numa_id);
  config.subpool_count = numa_num_configured_nodes();
  config.subpool_numa_map.resize(config.subpool_count);
  config.subpool_thread_count.resize(config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    config.subpool_numa_map[i] = single_numa_id;
    config.subpool_thread_count[i] = total_threads / config.subpool_count;
  }
  init(config);
}

WorkerPool::~WorkerPool() {}

int WorkerPool::get_thread_num() { return total_thread_count; }

void WorkerPool::set_restricted_worker_count(int count) {
  for (int i = 0; i < numa_count; i++) {
    numa_worker_pools[i]->set_restricted_worker_count(threads_per_numa);
  }
}

InNumaPool* WorkerPool::get_subpool(int numa_id) { return numa_worker_pools[numa_id].get(); }

NumaJobDistributor* WorkerPool::dispense_backend() { return distributor.get(); }

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func,
                                      const char* task_name, int block_size, bool async) {
  numa_worker_pools[0]->do_work_stealing_job(task_num, init_func, compute_func, finalize_func, task_name, block_size,
                                             async);
}

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func, const char* task_name,
                                      int block_size, bool async) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr, task_name, block_size, async);
}

void WorkerPool::wait() { numa_worker_pools[0]->wait(); }