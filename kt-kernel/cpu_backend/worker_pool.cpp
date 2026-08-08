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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

#include "hwloc.h"

thread_local int WorkerPool::thread_local_id = -1;

// 自动检测被占用的 CPU: 采样 /proc/stat 两次，计算每个 CPU 的 idle 率。
// idle < threshold 的 CPU 被认为是"被占用"，绑定时跳过。
// 这样不需要手动设 KT_SKIP_CPUS，自动避开其他用户的 worker。
static std::set<int> detect_busy_cpus(double idle_threshold = 0.3, int sample_us = 200000) {
  std::set<int> busy;
  auto read_idle = [](std::vector<std::pair<long, long>>* out) {
    FILE* f = fopen("/proc/stat", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "cpu", 3) != 0 || !isdigit((unsigned char)line[3])) continue;
      int cpu_id;
      long user, nice, system, idle, iowait, irq, softirq, steal;
      int n = sscanf(line, "cpu%d %ld %ld %ld %ld %ld %ld %ld %ld", &cpu_id, &user, &nice,
                     &system, &idle, &iowait, &irq, &softirq, &steal);
      if (n >= 5) {
        long total = user + nice + system + idle + iowait + irq + softirq + steal;
        if (cpu_id >= (int)out->size()) out->resize(cpu_id + 1, {0, 0});
        (*out)[cpu_id] = {idle, total};
      }
    }
    fclose(f);
  };
  std::vector<std::pair<long, long>> first, second;
  read_idle(&first);
  usleep(sample_us);
  read_idle(&second);
  for (size_t i = 0; i < first.size() && i < second.size(); i++) {
    long d_idle = second[i].first - first[i].first;
    long d_total = second[i].second - first[i].second;
    if (d_total > 0) {
      double idle_rate = (double)d_idle / d_total;
      if (idle_rate < idle_threshold) {
        busy.insert((int)i);
      }
    }
  }
  return busy;
}

// 返回需要跳过的 CPU 集合。
// 优先用环境变量 KT_SKIP_CPUS (逗号分隔的物理 CPU ID)；
// 若未设置，则自动检测 (采样 /proc/stat，idle<30% 的 CPU)。
static const std::set<int>& get_skip_cpus() {
  static std::set<int> skip;
  static bool inited = false;
  if (!inited) {
    inited = true;
    const char* env = getenv("KT_SKIP_CPUS");
    if (env && *env) {
      std::stringstream ss(env);
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
          try {
            skip.insert(std::stoi(item));
          } catch (...) {
          }
        }
      }
      fprintf(stderr, "[MESH] KT_SKIP_CPUS manual (%zu cpus): ", skip.size());
      for (int c : skip) fprintf(stderr, "%d ", c);
      fprintf(stderr, "\n");
    } else {
      skip = detect_busy_cpus(0.30, 200000);
      fprintf(stderr, "[MESH] auto-detected %zu busy cpus (idle<30%%): ", skip.size());
      for (int c : skip) fprintf(stderr, "%d ", c);
      fprintf(stderr, "\n");
    }
  }
  return skip;
}

// 全局 core 分配器: 每个 NUMA 节点维护一个 next_core_idx，
// NumaJobDistributor (main worker) 和 InNumaPool (worker) 共享，
// 避免多个线程绑到同一个 core。
static std::mutex g_core_alloc_mtx;
static std::map<int, int> g_next_core_idx;  // numa_id -> next available core idx

static int alloc_core_idx(int numa_id, int hint_idx) {
  std::lock_guard<std::mutex> lock(g_core_alloc_mtx);
  auto it = g_next_core_idx.find(numa_id);
  int idx = (it == g_next_core_idx.end()) ? hint_idx : std::max(it->second, hint_idx);
  return idx;
}

static void advance_core_idx(int numa_id, int used_idx) {
  std::lock_guard<std::mutex> lock(g_core_alloc_mtx);
  if (g_next_core_idx[numa_id] <= used_idx) {
    g_next_core_idx[numa_id] = used_idx + 1;
  }
}

InNumaPool::InNumaPool(int max_thread_num) {
  printf("In Numa Worker Pool at NUMA %d, %d threads\n", numa_node_of_cpu(sched_getcpu()), max_thread_num);
  total_worker_count = max_thread_num;
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
  total_worker_count = max_thread_num;
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
    // 跳过被其他用户占用的 core: 用全局 alloc_core_idx 获取起始位置，
    // 确保不会和 NumaJobDistributor main worker 或其他 InNumaPool worker 重复。
    const auto& skip_cpus = get_skip_cpus();
    int core_search_idx = alloc_core_idx(numa_id, i + threads_id_start);
    int bound_cpu_id = -1;
    while (true) {
      core_obj = hwloc_get_obj_inside_cpuset_by_type(topology, numa_obj->cpuset, HWLOC_OBJ_CORE, core_search_idx);
      if (!core_obj) {
        break;
      }
      int cpu_id = hwloc_bitmap_first(core_obj->cpuset);
      if (cpu_id >= 0 && skip_cpus.count(cpu_id)) {
        fprintf(stderr, "[MESH] Skip occupied CPU %d (core idx %d) for worker %d\n", cpu_id, core_search_idx, i);
        core_search_idx++;
        continue;
      }
      bound_cpu_id = cpu_id;
      break;
    }
    if (!core_obj) {
      fprintf(stderr, "Core %d inside NUMA node %d not found (searched from %d)\n", i, numa_id, i + threads_id_start);
      // throw std::runtime_error("Core not found inside NUMA node");
      continue;
    }
    advance_core_idx(numa_id, core_search_idx);
    cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_copy(cpuset, core_obj->cpuset);
    hwloc_bitmap_singlify(cpuset);
    auto res = hwloc_set_thread_cpubind(topology, native_handle, cpuset, HWLOC_CPUBIND_STRICT);
    if (res != 0) {
      fprintf(stderr, "Failed to set thread CPU binding: %s\n", strerror(errno));
    } else {
      fprintf(stderr, "[MESH] numa_%d_t_%d -> CPU %d (core idx %d)\n", numa_id, i + threads_id_start, bound_cpu_id, core_search_idx);
    }
  }
}

InNumaPool::~InNumaPool() {
  for (int i = 0; i < total_worker_count; i++) {
    {
      std::lock_guard<std::mutex> lock(thread_state_[i].mutex);
      thread_state_[i].status.store(ThreadStatus::EXIT, std::memory_order_release);
    }
    thread_state_[i].cv.notify_one();
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
  auto wait_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < worker_count; i++) {
    while (thread_state_[i].status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }
  auto wait_end = std::chrono::high_resolution_clock::now();
  long wait_us = std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();

#ifdef PROFILE_BALANCE
  if (wait_us > 1000) {  // 超过 1ms，打印每个 worker 的唤醒延迟与执行时间分解
    fprintf(stderr, "[WAIT_SLOW] wait=%ldus wc=%d", wait_us, worker_count);
    for (int j = 0; j < worker_count; j++) {
      // wake_us: worker 检测到 WORKING 的唤醒延迟 (detect - work_set)
      // exec_us: worker process_tasks 执行耗时 (finish_ns)
      long wake_us = 0, exec_us = 0;
      if (thread_state_[j].work_set_ns > 0 && thread_state_[j].detect_ns >= thread_state_[j].work_set_ns) {
        wake_us = (long)((thread_state_[j].detect_ns - thread_state_[j].work_set_ns) / 1000);
      } else if (thread_state_[j].work_set_ns == 0) {
        wake_us = -1;  // work_set_ns 未被设置（主线程 thread 0 路径）
      } else {
        wake_us = -2;  // detect < work_set，时钟异常
      }
      exec_us = (long)(thread_state_[j].finish_ns / 1000);
      fprintf(stderr, " w[%d]{wake=%ld exec=%ld}us", j, wake_us, exec_us);
    }
    fprintf(stderr, "\n");
  }
#endif
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr);
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  do_work_stealing_job_async(task_num, init_func, compute_func, finalize_func);
  wait();
}

void InNumaPool::do_work_stealing_job_async(int task_num, std::function<void(int)> init_func,
                                            std::function<void(int)> compute_func,
                                            std::function<void(int)> finalize_func) {
  init_func_ = init_func;
  compute_func_ = compute_func;
  finalize_func_ = finalize_func;
  worker_count = std::min(restricted_worker_count, task_num);
  curr_.store(0, std::memory_order_release);
  end_ = task_num;
  for (int i = 0; i < worker_count; i++) {
    {
      std::lock_guard<std::mutex> lock(thread_state_[i].mutex);
      thread_state_[i].status.store(ThreadStatus::WORKING, std::memory_order_release);
#ifdef PROFILE_BALANCE
      thread_state_[i].work_set_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
    }
    thread_state_[i].cv.notify_one();
  }
  WorkerPool::thread_local_id = 0;
  process_tasks(0);
}

void InNumaPool::process_tasks(int thread_id) {
#ifdef PROFILE_BALANCE
  auto start = std::chrono::high_resolution_clock::now();
  thread_state_[thread_id].detect_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      start.time_since_epoch()).count();
  size_t init_ns = 0, compute_ns = 0, finalize_ns = 0;
#endif
  auto& s = thread_state_[thread_id];
  if (init_func_ != nullptr) {
#ifdef PROFILE_BALANCE
    auto init_start = std::chrono::high_resolution_clock::now();
#endif
    init_func_(thread_id);
#ifdef PROFILE_BALANCE
    init_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - init_start).count();
#endif
  }

  // omp-guided-style work scheduling
  while (true) {
    int old = curr_.load(std::memory_order_relaxed);
    int rem = end_ - old;
    if (rem <= 0) {
      break;
    }

    int block = (rem + worker_count - 1) / worker_count;
    block = 1;
    int task_id = curr_.fetch_add(block, std::memory_order_acq_rel);
    if (task_id >= end_) {
      break;
    }

    for (int i = 0; i < block; i++) {
      if (task_id + i >= end_) {
        break;
      }
#ifdef PROFILE_BALANCE
      auto comp_start = std::chrono::high_resolution_clock::now();
#endif
      compute_func_(task_id + i);
#ifdef PROFILE_BALANCE
      compute_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - comp_start).count();
#endif
    }
  }

  if (finalize_func_ != nullptr) {
#ifdef PROFILE_BALANCE
    auto fin_start = std::chrono::high_resolution_clock::now();
#endif
    finalize_func_(thread_id);
#ifdef PROFILE_BALANCE
    finalize_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - fin_start).count();
#endif
  }

  s.status.store(ThreadStatus::WAITING, std::memory_order_release);
#ifdef PROFILE_BALANCE
  s.finish_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
  s.init_ns = init_ns;
  s.compute_ns = compute_ns;
  s.finalize_ns = finalize_ns;
#endif
}

void InNumaPool::worker_thread(int thread_id, int numa_id) {
  if (numa_id >= 0) {
    set_memory_to_numa(numa_id);
  }
  auto start = std::chrono::high_resolution_clock::now();
  WorkerPool::thread_local_id = thread_id;  // 设置线程本地变量
  while (true) {
    ThreadStatus status = thread_state_[thread_id].status.load(std::memory_order_acquire);
    if (status == ThreadStatus::WORKING) {
      process_tasks(thread_id);
      start = std::chrono::high_resolution_clock::now();
    } else if (status == ThreadStatus::WAITING) {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      // 短暂 busy-wait 后进入 cv.wait：CPU 空闲时 cv.notify_one 唤醒延迟 <1ms，
      // 避免永久 busy-wait 导致几十个核空转。
      if (duration > 5) {
        std::unique_lock<std::mutex> lock(thread_state_[thread_id].mutex);
        thread_state_[thread_id].cv.wait(lock, [&] {
          return thread_state_[thread_id].status.load(std::memory_order_acquire) != ThreadStatus::WAITING;
        });
        start = std::chrono::high_resolution_clock::now();
      }
    } else if (status == ThreadStatus::EXIT) {
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
    mutexes.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
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
    mutexes.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
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
    // 跳过被占用的 core: NumaJobDistributor 主线程也要避开 djh 满载的 CPU。
    // 用全局 alloc_core_idx 确保和 InNumaPool worker 不冲突。
    const auto& skip_cpus = get_skip_cpus();
    int main_search_idx = alloc_core_idx(this_numa, start_id);
    int main_bound_cpu = -1;
    while (true) {
      core_obj = hwloc_get_obj_inside_cpuset_by_type(topology, numa_obj->cpuset, HWLOC_OBJ_CORE, main_search_idx);
      if (!core_obj) {
        break;
      }
      int cpu_id = hwloc_bitmap_first(core_obj->cpuset);
      if (cpu_id >= 0 && skip_cpus.count(cpu_id)) {
        fprintf(stderr, "[MESH] Skip occupied CPU %d (core idx %d) for numa_%d main worker\n", cpu_id, main_search_idx, this_numa);
        main_search_idx++;
        continue;
      }
      main_bound_cpu = cpu_id;
      break;
    }
    if (!core_obj) {
      fprintf(stderr, "Core %d inside NUMA node %d not found\n", 0, this_numa);
      // throw std::runtime_error("Core not found inside NUMA node");
      continue;
    }
    advance_core_idx(this_numa, main_search_idx);
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
    } else {
      fprintf(stderr, "[MESH] numa_%d_m_%d -> CPU %d (core idx %d)\n", this_numa, start_id, main_bound_cpu, main_search_idx);
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
    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::EXIT, std::memory_order_release);
    }
    cvs[i]->notify_one();
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

    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    cvs[i]->notify_one();
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
    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  for (int i = 0; i < numa_count; i++) {
    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }
}
#endif

void NumaJobDistributor::worker_thread(int numa_id) {
  auto start = std::chrono::high_resolution_clock::now();
  set_memory_to_numa(numa_id);
  status[numa_id] =
      std::move(std::unique_ptr<std::atomic<ThreadStatus>>(new std::atomic<ThreadStatus>(ThreadStatus::WAITING)));
  ready_bar->arrive_and_wait();
  while (true) {
    auto stat = status[numa_id]->load(std::memory_order_acquire);
    if (stat == ThreadStatus::WORKING) {
      auto me_numa = numa_node_of_cpu(sched_getcpu());
      // printf("numa work on %d, me %d\n", numa_id, me_numa);
      compute_func(numa_id);
      status[numa_id]->store(ThreadStatus::WAITING, std::memory_order_release);
      start = std::chrono::high_resolution_clock::now();
    } else if (stat == ThreadStatus::WAITING) {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      // 禁用 cv.wait：decode 阶段层间间隔可能 > 50ms（attention 慢），
      // worker 进入 cv.wait 后唤醒延迟 11-27ms，导致 decode 性能从 36 tok/s 暴跌到 1 tok/s。
      // 改为永久 busy-wait，确保 worker 在下次 do_work_stealing_job_async 时立即响应。
      if (duration > 3600000) {
        std::unique_lock<std::mutex> lock(*mutexes[numa_id]);
        cvs[numa_id]->wait(lock, [&] {
          return status[numa_id]->load(std::memory_order_acquire) != ThreadStatus::WAITING;
        });
      }
    } else if (stat == ThreadStatus::EXIT) {
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
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  numa_worker_pools[0]->do_work_stealing_job(task_num, init_func, compute_func, finalize_func);
}

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr);
}
