/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>  // 添加这个头文件

#include <cerrno>  // 添加这个头文件
#include <chrono>
#include <cstring>  // 添加这个头文件
#include <iostream>
#include <thread>
// 添加设置CPU亲和性的私有方法
bool TaskQueue::set_cpu_affinity(int cpu_core_id) {
  if (!workerThread.joinable()) {
    std::cerr << "Worker thread not running, cannot set affinity" << std::endl;
    return false;
  }

  // 检查CPU核心是否有效
  int num_cores = sysconf(_SC_NPROCESSORS_CONF);
  if (cpu_core_id < 0 || cpu_core_id >= num_cores) {
    std::cerr << "Invalid CPU core ID: " << cpu_core_id << " (system has " << num_cores << " cores)" << std::endl;
    return false;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_core_id, &cpuset);

  int result = pthread_setaffinity_np(workerThread.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (result != 0) {
    std::cerr << "Failed to set thread affinity for core " << cpu_core_id << ": " << strerror(errno) << std::endl;
    return false;
  }
  return true;
}

TaskQueue::TaskQueue(int cpu_core_id) : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  workerThread = std::thread(&TaskQueue::worker, this);

  // 设置CPU亲和性
  if (!set_cpu_affinity(cpu_core_id)) {
    std::cerr << "Warning: Failed to bind worker thread to core " << cpu_core_id << std::endl;
  }
}

TaskQueue::TaskQueue() : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  workerThread = std::thread(&TaskQueue::worker, this);
}

TaskQueue::~TaskQueue() {
  done.store(true, std::memory_order_release);
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
}

void TaskQueue::sync(size_t allow_n_pending) {
  // Spin until the pending task count drops to the allowed threshold.
  while (pending.load(std::memory_order_acquire) > allow_n_pending);
}

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      if (next->task) {
        next->task();
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      pending.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
}