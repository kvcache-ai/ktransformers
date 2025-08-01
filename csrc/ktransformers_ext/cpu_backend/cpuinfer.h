/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#ifdef KTRANSFORMERS_USE_CUDA
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#endif

#include "./vendors/vendor.h"
#include "task_queue.h"
#include "worker_pool.h"

#include "llama.cpp/ggml-impl.h"

class CPUInfer {
public:
  CPUInfer(int thread_num) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    delete backend_;
    delete task_queue_;
  }

  CPUInfer(const CPUInfer &) = delete;
  CPUInfer &operator=(const CPUInfer &) = delete;
  CPUInfer(CPUInfer &&) = delete;
  CPUInfer &operator=(CPUInfer &&) = delete;

  template <typename Func, typename Obj, typename... Args> void enqueue(Func f, Obj *obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void *) = (void (*)(void *))params.first;
    void *args = (void *)params.second;
    *((CPUInfer **)args) = this;
    func(args);
  }

  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
    void (*func)(void *) = (void (*)(void *))params.first;
    void *args = (void *)params.second;
    *((CPUInfer **)args) = this;
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
  }

  struct SyncArgs {
    CPUInfer *cpuinfer;
    size_t n;
  };

  static void sync_(void *sync_args) {
    SyncArgs *args = (SyncArgs *)sync_args;
    args->cpuinfer->task_queue_->sync(args->n);
  }

  void sync(size_t n = 0) {
    SyncArgs *args = new SyncArgs{this, n};
    sync_(args);
  }

  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t n = 0) {
    SyncArgs *args = new SyncArgs{this, n};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void *)args);
  }

public:
  WorkerPool *backend_;
  TaskQueue *task_queue_;
};

#endif