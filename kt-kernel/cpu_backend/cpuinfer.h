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
#elif KTRANSFORMERS_USE_MACA
#include "vendors/maca.h"
#endif

#include "./vendors/vendor.h"
#include "llama.cpp/ggml-impl.h"
#include "task_queue.h"
#include "worker_pool.h"

// B8: MESH Heat 批量传输回调声明（定义在 ext_bindings.cpp 中）
// 由 CPUInfer::submit_gating_scores_ 调用，避免 cpuinfer.h 直接依赖 mesh 头文件
#if defined(KT_ENABLE_MESH)
extern "C" void mesh_on_gating_scores_ready(void* mesh_residency,
                                              const float* gating_scores_cpu,
                                              int num_layers, int expert_num, int topk);
#endif

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

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || \
    defined(KTRANSFORMERS_USE_MACA)
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
#endif
  }
#endif

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
  };

  static void sync_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    args->cpuinfer->task_queue_->sync(args->allow_n_pending);
  }

  void sync(size_t allow_n_pending = 0) {
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    sync_(args);
  }

  // ===== MESH 插件：gating score 批量传输回调 =====
  // MESH 需要每 token 结束后将所有层的 gating 分数从 GPU 一次性传到 CPU。
  // 利用现有 submit_with_cuda_stream 机制，通过 cudaLaunchHostFunc 调度到 CUDA stream。
  // 原版 KT 的 moe gate 只传 top-8，MESH 改成传全长 vector（专家数长度）。
  struct GatingScoreArgs {
    CPUInfer* cpuinfer;
    void* mesh_residency;                  // mesh::MeshResidencyManager*
    const float* gating_scores_gpu;        // GPU 指针，所有层的 gating 分数
    int num_layers;
    int expert_num;
    int topk;                              // B8: top-k 参数（每层激活的专家数）
  };

  static void submit_gating_scores_(void* args_ptr) {
    GatingScoreArgs* args = (GatingScoreArgs*)args_ptr;
#if defined(KT_ENABLE_MESH)
    // B8: 从 GPU 拷贝 gating scores 到 CPU，然后调用 mesh 回调
    int total_floats = args->num_layers * args->expert_num;
    std::vector<float> gating_scores_cpu(total_floats);
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || \
    defined(KTRANSFORMERS_USE_MACA)
    cudaMemcpy(gating_scores_cpu.data(), args->gating_scores_gpu,
               total_floats * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    // 调用 mesh 回调，内部会组织数据并调用 on_decode_token_end
    mesh_on_gating_scores_ready(args->mesh_residency, gating_scores_cpu.data(),
                                 args->num_layers, args->expert_num, args->topk);
#endif
    delete args;
  }

#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_gating_scores_with_cuda_stream(
      intptr_t user_cuda_stream,
      void* mesh_residency,
      const float* gating_scores_gpu,
      int num_layers, int expert_num, int topk) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || \
    defined(KTRANSFORMERS_USE_MACA)
    GatingScoreArgs* args = new GatingScoreArgs{
      this, mesh_residency, gating_scores_gpu, num_layers, expert_num, topk
    };
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream,
                       (cudaHostFn_t)&submit_gating_scores_, (void*)args);
#endif
  }
#endif
#ifndef KTRANSFORMERS_CPU_ONLY
  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || \
    defined(KTRANSFORMERS_USE_MACA)
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
#endif
  }
#endif
 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;
};

#endif
