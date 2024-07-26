/**
 * @Description  :  
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:33:42
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

#include "backend.h"
#include "task_queue.h"

#include "llama.cpp/ggml-impl.h"

class CPUInfer {
   public:
    CPUInfer(int thread_num) {
        backend_ = new Backend(thread_num - 1);
        task_queue_ = new TaskQueue();
        for (int i = 0; i < (1 << 16); ++i) {
            ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
        }
    }

    ~CPUInfer() {
        delete backend_;
        delete task_queue_;
    }

    template <typename Func, typename Obj, typename... Args>
    void submit(Func f, Obj* obj, Args... args) {
        task_queue_->enqueue([=]() {
            std::invoke(f, *obj, args..., backend_);
        });
    }

    void sync() {
        task_queue_->sync();
    }

   public:
    Backend* backend_;
    TaskQueue* task_queue_;
};

#endif