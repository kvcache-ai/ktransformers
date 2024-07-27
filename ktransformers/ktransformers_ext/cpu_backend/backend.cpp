/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "backend.h"

Backend::Backend(int thread_num) {
    thread_num_ = thread_num;
    thread_state_.resize(thread_num);
    for (int i = 0; i < thread_num; i++) {
        thread_state_[i].curr = std::make_unique<std::atomic<int>>();
        thread_state_[i].status = std::make_unique<std::atomic<ThreadStatus>>(ThreadStatus::WAITING);
    }
    workers_.resize(thread_num);
    for (int i = 1; i < thread_num; i++) {
        workers_[i] = std::thread(&Backend::worker_thread, this, i);
    }
}

Backend::~Backend() {
    for (int i = 0; i < thread_num_; i++) {
        thread_state_[i].status->store(ThreadStatus::EXIT, std::memory_order_release);
    }
    for (int i = 1; i < thread_num_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }
}

int Backend::get_thread_num() {
    return thread_num_;
}

void Backend::do_work_stealing_job(int task_num, std::function<void(int)> func) {
    func_ = func;
    int base = task_num / thread_num_;
    int remain = task_num % thread_num_;
    thread_state_[0].end = base + (0 < remain);
    for (int i = 1; i < thread_num_; i++) {
        thread_state_[i].curr->store(thread_state_[i - 1].end, std::memory_order_relaxed);
        thread_state_[i].end = thread_state_[i - 1].end + base + (i < remain);
        thread_state_[i].status->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    thread_state_[0].curr->store(0, std::memory_order_relaxed);
    thread_state_[0].status->store(ThreadStatus::WORKING, std::memory_order_release);
    process_tasks(0);
    for (int i = 1; i < thread_num_; i++) {
        while (thread_state_[i].status->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
        }
    }
}

void Backend::process_tasks(int thread_id) {
    while (true) {
        int task_id = thread_state_[thread_id].curr->fetch_add(1, std::memory_order_acq_rel);
        if (task_id >= thread_state_[thread_id].end) {
            break;
        }
        func_(task_id);
    }
    for (int t_offset = 1; t_offset < thread_num_; t_offset++) {
        int t_i = (thread_id + t_offset) % thread_num_;
        if (thread_state_[t_i].status->load(std::memory_order_acquire) != ThreadStatus::WORKING) {
            continue;
        }
        while (true) {
            int task_id = thread_state_[t_i].curr->fetch_add(1, std::memory_order_acq_rel);
            if (task_id >= thread_state_[t_i].end) {
                break;
            }
            func_(task_id);
        }
    }
    thread_state_[thread_id].status->store(ThreadStatus::WAITING, std::memory_order_release);
}

void Backend::worker_thread(int thread_id) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
        ThreadStatus status = thread_state_[thread_id].status->load(std::memory_order_acquire);
        if (status == ThreadStatus::WORKING) {
            process_tasks(thread_id);
            start = std::chrono::steady_clock::now();
        } else if (status == ThreadStatus::WAITING) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (duration > 50) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
    }
}