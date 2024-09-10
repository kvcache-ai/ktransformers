/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-17 12:25:51
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:44
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"

TaskQueue::TaskQueue() {
    worker = std::thread(&TaskQueue::processTasks, this);
    sync_flag.store(true, std::memory_order_seq_cst);
    exit_flag.store(false, std::memory_order_seq_cst);
}

TaskQueue::~TaskQueue() {
    {
        std::unique_lock<std::mutex> lock(mutex);
        exit_flag.store(true, std::memory_order_seq_cst);
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void TaskQueue::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(mutex);
        tasks.push(task);
        sync_flag.store(false, std::memory_order_seq_cst);
    }
    cv.notify_one();
}

void TaskQueue::sync() {
    while (!sync_flag.load(std::memory_order_seq_cst))
        ;
}

void TaskQueue::processTasks() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [this]() { return !tasks.empty() || exit_flag.load(std::memory_order_seq_cst); });
            if (exit_flag.load(std::memory_order_seq_cst) && tasks.empty()) {
                return;
            }
            task = tasks.front();
            tasks.pop();
        }
        task();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (tasks.empty()) {
                sync_flag.store(true, std::memory_order_seq_cst);
            }
        }
    }
}
