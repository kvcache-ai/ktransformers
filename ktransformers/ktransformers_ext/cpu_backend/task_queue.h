/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:33:47
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_TASKQUEUE_H
#define CPUINFER_TASKQUEUE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class TaskQueue {
   public:
    TaskQueue();
    ~TaskQueue();

    void enqueue(std::function<void()>);

    void sync();

   private:
    void processTasks();

    std::queue<std::function<void()>> tasks;
    std::thread worker;
    std::mutex mutex;
    std::atomic<bool> sync_flag;
    std::atomic<bool> exit_flag;
};
#endif