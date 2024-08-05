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
#ifdef _WIN32
#include <windows.h>
#endif

class custom_mutex {
private:
#ifdef _WIN32
    HANDLE  global_mutex;
#elif
    std::mutex global_mutex;
#endif
    
public:
    custom_mutex()
    {
#ifdef _WIN32
        HANDLE  global_mutex;
#endif
    }

    void lock()
    {
#ifdef _WIN32
        WaitForSingleObject(global_mutex, INFINITE);
#elif
        global_mutex.lock();
#endif
    }

    void unlock()
    {
#ifdef _WIN32
        ReleaseMutex(global_mutex);
#elif
        global_mutex.lock();
#endif
    }
};

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
    custom_mutex mutex;
    std::atomic<bool> sync_flag;
    std::atomic<bool> exit_flag;
};
#endif