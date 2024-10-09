/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-16 10:43:18
 * @Version   : 1.0.0
 * @LastEditors : chenht
 * @LastEditTime : 2024-10-09 11:08:07
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
    CRITICAL_SECTION cs;
#else
    std::mutex mtx;
#endif

   public:
    custom_mutex() {
#ifdef _WIN32
        InitializeCriticalSection(&cs);
#else
        // No initialization required for std::mutex
#endif
    }

    ~custom_mutex() {
#ifdef _WIN32
        DeleteCriticalSection(&cs);
#endif
    }

    void lock() {
#ifdef _WIN32
        EnterCriticalSection(&cs);
#else
        mtx.lock();
#endif
    }

    void unlock() {
#ifdef _WIN32
        LeaveCriticalSection(&cs);
#else
        mtx.unlock();
#endif
    }

#ifdef _WIN32
    CRITICAL_SECTION* get_handle() {
        return &cs;
    }
#else
    std::mutex* get_handle() {
        return &mtx;
    }
#endif
};

class custom_condition_variable {
   private:
#ifdef _WIN32
    CONDITION_VARIABLE cond_var;
#else
    std::condition_variable cond_var;
#endif

   public:
    custom_condition_variable() {
#ifdef _WIN32
        InitializeConditionVariable(&cond_var);
#endif
    }

    template <typename Predicate>
    void wait(custom_mutex& mutex, Predicate pred) {
#ifdef _WIN32
        while (!pred()) {
            SleepConditionVariableCS(&cond_var, mutex.get_handle(), INFINITE);
        }
#else
        std::unique_lock<std::mutex> lock(*mutex.get_handle(), std::adopt_lock);
        cond_var.wait(lock, pred);
        lock.release();
#endif
    }

    void notify_one() {
#ifdef _WIN32
        WakeConditionVariable(&cond_var);
#else
        cond_var.notify_one();
#endif
    }

    void notify_all() {
#ifdef _WIN32
        WakeAllConditionVariable(&cond_var);
#else
        cond_var.notify_all();
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
    custom_mutex mutex;
    custom_condition_variable cv;
    std::thread worker;
    std::atomic<bool> sync_flag;
    std::atomic<bool> exit_flag;
};
#endif