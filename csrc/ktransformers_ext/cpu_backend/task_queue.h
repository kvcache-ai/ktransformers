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

class TaskQueue {
public:
  TaskQueue();
  ~TaskQueue();

  void enqueue(std::function<void()>);

  void sync(size_t n);

private:
  struct Node {
    std::function<void()> task;
    std::atomic<Node *> next;
    Node() : task(nullptr), next(nullptr) {}
    Node(const std::function<void()> &t) : task(t), next(nullptr) {}
  };

  std::atomic<Node *> head;
  std::atomic<Node *> tail;
  std::atomic<bool> done;
  std::atomic<size_t> pending;
  std::thread workerThread;

  void worker();
};

#endif