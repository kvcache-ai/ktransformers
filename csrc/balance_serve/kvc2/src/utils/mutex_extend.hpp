#ifndef __MUTEX_EXTEND_HPP_
#define __MUTEX_EXTEND_HPP_

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

class non_recursive_mutex {
 public:
  non_recursive_mutex() = default;

  // 使用 try_lock 实现非递归锁
  bool try_lock() {
    std::thread::id this_id = std::this_thread::get_id();

    // 检查当前线程是否已经持有该锁
    if (owner.load(std::memory_order_acquire) == this_id) {
      return false;  // 如果是当前线程，返回失败
    }

    // 尝试加锁
    if (mtx.try_lock()) {
      owner.store(this_id, std::memory_order_release);  // 设置锁的拥有者
      return true;
    }

    return false;
  }

  // lock 会阻塞，直到获得锁
  void lock() {
    std::thread::id this_id = std::this_thread::get_id();

    while (true) {
      // 检查当前线程是否已经持有该锁
      if (owner.load(std::memory_order_acquire) == this_id) {
        throw std::runtime_error("Thread is trying to lock a mutex it already holds");
      }

      // 尝试加锁
      if (mtx.try_lock()) {
        owner.store(this_id, std::memory_order_release);  // 设置锁的拥有者
        return;
      }

      // 如果锁未获得，则稍微等待，防止忙等
      std::this_thread::yield();
    }
  }

  // 解锁
  void unlock() {
    std::thread::id this_id = std::this_thread::get_id();

    // 确保只有持有锁的线程可以解锁
    if (owner.load(std::memory_order_acquire) == this_id) {
      owner.store(std::thread::id(), std::memory_order_release);  // 清除锁的拥有者
      mtx.unlock();
    } else {
      throw std::runtime_error("Thread attempting to unlock a mutex it doesn't own");
    }
  }

 private:
  std::mutex mtx;                      // 实际的互斥量
  std::atomic<std::thread::id> owner;  // 原子变量，记录当前锁的拥有者
};

#endif
