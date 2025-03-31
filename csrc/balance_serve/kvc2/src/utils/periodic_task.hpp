#ifndef PERIODIC_TASK_HPP
#define PERIODIC_TASK_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace periodic {

class PeriodicTask {
 public:
  explicit PeriodicTask(std::function<void()> func,
                        std::chrono::milliseconds interval_ms = std::chrono::milliseconds(100))
      : func_(std::move(func)), interval_(interval_ms), worker_([this](std::stop_token stoken) { this->run(stoken); }) {
    // std::cout << "PeriodicTask created with interval: " << interval_.count() << " ms" << std::endl;
  }

  ~PeriodicTask() {
    worker_.request_stop();
    cv_.notify_one();  // Ensure worker wakes up when destroyed
    // std::cout << "PeriodicTask destructor called, stopping worker." << std::endl;
  }

  void wakeUp() {
    {
      std::lock_guard<std::mutex> lock(wakeup_mutex_);
      wake_up_requested_ = true;
    }
    cv_.notify_one();  // Notify worker thread to wake up immediately
    // std::cout << "wakeUp() called: worker thread will wake up." << std::endl;
  }

  std::future<void> wakeUpWait() {
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    {
      std::lock_guard<std::mutex> lock(promise_mutex_);
      wakeup_promises_.push_back(std::move(promise));
    }
    wakeUp();
    return future;
  }

 private:
  void run(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
      std::unique_lock lock(mutex_);
      // Wait for either the time interval or a wake-up signal
      cv_.wait_for(lock, interval_, [this] { return wake_up_requested_.load(); });

      if (stoken.stop_requested())
        break;

      // If the wake-up was triggered, reset the flag and process the task
      {
        std::lock_guard<std::mutex> lock(wakeup_mutex_);
        wake_up_requested_ = false;
      }

      try {
        // std::cout << "Running task function." << std::endl;
        func_();
      } catch (...) {
        std::cerr << "Error in task function." << std::endl;
      }

      notifyPromises();
    }
  }

  void notifyPromises() {
    std::lock_guard<std::mutex> lock(promise_mutex_);
    // std::cout << "Notifying all waiting promises." << std::endl;
    for (auto& promise : wakeup_promises_) {
      promise.set_value();
    }
    wakeup_promises_.clear();
  }

  std::function<void()> func_;
  std::chrono::milliseconds interval_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::promise<void>> wakeup_promises_;
  std::mutex promise_mutex_;
  std::mutex wakeup_mutex_;
  std::atomic<bool> wake_up_requested_ = false;
  std::jthread worker_;
};

}  // namespace periodic

#endif  // PERIODIC_TASK_HPP
