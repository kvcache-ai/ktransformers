/**
 * @Description  :
 * @Author       : Xie Weiyu
 * @Date         : 2024-12-11 06:35:31
 * @Version      : 1.0.0
 * @LastEditors  : Xie Weiyu
 * @LastEditTime : 2024-12-11 06:50:55
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#pragma once
#include <atomic>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct BatchPromise {
  std::promise<void> promise;
  std::shared_future<void> fut;
  std::atomic_size_t count;

  inline BatchPromise(size_t count) : count(count) { fut = promise.get_future().share(); }

  inline void inc(size_t count = 1) { this->count.fetch_add(count, std::memory_order_seq_cst); }

  inline void set() {
    if (count.fetch_sub(1, std::memory_order_seq_cst) == 1) {
      promise.set_value();
    }
  }
  inline std::shared_future<void> get_shared_fut() { return fut; }
};

template <typename Lock>
struct TransferControl {
  Lock lock;

  std::optional<std::shared_future<void>> transfer_ok = std::nullopt;
  bool has_data = false;

  TransferControl() {}

  /*
   true, std::nullopt : Already has data
   false, shared_future : Transfer already started, should wait for the future
   false, std::nullopt : should transfer by you
   true, shared_future: Should not appear
  */
  std::pair<bool, std::optional<std::shared_future<void>>> has_data_or_transfer(std::shared_future<void> shared_fut) {
    std::lock_guard<Lock> lg(lock);
    if (has_data) {
      return {true, std::nullopt};
    } else {
      if (transfer_ok.has_value()) {
        return {false, transfer_ok};
      } else {
        transfer_ok = shared_fut;
        return {false, std::nullopt};
      }
    }
  }

  void set_has_data() {
    std::lock_guard<Lock> lg(lock);
    has_data = true;
    transfer_ok = std::nullopt;
  }

  bool get_has_data() {
    std::lock_guard<Lock> lg(lock);
    if (has_data) {
      return true;
    } else {
      return false;
    }
  }

  void reset() {
    std::lock_guard<Lock> lg(lock);
    transfer_ok = std::nullopt;
    has_data = false;
  }

  std::string debug() {
    std::lock_guard<Lock> lg(lock);
    return std::string("") + (has_data ? "has data" : "no data") + " " +
           (transfer_ok.has_value() ? "transfer " : "no transfer");
  }
};

struct ConcurrentController {
  std::atomic_bool dirty = false;
  std::atomic_size_t ref_count = 0;
  TransferControl<std::mutex> tc;
};

template <typename Unit>
struct IO_Helper {
  BatchPromise batch_promise;
  std::function<void(Unit*)> call_back_on_unit = nullptr;
  std::function<void()> call_back = nullptr;

  std::vector<std::shared_future<void>> futs;
  std::vector<Unit*> units_by_myself;

  IO_Helper(std::function<void(Unit*)> call_back_on_unit, std::function<void()> call_back = nullptr)
      : batch_promise(1), call_back_on_unit(call_back_on_unit), call_back(call_back) {}

  IO_Helper(const IO_Helper& other) = delete;
  IO_Helper& operator=(const IO_Helper& other) = delete;
  IO_Helper(IO_Helper&& other) = delete;
  IO_Helper& operator=(IO_Helper&& other) = delete;
  ~IO_Helper() {
    // std::cout<<"Destory IO helper"<<std::endl;
  }

  size_t total_task_count = 0;
  void new_task(size_t count = 1) {
    total_task_count += 1;
    batch_promise.inc(count);
  }
  void finish_add_taks() { batch_promise.set(); }

  bool absorb_tc(Unit* unit, TransferControl<std::mutex>& tc) {
    auto [ok, fut] = tc.has_data_or_transfer(batch_promise.get_shared_fut());
    if (ok) {
      return false;
    } else {
      if (fut.has_value()) {
        futs.push_back(fut.value());
        // printf("Transfer started\n");
        return false;
      } else {
        units_by_myself.push_back(unit);
        // printf("Not Transfer\n");
        return true;
      }
    }
  }

  void wait() {
    for (auto& fut : futs) {
      fut.wait();
    }
    batch_promise.get_shared_fut().wait();
    for (auto& b : units_by_myself) {
      call_back_on_unit(b);
    }
    if (call_back)
      call_back();
  }
};
