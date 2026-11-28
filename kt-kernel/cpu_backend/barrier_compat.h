// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#if __has_include(<barrier>)
#include <barrier>
#endif

#if !defined(KT_ARM_SFT) || (defined(__cpp_lib_barrier) && __cpp_lib_barrier >= 201907L)
namespace kt {
using Barrier = std::barrier<>;
}
#else
#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace kt {
// The worker pool needs only the reusable arrive-and-wait operation.
class Barrier {
 public:
  explicit Barrier(std::ptrdiff_t participants) : participants_(participants), remaining_(participants) {
    if (participants <= 0) throw std::invalid_argument("Barrier requires participants");
  }
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto generation = generation_;
    if (--remaining_ == 0) {
      remaining_ = participants_;
      ++generation_;
      condition_.notify_all();
    } else {
      condition_.wait(lock, [&] { return generation_ != generation; });
    }
  }

 private:
  const std::ptrdiff_t participants_;
  std::ptrdiff_t remaining_;
  size_t generation_ = 0;
  std::mutex mutex_;
  std::condition_variable condition_;
};
}  // namespace kt
#endif
