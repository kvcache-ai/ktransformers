// SPDX-License-Identifier: Apache-2.0
#ifndef CPUINFER_OPERATOR_SFT_REPACK_HPP
#define CPUINFER_OPERATOR_SFT_REPACK_HPP

#include <atomic>
#include <cstdint>
#include <future>
#include <mutex>
#include <utility>

namespace sft {

// SFT scratch pools are process-wide, and WorkerPool does not support concurrent
// job submission. Take this lease on the submitting thread, BEFORE dispatching
// NUMA work, and keep it until all CPU consumers finish. NUMA workers must never
// take it themselves. GPU work does not take the lease and can overlap repacking.
[[nodiscard]] inline std::unique_lock<std::mutex> acquire_cpu_execution() {
  static std::mutex mutex;
  return std::unique_lock<std::mutex>(mutex);
}

// One outstanding preparation task per layer. The future owns completion and
// exception propagation; there is no separate in-flight flag to keep in sync.
// submit()/wait() are called by the layer's serial host-side dispatcher.
class RepackTask {
 public:
  RepackTask() = default;
  RepackTask(const RepackTask&) = delete;
  RepackTask& operator=(const RepackTask&) = delete;

  template <class Prepare>
  void submit(Prepare&& prepare) {
    wait();
    task_ = std::async(std::launch::async, [prepare = std::forward<Prepare>(prepare)]() mutable {
      auto execution = acquire_cpu_execution();
      prepare();
    });
  }

  void wait() {
    if (task_.valid()) task_.get();
  }

  bool pending() const noexcept { return task_.valid(); }

  // A containing operator must drain BEFORE destroying data captured by the
  // task. Normal wait() reports errors; destruction only joins and cannot throw.
  void drain() noexcept {
    if (task_.valid()) task_.wait();
  }

  ~RepackTask() { drain(); }

 private:
  std::future<void> task_;
};

// Readiness of a shared backward-weight buffer, independent of its dtype or
// physical layout. Access is serialized by the submitting CPU execution lease;
// distinct NUMA partitions have distinct states and prepare in parallel.
class RepackState {
 public:
  using Version = std::uint64_t;

  // A version identifies an operator's weights, not a layer number: two models
  // (or two dtypes) may use the same layer number. Reloading gets a new version.
  static Version new_version() {
    static std::atomic<Version> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
  }

  template <class Prepare>
  void ensure_ready(Version version, Prepare&& prepare) {
    if (ready_version_ == version) return;
    // A failed/partial repack must not leave the old owner looking ready.
    ready_version_ = 0;
    std::forward<Prepare>(prepare)();
    ready_version_ = version;
  }

 private:
  Version ready_version_ = 0;
};

}  // namespace sft

#endif
