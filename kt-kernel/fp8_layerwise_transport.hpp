#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace kt::layerwise {

inline constexpr std::size_t kFP8LayerwiseControlBytes = 8192;
inline constexpr int kFP8LayerwiseMaxTPSize = 8;
inline constexpr int kFP8LayerwiseBufferKinds = 4;
inline constexpr int kFP8LayerwiseHostSlots = 2;

struct FP8LayerwiseStats {
  std::uint64_t epoch = 0;
  std::int64_t layer_id = -1;
  int expert_count = 0;
  int rank = -1;
  double writer_ms = 0.0;
  double slot_wait_ms = 0.0;
  double h2d_ms = 0.0;
  double total_ms = 0.0;
  std::uint64_t bytes = 0;
  bool poisoned = false;
  int error_code = 0;
  int error_rank = -1;
  std::string error_message;
};

// Initialize a page-aligned shared-memory region before constructing any
// transport. The region is process-shared POD; it deliberately contains no
// std::atomic objects. Synchronization is implemented with GCC/Clang
// __atomic builtins in the implementation.
void initialize_fp8_layerwise_control(std::uintptr_t control_ptr, std::size_t control_size, int tp_size);

class FP8LayerwiseTransport final {
 public:
  using Writer = std::function<void(int expert_id, const std::vector<std::uintptr_t>& w13_weight_ptrs,
                                    const std::vector<std::uintptr_t>& w13_scale_ptrs,
                                    const std::vector<std::uintptr_t>& w2_weight_ptrs,
                                    const std::vector<std::uintptr_t>& w2_scale_ptrs)>;

  FP8LayerwiseTransport(std::uintptr_t control_ptr, std::size_t control_size, int rank, int tp_size, int cuda_device,
                        const std::vector<std::uintptr_t>& local_host_ptrs,
                        const std::vector<std::uintptr_t>& local_gpu_ptrs,
                        const std::vector<std::uintptr_t>& all_rank_host_ptrs,
                        const std::vector<std::size_t>& expert_nbytes, int num_experts,
                        std::uint64_t timeout_ms = 60000);
  ~FP8LayerwiseTransport();

  FP8LayerwiseTransport(const FP8LayerwiseTransport&) = delete;
  FP8LayerwiseTransport& operator=(const FP8LayerwiseTransport&) = delete;

  void join(std::uint64_t epoch, std::int64_t layer_id, int expert_count);
  FP8LayerwiseStats wait(std::uint64_t epoch);
  void close();

  // Rank zero executes the producer. The hot expert loop stays native: write
  // one expert into a host slot, publish its generation, and wait for all
  // local-rank DMA consumers only when that same slot is about to be reused.
  void run_producer(std::uint64_t epoch, std::int64_t layer_id, int expert_count, const Writer& writer);

  int rank() const noexcept;
  int tp_size() const noexcept;
  int num_experts() const noexcept;
  bool closed() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kt::layerwise
