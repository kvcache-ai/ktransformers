#include "fp8_layerwise_transport.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#if defined(KTRANSFORMERS_USE_CUDA) || defined(USE_CUDA)
#include <cuda_runtime_api.h>
#define KT_FP8_LAYERWISE_HAS_CUDA 1
#else
#define KT_FP8_LAYERWISE_HAS_CUDA 0
#endif

namespace kt::layerwise {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kControlMagic = 0x4b544650384c5731ULL;  // "KTFP8LW1"
constexpr std::uint64_t kControlVersion = 2;
constexpr std::uint64_t kPoisonClaimed = std::numeric_limits<std::uint64_t>::max();

enum ErrorCode : int {
  kErrorNone = 0,
  kErrorProtocol = 1,
  kErrorTimeout = 2,
  kErrorCudaInitialization = 3,
  kErrorCudaCopy = 4,
  kErrorWriter = 5,
  kErrorClosedWhileActive = 6,
};

struct alignas(64) AtomicCell {
  std::uint64_t value;
  std::uint8_t padding[56];
};
static_assert(sizeof(AtomicCell) == 64);

struct alignas(64) ControlHeader {
  std::uint64_t magic;
  std::uint64_t version;
  std::uint64_t struct_bytes;
  std::uint64_t tp_size;
  std::uint8_t padding[32];
};
static_assert(sizeof(ControlHeader) == 64);

struct alignas(64) ReadySlot {
  AtomicCell sequence;
  AtomicCell epoch;
  AtomicCell expert_id;
};

// This type is intentionally POD. It lives in mmap'ed process-shared memory;
// all synchronization goes through the __atomic helpers below.
struct alignas(64) FP8LayerwiseControl {
  ControlHeader header;
  AtomicCell poison_state;  // 0 healthy, UINT64_MAX claimed/writing, 1 poisoned
  AtomicCell error_code;
  AtomicCell error_rank;
  alignas(64) char error_message[256];

  AtomicCell active_epoch;
  AtomicCell layer_id;
  AtomicCell expert_count;
  AtomicCell next_sequence;
  AtomicCell producer_done_epoch;

  AtomicCell join_epoch[kFP8LayerwiseMaxTPSize];
  AtomicCell consumer_done_epoch[kFP8LayerwiseMaxTPSize];
  AtomicCell wait_returned_epoch[kFP8LayerwiseMaxTPSize];

  ReadySlot ready[kFP8LayerwiseHostSlots];
  AtomicCell ack_sequence[kFP8LayerwiseMaxTPSize][kFP8LayerwiseHostSlots];

  AtomicCell layer_start_ns;
  AtomicCell writer_ns;
  AtomicCell slot_wait_ns;
  AtomicCell producer_total_ns;
  AtomicCell h2d_ns[kFP8LayerwiseMaxTPSize];
  AtomicCell bytes[kFP8LayerwiseMaxTPSize];
};

static_assert(std::is_trivial_v<FP8LayerwiseControl>);
static_assert(std::is_standard_layout_v<FP8LayerwiseControl>);
static_assert(sizeof(FP8LayerwiseControl) <= kFP8LayerwiseControlBytes,
              "FP8 layerwise control must fit in the shared control region");

std::uint64_t load_acquire(const AtomicCell& cell) {
  return __atomic_load_n(&cell.value, __ATOMIC_ACQUIRE);
}

std::uint64_t load_relaxed(const AtomicCell& cell) {
  return __atomic_load_n(&cell.value, __ATOMIC_RELAXED);
}

void store_release(AtomicCell& cell, std::uint64_t value) {
  __atomic_store_n(&cell.value, value, __ATOMIC_RELEASE);
}

void store_relaxed(AtomicCell& cell, std::uint64_t value) {
  __atomic_store_n(&cell.value, value, __ATOMIC_RELAXED);
}

std::uint64_t fetch_add_relaxed(AtomicCell& cell, std::uint64_t value) {
  return __atomic_fetch_add(&cell.value, value, __ATOMIC_RELAXED);
}

bool compare_exchange(AtomicCell& cell, std::uint64_t& expected, std::uint64_t desired) {
  return __atomic_compare_exchange_n(&cell.value, &expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

double ns_to_ms(std::uint64_t ns) { return static_cast<double>(ns) / 1.0e6; }

std::uint64_t signed_to_bits(std::int64_t value) { return static_cast<std::uint64_t>(value); }

std::int64_t bits_to_signed(std::uint64_t value) { return static_cast<std::int64_t>(value); }

void validate_control_region(std::uintptr_t control_ptr, std::size_t control_size) {
  if (control_ptr == 0) throw std::invalid_argument("FP8 layerwise control pointer is null");
  if ((control_ptr & 63U) != 0) throw std::invalid_argument("FP8 layerwise control pointer must be 64-byte aligned");
  if (control_size < kFP8LayerwiseControlBytes) {
    throw std::invalid_argument("FP8 layerwise control region must be at least 8192 bytes");
  }
}

FP8LayerwiseControl* checked_control(std::uintptr_t control_ptr, std::size_t control_size, int tp_size) {
  validate_control_region(control_ptr, control_size);
  auto* control = reinterpret_cast<FP8LayerwiseControl*>(control_ptr);
  const auto magic = __atomic_load_n(&control->header.magic, __ATOMIC_ACQUIRE);
  if (magic != kControlMagic || control->header.version != kControlVersion ||
      control->header.struct_bytes != sizeof(FP8LayerwiseControl)) {
    throw std::runtime_error("FP8 layerwise control is uninitialized or has an incompatible ABI");
  }
  if (control->header.tp_size != static_cast<std::uint64_t>(tp_size)) {
    throw std::invalid_argument("FP8 layerwise control TP size does not match transport TP size");
  }
  return control;
}

std::string poison_message(FP8LayerwiseControl* control) {
  // The process which wins the poison claim fills diagnostics before changing
  // kPoisonClaimed to 1. Wait briefly so readers normally avoid a partial
  // string, but never let a process dying mid-publication hang every rank.
  const auto claim_deadline = Clock::now() + std::chrono::milliseconds(10);
  while (load_acquire(control->poison_state) == kPoisonClaimed && Clock::now() < claim_deadline) {
    std::this_thread::yield();
  }
  if (load_acquire(control->poison_state) == kPoisonClaimed) {
    return "FP8 layerwise transport poisoned, but poison publication did not complete (claim owner may have died)";
  }
  std::ostringstream out;
  out << "FP8 layerwise transport poisoned (code=" << load_relaxed(control->error_code)
      << ", rank=" << bits_to_signed(load_relaxed(control->error_rank)) << ")";
  if (control->error_message[0] != '\0') out << ": " << control->error_message;
  return out.str();
}

void publish_poison(FP8LayerwiseControl* control, int code, int rank, const std::string& message) noexcept {
  std::uint64_t expected = 0;
  if (!compare_exchange(control->poison_state, expected, kPoisonClaimed)) return;
  store_relaxed(control->error_code, static_cast<std::uint64_t>(code));
  store_relaxed(control->error_rank, signed_to_bits(rank));
  std::snprintf(control->error_message, sizeof(control->error_message), "%s", message.c_str());
  __atomic_thread_fence(__ATOMIC_RELEASE);
  store_release(control->poison_state, 1);
}

void throw_if_poisoned(FP8LayerwiseControl* control) {
  if (load_acquire(control->poison_state) != 0) throw std::runtime_error(poison_message(control));
}

std::uint64_t fold_progress(std::uint64_t seed, std::uint64_t value) {
  // A cheap mixer used only to notice progress and refresh a timeout deadline.
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

}  // namespace

void initialize_fp8_layerwise_control(std::uintptr_t control_ptr, std::size_t control_size, int tp_size) {
  validate_control_region(control_ptr, control_size);
  if (tp_size <= 0 || tp_size > kFP8LayerwiseMaxTPSize) {
    throw std::invalid_argument("FP8 layerwise transport supports TP sizes 1 through 8");
  }
  auto* control = reinterpret_cast<FP8LayerwiseControl*>(control_ptr);
  std::memset(control, 0, kFP8LayerwiseControlBytes);
  control->header.version = kControlVersion;
  control->header.struct_bytes = sizeof(FP8LayerwiseControl);
  control->header.tp_size = static_cast<std::uint64_t>(tp_size);
  store_relaxed(control->error_rank, signed_to_bits(-1));
  __atomic_store_n(&control->header.magic, kControlMagic, __ATOMIC_RELEASE);
}

class FP8LayerwiseTransport::Impl {
 public:
  Impl(std::uintptr_t control_ptr, std::size_t control_size, int rank, int tp_size, int cuda_device,
       const std::vector<std::uintptr_t>& local_host_ptrs, const std::vector<std::uintptr_t>& local_gpu_ptrs,
       const std::vector<std::uintptr_t>& all_rank_host_ptrs, const std::vector<std::size_t>& expert_nbytes,
       int num_experts, std::uint64_t timeout_ms)
      : control_(checked_control(control_ptr, control_size, tp_size)),
        rank_(rank),
        tp_size_(tp_size),
        cuda_device_(cuda_device),
        num_experts_(num_experts),
        timeout_(timeout_ms) {
    if (rank < 0 || rank >= tp_size) throw std::invalid_argument("FP8 layerwise rank is outside TP range");
    if (num_experts <= 0) throw std::invalid_argument("FP8 layerwise num_experts must be positive");
    if (timeout_ms == 0) throw std::invalid_argument("FP8 layerwise timeout_ms must be positive");
    if (local_host_ptrs.size() != kFP8LayerwiseHostSlots * kFP8LayerwiseBufferKinds) {
      throw std::invalid_argument("FP8 layerwise local_host_ptrs must contain 8 pointers in [slot][kind] order");
    }
    if (local_gpu_ptrs.size() != kFP8LayerwiseBufferKinds) {
      throw std::invalid_argument("FP8 layerwise local_gpu_ptrs must contain 4 pointers");
    }
    if (expert_nbytes.size() != kFP8LayerwiseBufferKinds) {
      throw std::invalid_argument("FP8 layerwise expert_nbytes must contain 4 sizes");
    }
    const std::size_t expected_all_rank = kFP8LayerwiseHostSlots * tp_size * kFP8LayerwiseBufferKinds;
    if ((rank == 0 && all_rank_host_ptrs.size() != expected_all_rank) || (rank != 0 && !all_rank_host_ptrs.empty())) {
      throw std::invalid_argument(
          "FP8 layerwise all_rank_host_ptrs must be [slot][rank][kind] on rank zero and empty on other ranks");
    }
    if (std::any_of(local_host_ptrs.begin(), local_host_ptrs.end(), [](auto ptr) { return ptr == 0; }) ||
        std::any_of(local_gpu_ptrs.begin(), local_gpu_ptrs.end(), [](auto ptr) { return ptr == 0; }) ||
        std::any_of(expert_nbytes.begin(), expert_nbytes.end(), [](auto size) { return size == 0; }) ||
        (rank == 0 &&
         std::any_of(all_rank_host_ptrs.begin(), all_rank_host_ptrs.end(), [](auto ptr) { return ptr == 0; }))) {
      throw std::invalid_argument("FP8 layerwise buffer pointers and expert byte sizes must be non-zero");
    }

    std::copy(local_host_ptrs.begin(), local_host_ptrs.end(), local_host_ptrs_.begin());
    std::copy(local_gpu_ptrs.begin(), local_gpu_ptrs.end(), local_gpu_ptrs_.begin());
    std::copy(expert_nbytes.begin(), expert_nbytes.end(), expert_nbytes_.begin());
    all_rank_host_ptrs_ = all_rank_host_ptrs;
    for (int slot = 0; slot < kFP8LayerwiseHostSlots; ++slot) {
      last_ready_sequence_[slot] = load_acquire(control_->ready[slot].sequence);
    }

#if KT_FP8_LAYERWISE_HAS_CUDA
    worker_ = std::thread(&Impl::consumer_main, this);
    std::unique_lock lock(init_mutex_);
    init_cv_.wait(lock, [this] { return init_complete_; });
    if (!init_error_.empty()) {
      lock.unlock();
      stopping_.store(true, std::memory_order_release);
      if (worker_.joinable()) worker_.join();
      throw std::runtime_error(init_error_);
    }
#else
    throw std::runtime_error("FP8 layerwise transport requires a CUDA-enabled kt-kernel build");
#endif
  }

  ~Impl() { close(); }

  void join(std::uint64_t epoch, std::int64_t layer_id, int expert_count) {
    ensure_open();
    throw_if_poisoned(control_);
    validate_call(epoch, layer_id, expert_count);

    if (rank_ == 0) {
      const std::uint64_t previous_epoch = load_acquire(control_->active_epoch);
      if (epoch <= previous_epoch) {
        poison_and_throw(kErrorProtocol, "rank zero attempted to reuse or decrease a layerwise epoch");
      }
      if (previous_epoch != 0) {
        wait_until(
            "all ranks returning from the previous layer",
            [&] {
              for (int rank = 0; rank < tp_size_; ++rank) {
                if (load_acquire(control_->wait_returned_epoch[rank]) < previous_epoch) return false;
              }
              return true;
            },
            [&] {
              std::uint64_t progress = 0;
              for (int rank = 0; rank < tp_size_; ++rank) {
                progress = fold_progress(progress, load_relaxed(control_->wait_returned_epoch[rank]));
              }
              return progress;
            });
      }

      store_relaxed(control_->layer_id, signed_to_bits(layer_id));
      store_relaxed(control_->expert_count, static_cast<std::uint64_t>(expert_count));
      store_relaxed(control_->layer_start_ns, now_ns());
      store_relaxed(control_->writer_ns, 0);
      store_relaxed(control_->slot_wait_ns, 0);
      store_relaxed(control_->producer_total_ns, 0);
      for (int rank = 0; rank < tp_size_; ++rank) {
        store_relaxed(control_->h2d_ns[rank], 0);
        store_relaxed(control_->bytes[rank], 0);
      }
      __atomic_thread_fence(__ATOMIC_RELEASE);
      store_release(control_->active_epoch, epoch);
    } else {
      wait_until(
          "rank zero publishing the requested epoch",
          [&] {
            const auto active = load_acquire(control_->active_epoch);
            if (active > epoch) poison_and_throw(kErrorProtocol, "rank joined an epoch older than the active epoch");
            return active == epoch;
          },
          [&] { return load_relaxed(control_->active_epoch); });
      if (bits_to_signed(load_relaxed(control_->layer_id)) != layer_id ||
          load_relaxed(control_->expert_count) != static_cast<std::uint64_t>(expert_count)) {
        poison_and_throw(kErrorProtocol, "ranks supplied inconsistent layer_id or expert_count");
      }
    }

    store_release(control_->join_epoch[rank_], epoch);
  }

  FP8LayerwiseStats wait(std::uint64_t epoch) {
    ensure_open();
    throw_if_poisoned(control_);
    if (epoch == 0 || load_acquire(control_->join_epoch[rank_]) != epoch) {
      poison_and_throw(kErrorProtocol, "wait called before this rank joined the epoch");
    }

    wait_until(
        "local H2D consumer and producer completion",
        [&] {
          return load_acquire(control_->consumer_done_epoch[rank_]) >= epoch &&
                 load_acquire(control_->producer_done_epoch) >= epoch;
        },
        [&] {
          return fold_progress(load_relaxed(control_->consumer_done_epoch[rank_]),
                               load_relaxed(control_->producer_done_epoch));
        });

    FP8LayerwiseStats stats;
    stats.epoch = epoch;
    stats.layer_id = bits_to_signed(load_relaxed(control_->layer_id));
    stats.expert_count = static_cast<int>(load_relaxed(control_->expert_count));
    stats.rank = rank_;
    stats.writer_ms = ns_to_ms(load_relaxed(control_->writer_ns));
    stats.slot_wait_ms = ns_to_ms(load_relaxed(control_->slot_wait_ns));
    stats.h2d_ms = ns_to_ms(load_relaxed(control_->h2d_ns[rank_]));
    stats.total_ms = ns_to_ms(load_relaxed(control_->producer_total_ns));
    stats.bytes = load_relaxed(control_->bytes[rank_]);
    stats.poisoned = load_acquire(control_->poison_state) != 0;
    stats.error_code = static_cast<int>(load_relaxed(control_->error_code));
    stats.error_rank = static_cast<int>(bits_to_signed(load_relaxed(control_->error_rank)));
    stats.error_message = control_->error_message;
    store_release(control_->wait_returned_epoch[rank_], epoch);
    return stats;
  }

  void run_producer(std::uint64_t epoch, std::int64_t layer_id, int expert_count, const Writer& writer) {
    ensure_open();
    throw_if_poisoned(control_);
    if (rank_ != 0) throw std::invalid_argument("run_layerwise_fp8_batch is rank-zero only");
    if (!writer) throw std::invalid_argument("FP8 layerwise writer callback is empty");
    validate_call(epoch, layer_id, expert_count);
    if (load_acquire(control_->active_epoch) != epoch || bits_to_signed(load_relaxed(control_->layer_id)) != layer_id ||
        load_relaxed(control_->expert_count) != static_cast<std::uint64_t>(expert_count) ||
        load_acquire(control_->join_epoch[0]) != epoch) {
      poison_and_throw(kErrorProtocol, "producer arguments do not match the joined layer");
    }

    const auto producer_begin = Clock::now();
    try {
      wait_until(
          "all ranks joining the layer",
          [&] {
            for (int rank = 0; rank < tp_size_; ++rank) {
              if (load_acquire(control_->join_epoch[rank]) != epoch) return false;
            }
            return true;
          },
          [&] {
            std::uint64_t progress = 0;
            for (int rank = 0; rank < tp_size_; ++rank) {
              progress = fold_progress(progress, load_relaxed(control_->join_epoch[rank]));
            }
            return progress;
          });

      std::array<std::uint64_t, kFP8LayerwiseHostSlots> published_sequence{};
      std::array<std::vector<std::uintptr_t>, kFP8LayerwiseBufferKinds> ptrs;
      for (auto& values : ptrs) values.resize(tp_size_);
      std::uint64_t writer_ns = 0;
      std::uint64_t slot_wait_ns = 0;
      for (int expert = 0; expert < expert_count; ++expert) {
        const int slot = expert % kFP8LayerwiseHostSlots;
        if (published_sequence[slot] != 0) {
          const auto wait_begin = Clock::now();
          wait_for_slot_ack(slot, published_sequence[slot]);
          slot_wait_ns += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - wait_begin).count());
        }

        for (int rank = 0; rank < tp_size_; ++rank) {
          for (int kind = 0; kind < kFP8LayerwiseBufferKinds; ++kind) {
            ptrs[kind][rank] =
                all_rank_host_ptrs_[((slot * tp_size_ + rank) * kFP8LayerwiseBufferKinds) + kind];
          }
        }

        const auto writer_begin = Clock::now();
        writer(expert, ptrs[0], ptrs[1], ptrs[2], ptrs[3]);
        writer_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - writer_begin).count());

        const std::uint64_t sequence = fetch_add_relaxed(control_->next_sequence, 1) + 1;
        store_relaxed(control_->ready[slot].epoch, epoch);
        store_relaxed(control_->ready[slot].expert_id, static_cast<std::uint64_t>(expert));
        __atomic_thread_fence(__ATOMIC_RELEASE);
        store_release(control_->ready[slot].sequence, sequence);
        published_sequence[slot] = sequence;
      }

      for (int slot = 0; slot < kFP8LayerwiseHostSlots; ++slot) {
        if (published_sequence[slot] == 0) continue;
        const auto wait_begin = Clock::now();
        wait_for_slot_ack(slot, published_sequence[slot]);
        slot_wait_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - wait_begin).count());
      }

      const auto total_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - producer_begin).count());
      store_relaxed(control_->writer_ns, writer_ns);
      store_relaxed(control_->slot_wait_ns, slot_wait_ns);
      store_relaxed(control_->producer_total_ns, total_ns);
      __atomic_thread_fence(__ATOMIC_RELEASE);
      store_release(control_->producer_done_epoch, epoch);
    } catch (const std::exception& error) {
      if (load_acquire(control_->poison_state) == 0) publish_poison(control_, kErrorWriter, rank_, error.what());
      throw;
    } catch (...) {
      publish_poison(control_, kErrorWriter, rank_, "unknown exception in FP8 layerwise writer");
      throw;
    }
  }

  void close() noexcept {
    bool expected = false;
    if (!closed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
    const auto active = load_acquire(control_->active_epoch);
    if (active != 0 && load_acquire(control_->consumer_done_epoch[rank_]) < active &&
        load_acquire(control_->poison_state) == 0) {
      publish_poison(control_, kErrorClosedWhileActive, rank_, "transport closed while a layer was active");
    }
    stopping_.store(true, std::memory_order_release);
    if (worker_.joinable()) worker_.join();
  }

  int rank() const noexcept { return rank_; }
  int tp_size() const noexcept { return tp_size_; }
  int num_experts() const noexcept { return num_experts_; }
  bool closed() const noexcept { return closed_.load(std::memory_order_acquire); }

 private:
  void validate_call(std::uint64_t epoch, std::int64_t layer_id, int expert_count) const {
    if (epoch == 0) throw std::invalid_argument("FP8 layerwise epoch zero is reserved");
    if (layer_id < 0) throw std::invalid_argument("FP8 layerwise layer_id must be non-negative");
    if (expert_count <= 0 || expert_count > num_experts_) {
      throw std::invalid_argument("FP8 layerwise expert_count is outside the configured expert range");
    }
  }

  void ensure_open() const {
    if (closed_.load(std::memory_order_acquire)) throw std::runtime_error("FP8 layerwise transport is closed");
  }

  [[noreturn]] void poison_and_throw(int code, const std::string& message) {
    publish_poison(control_, code, rank_, message);
    throw std::runtime_error(message);
  }

  template <typename Predicate, typename Progress>
  void wait_until(const char* description, Predicate&& predicate, Progress&& progress) {
    auto last_progress_time = Clock::now();
    std::uint64_t last_progress = progress();
    std::uint32_t spins = 0;
    while (!predicate()) {
      throw_if_poisoned(control_);
      if (stopping_.load(std::memory_order_acquire)) throw std::runtime_error("FP8 layerwise transport is stopping");
      const auto current_progress = progress();
      if (current_progress != last_progress) {
        last_progress = current_progress;
        last_progress_time = Clock::now();
      } else if (Clock::now() - last_progress_time >= timeout_) {
        std::ostringstream message;
        message << "timed out after " << timeout_.count() << " ms without progress while waiting for " << description;
        poison_and_throw(kErrorTimeout, message.str());
      }
      if (++spins < 2048) {
        std::this_thread::yield();
      } else {
        spins = 0;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
    throw_if_poisoned(control_);
  }

  void wait_for_slot_ack(int slot, std::uint64_t sequence) {
    wait_until(
        "all ranks acknowledging a reused host slot",
        [&] {
          for (int rank = 0; rank < tp_size_; ++rank) {
            if (load_acquire(control_->ack_sequence[rank][slot]) < sequence) return false;
          }
          return true;
        },
        [&] {
          std::uint64_t progress = 0;
          for (int rank = 0; rank < tp_size_; ++rank) {
            progress = fold_progress(progress, load_relaxed(control_->ack_sequence[rank][slot]));
          }
          return progress;
        });
  }

#if KT_FP8_LAYERWISE_HAS_CUDA
  static void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }

  void initialize_cuda_worker() {
    check_cuda(cudaSetDevice(cuda_device_), "cudaSetDevice");
    check_cuda(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    for (int slot = 0; slot < kFP8LayerwiseHostSlots; ++slot) {
      check_cuda(cudaEventCreateWithFlags(&slot_events_[slot], cudaEventDisableTiming), "cudaEventCreateWithFlags");
    }
  }

  void destroy_cuda_worker() noexcept {
    for (auto& event : slot_events_) {
      if (event != nullptr) {
        cudaEventDestroy(event);
        event = nullptr;
      }
    }
    if (copy_stream_ != nullptr) {
      cudaStreamDestroy(copy_stream_);
      copy_stream_ = nullptr;
    }
  }

  void copy_expert(int expert, int slot) {
    const auto begin = Clock::now();
    for (int kind = 0; kind < kFP8LayerwiseBufferKinds; ++kind) {
      if (expert_nbytes_[kind] > std::numeric_limits<std::uintptr_t>::max() / static_cast<std::size_t>(num_experts_)) {
        throw std::overflow_error("FP8 layerwise GPU expert offset overflows uintptr_t");
      }
      const auto destination = local_gpu_ptrs_[kind] + static_cast<std::uintptr_t>(expert) * expert_nbytes_[kind];
      const auto source = local_host_ptrs_[slot * kFP8LayerwiseBufferKinds + kind];
      check_cuda(cudaMemcpyAsync(reinterpret_cast<void*>(destination), reinterpret_cast<const void*>(source),
                                 expert_nbytes_[kind], cudaMemcpyHostToDevice, copy_stream_),
                 "cudaMemcpyAsync(H2D)");
    }
    check_cuda(cudaEventRecord(slot_events_[slot], copy_stream_), "cudaEventRecord");
    wait_until(
        "the local H2D completion event",
        [&] {
          const auto status = cudaEventQuery(slot_events_[slot]);
          if (status == cudaSuccess) return true;
          if (status == cudaErrorNotReady) return false;
          check_cuda(status, "cudaEventQuery");
          return false;
        },
        [] { return std::uint64_t{0}; });
    local_h2d_ns_ += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
    for (auto size : expert_nbytes_) local_bytes_ += size;
  }
#endif

  void consumer_main() noexcept {
#if KT_FP8_LAYERWISE_HAS_CUDA
    try {
      initialize_cuda_worker();
      {
        std::lock_guard lock(init_mutex_);
        init_complete_ = true;
      }
      init_cv_.notify_one();

      std::uint64_t completed_epoch = load_acquire(control_->consumer_done_epoch[rank_]);
      while (!stopping_.load(std::memory_order_acquire)) {
        const auto epoch = load_acquire(control_->active_epoch);
        if (epoch == 0 || epoch <= completed_epoch || load_acquire(control_->join_epoch[rank_]) != epoch) {
          throw_if_poisoned(control_);
          std::this_thread::sleep_for(std::chrono::microseconds(50));
          continue;
        }

        const int expert_count = static_cast<int>(load_relaxed(control_->expert_count));
        if (expert_count <= 0 || expert_count > num_experts_) {
          poison_and_throw(kErrorProtocol, "consumer observed an invalid expert_count");
        }
        local_h2d_ns_ = 0;
        local_bytes_ = 0;

        for (int expert = 0; expert < expert_count; ++expert) {
          const int slot = expert % kFP8LayerwiseHostSlots;
          std::uint64_t observed_sequence = 0;
          wait_until(
              "the next expert generation",
              [&] {
                const auto active = load_acquire(control_->active_epoch);
                if (active > epoch) poison_and_throw(kErrorProtocol, "active epoch advanced before consumer completed");
                const auto sequence = load_acquire(control_->ready[slot].sequence);
                if (sequence == last_ready_sequence_[slot]) return false;
                if (load_relaxed(control_->ready[slot].epoch) != epoch ||
                    load_relaxed(control_->ready[slot].expert_id) != static_cast<std::uint64_t>(expert)) {
                  return false;
                }
                observed_sequence = sequence;
                return true;
              },
              [&] { return load_relaxed(control_->ready[slot].sequence); });

          copy_expert(expert, slot);
          last_ready_sequence_[slot] = observed_sequence;
          __atomic_thread_fence(__ATOMIC_RELEASE);
          store_release(control_->ack_sequence[rank_][slot], observed_sequence);
        }

        store_relaxed(control_->h2d_ns[rank_], local_h2d_ns_);
        store_relaxed(control_->bytes[rank_], local_bytes_);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        store_release(control_->consumer_done_epoch[rank_], epoch);
        completed_epoch = epoch;
      }
    } catch (const std::exception& error) {
      bool initialization_failure = false;
      {
        std::lock_guard lock(init_mutex_);
        if (!init_complete_) {
          initialization_failure = true;
          init_error_ = error.what();
          init_complete_ = true;
        }
      }
      if (initialization_failure) init_cv_.notify_one();
      if (!stopping_.load(std::memory_order_acquire)) {
        publish_poison(control_, initialization_failure ? kErrorCudaInitialization : kErrorCudaCopy, rank_,
                       error.what());
      }
    } catch (...) {
      bool initialization_failure = false;
      {
        std::lock_guard lock(init_mutex_);
        if (!init_complete_) {
          initialization_failure = true;
          init_error_ = "unknown CUDA worker initialization error";
          init_complete_ = true;
        }
      }
      if (initialization_failure) init_cv_.notify_one();
      if (!stopping_.load(std::memory_order_acquire)) {
        publish_poison(control_, initialization_failure ? kErrorCudaInitialization : kErrorCudaCopy, rank_,
                       initialization_failure ? "unknown CUDA worker initialization error"
                                              : "unknown exception in FP8 layerwise H2D consumer");
      }
    }
    destroy_cuda_worker();
#endif
  }

  FP8LayerwiseControl* control_;
  int rank_;
  int tp_size_;
  int cuda_device_;
  int num_experts_;
  std::chrono::milliseconds timeout_;
  std::array<std::uintptr_t, kFP8LayerwiseHostSlots * kFP8LayerwiseBufferKinds> local_host_ptrs_{};
  std::array<std::uintptr_t, kFP8LayerwiseBufferKinds> local_gpu_ptrs_{};
  std::vector<std::uintptr_t> all_rank_host_ptrs_;
  std::array<std::size_t, kFP8LayerwiseBufferKinds> expert_nbytes_{};
  std::array<std::uint64_t, kFP8LayerwiseHostSlots> last_ready_sequence_{};

  std::atomic<bool> stopping_{false};
  std::atomic<bool> closed_{false};
  std::thread worker_;
  std::mutex init_mutex_;
  std::condition_variable init_cv_;
  bool init_complete_ = false;
  std::string init_error_;
  std::uint64_t local_h2d_ns_ = 0;
  std::uint64_t local_bytes_ = 0;

#if KT_FP8_LAYERWISE_HAS_CUDA
  cudaStream_t copy_stream_ = nullptr;
  std::array<cudaEvent_t, kFP8LayerwiseHostSlots> slot_events_{};
#endif
};

FP8LayerwiseTransport::FP8LayerwiseTransport(
    std::uintptr_t control_ptr, std::size_t control_size, int rank, int tp_size, int cuda_device,
    const std::vector<std::uintptr_t>& local_host_ptrs, const std::vector<std::uintptr_t>& local_gpu_ptrs,
    const std::vector<std::uintptr_t>& all_rank_host_ptrs, const std::vector<std::size_t>& expert_nbytes,
    int num_experts, std::uint64_t timeout_ms)
    : impl_(std::make_unique<Impl>(control_ptr, control_size, rank, tp_size, cuda_device, local_host_ptrs,
                                   local_gpu_ptrs, all_rank_host_ptrs, expert_nbytes, num_experts, timeout_ms)) {}

FP8LayerwiseTransport::~FP8LayerwiseTransport() = default;

void FP8LayerwiseTransport::join(std::uint64_t epoch, std::int64_t layer_id, int expert_count) {
  impl_->join(epoch, layer_id, expert_count);
}

FP8LayerwiseStats FP8LayerwiseTransport::wait(std::uint64_t epoch) { return impl_->wait(epoch); }

void FP8LayerwiseTransport::close() { impl_->close(); }

void FP8LayerwiseTransport::run_producer(std::uint64_t epoch, std::int64_t layer_id, int expert_count,
                                         const Writer& writer) {
  impl_->run_producer(epoch, layer_id, expert_count, writer);
}

int FP8LayerwiseTransport::rank() const noexcept { return impl_->rank(); }

int FP8LayerwiseTransport::tp_size() const noexcept { return impl_->tp_size(); }

int FP8LayerwiseTransport::num_experts() const noexcept { return impl_->num_experts(); }

bool FP8LayerwiseTransport::closed() const noexcept { return impl_->closed(); }

}  // namespace kt::layerwise
