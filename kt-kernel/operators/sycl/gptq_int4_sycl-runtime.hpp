/**
 * @Description : Common SYCL runtime support for the GPTQ INT4 backend
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 */
#ifndef CPUINFER_OPERATOR_SYCL_GPTQ_INT4_RUNTIME_H
#define CPUINFER_OPERATOR_SYCL_GPTQ_INT4_RUNTIME_H

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>

namespace sycl_int4 {

constexpr int kSubgroupSize = 16;
constexpr int kDownRowsPerWorkGroup = 2;
constexpr int kPrefillSparseRows = 2;
constexpr int kPrefillDenseRows = 8;
constexpr int kPrefillDenseThreshold = 33;

class AsyncErrorState {
 public:
  void capture(sycl::exception_list exceptions) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_) return;
    for (const auto& exception : exceptions) {
      pending_ = exception;
      break;
    }
  }

  void rethrow_if_pending() {
    std::exception_ptr pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending.swap(pending_);
    }
    if (pending) std::rethrow_exception(pending);
  }

 private:
  std::mutex mutex_;
  std::exception_ptr pending_;
};

inline AsyncErrorState& async_error_state() {
  static AsyncErrorState state;
  return state;
}

inline sycl::queue& queue() {
  // Construct error storage first so it outlives the queue and its handler.
  (void)async_error_state();
  static sycl::queue q([] {
    auto async_handler = [](sycl::exception_list exceptions) { async_error_state().capture(std::move(exceptions)); };

    try {
      const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
      if (devices.empty()) {
        throw std::runtime_error("No SYCL GPU device is available");
      }
      for (const auto& device : devices) {
        if (device.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero) {
          return sycl::queue(device, async_handler);
        }
      }
      return sycl::queue(devices.front(), async_handler);
    } catch (const sycl::exception& error) {
      throw std::runtime_error(std::string("Failed to create the SYCL GPTQ INT4 queue. Check sycl-ls, "
                                           "the render-group permission, or ONEAPI_DEVICE_SELECTOR. Original error: ") +
                               error.what());
    }
  }());
  return q;
}

inline void wait_and_throw(sycl::event& event) {
  event.wait_and_throw();
  async_error_state().rethrow_if_pending();
}

template <typename T>
inline T* usm_alloc(size_t elements, const char* name) {
  elements = std::max<size_t>(elements, 1);
  T* pointer = sycl::malloc_shared<T>(elements, queue());
  if (pointer == nullptr) {
    throw std::runtime_error(std::string("SYCL shared-USM allocation failed for ") + name);
  }
  return pointer;
}

inline void usm_free(void* pointer) {
  if (pointer != nullptr) sycl::free(pointer, queue());
}

inline float bf16_to_fp32(uint16_t value) { return sycl::bit_cast<float>(static_cast<uint32_t>(value) << 16); }

inline uint16_t fp32_to_bf16(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

inline int32_t unpack_i4x4_to_i8x4(uint32_t packed) {
  uint32_t value = packed & 0xffffu;
  value = (value | (value << 8)) & 0x00ff00ffu;
  value = (value | (value << 4)) & 0x0f0f0f0fu;
  const uint32_t negative = (~value) & 0x08080808u;
  value = (value & 0x07070707u) | negative | (negative << 1) | (negative << 2) | (negative << 3) | (negative << 4);
  return sycl::bit_cast<int32_t>(value);
}

}  // namespace sycl_int4

#endif  // CPUINFER_OPERATOR_SYCL_GPTQ_INT4_RUNTIME_H
